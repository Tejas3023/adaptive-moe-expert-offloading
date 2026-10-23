
import time

import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher


LAYER_ID = 2
PREFETCH_START_LAYER = 1
PROMPT = "The future of artificial intelligence is"


class PrefetchedWeightMap:
    def __init__(self, original, staged):
        self.original = original
        self.staged = staged
        self.staged_hits = 0
        self.fallback_reads = 0

    def __getitem__(self, key):
        if key in self.staged:
            self.staged_hits += 1
            return self.staged[key]
        self.fallback_reads += 1
        return self.original[key]

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)

    def keys(self):
        return self.original.keys()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prime_predictor(model, predictor, input_ids):
    moe = model.model.layers[LAYER_ID].mlp

    def capture(module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return

        router_logits = output[1]
        if not isinstance(router_logits, torch.Tensor):
            return

        selected = torch.topk(
            router_logits.reshape(-1, router_logits.shape[-1]),
            k=8,
            dim=-1,
        ).indices

        for row in selected:
            predictor.observe(LAYER_ID, row.detach().cpu().tolist())

    handle = moe.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, output_router_logits=True)
    finally:
        handle.remove()


def get_projection_modules(model, expert_id):
    expert = model.model.layers[LAYER_ID].mlp.experts[expert_id]
    return {
        "gate_proj": expert.gate_proj,
        "up_proj": expert.up_proj,
        "down_proj": expert.down_proj,
    }


def start_staging(model, expert_id):
    """
    Prepare CPU weights and enqueue copies on a separate CUDA stream.
    Returns before the copies necessarily finish.
    """
    modules = get_projection_modules(model, expert_id)
    originals = {}
    wrappers = {}
    pinned_sources = []
    staged_tensors = []

    stream = torch.cuda.Stream()

    try:
        with torch.cuda.stream(stream):
            for name, module in modules.items():
                hook = module._hf_hook
                if not hasattr(hook, "weights_map"):
                    raise RuntimeError(
                        f"{name} has no Accelerate weights_map."
                    )

                original = hook.weights_map
                source = original["weight"]

                if source.device.type == "meta":
                    raise RuntimeError(
                        f"{name} source weight is meta."
                    )

                if source.device.type != "cpu":
                    source = source.cpu()

                pinned = source.contiguous().pin_memory()
                staged = pinned.to("cuda", non_blocking=True)

                originals[module] = original
                wrappers[name] = PrefetchedWeightMap(
                    original, {"weight": staged}
                )
                pinned_sources.append(pinned)
                staged_tensors.append(staged)

        event = torch.cuda.Event()
        event.record(stream)

        return {
            "originals": originals,
            "wrappers": wrappers,
            "pinned_sources": pinned_sources,
            "staged_tensors": staged_tensors,
            "stream": stream,
            "event": event,
            "expert_id": expert_id,
        }

    except Exception:
        for module, original in originals.items():
            module._hf_hook.weights_map = original
        raise


def install_staged_maps(model, staging):
    # Do not expose staged tensors to Accelerate until their copies finish.
    staging["event"].synchronize()

    modules = get_projection_modules(model, staging["expert_id"])
    for name, module in modules.items():
        module._hf_hook.weights_map = staging["wrappers"][name]


def restore_staged_maps(staging):
    for module, original in staging["originals"].items():
        module._hf_hook.weights_map = original


def run_forward(model, input_ids):
    with torch.inference_mode():
        return model(input_ids=input_ids).logits.detach().float().cpu()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    print("Loading model...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    prompt_ids = tokenizer(
        PROMPT, return_tensors="pt"
    )["input_ids"].to("cuda")
    target_ids = prompt_ids[:, -1:]

    predictor = LayerAwareHistoryPrefetcher(
        history_size=8, prefetch_size=1
    )
    print("Priming routing history...")
    prime_predictor(model, predictor, prompt_ids)

    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("No predicted expert available.")

    expert_id = candidates[0]
    print(f"Predicted expert for layer {LAYER_ID}: {expert_id}")

    # Baseline reference, for correctness comparison.
    sync()
    start = time.perf_counter()
    baseline_logits = run_forward(model, target_ids)
    sync()
    baseline_time = time.perf_counter() - start
    print(f"Baseline forward: {baseline_time:.3f} s")

    # Demonstration of separate-stream staging during earlier-layer compute.
    # This runs the model up to the target layer, then begins staging.
    # We time the whole operation, including staging and the target layer.
    staging = None
    pre_handle = None
    layer1 = model.model.layers[PREFETCH_START_LAYER]
    layer2 = model.model.layers[LAYER_ID]

    state = {"started": False}

    def prefetch_during_layer1(module, inputs):
        if state["started"]:
            return
        state["started"] = True
        print("Starting asynchronous staging during layer 1...")
        state["staging_start"] = time.perf_counter()
        state["staging"] = start_staging(model, expert_id)

    def install_before_layer2(module, inputs):
        if not state["started"]:
            raise RuntimeError("Prefetch was not started.")
        install_staged_maps(model, state["staging"])

    def restore_after_layer2(module, inputs, output):
        if state.get("staging") is not None:
            restore_staged_maps(state["staging"])

    # Pre-hook on layer 1 begins staging after layer 1 receives its input.
    # Layer 1's computation then proceeds while the copy is enqueued.
    pre_handle = layer1.register_forward_pre_hook(prefetch_during_layer1)
    install_handle = layer2.register_forward_pre_hook(install_before_layer2)
    restore_handle = layer2.register_forward_hook(restore_after_layer2)

    sync()
    start = time.perf_counter()
    try:
        prefetch_logits = run_forward(model, target_ids)
        sync()
        prefetch_time = time.perf_counter() - start
    finally:
        pre_handle.remove()
        install_handle.remove()
        restore_handle.remove()

        if state.get("staging") is not None:
            restore_staged_maps(state["staging"])
            state["staging"]["event"].synchronize()

    diff = (baseline_logits - prefetch_logits).abs().max().item()

    print("\n=== ASYNC PREFETCH RESULT ===")
    print(f"Predicted expert: {expert_id}")
    print(f"Baseline: {baseline_time:.3f} s")
    print(f"Async path: {prefetch_time:.3f} s")
    print(f"Max logit difference: {diff:.8g}")

    if state.get("staging") is not None:
        staging = state["staging"]
        print("\nProjection usage:")
        for name, wrapper in staging["wrappers"].items():
            print(
                f"{name}: hits={wrapper.staged_hits}, "
                f"fallbacks={wrapper.fallback_reads}"
            )

    print(
        "\nNote: this prototype tests separate-stream staging. "
        "It does not yet prove a speedup; the CPU weight lookup and "
        "pinning still happen synchronously."
    )


if __name__ == "__main__":
    main()
