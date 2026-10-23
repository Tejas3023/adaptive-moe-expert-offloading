import statistics
import time

import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher


LAYER_ID = 2
NUM_TRIALS = 3
PROMPT = "The future of artificial intelligence is"


class PrefetchedWeightMap:
    """Serve staged weights and fall back to the original weight map."""

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
    """Observe real router selections from the prompt."""

    moe = model.model.layers[LAYER_ID].mlp

    def capture_routing(module, inputs, output):
        if not isinstance(output, tuple) or len(output) < 2:
            return

        router_logits = output[1]
        if not isinstance(router_logits, torch.Tensor):
            return

        # Router logits: [tokens, experts] or [batch, tokens, experts].
        logits = router_logits.reshape(-1, router_logits.shape[-1])
        selected = torch.topk(logits, k=8, dim=-1).indices

        for row in selected:
            predictor.observe(
                LAYER_ID,
                row.detach().cpu().tolist(),
            )

    handle = moe.register_forward_hook(capture_routing)

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, output_router_logits=True)
    finally:
        handle.remove()


def stage_expert(model, expert_id):
    """Stage all three projection weights and temporarily wrap their maps."""

    expert = model.model.layers[LAYER_ID].mlp.experts[expert_id]
    projections = {
        "gate_proj": expert.gate_proj,
        "up_proj": expert.up_proj,
        "down_proj": expert.down_proj,
    }

    originals = {}
    wrappers = {}
    staged_tensors = []

    try:
        for name, module in projections.items():
            hook = module._hf_hook

            if not hasattr(hook, "weights_map"):
                raise RuntimeError(
                    f"{name} has no Accelerate weights_map."
                )

            original_map = hook.weights_map
            source = original_map["weight"]

            if source.device.type == "meta":
                raise RuntimeError(
                    f"{name} source weight is on meta; cannot stage it."
                )

            if source.device.type != "cpu":
                source = source.cpu()

            # Pinning enables an asynchronous host-to-GPU copy.
            pinned = source.contiguous().pin_memory()
            staged = pinned.to("cuda", non_blocking=True)
            staged_tensors.append(staged)

            wrapper = PrefetchedWeightMap(
                original_map,
                {"weight": staged},
            )

            originals[module] = original_map
            wrappers[name] = wrapper
            hook.weights_map = wrapper

        sync()
        return originals, wrappers, staged_tensors

    except Exception:
        for module, original in originals.items():
            module._hf_hook.weights_map = original
        raise


def restore_maps(originals):
    for module, original in originals.items():
        module._hf_hook.weights_map = original


def timed_forward(model, input_ids):
    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model(input_ids=input_ids)

    sync()
    elapsed = time.perf_counter() - start

    return output.logits.detach().float().cpu(), elapsed


def timed_prefetch_forward(model, input_ids, expert_id):
    sync()
    start = time.perf_counter()

    originals = {}
    wrappers = {}
    staged_tensors = []

    try:
        originals, wrappers, staged_tensors = stage_expert(
            model, expert_id
        )

        with torch.inference_mode():
            output = model(input_ids=input_ids)

        sync()
        elapsed = time.perf_counter() - start
        logits = output.logits.detach().float().cpu()

        usage = {
            name: {
                "hits": wrapper.staged_hits,
                "fallbacks": wrapper.fallback_reads,
            }
            for name, wrapper in wrappers.items()
        }

        return logits, elapsed, usage

    finally:
        restore_maps(originals)
        wrappers.clear()
        staged_tensors.clear()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark expects a GPU.")

    print("Loading tokenizer and model...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    prompt_ids = tokenizer(
        PROMPT,
        return_tensors="pt",
    )["input_ids"].to("cuda")

    # The experiment compares the same final-token input every time.
    target_ids = prompt_ids[:, -1:]

    predictor = LayerAwareHistoryPrefetcher(
        history_size=8,
        prefetch_size=1,
    )

    print("\nPriming the predictor with the prompt...")
    prime_predictor(model, predictor, prompt_ids)

    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("Predictor produced no expert candidate.")

    predicted_expert = candidates[0]

    print(f"Layer: {LAYER_ID}")
    print(f"Frozen predicted expert: {predicted_expert}")
    print(f"History entries: {len(predictor)}")

    # Warm up both paths before collecting measurements.
    print("\nWarm-up: baseline...")
    baseline_reference, _ = timed_forward(model, target_ids)

    print("Warm-up: staged expert...")
    staged_reference, _, warmup_usage = timed_prefetch_forward(
        model, target_ids, predicted_expert
    )

    warmup_diff = (
        baseline_reference - staged_reference
    ).abs().max().item()

    print(f"Warm-up max logit difference: {warmup_diff:.8g}")
    print(f"Warm-up staged usage: {warmup_usage}")

    baseline_times = []
    prefetch_times = []
    all_usages = []
    max_logit_diff = 0.0

    # Alternate order to reduce systematic order/cache bias.
    for trial in range(NUM_TRIALS):
        print(f"\nTrial {trial + 1}/{NUM_TRIALS}")

        if trial % 2 == 0:
            baseline_logits, baseline_s = timed_forward(
                model, target_ids
            )
            prefetch_logits, prefetch_s, usage = (
                timed_prefetch_forward(
                    model, target_ids, predicted_expert
                )
            )
        else:
            prefetch_logits, prefetch_s, usage = (
                timed_prefetch_forward(
                    model, target_ids, predicted_expert
                )
            )
            baseline_logits, baseline_s = timed_forward(
                model, target_ids
            )

        diff = (
            baseline_logits - prefetch_logits
        ).abs().max().item()

        max_logit_diff = max(max_logit_diff, diff)
        baseline_times.append(baseline_s)
        prefetch_times.append(prefetch_s)
        all_usages.append(usage)

        print(f"Baseline: {baseline_s:.3f} s")
        print(f"Prefetch path (including staging): {prefetch_s:.3f} s")
        print(f"Max logit difference: {diff:.8g}")
        print(f"Staged usage: {usage}")

    baseline_median = statistics.median(baseline_times)
    prefetch_median = statistics.median(prefetch_times)

    improvement = (
        (baseline_median - prefetch_median) / baseline_median * 100
    )

    print("\n" + "=" * 65)
    print("CONTROLLED BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Frozen candidate: expert {predicted_expert}, layer {LAYER_ID}")
    print(f"Trials per path: {NUM_TRIALS}")
    print(f"Baseline times: {[round(x, 3) for x in baseline_times]}")
    print(f"Prefetch times: {[round(x, 3) for x in prefetch_times]}")
    print(f"Baseline median: {baseline_median:.3f} s")
    print(f"Prefetch median: {prefetch_median:.3f} s")
    print(f"Median time change: {improvement:+.2f}%")
    print(f"Maximum logit difference: {max_logit_diff:.8g}")

    print("\nMedian-based interpretation:")
    if improvement > 0:
        print("The staged path was faster in this small benchmark.")
    elif improvement < 0:
        print("The staged path was slower in this small benchmark.")
    else:
        print("The median times were equal.")

    print(
        "\nImportant: this measures synchronous staging plus inference. "
        "It does not demonstrate transfer/compute overlap or prove "
        "a general end-to-end speedup."
    )


if __name__ == "__main__":
    main()