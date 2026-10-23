
import time
import statistics
import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher

LAYER_ID = 2
PREFETCH_START_LAYER = 1
PROMPT = "The future of artificial intelligence is"
TRIALS = 3


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
    modules = get_projection_modules(model, expert_id)
    originals = {}
    wrappers = {}
    pinned_sources = []
    staged_tensors = []
    stream = torch.cuda.Stream()

    # CPU lookup and pinning are measured separately from the GPU copies.
    prep_start = time.perf_counter()
    sources = {}

    for name, module in modules.items():
        hook = module._hf_hook
        if not hasattr(hook, "weights_map"):
            raise RuntimeError(f"{name} has no Accelerate weights_map.")

        original = hook.weights_map
        source = original["weight"]

        if source.device.type == "meta":
            raise RuntimeError(f"{name} source weight is meta.")
        if source.device.type != "cpu":
            source = source.cpu()

        pinned = source.contiguous().pin_memory()
        sources[name] = (module, original, pinned)
        pinned_sources.append(pinned)

    cpu_prepare_ms = (time.perf_counter() - prep_start) * 1000

    # CUDA events measure time spent executing copies on this stream.
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    enqueue_start = time.perf_counter()

    try:
        with torch.cuda.stream(stream):
            copy_start.record(stream)

            for name, (module, original, pinned) in sources.items():
                staged = pinned.to("cuda", non_blocking=True)

                originals[module] = original
                wrappers[name] = PrefetchedWeightMap(
                    original, {"weight": staged}
                )
                staged_tensors.append(staged)

            copy_end.record(stream)

        enqueue_ms = (time.perf_counter() - enqueue_start) * 1000

        return {
            "originals": originals,
            "wrappers": wrappers,
            "pinned_sources": pinned_sources,
            "staged_tensors": staged_tensors,
            "stream": stream,
            "copy_start": copy_start,
            "copy_end": copy_end,
            "cpu_prepare_ms": cpu_prepare_ms,
            "enqueue_ms": enqueue_ms,
            "wait_ms": None,
            "copy_ms": None,
            "expert_id": expert_id,
        }

    except Exception:
        for module, original in originals.items():
            module._hf_hook.weights_map = original
        raise


def install_staged_maps(staging):
    wait_start = time.perf_counter()

    # Wait only when layer 2 is about to need these weights.
    staging["copy_end"].synchronize()
    staging["wait_ms"] = (time.perf_counter() - wait_start) * 1000
    staging["copy_ms"] = staging["copy_start"].elapsed_time(
        staging["copy_end"]
    )

    modules = get_projection_modules_for_staging(
        staging, staging["expert_id"]
    )
    for name, module in modules.items():
        module._hf_hook.weights_map = staging["wrappers"][name]


def get_projection_modules_for_staging(staging, expert_id):
    # Recover the three modules from the original hook-map keys.
    return {
        name: module
        for name, module in staging["module_by_name"].items()
    }


def restore_staged_maps(staging):
    for module, original in staging["originals"].items():
        module._hf_hook.weights_map = original


def run_forward(model, input_ids):
    with torch.inference_mode():
        return model(input_ids=input_ids).logits.detach().float().cpu()


def run_async_trial(model, input_ids, expert_id):
    layer1 = model.model.layers[PREFETCH_START_LAYER]
    layer2 = model.model.layers[LAYER_ID]
    state = {"staging": None}

    # Save module references for installation and diagnostics.
    modules = get_projection_modules(model, expert_id)

    def start_hook(module, inputs):
        state["staging"] = start_staging(model, expert_id)
        state["staging"]["module_by_name"] = modules

    def install_hook(module, inputs):
        install_staged_maps(state["staging"])

    def restore_hook(module, inputs, output):
        restore_staged_maps(state["staging"])

    h1 = layer1.register_forward_pre_hook(start_hook)
    h2 = layer2.register_forward_pre_hook(install_hook)
    h3 = layer2.register_forward_hook(restore_hook)

    sync()
    start = time.perf_counter()

    try:
        logits = run_forward(model, input_ids)
        sync()
        total_ms = (time.perf_counter() - start) * 1000
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

        if state["staging"] is not None:
            restore_staged_maps(state["staging"])
            state["staging"]["copy_end"].synchronize()

    return logits, total_ms, state["staging"]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("Loading OLMoE...")
    tokenizer = load_tokenizer()
    model = load_olmoe_model()
    model.eval()

    input_ids = tokenizer(
        PROMPT, return_tensors="pt"
    )["input_ids"].to("cuda")
    target_ids = input_ids[:, -1:]

    predictor = LayerAwareHistoryPrefetcher(
        history_size=8, prefetch_size=1
    )
    print("Priming predictor...")
    prime_predictor(model, predictor, input_ids)

    candidates = predictor.predict(LAYER_ID)
    if not candidates:
        raise RuntimeError("Predictor produced no candidates.")

    expert_id = candidates[0]
    print(f"Predicted expert: {expert_id}")

    print("Warming up baseline...")
    run_forward(model, target_ids)
    sync()

    baseline_times = []
    async_times = []
    max_diffs = []
    diagnostic_rows = []

    for trial in range(1, TRIALS + 1):
        print(f"\nTrial {trial}/{TRIALS}")

        sync()
        start = time.perf_counter()
        baseline_logits = run_forward(model, target_ids)
        sync()
        baseline_ms = (time.perf_counter() - start) * 1000
        baseline_times.append(baseline_ms)

        async_logits, async_ms, staging = run_async_trial(
            model, target_ids, expert_id
        )
        async_times.append(async_ms)

        diff = (baseline_logits - async_logits).abs().max().item()
        max_diffs.append(diff)

        diagnostic_rows.append({
            "cpu_prepare_ms": staging["cpu_prepare_ms"],
            "enqueue_ms": staging["enqueue_ms"],
            "copy_ms": staging["copy_ms"],
            "wait_ms": staging["wait_ms"],
        })

        print(f"  Baseline: {baseline_ms / 1000:.3f} s")
        print(f"  Async:    {async_ms / 1000:.3f} s")
        print(f"  Max logit difference: {diff:.8g}")
        print(
            "  Staging: "
            f"CPU prep={staging['cpu_prepare_ms']:.2f} ms, "
            f"copy enqueue={staging['enqueue_ms']:.2f} ms, "
            f"GPU copies={staging['copy_ms']:.2f} ms, "
            f"wait at layer 2={staging['wait_ms']:.2f} ms"
        )

        for name, wrapper in staging["wrappers"].items():
            print(
                f"  {name}: hits={wrapper.staged_hits}, "
                f"fallbacks={wrapper.fallback_reads}"
            )

    baseline_median = statistics.median(baseline_times)
    async_median = statistics.median(async_times)
    change_pct = (async_median / baseline_median - 1) * 100

    print("\n=== ASYNC PREFETCH DIAGNOSTICS ===")
    print(f"Predicted expert: {expert_id}")
    print(f"Baseline median: {baseline_median / 1000:.3f} s")
    print(f"Async median:    {async_median / 1000:.3f} s")
    print(f"Async latency change: {change_pct:+.2f}%")
    print(f"Maximum logit difference across trials: {max(max_diffs):.8g}")
    print("\nMedian staging metrics:")

    for key in ("cpu_prepare_ms", "enqueue_ms", "copy_ms", "wait_ms"):
        value = statistics.median(row[key] for row in diagnostic_rows)
        print(f"  {key}: {value:.2f} ms")

    print(
        "\nInterpretation: CUDA-event copy time measures GPU-stream copy "
        "duration. The wait measurement is how long the CPU blocks at the "
        "layer-2 synchronization point; it is not a direct measurement of "
        "overlap efficiency. Results are diagnostic, not proof of a speedup."
    )


if __name__ == "__main__":
    main()
