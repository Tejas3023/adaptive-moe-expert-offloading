
import time
import statistics
import torch

from src.modeling.olmoe_loader import load_olmoe_model, load_tokenizer
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher
from scripts.diagnose_async_prefetch import (
    prime_predictor,
    run_forward,
    run_async_trial,
    start_staging,
    sync,
)

LAYER_ID = 2
PREFETCH_START_LAYER = 1
PROMPT = "The future of artificial intelligence is"
TRIALS = 5


def timed_baseline(model, input_ids):
    sync()
    start = time.perf_counter()
    logits = run_forward(model, input_ids)
    sync()
    return logits, (time.perf_counter() - start) * 1000


def timed_transfer_only(model, input_ids, expert_id):
    """Stage weights during layer 1, but never install staged maps."""
    layer1 = model.model.layers[PREFETCH_START_LAYER]
    state = {"staging": None}

    def start_hook(module, inputs):
        state["staging"] = start_staging(model, expert_id)

    handle = layer1.register_forward_pre_hook(start_hook)
    sync()
    start = time.perf_counter()

    try:
        logits = run_forward(model, input_ids)
        sync()  # Ensure the asynchronous copies have finished.
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        handle.remove()
        staging = state["staging"]
        if staging is not None:
            # No maps were replaced in this mode. Ensure copies are done
            # before pinned buffers and staged tensors can be released.
            staging["copy_end"].synchronize()

    return logits, elapsed_ms, state["staging"]


def run_mode(model, input_ids, expert_id, mode):
    if mode == "A":
        logits, elapsed_ms = timed_baseline(model, input_ids)
        return logits, elapsed_ms, None

    if mode == "B":
        return timed_transfer_only(model, input_ids, expert_id)

    if mode == "C":
        logits, elapsed_ms, staging = run_async_trial(
            model, input_ids, expert_id
        )
        return logits, elapsed_ms, staging

    raise ValueError(f"Unknown mode: {mode}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

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

    labels = {
        "A": "Baseline",
        "B": "Transfer only",
        "C": "Staged weights",
    }

    print("Warming up all three modes...")
    warmup_logits = {}
    for mode in ("A", "B", "C"):
        logits, _, _ = run_mode(model, target_ids, expert_id, mode)
        warmup_logits[mode] = logits
    sync()

    print("\nWarm-up output differences:")
    for mode in ("B", "C"):
        diff = (warmup_logits["A"] - warmup_logits[mode]).abs().max().item()
        print(f"  A vs {mode}: {diff:.8g}")

    measurements = {"A": [], "B": [], "C": []}
    max_diffs = []

    # Rotate order to reduce systematic first/last execution bias.
    orders = [
        ("A", "B", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("A", "C", "B"),
        ("B", "A", "C"),
    ]

    for trial, order in enumerate(orders, start=1):
        print(f"\nTrial {trial}/{TRIALS}: " +
              " -> ".join(f"{m} ({labels[m]})" for m in order))
        trial_logits = {}

        for mode in order:
            logits, elapsed_ms, staging = run_mode(
                model, target_ids, expert_id, mode
            )
            measurements[mode].append(elapsed_ms)
            trial_logits[mode] = logits

            print(f"  {labels[mode]}: {elapsed_ms / 1000:.3f} s")

            if staging is not None:
                hits = sum(
                    wrapper.staged_hits
                    for wrapper in staging["wrappers"].values()
                )
                fallbacks = sum(
                    wrapper.fallback_reads
                    for wrapper in staging["wrappers"].values()
                )
                print(
                    f"    CPU prep: {staging['cpu_prepare_ms']:.2f} ms; "
                    f"GPU copy: {staging['copy_start'].elapsed_time(staging['copy_end']):.2f} ms; "
                    f"staged hits: {hits}; fallbacks: {fallbacks}"
                )

        for mode in ("B", "C"):
            diff = (
                trial_logits["A"] - trial_logits[mode]
            ).abs().max().item()
            max_diffs.append(diff)
            print(f"  Max logit difference A vs {mode}: {diff:.8g}")

    medians = {
        mode: statistics.median(values)
        for mode, values in measurements.items()
    }

    print("\n=== THREE-MODE PREFETCH COMPARISON ===")
    for mode in ("A", "B", "C"):
        values = measurements[mode]
        print(
            f"{mode} — {labels[mode]}: "
            f"median={medians[mode] / 1000:.3f} s; "
            f"range={min(values) / 1000:.3f}–{max(values) / 1000:.3f} s"
        )

    for mode in ("B", "C"):
        change_pct = (medians[mode] / medians["A"] - 1) * 100
        print(
            f"{labels[mode]} vs baseline median: "
            f"{change_pct:+.2f}%"
        )

    print(f"Maximum logit difference: {max(max_diffs):.8g}")
    print(
        "\nInterpretation: B measures the full forward with concurrent "
        "staging but without using staged weights; C uses staged weights. "
        "Differences also include runtime variability and cache effects, "
        "so compare the distributions, not just one median."
    )


if __name__ == "__main__":
    main()
