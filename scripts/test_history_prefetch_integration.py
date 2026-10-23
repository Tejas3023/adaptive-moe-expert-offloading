import time
import torch
from transformers import AutoTokenizer

from src.config import MODEL_NAME
from src.modeling.olmoe_loader import load_olmoe_model
from src.prefetch.layer_aware_prefetcher import LayerAwareHistoryPrefetcher


class PrefetchedWeightMap:
    """Return staged GPU weights while preserving the original weight map."""

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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    layer_id = 2
    prefetch_size = 1

    predictor = LayerAwareHistoryPrefetcher(
        history_size=8,
        prefetch_size=prefetch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_olmoe_model()
    model.eval()

    prompt = "The future of artificial intelligence is"
    encoded = tokenizer(prompt, return_tensors="pt")

    input_device = model.get_input_embeddings().weight.device
    prompt_ids = encoded["input_ids"].to(input_device)
    target_ids = prompt_ids[:, -1:]

    modules = dict(model.named_modules())
    moe = modules[f"model.layers.{layer_id}.mlp"]

    state = {
        "enabled": True,
        "predicted": [],
        "actual": [],
        "wrappers": {},
        "original_maps": {},
        "staged_tensors": [],
        "usage": [],
    }

    def cleanup_staging():
        """Restore original maps and preserve usage counters."""
        for name, wrapper in list(state["wrappers"].items()):
            state["usage"].append(
                (
                    name,
                    wrapper.staged_hits,
                    wrapper.fallback_reads,
                )
            )

        for name, original in list(state["original_maps"].items()):
            modules[name]._hf_hook.weights_map = original

        state["wrappers"].clear()
        state["original_maps"].clear()
        state["staged_tensors"].clear()

    def before_moe(module, args):
        if not state["enabled"]:
            return

        # Predict using previously observed routing history only.
        predictions = predictor.predict(layer_id)
        state["predicted"] = predictions

        if not predictions:
            print("No prediction available yet.")
            return

        expert_id = predictions[0]
        print(f"Predicted layer-{layer_id} expert: {expert_id}")

        try:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = (
                    f"model.layers.{layer_id}.mlp.experts."
                    f"{expert_id}.{projection}"
                )

                projection_module = modules[name]
                hook = projection_module._hf_hook

                if not getattr(hook, "offload", False):
                    raise RuntimeError(
                        f"Expected an offloaded projection: {name}"
                    )

                original = hook.weights_map

                # PrefixedDataset adds the module prefix automatically.
                cpu_weight = original["weight"]
                pinned_weight = cpu_weight.pin_memory()
                gpu_weight = pinned_weight.to(
                    "cuda:0",
                    non_blocking=True,
                )

                wrapper = PrefetchedWeightMap(
                    original,
                    {"weight": gpu_weight},
                )

                state["original_maps"][name] = original
                state["wrappers"][name] = wrapper
                state["staged_tensors"].extend(
                    [pinned_weight, gpu_weight]
                )

                hook.weights_map = wrapper

            # Ensure all staging transfers have completed.
            torch.cuda.synchronize()

        except Exception:
            cleanup_staging()
            raise

    def after_moe(module, args, output):
        try:
            if not state["enabled"]:
                return

            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    "Unexpected MoE output; expected hidden states "
                    "and router logits."
                )

            router_logits = output[1].detach()

            if router_logits.ndim == 3:
                router_logits = router_logits.reshape(
                    -1, router_logits.shape[-1]
                )

            actual = torch.topk(
                router_logits,
                k=8,
                dim=-1,
            ).indices.cpu().tolist()

            state["actual"] = actual

            for experts in actual:
                predictor.observe(layer_id, experts)

        finally:
            cleanup_staging()

    pre_handle = moe.register_forward_pre_hook(before_moe)
    post_handle = moe.register_forward_hook(after_moe)

    try:
        print("\n1. Observing prompt routing to build history...")

        with torch.inference_mode():
            model(input_ids=prompt_ids)

        print("History entries:", len(predictor))
        print(
            "Recent layer-2 history:",
            predictor.history_contents(layer_id),
        )

        if not len(predictor):
            raise RuntimeError("No routing history was recorded.")

        # Baseline: disable prediction and staging.
        state["enabled"] = False

        print("\n2. Running baseline inference for target token...")

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            baseline = model(input_ids=target_ids)

        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - start) * 1000
        baseline_logits = baseline.logits.detach().float().cpu()

        print(f"Baseline pass time: {baseline_ms:.3f} ms")

        # Predictor-driven staging pass.
        state["enabled"] = True
        print("\n3. Running predictor-driven staging...")

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            result = model(input_ids=target_ids)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        result_logits = result.logits.detach().float().cpu()

        difference = (baseline_logits - result_logits).abs()

        predicted = set(state["predicted"])
        actual = {
            expert
            for token_experts in state["actual"]
            for expert in token_experts
        }

        print("\nRESULTS")
        print("Predicted experts:", state["predicted"])
        print("Actual layer-2 experts:", sorted(actual))
        print(
            "Prediction matched actual routing:",
            bool(predicted & actual),
        )
        print("Baseline logits shape:", tuple(baseline_logits.shape))
        print("Prefetched logits shape:", tuple(result_logits.shape))
        print("Maximum absolute difference:", difference.max().item())
        print("Mean absolute difference:", difference.mean().item())
        print(
            "Outputs close:",
            torch.allclose(
                baseline_logits,
                result_logits,
                rtol=1e-3,
                atol=1e-3,
            ),
        )

        print("\nSTAGED WEIGHT USAGE")
        for name, hits, fallback in state["usage"]:
            print(
                f"{name}: staged_hits={hits}, "
                f"fallback_reads={fallback}"
            )

        print("\nTIMING")
        print(f"Baseline pass: {baseline_ms:.3f} ms")
        print(f"Prefetch-path pass: {elapsed_ms:.3f} ms")
        print(
            "Difference (prefetch - baseline): "
            f"{elapsed_ms - baseline_ms:.3f} ms"
        )

        print(
            "\nNOTE: These are single-pass measurements, "
            "not a controlled performance benchmark."
        )

    finally:
        state["enabled"] = False
        cleanup_staging()
        pre_handle.remove()
        post_handle.remove()
        torch.cuda.synchronize()
        print("\nOriginal weight maps restored; temporary hooks removed.")


if __name__ == "__main__":
    main()
