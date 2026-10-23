import time
import torch
from transformers import AutoTokenizer
from src.config import MODEL_NAME
from src.modeling.olmoe_loader import load_olmoe_model


class PrefetchedWeightMap:
    """Track whether Accelerate actually retrieves staged weights."""

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_olmoe_model()
    model.eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    print("\nRunning baseline inference...")
    with torch.inference_mode():
        baseline = model(**inputs, output_router_logits=True)

    baseline_logits = baseline.logits.detach().float().cpu()
    router_logits = baseline.router_logits

    if router_logits is None or len(router_logits) <= 2:
        raise RuntimeError(
            "Router logits were not returned. Stop here; do not alter model hooks."
        )

    # Select an expert that was actually routed to in layer 2.
    layer_logits = router_logits[2].detach()
    if layer_logits.ndim == 3:
        layer_logits = layer_logits[0]

    selected = torch.topk(layer_logits, k=8, dim=-1).indices
    expert_id = int(selected.flatten()[0].item())
    layer_id = 2

    print(f"Selected layer-2 expert for experiment: {expert_id}")

    modules = dict(model.named_modules())
    staged_by_module = {}
    original_maps = {}

    print("Staging the expert's three projection weights...")
    start = time.perf_counter()

    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = (
            f"model.layers.{layer_id}.mlp.experts."
            f"{expert_id}.{projection}"
        )
        module = modules[name]
        hook = module._hf_hook

        if not getattr(hook, "offload", False):
            raise RuntimeError(f"Expected an offloaded module: {name}")

        # The map adds the module prefix, so its local key is "weight".
        cpu_weight = hook.weights_map["weight"]
        pinned_weight = cpu_weight.pin_memory()
        gpu_weight = pinned_weight.to("cuda:0", non_blocking=True)

        staged_by_module[name] = {"weight": gpu_weight}
        original_maps[name] = hook.weights_map

    torch.cuda.synchronize()
    staging_ms = (time.perf_counter() - start) * 1000
    print(f"Staging time for all three projections: {staging_ms:.3f} ms")

    try:
        wrappers = {}

        for name, staged in staged_by_module.items():
            wrapper = PrefetchedWeightMap(original_maps[name], staged)
            modules[name]._hf_hook.weights_map = wrapper
            wrappers[name] = wrapper
        

        print("Running inference with staged weights...")
        with torch.inference_mode():
            prefetched = model(**inputs, output_router_logits=True)

        prefetched_logits = prefetched.logits.detach().float().cpu()
        print("\nSTAGED WEIGHT USAGE")
        for name, wrapper in wrappers.items():
            print(
                f"{name}: "
                f"staged_hits={wrapper.staged_hits}, "
                f"fallback_reads={wrapper.fallback_reads}"
        )
        difference = (baseline_logits - prefetched_logits).abs()

        print("\nCORRECTNESS RESULTS")
        print("Baseline logits shape:", tuple(baseline_logits.shape))
        print("Prefetched logits shape:", tuple(prefetched_logits.shape))
        print("Maximum absolute difference:", difference.max().item())
        print("Mean absolute difference:", difference.mean().item())
        print(
            "Outputs close:",
            torch.allclose(
                baseline_logits, prefetched_logits, rtol=1e-3, atol=1e-3
            ),
        )
        print(
            "GPU memory allocated (MiB):",
            round(torch.cuda.memory_allocated() / (1024**2), 2),
        )
        print(
            "\nNOTE: This tests staged-weight correctness, not latency improvement."
        )

    finally:
        # Restore original maps and release staging buffers.
        for name, original in original_maps.items():
            modules[name]._hf_hook.weights_map = original

        staged_by_module.clear()
        original_maps.clear()
        torch.cuda.synchronize()

    print("Original weight maps restored.")


if __name__ == "__main__":
    main()
