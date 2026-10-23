import statistics
import time
import torch
from src.modeling.olmoe_loader import load_olmoe_model

model = load_olmoe_model()
module = dict(model.named_modules())[
    "model.layers.2.mlp.experts.0.gate_proj"
]
cpu_weight = module._hf_hook.weights_map["weight"]

def measure_copy(source, non_blocking, trials=10):
    # Warm up CUDA and exclude first-use initialization.
    target = source.to("cuda:0", non_blocking=non_blocking)
    torch.cuda.synchronize()
    del target

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        target = source.to("cuda:0", non_blocking=non_blocking)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        del target

    return times

print("Tensor size:", cpu_weight.numel() * cpu_weight.element_size() / (1024**2), "MiB")

regular = measure_copy(cpu_weight, False)
print("\nRegular CPU-to-GPU copy:")
print("Times (ms):", [round(x, 3) for x in regular])
print("Median (ms):", round(statistics.median(regular), 3))

try:
    pinned = cpu_weight.pin_memory()
    pinned_times = measure_copy(pinned, True)
    print("\nPinned-memory non-blocking copy:")
    print("Times (ms):", [round(x, 3) for x in pinned_times])
    print("Median (ms):", round(statistics.median(pinned_times), 3))
except Exception as exc:
    print("\nPinned-memory test unavailable:", type(exc).__name__, str(exc))

print("\nOriginal parameter device:", module.weight.device)
