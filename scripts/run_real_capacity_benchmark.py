import json
from collections import defaultdict

from src.cache.multi_tier_cache import MultiTierExpertCache


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"

EXPERT_SIZE_MB = 12.0

GPU_MEMORY_MB = 3072
CPU_MEMORY_MB = 2048


def capacity_from_memory(memory_mb):

    return int(
        memory_mb // EXPERT_SIZE_MB
    )


def load_traces():

    data = defaultdict(
        lambda: defaultdict(dict)
    )

    with open(
        TRACE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            trace = json.loads(line)

            sample = trace["sample_id"]
            layer = trace["layer_id"]
            token = trace["token_position"]

            data[sample][layer][token] = (
                trace["selected_experts"]
            )

    return data


def run(data, gpu_capacity, cpu_capacity):

    cache = MultiTierExpertCache(
        gpu_capacity=gpu_capacity,
        cpu_capacity=cpu_capacity,
    )

    for sample_layers in data.values():

        for layer_id, token_layers in sample_layers.items():

            for token in sorted(token_layers):

                for expert_id in token_layers[token]:

                    cache.request(
                        layer_id,
                        expert_id,
                    )

    return cache.statistics()


def main():

    print("=" * 90)
    print("REAL MEMORY-CONSTRAINED MULTI-TIER BENCHMARK")
    print("=" * 90)

    gpu_capacity = capacity_from_memory(
        GPU_MEMORY_MB
    )

    cpu_capacity = capacity_from_memory(
        CPU_MEMORY_MB
    )

    print()
    print(f"Expert size: {EXPERT_SIZE_MB:.2f} MB")

    print(
        f"GPU memory: {GPU_MEMORY_MB} MB"
    )

    print(
        f"CPU memory: {CPU_MEMORY_MB} MB"
    )

    print()
    print(
        f"GPU expert capacity: "
        f"{gpu_capacity}"
    )

    print(
        f"CPU expert capacity: "
        f"{cpu_capacity}"
    )

    print()
    print("Loading traces...")

    data = load_traces()

    result = run(
        data,
        gpu_capacity,
        cpu_capacity,
    )

    print()
    print("-" * 90)
    print("RESULTS")
    print("-" * 90)

    print(
        f"GPU hits:       "
        f"{result['gpu_hits']}"
    )

    print(
        f"CPU hits:       "
        f"{result['cpu_hits']}"
    )

    print(
        f"Disk fetches:   "
        f"{result['disk_fetches']}"
    )

    print(
        f"GPU evictions:  "
        f"{result['gpu_evictions']}"
    )

    print(
        f"CPU evictions:  "
        f"{result['cpu_evictions']}"
    )

    print(
        f"Total requests: "
        f"{result['total_requests']}"
    )

    print(
        f"Total latency:  "
        f"{result['total_latency_ms']:.2f} ms"
    )

    print(
        f"Average latency:"
        f" {result['average_latency_ms']:.4f} ms"
    )

    print()
    print("=" * 90)
    print("REAL CAPACITY BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()