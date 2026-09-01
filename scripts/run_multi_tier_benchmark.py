import json
from collections import defaultdict

from src.cache.multi_tier_cache import MultiTierExpertCache


TRACE_FILE = "results/traces/wikitext_10_samples.jsonl"

CONFIGS = [
    (8, 16),
    (16, 32),
    (32, 32),
    (32, 64),
    (64, 64),
]


def load_traces():

    data = defaultdict(lambda: defaultdict(dict))

    with open(TRACE_FILE, "r", encoding="utf-8") as f:

        for line in f:

            trace = json.loads(line)

            sample = trace["sample_id"]
            layer = trace["layer_id"]
            token = trace["token_position"]

            data[sample][layer][token] = (
                trace["selected_experts"]
            )

    return data


def run_experiment(data, gpu_capacity, cpu_capacity):

    cache = MultiTierExpertCache(
        gpu_capacity=gpu_capacity,
        cpu_capacity=cpu_capacity,
    )

    for sample_layers in data.values():

        for token_layers in sample_layers.values():

            for token in sorted(token_layers):

                experts = token_layers[token]

                for expert_id in experts:

                    cache.request(
                        layer_id=0,
                        expert_id=expert_id,
                    )

    return cache.statistics()


def main():

    print("=" * 90)
    print("MULTI-TIER EXPERT CACHE BENCHMARK")
    print("=" * 90)

    print()
    print("Trace file:")
    print(TRACE_FILE)

    data = load_traces()

    print()
    print("-" * 90)
    print("CONFIGURATIONS")
    print("-" * 90)

    print(
        f"{'GPU':<10}"
        f"{'CPU':<10}"
        f"{'GPU Hits':<12}"
        f"{'CPU Hits':<12}"
        f"{'Disk':<12}"
        f"{'Avg Latency':<15}"
    )

    print("-" * 90)

    for gpu_capacity, cpu_capacity in CONFIGS:

        result = run_experiment(
            data,
            gpu_capacity,
            cpu_capacity,
        )

        print(
            f"{gpu_capacity:<10}"
            f"{cpu_capacity:<10}"
            f"{result['gpu_hits']:<12}"
            f"{result['cpu_hits']:<12}"
            f"{result['disk_fetches']:<12}"
            f"{result['average_latency_ms']:<15.4f}"
        )

    print()
    print("=" * 90)
    print("MULTI-TIER BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()