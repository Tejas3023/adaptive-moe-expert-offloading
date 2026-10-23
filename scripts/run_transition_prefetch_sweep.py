
import csv
import json

from scripts import run_heldout_strategy_comparison as benchmark


PREFETCH_SIZES = [1, 2, 4, 8]

OUTPUT_DIR = (
    benchmark.PROJECT_ROOT
    / "results"
    / "experiments"
)


def main():
    all_results = []

    print("=" * 100)
    print("HELD-OUT TRANSITION PREFETCH-SIZE SWEEP")
    print("Training: 80 samples | Testing: 20 samples per workload")
    print(
        f"GPU capacity={benchmark.GPU_CAPACITY}, "
        f"CPU capacity={benchmark.CPU_CAPACITY}"
    )
    print(f"Prefetch sizes: {PREFETCH_SIZES}")
    print("=" * 100)

    for display_name, prefix in benchmark.WORKLOADS.items():
        train_path = (
            benchmark.PROJECT_ROOT
            / "results"
            / "dataset"
            / f"{prefix}_train.jsonl"
        )
        test_path = (
            benchmark.PROJECT_ROOT
            / "results"
            / "dataset"
            / f"{prefix}_test.jsonl"
        )

        train_records = benchmark.load_jsonl(train_path)
        test_records = benchmark.load_jsonl(test_path)

        train_ids = {r["sample_id"] for r in train_records}
        test_ids = {r["sample_id"] for r in test_records}

        if train_ids & test_ids:
            raise RuntimeError(
                f"Train/test leakage in {display_name}"
            )

        print(f"\n{display_name}")

        # LRU baseline: run once per workload.
        benchmark.PREFETCH_SIZE = 8
        baseline = benchmark.run_test(
            test_records,
            "LRU",
        )
        baseline["workload"] = display_name
        baseline["prefetch_size"] = 0
        all_results.append(baseline)

        print(
            f"  LRU        "
            f"disk misses={baseline['demand_disk_misses']:,}  "
            f"cost/request="
            f"{baseline['modeled_latency_per_demand_request_ms']:.4f} ms"
        )

        for size in PREFETCH_SIZES:
            benchmark.PREFETCH_SIZE = size

            # Train a separate transition model for this size.
            transition = benchmark.train_transition_model(
                train_records
            )

            result = benchmark.run_test(
                test_records,
                "TRANSITION",
                trained_transition=transition,
            )

            result["workload"] = display_name
            result["prefetch_size"] = size
            all_results.append(result)

            print(
                f"  Transition p={size:<2} "
                f"precision={result['precision_pct'] if result['precision_pct'] is not None else 'N/A'} "
                f"coverage={result['coverage_pct']:.2f}%  "
                f"disk misses={result['demand_disk_misses']:,}  "
                f"prefetch disk loads={result['prefetch_disk_loads']:,}  "
                f"transfers={result['total_prefetch_transfers']:,}  "
                f"useful={result['useful_prefetch_transfers']:,}  "
                f"wasted={result['wasted_prefetch_transfers']:,}  "
                f"cost/request="
                f"{result['modeled_latency_per_demand_request_ms']:.4f} ms"
            )

    json_path = (
        OUTPUT_DIR / "transition_prefetch_size_sweep.json"
    )
    csv_path = (
        OUTPUT_DIR / "transition_prefetch_size_sweep.csv"
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\nSweep complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Reminder: costs are simulated cache costs, "
        "not end-to-end model latency."
    )


if __name__ == "__main__":
    main()
