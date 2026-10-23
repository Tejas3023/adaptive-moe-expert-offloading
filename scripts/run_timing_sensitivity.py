"""Sensitivity analysis for timing-aware held-out strategy comparison.

Run from project root:
    python -m scripts.run_timing_sensitivity

The script repeats validation selection and held-out testing at several assumed
compute intervals. These are simulation assumptions, not measured OLMoE timings.
"""

import csv
import json
from pathlib import Path

from scripts import run_common_prefix_comparison as common
from scripts import run_timing_aware_comparison as timing


COMPUTE_INTERVALS_MS = [0.0, 0.1, 0.5, 1.0]
METHODS = timing.METHODS


def prepare_workload(workload, prefix_name):
    root = common.ROOT
    train_records = common.load_jsonl(
        root / "results" / "dataset" / f"{prefix_name}_train.jsonl"
    )
    test_records = common.load_jsonl(
        root / "results" / "dataset" / f"{prefix_name}_test.jsonl"
    )
    train_all = common.group_by_sample(train_records)
    test_samples = common.group_by_sample(test_records)

    if set(train_all) & set(test_samples):
        raise RuntimeError(f"Train/test overlap for {workload}")

    model_ids, validation_ids = common.split_training_ids(train_all)
    model_train = {sid: train_all[sid] for sid in model_ids}
    validation = {sid: train_all[sid] for sid in validation_ids}

    validation_transition = common.make_transition_model(model_train)
    validation_prepared = {
        method: common.precompute_scores(
            validation, method, model_train, validation_transition
        )
        for method in METHODS
    }

    final_transition = common.make_transition_model(train_all)
    test_prepared = {
        method: common.precompute_scores(
            test_samples, method, train_all, final_transition
        )
        for method in METHODS
    }
    return validation_prepared, test_prepared


def run():
    output_dir = common.ROOT / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print("=" * 110)
    print("TIMING MODEL SENSITIVITY ANALYSIS")
    print(f"Compute intervals (assumptions): {COMPUTE_INTERVALS_MS} ms per future routing record")
    print("All timing results are simulated, not measured hardware latency.")
    print("=" * 110)

    for workload, prefix_name in common.WORKLOADS.items():
        validation_prepared, test_prepared = prepare_workload(
            workload, prefix_name
        )
        print(f"\n{'=' * 28} {workload} {'=' * 28}")

        for compute_ms in COMPUTE_INTERVALS_MS:
            selected = {}
            for method in METHODS:
                trials = []
                for size, threshold in timing.config_grid(method):
                    result = timing.evaluate_timing(
                        validation_prepared[method],
                        method,
                        size,
                        threshold,
                        compute_ms,
                    )
                    trials.append(result)
                best = min(
                    trials,
                    key=lambda row: row["simulated_future_wall_ms"],
                )
                selected[method] = (
                    best["prefetch_size"],
                    best["selective_threshold"] or 0.0,
                )

            for method in METHODS:
                size, threshold = selected[method]
                result = timing.evaluate_timing(
                    test_prepared[method],
                    method,
                    size,
                    threshold,
                    compute_ms,
                )
                result.update({
                    "workload": workload,
                    "split": "test",
                    "compute_ms_per_record": compute_ms,
                    "configuration": "validation_selected",
                })
                all_rows.append(result)

                print(
                    f"compute={compute_ms:>3.1f} ms | "
                    f"{method:<10} p={size:<2} "
                    f"threshold={threshold:.3f} "
                    f"wall={result['simulated_future_wall_ms']:>12.2f} ms "
                    f"stall={result['total_demand_stall_ms']:>12.2f} ms "
                    f"disk_miss={result['demand_disk_misses']:>7,} "
                    f"ready={result['prefetch_ready_before_demand']:>5,} "
                    f"late={result['prefetch_late_at_demand']:>5,} "
                    f"not_ready={result['prefetch_not_ready_at_demand']:>5,}"
                )

    json_path = output_dir / "timing_sensitivity_analysis.json"
    csv_path = output_dir / "timing_sensitivity_analysis.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)

    columns = sorted({key for row in all_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSensitivity analysis complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Prefetch accounting: ready = resident in GPU before demand; "
        "late = still transferring when demand arrived; not_ready = no longer "
        "resident/ready at demand; unconsumed prefetches are reported separately in CSV/JSON."
    )
    print(
        "Use this analysis to assess whether strategy rankings change when "
        "the assumed compute interval changes. All timings remain simulated."
    )


if __name__ == "__main__":
    run()
