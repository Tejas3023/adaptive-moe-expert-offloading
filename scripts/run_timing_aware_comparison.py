"""Timing-aware held-out comparison for Adaptive MoE Expert Offloading.

Run from project root:
    python -m scripts.run_timing_aware_comparison --compute-ms-per-record 0.1

This is a discrete-event simulation, not measured OLMoE inference latency.
It reuses the existing common-prefix evaluator's data splits and predictor
scoring logic, but evaluates cache transfers with bounded tiers and stalls.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_common_prefix_comparison as common
from src.cache.cost_model import CacheCostModel
from src.cache.timing_model import TimingAwareSimulator


METHODS = ("LRU", "HISTORY", "TRANSITION", "SELECTIVE", "EAM")


def evaluate_timing(prepared, method, size, threshold, compute_ms_per_record):
    """Evaluate one strategy using a fresh bounded cache per sample.

    Cache tier contents carry between samples. In-flight transfers do not:
    each sample's future horizon ends independently, so unfinished transfers
    at that boundary are counted as unfinished/wasted work.
    """
    cost_model = CacheCostModel()
    gpu_keys = []
    cpu_keys = []

    prediction_count = 0
    correct_unique_pairs = 0
    predicted_pair_count = 0
    actual_future_events = 0
    covered_future_events = 0

    prefetch_scheduled = 0
    prefetch_disk = 0
    prefetch_cpu = 0
    prefetch_already_gpu = 0

    prefetch_ready_before_demand = 0
    prefetch_late_at_demand = 0
    prefetch_not_ready_at_demand = 0
    wasted_prefetches = 0

    total_stall_ms = 0.0
    total_future_wall_ms = 0.0
    total_transfer_ms = 0.0
    total_prefetch_transfer_ms = 0.0
    total_demand_transfer_ms = 0.0

    total_demand_gpu_hits = 0
    total_demand_cpu_hits = 0
    total_demand_disk_misses = 0
    total_gpu_evictions = 0
    total_cpu_evictions = 0
    total_demand_requests = 0

    # Queue-wait diagnostics collected during the future evaluation horizon.
    total_prefetch_queue_wait_ms = 0.0
    total_demand_queue_wait_ms = 0.0
    total_prefetch_queue_wait_count = 0
    total_demand_queue_wait_count = 0

    # Demand wait on transfers that were already in flight at request time.
    total_demand_waiting_on_prefetch_ms = 0.0
    total_demand_waiting_on_prefetch_count = 0
    total_demand_waiting_on_demand_transfer_ms = 0.0
    total_demand_waiting_on_demand_transfer_count = 0

    for sample_id in sorted(prepared):
        item = prepared[sample_id]
        prefix = item["prefix"]
        future = item["future"]
        scores = item["scores"]

        selective = method == "SELECTIVE"
        candidates = common.choose_candidates(
            scores, size, threshold=threshold, selective=selective
        )
        candidate_set = set(candidates)

        future_pairs = {
            (record["layer_id"], expert_id)
            for record in future
            for expert_id in record["selected_experts"]
        }

        actual_future_events += sum(
            len(record["selected_experts"]) for record in future
        )
        correct_unique_pairs += len(candidate_set & future_pairs)
        predicted_pair_count += len(candidate_set)
        prediction_count += len(candidate_set)

        covered_future_events += sum(
            1
            for record in future
            for expert_id in record["selected_experts"]
            if (record["layer_id"], expert_id) in candidate_set
        )

        sim = TimingAwareSimulator(
            cost_model=cost_model,
            gpu_capacity=common.GPU_CAPACITY,
            cpu_capacity=common.CPU_CAPACITY,
            initially_resident=gpu_keys,
            initially_cpu=cpu_keys,
        )

        # Warm the cache using the same observed prefix for every method.
        for record in prefix:
            for expert_id in sorted(set(record["selected_experts"])):
                sim.request(record["layer_id"], expert_id)

        boundary_time = sim.current_time_ms
        boundary_stats = sim.statistics()
        future_demand_requests = 0
        pending_prefetches = set()

        # Schedule all predictions at the observation boundary.
        for layer_id, expert_id in candidates:
            outcome = sim.prefetch(layer_id, expert_id)
            key = (layer_id, expert_id)

            if outcome["status"] == "scheduled":
                prefetch_scheduled += 1
                pending_prefetches.add(key)

                if outcome["event"].source == "disk":
                    prefetch_disk += 1
                elif outcome["event"].source == "cpu":
                    prefetch_cpu += 1

            elif outcome["status"] == "already_resident":
                prefetch_already_gpu += 1

        # Replay future routing records. The configurable compute interval is
        # an explicit sensitivity assumption, not a measured trace field.
        for record in future:
            sim.advance_compute(compute_ms_per_record)
            layer_id = record["layer_id"]

            for expert_id in sorted(set(record["selected_experts"])):
                future_demand_requests += 1
                key = (layer_id, expert_id)

                if key in pending_prefetches:
                    # advance_compute() completes transfers whose finish time
                    # is <= current_time_ms. GPU residency means the prefetch
                    # was ready before this demand.
                    if key in sim.gpu:
                        prefetch_ready_before_demand += 1

                    elif (
                        key in sim.in_flight
                        and sim.in_flight[key].kind == "prefetch"
                        and sim.in_flight[key].finish_ms > sim.current_time_ms
                    ):
                        # The demand will have to wait for this prefetch.
                        prefetch_late_at_demand += 1

                    else:
                        # For example, the transfer completed but its expert
                        # was evicted before the demand arrived.
                        prefetch_not_ready_at_demand += 1

                    pending_prefetches.remove(key)

                sim.request(layer_id, expert_id)

        # Remaining candidates were not consumed in this sample's horizon.
        wasted_prefetches += len(pending_prefetches)

        stats = sim.statistics()

        # Account only for work after the observation boundary, excluding
        # prefix warming from the future-horizon measurements.
        total_future_wall_ms += max(
            0.0, sim.current_time_ms - boundary_time
        )

        total_prefetch_queue_wait_ms += (
            stats["prefetch_queue_wait_ms"]
            - boundary_stats["prefetch_queue_wait_ms"]
        )
        total_demand_queue_wait_ms += (
            stats["demand_queue_wait_ms"]
            - boundary_stats["demand_queue_wait_ms"]
        )
        total_prefetch_queue_wait_count += (
            stats["prefetch_queue_wait_count"]
            - boundary_stats["prefetch_queue_wait_count"]
        )
        total_demand_queue_wait_count += (
            stats["demand_queue_wait_count"]
            - boundary_stats["demand_queue_wait_count"]
        )

        total_demand_waiting_on_prefetch_ms += (
            stats["demand_waiting_on_prefetch_ms"]
            - boundary_stats["demand_waiting_on_prefetch_ms"]
        )
        total_demand_waiting_on_prefetch_count += (
            stats["demand_waiting_on_prefetch_count"]
            - boundary_stats["demand_waiting_on_prefetch_count"]
        )
        total_demand_waiting_on_demand_transfer_ms += (
            stats["demand_waiting_on_demand_transfer_ms"]
            - boundary_stats["demand_waiting_on_demand_transfer_ms"]
        )
        total_demand_waiting_on_demand_transfer_count += (
            stats["demand_waiting_on_demand_transfer_count"]
            - boundary_stats["demand_waiting_on_demand_transfer_count"]
        )

        total_stall_ms += (
            stats["total_stall_ms"] - boundary_stats["total_stall_ms"]
        )
        total_transfer_ms += (
            stats["total_transfer_ms"] - boundary_stats["total_transfer_ms"]
        )
        total_prefetch_transfer_ms += (
            stats["prefetch_transfer_ms"]
            - boundary_stats["prefetch_transfer_ms"]
        )
        total_demand_transfer_ms += (
            stats["demand_transfer_ms"]
            - boundary_stats["demand_transfer_ms"]
        )

        total_demand_gpu_hits += (
            stats["demand_gpu_hits"] - boundary_stats["demand_gpu_hits"]
        )
        total_demand_cpu_hits += (
            stats["demand_cpu_hits"] - boundary_stats["demand_cpu_hits"]
        )
        total_demand_disk_misses += (
            stats["demand_disk_misses"] - boundary_stats["demand_disk_misses"]
        )
        total_gpu_evictions += (
            stats["gpu_evictions"] - boundary_stats["gpu_evictions"]
        )
        total_cpu_evictions += (
            stats["cpu_evictions"] - boundary_stats["cpu_evictions"]
        )
        total_demand_requests += future_demand_requests

        # Carry completed cache contents forward, but not in-flight transfers.
        gpu_keys = sim.cached_gpu()
        cpu_keys = sim.cached_cpu()

    # Every scheduled prefetch must be classified exactly once.
    classified_prefetches = (
        prefetch_ready_before_demand
        + prefetch_late_at_demand
        + prefetch_not_ready_at_demand
        + wasted_prefetches
    )

    if classified_prefetches != prefetch_scheduled:
        raise RuntimeError(
            "Prefetch accounting mismatch: "
            f"scheduled={prefetch_scheduled}, "
            f"classified={classified_prefetches}"
        )

    return {
        "method": method,
        "prefetch_size": size,
        "selective_threshold": threshold if method == "SELECTIVE" else None,
        "compute_ms_per_record": compute_ms_per_record,
        "predicted_layer_expert_pairs": predicted_pair_count,
        "correct_predicted_pairs": correct_unique_pairs,
        "future_expert_selections": actual_future_events,
        "precision_pct": (
            100.0 * correct_unique_pairs / predicted_pair_count
            if predicted_pair_count
            else None
        ),
        "coverage_pct": (
            100.0 * covered_future_events / actual_future_events
            if actual_future_events
            else 0.0
        ),
        "total_demand_requests": total_demand_requests,
        "demand_gpu_hits": total_demand_gpu_hits,
        "demand_cpu_hits": total_demand_cpu_hits,
        "demand_disk_misses": total_demand_disk_misses,
        "prefetch_transfers_scheduled": prefetch_scheduled,
        "prefetch_disk_transfers": prefetch_disk,
        "prefetch_cpu_transfers": prefetch_cpu,
        "prefetch_already_gpu": prefetch_already_gpu,
        "prefetch_ready_before_demand": prefetch_ready_before_demand,
        "prefetch_late_at_demand": prefetch_late_at_demand,
        "prefetch_not_ready_at_demand": prefetch_not_ready_at_demand,
        "useful_prefetches": prefetch_ready_before_demand,
        "wasted_or_unconsumed_prefetches": wasted_prefetches,
        "prefetch_utilization_pct": (
            100.0 * prefetch_ready_before_demand / prefetch_scheduled
            if prefetch_scheduled
            else 0.0
        ),
        "simulated_future_wall_ms": total_future_wall_ms,
        "simulated_future_ms_per_demand_request": (
            total_future_wall_ms / total_demand_requests
            if total_demand_requests
            else 0.0
        ),
        "total_demand_stall_ms": total_stall_ms,
        "total_transfer_work_ms": total_transfer_ms,
        "prefetch_transfer_work_ms": total_prefetch_transfer_ms,
        "demand_transfer_work_ms": total_demand_transfer_ms,

        # Queue-wait diagnostics.
        "prefetch_queue_wait_ms": total_prefetch_queue_wait_ms,
        "demand_queue_wait_ms": total_demand_queue_wait_ms,
        "prefetch_queue_wait_count": total_prefetch_queue_wait_count,
        "demand_queue_wait_count": total_demand_queue_wait_count,
        "avg_prefetch_queue_wait_ms": (
            total_prefetch_queue_wait_ms / total_prefetch_queue_wait_count
            if total_prefetch_queue_wait_count
            else 0.0
        ),
        "avg_demand_queue_wait_ms": (
            total_demand_queue_wait_ms / total_demand_queue_wait_count
            if total_demand_queue_wait_count
            else 0.0
        ),

        # Demand wait caused by transfers already in flight at request time.
        "demand_waiting_on_prefetch_ms": (
            total_demand_waiting_on_prefetch_ms
        ),
        "demand_waiting_on_prefetch_count": (
            total_demand_waiting_on_prefetch_count
        ),
        "demand_waiting_on_demand_transfer_ms": (
            total_demand_waiting_on_demand_transfer_ms
        ),
        "demand_waiting_on_demand_transfer_count": (
            total_demand_waiting_on_demand_transfer_count
        ),

        "gpu_evictions": total_gpu_evictions,
        "cpu_evictions": total_cpu_evictions,
    }


def config_grid(method):
    if method == "LRU":
        return [(0, 0.0)]

    if method == "SELECTIVE":
        return [
            (size, threshold)
            for size in common.PREFETCH_SIZES
            for threshold in common.SELECTIVE_THRESHOLDS
        ]

    return [(size, 0.0) for size in common.PREFETCH_SIZES]


def pct(value):
    return "N/A" if value is None else f"{value:.2f}%"


def print_result(label, result):
    print(
        f"{label:<25} precision={pct(result['precision_pct']):>8} "
        f"coverage={result['coverage_pct']:6.2f}% "
        f"stall={result['total_demand_stall_ms']:10.2f} ms "
        f"future={result['simulated_future_wall_ms']:10.2f} ms "
        f"disk_miss={result['demand_disk_misses']:>7,} "
        f"prefetch={result['prefetch_transfers_scheduled']:>6,} "
        f"ready={result['prefetch_ready_before_demand']:>5,} "
        f"late={result['prefetch_late_at_demand']:>5,}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-ms-per-record",
        type=float,
        default=0.1,
        help=(
            "Assumed compute time available before each future routing record "
            "(default 0.1 ms). This is a simulation assumption, not measured."
        ),
    )
    args = parser.parse_args()

    if args.compute_ms_per_record < 0:
        parser.error("--compute-ms-per-record must be non-negative")

    output_dir = ROOT / "results" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=" * 120)
    print("TIMING-AWARE COMMON-PREFIX HELD-OUT COMPARISON")
    print(
        f"Assumed compute interval={args.compute_ms_per_record:.4f} ms "
        "per future routing record"
    )
    print(
        f"GPU={common.GPU_CAPACITY}, CPU={common.CPU_CAPACITY}; "
        "transfer channel is serialized; timing is simulated, not measured."
    )
    print("=" * 120)

    for workload, prefix_name in common.WORKLOADS.items():
        train_path = (
            ROOT / "results" / "dataset" / f"{prefix_name}_train.jsonl"
        )
        test_path = (
            ROOT / "results" / "dataset" / f"{prefix_name}_test.jsonl"
        )

        train_records = common.load_jsonl(train_path)
        test_records = common.load_jsonl(test_path)
        train_samples_all = common.group_by_sample(train_records)
        test_samples = common.group_by_sample(test_records)

        if set(train_samples_all) & set(test_samples):
            raise RuntimeError(f"Train/test sample overlap for {workload}")

        model_train_ids, validation_ids = common.split_training_ids(
            train_samples_all
        )
        model_train_samples = {
            sid: train_samples_all[sid] for sid in model_train_ids
        }
        validation_samples = {
            sid: train_samples_all[sid] for sid in validation_ids
        }

        print(f"\n{'=' * 36} {workload} {'=' * 36}")
        print(
            f"Model train={len(model_train_samples)}, "
            f"validation={len(validation_samples)}, test={len(test_samples)}"
        )

        validation_transition = common.make_transition_model(
            model_train_samples
        )
        validation_prepared = {}

        for method in METHODS:
            validation_prepared[method] = common.precompute_scores(
                validation_samples,
                method,
                model_train_samples,
                validation_transition,
            )

        selected = {}
        print("\nValidation tuning by simulated future wall-clock time:")

        for method in METHODS:
            trials = []

            for size, threshold in config_grid(method):
                result = evaluate_timing(
                    validation_prepared[method],
                    method,
                    size,
                    threshold,
                    args.compute_ms_per_record,
                )
                result.update({
                    "workload": workload,
                    "split": "validation",
                })
                trials.append(result)

            best = min(
                trials,
                key=lambda row: row["simulated_future_wall_ms"],
            )
            selected[method] = (
                best["prefetch_size"],
                best["selective_threshold"] or 0.0,
            )
            all_results.extend(trials)

            label = f"{method} p={best['prefetch_size']}"
            if method == "SELECTIVE":
                label += f" t={best['selective_threshold']:.3f}"

            print_result(label, best)

        # Refit on all training samples and evaluate once on the untouched
        # test set. The test set is not used to select hyperparameters.
        final_transition = common.make_transition_model(train_samples_all)
        test_prepared = {}

        for method in METHODS:
            test_prepared[method] = common.precompute_scores(
                test_samples,
                method,
                train_samples_all,
                final_transition,
            )

        print("\nFinal held-out test:")

        for method in METHODS:
            size, threshold = selected[method]
            result = evaluate_timing(
                test_prepared[method],
                method,
                size,
                threshold,
                args.compute_ms_per_record,
            )
            result.update({
                "workload": workload,
                "split": "test",
                "configuration": "validation_selected",
            })
            all_results.append(result)

            label = f"{method} p={size}"
            if method == "SELECTIVE":
                label += f" t={threshold:.3f}"

            print_result(label, result)

    json_path = output_dir / "timing_aware_strategy_comparison.json"
    csv_path = output_dir / "timing_aware_strategy_comparison.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    columns = sorted({key for row in all_results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_results)

    print("\nTiming-aware comparison complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Prefetch accounting: ready means resident in GPU before demand; "
        "late means still transferring when demand arrived."
    )
    print(
        "Queue-wait fields report channel waits for newly scheduled transfers. "
        "Demand-waiting fields separately report requests that joined an "
        "already-running prefetch or demand transfer."
    )
    print(
        "Interpretation: results depend on assumed compute intervals and "
        "fixed transfer costs; they are not real hardware latency measurements."
    )


if __name__ == "__main__":
    main()