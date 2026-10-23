
import csv
import json
import random
from pathlib import Path

import numpy as np

from scripts import eam_predictor as eam
from src.cache.multi_tier_cache import MultiTierExpertCache


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "experiments"

WORKLOADS = {
    "WIKITEXT": "wikitext",
    "GSM8K": "gsm8k",
    "MBPP": "mbpp",
}

PREFETCH_SIZES = [1, 2, 4, 8]
OBSERVED_TOKENS = 10
NUM_MATCHES = 5
VALIDATION_SEED = 123

GPU_CAPACITY = 256
CPU_CAPACITY = 170


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def split_ids(sample_ids):
    sample_ids = sorted(sample_ids)

    if len(sample_ids) != 80:
        raise ValueError(
            f"Expected 80 training samples, got {len(sample_ids)}"
        )

    shuffled = sample_ids.copy()
    random.Random(VALIDATION_SEED).shuffle(shuffled)

    validation_ids = set(shuffled[:16])
    model_train_ids = set(shuffled[16:])

    return model_train_ids, validation_ids


def make_eam_database(train_samples, train_ids):
    ids = sorted(train_ids)

    eams = [
        eam.build_eam(train_samples[sample_id])
        for sample_id in ids
    ]

    return eams, ids


def score_future_layer_experts(
    matches,
    train_samples,
    current_token,
):
    """Return a separate score for every (layer, expert) pair."""
    scores = np.zeros(
        (eam.NUM_LAYERS, eam.NUM_EXPERTS),
        dtype=np.float64,
    )

    for match in matches:
        sample_id = match["sample_id"]
        similarity = match["similarity"]

        for record in train_samples[sample_id]:
            # The partial EAM contains tokens before current_token.
            # Predict from current_token onward.
            if record["token_position"] < current_token:
                continue

            layer_id = record["layer_id"]

            for expert_id in record["selected_experts"]:
                scores[layer_id, expert_id] += similarity

    return scores


def choose_candidates(scores, size):
    """Select up to 'size' candidates independently per layer."""
    candidates = []

    if size <= 0:
        return candidates

    for layer_id in range(eam.NUM_LAYERS):
        ranked = sorted(
            range(eam.NUM_EXPERTS),
            key=lambda expert_id: (
                -scores[layer_id, expert_id],
                expert_id,
            ),
        )

        selected = [
            expert_id
            for expert_id in ranked
            if scores[layer_id, expert_id] > 0
        ][:size]

        for expert_id in selected:
            candidates.append(
                (layer_id, expert_id, scores[layer_id, expert_id])
            )

    return candidates


def evaluate_eam(
    train_samples,
    eval_samples,
    prefetch_size,
):
    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY,
    )

    predicted_pairs = 0
    correct_predicted_pairs = 0
    actual_future_selections = 0
    covered_future_selections = 0

    prefetch_disk_loads = 0
    prefetch_cpu_loads = 0
    prefetch_already_on_gpu = 0
    total_transfers = 0

    pending_transfers = set()
    useful_transfers = 0
    wasted_transfers = 0

    def resolve_evicted_transfers():
        nonlocal wasted_transfers

        for key in list(pending_transfers):
            if key not in cache.gpu and key not in cache.cpu:
                pending_transfers.remove(key)
                wasted_transfers += 1

    for sample_id in sorted(eval_samples):
        records = eval_samples[sample_id]

        token_positions = sorted({
            record["token_position"]
            for record in records
        })

        if not token_positions:
            continue

        observed_tokens = min(
            OBSERVED_TOKENS,
            max(len(token_positions) - 1, 0),
        )

        prefix = [
            record for record in records
            if record["token_position"] < observed_tokens
        ]

        future = [
            record for record in records
            if record["token_position"] >= observed_tokens
        ]

        # First replay the observed prefix to warm the cache.
        for record in prefix:
            for expert_id in sorted(record["selected_experts"]):
                cache.request(record["layer_id"], expert_id)
                resolve_evicted_transfers()

        # Construct the partial EAM using observed tokens only.
        partial_eam = eam.build_partial_eam(
            records,
            observed_tokens,
        )

        train_eams, train_ids = make_eam_database(
            train_samples,
            set(train_samples),
        )

        matches = eam.find_similar_eams(
            partial_eam,
            train_eams,
            train_ids,
            NUM_MATCHES,
        )

        scores = score_future_layer_experts(
            matches,
            train_samples,
            observed_tokens,
        )

        candidates = choose_candidates(
            scores,
            prefetch_size,
        )

        predicted_keys = {
            (layer_id, expert_id)
            for layer_id, expert_id, _ in candidates
        }

        actual_future_pairs = {
            (record["layer_id"], expert_id)
            for record in future
            for expert_id in record["selected_experts"]
        }

        predicted_pairs += len(predicted_keys)
        correct_predicted_pairs += len(
            predicted_keys & actual_future_pairs
        )

        for record in future:
            actual_future_selections += len(
                record["selected_experts"]
            )
            covered_future_selections += sum(
                (
                    record["layer_id"],
                    expert_id,
                ) in predicted_keys
                for expert_id in record["selected_experts"]
            )

        # Prefetch predicted pairs before replaying future demand.
        for layer_id, expert_id, _ in candidates:
            key = (layer_id, expert_id)

            # A transfer that was evicted before being used is wasted.
            if key in pending_transfers and key in cache.cpu:
                pending_transfers.remove(key)
                wasted_transfers += 1

            tier = cache.prefetch(layer_id, expert_id)

            if tier == "disk":
                prefetch_disk_loads += 1
                total_transfers += 1
                pending_transfers.add(key)

            elif tier == "cpu":
                prefetch_cpu_loads += 1
                total_transfers += 1
                pending_transfers.add(key)

            else:
                prefetch_already_on_gpu += 1

            resolve_evicted_transfers()

        # Replay the actual future experts in inference order.
        for record in future:
            layer_id = record["layer_id"]

            for expert_id in sorted(record["selected_experts"]):
                key = (layer_id, expert_id)

                if key in pending_transfers:
                    if key in cache.gpu or key in cache.cpu:
                        useful_transfers += 1
                    else:
                        wasted_transfers += 1

                    pending_transfers.remove(key)

                cache.request(layer_id, expert_id)
                resolve_evicted_transfers()

    # Any transfer never used during the evaluated stream is unused.
    wasted_transfers += len(pending_transfers)

    stats = cache.statistics()

    return {
        "prefetch_size_per_layer": prefetch_size,
        "predicted_layer_expert_pairs": predicted_pairs,
        "correct_predicted_pairs": correct_predicted_pairs,
        "actual_future_selections": actual_future_selections,
        "covered_future_selections": covered_future_selections,
        "precision_pct": (
            100 * correct_predicted_pairs / predicted_pairs
            if predicted_pairs else None
        ),
        "coverage_pct": (
            100 * covered_future_selections / actual_future_selections
            if actual_future_selections else 0.0
        ),
        "gpu_hits": stats["gpu_hits"],
        "cpu_hits": stats["cpu_hits"],
        "demand_disk_misses": stats["disk_fetches"],
        "gpu_evictions": stats["gpu_evictions"],
        "cpu_evictions": stats["cpu_evictions"],
        "prefetch_disk_loads": prefetch_disk_loads,
        "prefetch_cpu_loads": prefetch_cpu_loads,
        "prefetch_already_on_gpu": prefetch_already_on_gpu,
        "total_prefetch_transfers": total_transfers,
        "useful_prefetch_transfers": useful_transfers,
        "wasted_prefetch_transfers": wasted_transfers,
        "transfer_utilization_pct": (
            100 * useful_transfers / total_transfers
            if total_transfers else 0.0
        ),
        "total_modeled_cost_ms": stats["total_latency_ms"],
        "modeled_cost_per_demand_request_ms": stats["average_latency_ms"],
    }


def print_result(label, result):
    precision = result["precision_pct"]
    precision_text = (
        f"{precision:.2f}%"
        if precision is not None else "N/A"
    )

    print(
        f"{label:<23}"
        f" precision={precision_text:>8}"
        f" coverage={result['coverage_pct']:6.2f}%"
        f" disk_misses={result['demand_disk_misses']:>7,}"
        f" transfers={result['total_prefetch_transfers']:>7,}"
        f" useful={result['useful_prefetch_transfers']:>7,}"
        f" wasted={result['wasted_prefetch_transfers']:>7,}"
        f" cost/request="
        f"{result['modeled_cost_per_demand_request_ms']:.4f} ms"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 100)
    print("LAYER-AWARE EAM HELD-OUT EVALUATION")
    print("Training=64 | Validation=16 | Final test=20")
    print(
        f"GPU={GPU_CAPACITY} experts, CPU={CPU_CAPACITY} experts"
    )
    print(
        f"Observed tokens={OBSERVED_TOKENS}, "
        f"historical matches={NUM_MATCHES}"
    )
    print("=" * 100)

    for workload, prefix in WORKLOADS.items():
        dataset_dir = ROOT / "results" / "dataset"

        train_records = load_jsonl(
            dataset_dir / f"{prefix}_train.jsonl"
        )
        test_records = load_jsonl(
            dataset_dir / f"{prefix}_test.jsonl"
        )

        train_samples = eam.group_by_sample(train_records)
        test_samples = eam.group_by_sample(test_records)

        train_ids = set(train_samples)
        test_ids = set(test_samples)

        if train_ids & test_ids:
            raise RuntimeError(
                f"Train/test leakage for {workload}"
            )

        model_train_ids, validation_ids = split_ids(train_ids)

        model_train_samples = {
            sid: train_samples[sid]
            for sid in model_train_ids
        }
        validation_samples = {
            sid: train_samples[sid]
            for sid in validation_ids
        }

        print(f"\n{'=' * 30} {workload} {'=' * 30}")

        validation_results = []

        # Select prefetch size on validation only.
        for size in PREFETCH_SIZES:
            result = evaluate_eam(
                model_train_samples,
                validation_samples,
                size,
            )
            result.update({
                "workload": workload,
                "stage": "validation",
            })
            results.append(result)
            validation_results.append(result)

            print_result(f"Validation p={size}", result)

        best = min(
            validation_results,
            key=lambda r: (
                r["modeled_cost_per_demand_request_ms"],
                r["demand_disk_misses"],
                -r["coverage_pct"],
            ),
        )
        selected_size = best["prefetch_size_per_layer"]

        print(f"\nValidation-selected size: {selected_size}")

        # Final test: use all 80 original training samples.
        lru = evaluate_eam(
            train_samples,
            test_samples,
            0,
        )
        lru.update({
            "workload": workload,
            "stage": "test",
            "policy": "LRU",
        })
        results.append(lru)
        print_result("Test LRU", lru)

        for label, size in (
            ("EAM fixed p=8", 8),
            ("EAM selected", selected_size),
        ):
            result = evaluate_eam(
                train_samples,
                test_samples,
                size,
            )
            result.update({
                "workload": workload,
                "stage": "test",
                "policy": label,
            })
            results.append(result)
            print_result(label, result)

    json_path = OUTPUT_DIR / "eam_heldout_evaluation.json"
    csv_path = OUTPUT_DIR / "eam_heldout_evaluation.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = list(dict.fromkeys(
        key
        for result in results
        for key in result
    ))

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nEAM evaluation complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Costs are simulated cache costs, not measured "
        "end-to-end OLMoE latency."
    )


if __name__ == "__main__":
    main()
