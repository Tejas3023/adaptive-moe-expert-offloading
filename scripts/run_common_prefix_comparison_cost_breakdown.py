"""Common-prefix, held-out comparison of LRU, History, Transition, Selective and EAM.

All predictors observe the first OBSERVED_TOKENS of each sample and make one
prediction for the remaining horizon. Prediction sizes are selected on a
validation split of the training samples; the final test samples remain held out.
This is a simulated cache-cost experiment, not end-to-end model inference.
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eam_predictor as eam
from src.cache.multi_tier_cache import MultiTierExpertCache
from src.prefetch.transition_prefetcher import TransitionPrefetcher

WORKLOADS = {"WIKITEXT": "wikitext", "GSM8K": "gsm8k", "MBPP": "mbpp"}
OUTPUT_DIR = ROOT / "results" / "experiments"
NUM_LAYERS = 16
NUM_EXPERTS = 64
TOP_K = 8
OBSERVED_TOKENS = 10
NUM_MATCHES = 5
HISTORY_WINDOW = 8
PREFETCH_SIZES = [1, 2, 4, 8]
SELECTIVE_THRESHOLDS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.105]
VALIDATION_SEED = 123
GPU_CAPACITY = 256
CPU_CAPACITY = 170


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_by_sample(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["sample_id"]].append(record)
    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda r: (r["token_position"], r["layer_id"]))
    return dict(grouped)


def split_training_ids(sample_ids):
    sample_ids = sorted(sample_ids)
    if len(sample_ids) != 80:
        raise ValueError(f"Expected 80 training sample IDs, found {len(sample_ids)}")
    shuffled = sample_ids.copy()
    random.Random(VALIDATION_SEED).shuffle(shuffled)
    validation_ids = set(shuffled[:16])
    model_train_ids = set(shuffled[16:])
    return model_train_ids, validation_ids


def make_transition_model(train_samples):
    predictor = TransitionPrefetcher(
        num_experts=NUM_EXPERTS, top_k=TOP_K, prefetch_size=max(PREFETCH_SIZES)
    )
    records = [r for sample in train_samples.values() for r in sample]
    records.sort(key=lambda r: (r["sample_id"], r["layer_id"], r["token_position"]))
    previous_sample = None
    for record in records:
        sample_id = record["sample_id"]
        if sample_id != previous_sample:
            predictor.previous_experts.clear()
            previous_sample = sample_id
        predictor.update(record["layer_id"], record["selected_experts"])
    predictor.previous_experts.clear()
    return predictor


def prefix_and_future(records):
    token_positions = sorted({r["token_position"] for r in records})
    observed_count = min(OBSERVED_TOKENS, max(len(token_positions) - 1, 0))
    observed_positions = set(token_positions[:observed_count])
    prefix = [r for r in records if r["token_position"] in observed_positions]
    future = [r for r in records if r["token_position"] not in observed_positions]
    return prefix, future, observed_count


def score_history(prefix):
    scores = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    by_layer = defaultdict(list)
    for record in prefix:
        by_layer[record["layer_id"]].append(record)
    for layer_id, layer_records in by_layer.items():
        layer_records = sorted(layer_records, key=lambda r: r["token_position"])[-HISTORY_WINDOW:]
        for record in layer_records:
            for expert_id in record["selected_experts"]:
                scores[layer_id, expert_id] += 1.0
    return scores


def score_transition(prefix, transition_model):
    """Aggregate learned outgoing-transition counts over observed prefix contexts."""
    scores = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    for record in prefix:
        layer_id = record["layer_id"]
        layer_table = transition_model.transition_counts.get(layer_id, {})
        for current_expert in record["selected_experts"]:
            outgoing = layer_table.get(current_expert, {})
            for next_expert, count in outgoing.items():
                scores[layer_id, next_expert] += float(count)
    return scores


def score_eam(prefix, future, train_samples, train_eams, train_ids):
    observed_count = len({r["token_position"] for r in prefix})
    partial = eam.build_partial_eam(prefix + future, observed_count)
    matches = eam.find_similar_eams(partial, train_eams, train_ids, NUM_MATCHES)
    scores = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
    for match in matches:
        sample_id = match["sample_id"]
        similarity = float(match["similarity"])
        for record in train_samples[sample_id]:
            if record["token_position"] < observed_count:
                continue
            for expert_id in record["selected_experts"]:
                scores[record["layer_id"], expert_id] += similarity
    return scores


def precompute_scores(samples, method, train_samples, transition_model=None):
    """Compute score matrices once per sample so hyperparameter sweeps are cheap."""
    train_ids = sorted(train_samples)
    train_eams = None
    if method == "EAM":
        train_eams = [eam.build_eam(train_samples[sample_id]) for sample_id in train_ids]

    prepared = {}
    for sample_id, records in samples.items():
        prefix, future, observed_count = prefix_and_future(records)
        if method == "LRU":
            scores = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)
        elif method == "HISTORY":
            scores = score_history(prefix)
        elif method in ("TRANSITION", "SELECTIVE"):
            scores = score_transition(prefix, transition_model)
        elif method == "EAM":
            scores = score_eam(prefix, future, train_samples, train_eams, train_ids)
        else:
            raise ValueError(f"Unknown method: {method}")
        prepared[sample_id] = {
            "records": records,
            "prefix": prefix,
            "future": future,
            "observed_count": observed_count,
            "scores": scores,
        }
    return prepared


def choose_candidates(scores, size, threshold=0.0, selective=False):
    candidates = []
    if size <= 0:
        return candidates
    for layer_id in range(NUM_LAYERS):
        row = scores[layer_id]
        ranked = sorted(range(NUM_EXPERTS), key=lambda expert: (-row[expert], expert))
        positive = [expert for expert in ranked if row[expert] > 0]
        total_score = float(sum(row[expert] for expert in positive))
        selected = []
        for expert in positive:
            if selective:
                relative_score = float(row[expert]) / total_score if total_score else 0.0
                if relative_score < threshold:
                    continue
            selected.append(expert)
            if len(selected) >= size:
                break
        candidates.extend((layer_id, expert) for expert in selected)
    return candidates


def evaluate(prepared, method, size, threshold=0.0):
    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY, cpu_capacity=CPU_CAPACITY
    )
    prediction_count = 0
    correct_unique_pairs = 0
    predicted_pair_count = 0
    actual_future_events = 0
    covered_future_events = 0
    prefetch_disk_loads = 0
    prefetch_cpu_loads = 0
    prefetch_already_gpu = 0
    total_transfers = 0
    useful_transfers = 0
    wasted_transfers = 0

    for sample_id in sorted(prepared):
        item = prepared[sample_id]
        prefix = item["prefix"]
        future = item["future"]
        scores = item["scores"]
        selective = method == "SELECTIVE"
        candidate_list = choose_candidates(scores, size, threshold, selective)
        candidate_set = set(candidate_list)
        future_pairs = {
            (record["layer_id"], expert)
            for record in future for expert in record["selected_experts"]
        }
        actual_future_events += sum(len(record["selected_experts"]) for record in future)
        correct_unique_pairs += len(candidate_set & future_pairs)
        predicted_pair_count += len(candidate_set)
        prediction_count += len(candidate_set)
        covered_future_events += sum(
            1 for record in future for expert in record["selected_experts"]
            if (record["layer_id"], expert) in candidate_set
        )

        # Warm the same cache with the observed prefix for every strategy.
        for record in prefix:
            for expert_id in sorted(set(record["selected_experts"])):
                cache.request(record["layer_id"], expert_id)

        pending = set()

        def resolve_evicted_pending():
            nonlocal wasted_transfers
            for key in list(pending):
                if key not in cache.gpu and key not in cache.cpu:
                    pending.remove(key)
                    wasted_transfers += 1

        # One-shot prefetch at the shared observation boundary.
        for layer_id, expert_id in candidate_list:
            tier = cache.prefetch(layer_id, expert_id)
            key = (layer_id, expert_id)
            if tier == "disk":
                prefetch_disk_loads += 1
                total_transfers += 1
                pending.add(key)
            elif tier == "cpu":
                prefetch_cpu_loads += 1
                total_transfers += 1
                pending.add(key)
            elif tier == "gpu":
                prefetch_already_gpu += 1
            resolve_evicted_pending()

        # Score only the common future horizon. Prefetches are useful if still
        # resident when the matching future expert is demanded.
        for record in future:
            layer_id = record["layer_id"]
            for expert_id in sorted(set(record["selected_experts"])):
                key = (layer_id, expert_id)
                if key in pending:
                    if key in cache.gpu or key in cache.cpu:
                        useful_transfers += 1
                    else:
                        wasted_transfers += 1
                    pending.discard(key)
                cache.request(layer_id, expert_id)
                resolve_evicted_pending()

        # A sample's predictions target only that sample's future horizon.
        wasted_transfers += len(pending)

    stats = cache.statistics()
    return {
        "method": method,
        "prefetch_size": size,
        "selective_threshold": threshold if method == "SELECTIVE" else None,
        "predicted_layer_expert_pairs": predicted_pair_count,
        "correct_predicted_pairs": correct_unique_pairs,
        "future_expert_selections": actual_future_events,
        "precision_pct": 100.0 * correct_unique_pairs / predicted_pair_count if predicted_pair_count else None,
        "coverage_pct": 100.0 * covered_future_events / actual_future_events if actual_future_events else 0.0,
        "demand_disk_misses": stats["disk_fetches"],
        "gpu_hits": stats["gpu_hits"],
        "cpu_hits": stats["cpu_hits"],
        "prefetch_disk_loads": prefetch_disk_loads,
        "prefetch_cpu_loads": prefetch_cpu_loads,
        "prefetch_already_gpu": prefetch_already_gpu,
        "total_prefetch_transfers": total_transfers,
        "useful_prefetch_transfers": useful_transfers,
        "wasted_prefetch_transfers": wasted_transfers,
        "transfer_utilization_pct": 100.0 * useful_transfers / total_transfers if total_transfers else 0.0,
        # Separated simulated cost accounting from MultiTierExpertCache.
        "demand_latency_ms": stats["demand_latency_ms"],
        "prefetch_transfer_cost_ms": stats["prefetch_transfer_cost_ms"],
        "average_demand_latency_ms": stats["average_demand_latency_ms"],
        "prefetch_transfer_cost_per_demand_request_ms": stats["prefetch_transfer_cost_per_demand_request_ms"],
        # Backward-compatible combined cost metrics.
        "modeled_cost_per_demand_request_ms": stats["average_latency_ms"],
        "total_modeled_cost_ms": stats["total_latency_ms"],
    }


def config_grid(method):
    if method == "LRU":
        return [(0, 0.0)]
    if method == "SELECTIVE":
        return [(size, threshold) for size in PREFETCH_SIZES for threshold in SELECTIVE_THRESHOLDS]
    return [(size, 0.0) for size in PREFETCH_SIZES]


def format_pct(value):
    return "N/A" if value is None else f"{value:.2f}%"


def print_result(label, result):
    print(
        f"{label:<24} precision={format_pct(result['precision_pct']):>8} "
        f"coverage={result['coverage_pct']:6.2f}% "
        f"disk_misses={result['demand_disk_misses']:>7,} "
        f"transfers={result['total_prefetch_transfers']:>6,} "
        f"useful={result['useful_prefetch_transfers']:>6,} "
        f"wasted={result['wasted_prefetch_transfers']:>6,} "
        f"demand/request={result['average_demand_latency_ms']:.4f} ms "
        f"prefetch/request={result['prefetch_transfer_cost_per_demand_request_ms']:.4f} ms "
        f"combined/request={result['modeled_cost_per_demand_request_ms']:.4f} ms"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    print("=" * 112)
    print("COMMON-PREFIX HELD-OUT PREFETCH COMPARISON")
    print(f"Shared observation: first {OBSERVED_TOKENS} tokens | same remaining-token horizon")
    print(f"Validation: 16 samples | final test: 20 samples | GPU={GPU_CAPACITY}, CPU={CPU_CAPACITY}")
    print("All methods prefetch once at the observation boundary; costs are simulated, not end-to-end latency.")
    print("Reported cost separates demand service from prefetch transfers; combined cost is their sum per demand request.")
    print("=" * 112)

    for workload, prefix_name in WORKLOADS.items():
        train_path = ROOT / "results" / "dataset" / f"{prefix_name}_train.jsonl"
        test_path = ROOT / "results" / "dataset" / f"{prefix_name}_test.jsonl"
        train_records = load_jsonl(train_path)
        test_records = load_jsonl(test_path)
        train_samples_all = group_by_sample(train_records)
        test_samples = group_by_sample(test_records)
        if set(train_samples_all) & set(test_samples):
            raise RuntimeError(f"Train/test sample overlap for {workload}")
        model_train_ids, validation_ids = split_training_ids(train_samples_all)
        model_train_samples = {sid: train_samples_all[sid] for sid in model_train_ids}
        validation_samples = {sid: train_samples_all[sid] for sid in validation_ids}

        print(f"\n{'=' * 32} {workload} {'=' * 32}")
        print(f"Train pool={len(train_samples_all)} | model train={len(model_train_samples)} | validation={len(validation_samples)} | test={len(test_samples)}")
        validation_transition = make_transition_model(model_train_samples)
        validation_prepared = {}
        for method in ("LRU", "HISTORY", "TRANSITION", "SELECTIVE", "EAM"):
            validation_prepared[method] = precompute_scores(
                validation_samples, method, model_train_samples, validation_transition
            )

        selected = {}
        print("\nValidation tuning (select lowest modeled cost/request):")
        for method in ("LRU", "HISTORY", "TRANSITION", "SELECTIVE", "EAM"):
            trials = []
            for size, threshold in config_grid(method):
                result = evaluate(validation_prepared[method], method, size, threshold)
                result.update({"workload": workload, "split": "validation"})
                trials.append(result)
            best = min(trials, key=lambda r: r["modeled_cost_per_demand_request_ms"])
            selected[method] = (best["prefetch_size"], best["selective_threshold"] or 0.0)
            print_result(f"{method} selected p={best['prefetch_size']}" + (f" t={best['selective_threshold']:.3f}" if method == "SELECTIVE" else ""), best)
            all_results.extend(trials)

        # Retrain statistical models / EAM database on all 80 training samples;
        # evaluate once on untouched final test samples.
        final_transition = make_transition_model(train_samples_all)
        test_prepared = {}
        for method in ("LRU", "HISTORY", "TRANSITION", "SELECTIVE", "EAM"):
            test_prepared[method] = precompute_scores(
                test_samples, method, train_samples_all, final_transition
            )

        print("\nFinal held-out test:")
        for method in ("LRU", "HISTORY", "TRANSITION", "SELECTIVE", "EAM"):
            size, threshold = selected[method]
            result = evaluate(test_prepared[method], method, size, threshold)
            result.update({"workload": workload, "split": "test", "configuration": "validation_selected"})
            all_results.append(result)
            print_result(f"{method} selected p={size}" + (f" t={threshold:.3f}" if method == "SELECTIVE" else ""), result)

        # Add p=8 reference rows to show the effect of a common maximum budget.
        print("\nFixed maximum budget reference (p=8):")
        for method in ("HISTORY", "TRANSITION", "SELECTIVE", "EAM"):
            threshold = 0.0
            result = evaluate(test_prepared[method], method, 8, threshold)
            result.update({"workload": workload, "split": "test", "configuration": "fixed_p8_threshold0"})
            all_results.append(result)
            print_result(f"{method} fixed p=8", result)

    json_path = OUTPUT_DIR / "common_prefix_strategy_comparison.json"
    csv_path = OUTPUT_DIR / "common_prefix_strategy_comparison.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    columns = sorted({key for row in all_results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_results)

    print("\nComparison complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("Interpret results as simulated cache costs, not measured end-to-end OLMoE latency.")


if __name__ == "__main__":
    main()
