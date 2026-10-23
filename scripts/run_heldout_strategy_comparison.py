import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache.multi_tier_cache import MultiTierExpertCache
from src.prefetch.transition_prefetcher import TransitionPrefetcher
from src.prefetch.selective_prefetcher import SelectiveTransitionPrefetcher


GPU_CAPACITY = 256
CPU_CAPACITY = 170
HISTORY_WINDOW = 8
PREFETCH_SIZE = 8
SELECTIVE_THRESHOLD = 0.105

WORKLOADS = {
    "WIKITEXT": "wikitext",
    "GSM8K": "gsm8k",
    "MBPP": "mbpp",
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def layer_major_records(records):
    """Chronological token order within each layer, for predictor training."""
    return sorted(
        records,
        key=lambda r: (
            r["sample_id"],
            r["layer_id"],
            r["token_position"],
        ),
    )


def ordered_records(records):
    """Process each token through its layers in inference order."""
    return sorted(
        records,
        key=lambda r: (
            r["sample_id"],
            r["token_position"],
            r["layer_id"],
        ),
    )


def train_transition_model(records):
    predictor = TransitionPrefetcher(
        num_experts=64,
        top_k=8,
        prefetch_size=PREFETCH_SIZE,
    )

    records = layer_major_records(records)
    previous_sample = None

    for record in records:
        sample_id = record["sample_id"]

        if sample_id != previous_sample:
            # Clear only sequence history, not learned counts.
            predictor.previous_experts.clear()
            previous_sample = sample_id

        predictor.update(
            record["layer_id"],
            record["selected_experts"],
        )

    # The training sequence's final experts must not leak into testing.
    predictor.previous_experts.clear()
    return predictor


class HistoryPredictor:
    def __init__(self):
        self.history = {}

    def reset(self):
        self.history.clear()

    def predict(self, layer_id):
        recent = self.history.get(layer_id, [])
        counts = Counter(
            expert
            for expert_set in recent
            for expert in expert_set
        )
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [expert for expert, _ in ranked[:PREFETCH_SIZE]]

    def update(self, layer_id, experts):
        if layer_id not in self.history:
            self.history[layer_id] = deque(maxlen=HISTORY_WINDOW)
        self.history[layer_id].append(list(experts))


def run_test(records, strategy, trained_transition=None):
    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY,
    )

    history = HistoryPredictor()
    previous_by_layer = {}

    selective = None
    if strategy == "SELECTIVE":
        selective = SelectiveTransitionPrefetcher(
            num_experts=64,
            top_k=8,
            max_prefetch_size=PREFETCH_SIZE,
            confidence_threshold=SELECTIVE_THRESHOLD,
        )
        # Reuse the training counts, but no training sequence history.
        selective.transition_prefetcher.transition_counts = (
            trained_transition.transition_counts
        )
        selective.transition_prefetcher.previous_experts.clear()

    total_predictions = 0
    correct_predictions = 0
    total_actual = 0
    
    prefetch_disk_loads = 0
    prefetch_cpu_loads = 0
    prefetch_gpu_already_resident = 0

    # Tracks transferred prefetches that have not yet been resolved.
    pending_prefetch_keys = set()
    useful_prefetch_transfers = 0
    wasted_prefetch_transfers = 0
    total_prefetch_transfers = 0


    records = ordered_records(records)
    previous_sample = None

    for record in records:
        sample_id = record["sample_id"]
        layer_id = record["layer_id"]
        actual = set(record["selected_experts"])

        if sample_id != previous_sample:
            history.reset()
            previous_by_layer.clear()
            previous_sample = sample_id

        # Predict the current token using only earlier information.
        if strategy == "LRU":
            predictions = []

        elif strategy == "HISTORY":
            predictions = history.predict(layer_id)

        elif strategy == "TRANSITION":
            context = previous_by_layer.get(layer_id, [])
            predictions = (
                trained_transition.predict(layer_id, context)
                if context else []
            )

        elif strategy == "SELECTIVE":
            context = previous_by_layer.get(layer_id, [])
            predictions = (
                selective.predict(layer_id, context)
                if context else []
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        predictions = list(dict.fromkeys(predictions))
        total_predictions += len(predictions)
        correct_predictions += len(set(predictions) & actual)
        total_actual += len(actual)

        
        # Prefetch before requesting the current token's actual experts.
        for expert_id in predictions:
            key = (layer_id, expert_id)

            # If a previously transferred expert is no longer in either
            # tier, or must be transferred again from CPU, resolve the
            # previous transfer as wasted.
            if key in pending_prefetch_keys:
                in_gpu = key in cache.gpu
                in_cpu = key in cache.cpu

                if not in_gpu:
                    if not in_cpu:
                        wasted_prefetch_transfers += 1
                        pending_prefetch_keys.discard(key)
                    else:
                        # A new CPU-to-GPU transfer supersedes the
                        # previous unresolved prefetch episode.
                        wasted_prefetch_transfers += 1
                        pending_prefetch_keys.discard(key)

            tier = cache.prefetch(layer_id, expert_id)

            if tier == "disk":
                prefetch_disk_loads += 1
                total_prefetch_transfers += 1
                pending_prefetch_keys.add(key)

            elif tier == "cpu":
                prefetch_cpu_loads += 1
                total_prefetch_transfers += 1
                pending_prefetch_keys.add(key)

            elif tier == "gpu":
                prefetch_gpu_already_resident += 1


        
        # Demand requests for the experts actually selected by OLMoE.
        for expert_id in sorted(actual):
            key = (layer_id, expert_id)

            if key in pending_prefetch_keys:
                if key in cache.gpu or key in cache.cpu:
                    # The transferred expert was still cached when needed.
                    useful_prefetch_transfers += 1
                else:
                    # It was no longer cached when requested.
                    wasted_prefetch_transfers += 1

                pending_prefetch_keys.discard(key)

            cache.request(layer_id, expert_id)


        # Update only the online context/history, not learned test counts.
        history.update(layer_id, sorted(actual))
        previous_by_layer[layer_id] = sorted(actual)

        # Transfers still pending at the end were never used during this test.
    wasted_prefetch_transfers += len(pending_prefetch_keys)
    stats = cache.statistics()

    return {
        "workload": "",
        "strategy": strategy,
        "test_records": len(records),
        "predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "actual_expert_selections": total_actual,
        "precision_pct": (
    100 * correct_predictions / total_predictions
    if total_predictions else None
),
        "coverage_pct": (
            100 * correct_predictions / total_actual
            if total_actual else 0.0
        ),
        "gpu_hits": stats["gpu_hits"],
        "cpu_hits": stats["cpu_hits"],
        "demand_disk_misses": stats["disk_fetches"],
        "prefetch_operations": stats["prefetches"],
        "prefetch_disk_loads": prefetch_disk_loads,
        "prefetch_cpu_loads": prefetch_cpu_loads,
        "prefetch_already_on_gpu": prefetch_gpu_already_resident,
        
        "total_prefetch_transfers": total_prefetch_transfers,
        "useful_prefetch_transfers": useful_prefetch_transfers,
        "wasted_prefetch_transfers": wasted_prefetch_transfers,
        "prefetch_transfer_utilization_pct": (
            100 * useful_prefetch_transfers / total_prefetch_transfers
            if total_prefetch_transfers else 0.0
        ),
        "wasted_prefetch_transfer_pct": (
            100 * wasted_prefetch_transfers / total_prefetch_transfers
            if total_prefetch_transfers else 0.0
        ),

        "gpu_evictions": stats["gpu_evictions"],
        "cpu_evictions": stats["cpu_evictions"],
        "total_modeled_latency_ms": stats["total_latency_ms"],
        "modeled_latency_per_demand_request_ms": stats["average_latency_ms"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=" * 100)
    print("HELD-OUT MoE PREFETCH STRATEGY COMPARISON")
    print("Training: 80 samples | Testing: 20 samples per workload")
    print(f"GPU capacity={GPU_CAPACITY}, CPU capacity={CPU_CAPACITY}")
    print(f"History window={HISTORY_WINDOW}, prefetch size={PREFETCH_SIZE}")
    print(f"Selective relative-score threshold={SELECTIVE_THRESHOLD}")
    print("=" * 100)

    for display_name, prefix in WORKLOADS.items():
        train_path = PROJECT_ROOT / "results" / "dataset" / f"{prefix}_train.jsonl"
        test_path = PROJECT_ROOT / "results" / "dataset" / f"{prefix}_test.jsonl"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Missing train/test files for {display_name}: "
                f"{train_path} or {test_path}"
            )

        train_records = load_jsonl(train_path)
        test_records = load_jsonl(test_path)

        train_ids = {r["sample_id"] for r in train_records}
        test_ids = {r["sample_id"] for r in test_records}
        if train_ids & test_ids:
            raise RuntimeError(f"Train/test leakage in {display_name}")

        print(f"\n{display_name}: train records={len(train_records):,}, "
              f"test records={len(test_records):,}, "
              f"train samples={len(train_ids)}, test samples={len(test_ids)}")

        transition = train_transition_model(train_records)

        for strategy in ("LRU", "HISTORY", "TRANSITION", "SELECTIVE"):
            result = run_test(
                test_records,
                strategy,
                trained_transition=transition,
            )
            result["workload"] = display_name
            all_results.append(result)

            
            precision = result["precision_pct"]
            precision_text = (
                f"{precision:.2f}%"
                if precision is not None
                else "N/A"
            )

            print(
                f"{strategy:<12} "
                f"precision={precision_text:>8}  "
                f"coverage={result['coverage_pct']:6.2f}%  "
                f"disk misses={result['demand_disk_misses']:,}  "
                f"prefetch disk loads={result['prefetch_disk_loads']:,}  "
                f"modeled cost/request="
                f"{result['modeled_latency_per_demand_request_ms']:.4f} ms"
            )
            
            print(
                f"  transfers={result['total_prefetch_transfers']:,}  "
                f"useful={result['useful_prefetch_transfers']:,}  "
                f"wasted={result['wasted_prefetch_transfers']:,}  "
                f"utilization="
                f"{result['prefetch_transfer_utilization_pct']:.2f}%"
            )



    json_path = OUTPUT_DIR / "heldout_strategy_comparison.json"
    csv_path = OUTPUT_DIR / "heldout_strategy_comparison.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print("\nBenchmark complete.")
    print(f"JSON results: {json_path}")
    print(f"CSV results:  {csv_path}")
    print("Reminder: latency values are modeled cache costs, not end-to-end OLMoE latency.")


if __name__ == "__main__":
    main()
