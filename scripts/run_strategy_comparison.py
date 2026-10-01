import json
import sys
from pathlib import Path
from collections import deque, Counter


# ==============================================================
# PROJECT ROOT
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from src.cache.multi_tier_cache import MultiTierExpertCache

from src.prefetch.transition_prefetcher import (
    TransitionPrefetcher
)

from src.prefetch.selective_prefetcher import (
    SelectiveTransitionPrefetcher
)


# ==============================================================
# CONFIGURATION
# ==============================================================

TRACE_FILES = {
    "WIKITEXT": (
        PROJECT_ROOT
        / "results"
        / "traces"
        / "wikitext_10_samples.jsonl"
    ),

    "GSM8K": (
        PROJECT_ROOT
        / "results"
        / "traces"
        / "gsm8k_10_samples.jsonl"
    ),

    "MBPP": (
        PROJECT_ROOT
        / "results"
        / "traces"
        / "mbpp_10_samples.jsonl"
    ),
}


GPU_CAPACITY = 256
CPU_CAPACITY = 170

HISTORY_WINDOW = 8

PREFETCH_SIZE = 8

SELECTIVE_THRESHOLD = 0.105


# ==============================================================
# LOAD TRACE
# ==============================================================

def load_trace(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ==============================================================
# HISTORY PREDICTOR
# ==============================================================

class HistoryPrefetcher:

    def __init__(
        self,
        history_window=8,
        prefetch_size=8,
    ):

        self.history_window = history_window
        self.prefetch_size = prefetch_size

        self.history = {}

    def reset(self):

        self.history.clear()

    def update(
        self,
        layer_id,
        experts,
    ):

        if layer_id not in self.history:

            self.history[layer_id] = deque(
                maxlen=self.history_window
            )

        self.history[layer_id].append(
            list(experts)
        )

    def predict(
        self,
        layer_id,
    ):

        if layer_id not in self.history:

            return []

        counter = Counter()

        for expert_list in self.history[layer_id]:

            for expert_id in expert_list:

                counter[expert_id] += 1

        ranked = sorted(
            counter.items(),
            key=lambda x: (-x[1], x[0])
        )

        return [
            expert_id
            for expert_id, _ in ranked[
                :self.prefetch_size
            ]
        ]


# ==============================================================
# GENERIC CACHE BENCHMARK
# ==============================================================

def run_strategy(
    records,
    strategy_name,
):

    # ----------------------------------------------------------
    # Fresh cache for every strategy
    # ----------------------------------------------------------

    cache = MultiTierExpertCache(
        gpu_capacity=GPU_CAPACITY,
        cpu_capacity=CPU_CAPACITY,
    )

    # ----------------------------------------------------------
    # Fresh predictors for every strategy
    # ----------------------------------------------------------

    history = HistoryPrefetcher(
        history_window=HISTORY_WINDOW,
        prefetch_size=PREFETCH_SIZE,
    )

    transition = TransitionPrefetcher(
        num_experts=64,
        top_k=8,
        prefetch_size=PREFETCH_SIZE,
    )

    selective = SelectiveTransitionPrefetcher(
        num_experts=64,
        top_k=8,
        max_prefetch_size=PREFETCH_SIZE,
        confidence_threshold=SELECTIVE_THRESHOLD,
    )

    # ----------------------------------------------------------
    # Predictor metrics
    # ----------------------------------------------------------

    total_actual = 0
    total_predictions = 0
    correct_predictions = 0

    # ----------------------------------------------------------
    # Process trace
    # ----------------------------------------------------------

    for record in records:

        layer_id = record["layer_id"]

        actual_experts = set(
            record["selected_experts"]
        )

        total_actual += len(
            actual_experts
        )

        # ------------------------------------------------------
        # Generate predictions
        # ------------------------------------------------------

        predictions = []

        if strategy_name == "LRU":

            predictions = []

        elif strategy_name == "HISTORY":

            predictions = history.predict(
                layer_id
            )

        elif strategy_name == "TRANSITION":

            previous = (
                transition.previous_experts.get(
                    layer_id,
                    []
                )
            )

            predictions = transition.predict(
                layer_id,
                previous
            )

        elif strategy_name == "SELECTIVE":

            previous = (
                selective
                .transition_prefetcher
                .previous_experts
                .get(
                    layer_id,
                    []
                )
            )

            predictions = selective.predict(
                layer_id,
                previous
            )

        # ------------------------------------------------------
        # Predictor metrics
        # ------------------------------------------------------

        for expert_id in predictions:

            total_predictions += 1

            if expert_id in actual_experts:

                correct_predictions += 1

        # ------------------------------------------------------
        # Prefetch predicted experts
        # ------------------------------------------------------

        for expert_id in predictions:

            cache.prefetch(
                layer_id,
                expert_id
            )

        # ------------------------------------------------------
        # Request actual experts
        # ------------------------------------------------------

        for expert_id in actual_experts:

            cache.request(
                layer_id,
                expert_id
            )

        # ------------------------------------------------------
        # Update predictor AFTER observing actual routing
        # ------------------------------------------------------

        if strategy_name == "HISTORY":

            history.update(
                layer_id,
                list(actual_experts)
            )

        elif strategy_name == "TRANSITION":

            transition.update(
                layer_id,
                list(actual_experts)
            )

        elif strategy_name == "SELECTIVE":

            selective.update(
                layer_id,
                list(actual_experts)
            )

    # ==========================================================
    # CACHE STATISTICS
    # ==========================================================

    stats = cache.statistics()

    # ----------------------------------------------------------
    # Precision
    # ----------------------------------------------------------

    precision = (
        correct_predictions / total_predictions
        if total_predictions > 0
        else 0.0
    )

    # ----------------------------------------------------------
    # Coverage
    # ----------------------------------------------------------

    coverage = (
        correct_predictions / total_actual
        if total_actual > 0
        else 0.0
    )

    # ----------------------------------------------------------
    # Return result
    # ----------------------------------------------------------

    return {

        "strategy": strategy_name,

        "records": len(records),

        "predictions": total_predictions,

        "correct": correct_predictions,

        "precision": precision,

        "coverage": coverage,

        "gpu_hits": stats["gpu_hits"],

        "cpu_hits": stats["cpu_hits"],

        "disk_fetches": stats["disk_fetches"],

        "gpu_evictions": stats[
            "gpu_evictions"
        ],

        "cpu_evictions": stats[
            "cpu_evictions"
        ],

        "prefetches": stats[
            "prefetches"
        ],

        "prefetch_hits": stats[
            "prefetch_hits"
        ],

        "total_latency_ms": stats[
            "total_latency_ms"
        ],

        "average_latency_ms": stats[
            "average_latency_ms"
        ],
    }


# ==============================================================
# PRINT WORKLOAD RESULTS
# ==============================================================

def print_workload_results(
    workload_name,
    results,
):

    print()
    print("=" * 150)
    print(
        f"{workload_name} — FINAL COMPARISON"
    )
    print("=" * 150)

    print(
        f"{'Strategy':<16}"
        f"{'Pred':>10}"
        f"{'Correct':>10}"
        f"{'Precision':>12}"
        f"{'Coverage':>12}"
        f"{'GPU Hits':>12}"
        f"{'CPU Hits':>12}"
        f"{'Disk':>10}"
        f"{'Prefetch':>12}"
        f"{'Latency':>14}"
    )

    print("-" * 150)

    for result in results:

        print(
            f"{result['strategy']:<16}"
            f"{result['predictions']:>10,}"
            f"{result['correct']:>10,}"
            f"{result['precision'] * 100:>11.2f}%"
            f"{result['coverage'] * 100:>11.2f}%"
            f"{result['gpu_hits']:>12,}"
            f"{result['cpu_hits']:>12,}"
            f"{result['disk_fetches']:>10,}"
            f"{result['prefetches']:>12,}"
            f"{result['average_latency_ms']:>13.4f}"
        )

    print("=" * 150)


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 120)
    print(
        "PHASE 24 — CROSS-WORKLOAD "
        "MoE PREFETCH STRATEGY COMPARISON"
    )
    print("=" * 120)

    print()

    print(
        f"GPU capacity: "
        f"{GPU_CAPACITY} experts"
    )

    print(
        f"CPU capacity: "
        f"{CPU_CAPACITY} experts"
    )

    print(
        f"History window: "
        f"{HISTORY_WINDOW}"
    )

    print(
        f"Prefetch size: "
        f"{PREFETCH_SIZE}"
    )

    print(
        f"Selective threshold: "
        f"{SELECTIVE_THRESHOLD}"
    )

    print()

    strategies = [
        "LRU",
        "HISTORY",
        "TRANSITION",
        "SELECTIVE",
    ]

    # ==========================================================
    # ALL WORKLOADS
    # ==========================================================

    all_results = {}

    for workload_name, trace_path in TRACE_FILES.items():

        print()
        print("#" * 120)
        print(
            f"WORKLOAD: {workload_name}"
        )
        print("#" * 120)

        print(
            f"\nTrace: {trace_path}"
        )

        # ------------------------------------------------------
        # Check trace exists
        # ------------------------------------------------------

        if not trace_path.exists():

            print(
                f"\nERROR: Trace file not found:"
            )

            print(trace_path)

            continue

        # ------------------------------------------------------
        # Load trace
        # ------------------------------------------------------

        records = load_trace(
            trace_path
        )

        print(
            f"Loaded "
            f"{len(records):,} routing records."
        )

        if len(records) == 0:

            print(
                "WARNING: Trace is empty."
            )

            continue

        # ------------------------------------------------------
        # Run strategies
        # ------------------------------------------------------

        results = []

        for strategy in strategies:

            print()
            print(
                f"Running "
                f"{workload_name} / "
                f"{strategy}..."
            )

            result = run_strategy(
                records,
                strategy
            )

            result["workload"] = (
                workload_name
            )

            results.append(
                result
            )

            print(
                f"  Predictions: "
                f"{result['predictions']:,}"
            )

            print(
                f"  Precision: "
                f"{result['precision'] * 100:.2f}%"
            )

            print(
                f"  Coverage: "
                f"{result['coverage'] * 100:.2f}%"
            )

            print(
                f"  GPU hits: "
                f"{result['gpu_hits']:,}"
            )

            print(
                f"  CPU hits: "
                f"{result['cpu_hits']:,}"
            )

            print(
                f"  Disk fetches: "
                f"{result['disk_fetches']:,}"
            )

            print(
                f"  Average latency: "
                f"{result['average_latency_ms']:.4f} ms"
            )

        all_results[
            workload_name
        ] = results

        print_workload_results(
            workload_name,
            results
        )

    # ==========================================================
    # CROSS-WORKLOAD SUMMARY
    # ==========================================================

    print()
    print("=" * 160)
    print(
        "PHASE 24 — CROSS-WORKLOAD SUMMARY"
    )
    print("=" * 160)

    print(
        f"{'Workload':<12}"
        f"{'Strategy':<16}"
        f"{'Records':>10}"
        f"{'Pred':>10}"
        f"{'Precision':>12}"
        f"{'Coverage':>12}"
        f"{'GPU Hits':>12}"
        f"{'Disk':>10}"
        f"{'Latency':>14}"
    )

    print("-" * 160)

    for workload_name, results in all_results.items():

        for result in results:

            print(
                f"{workload_name:<12}"
                f"{result['strategy']:<16}"
                f"{result['records']:>10,}"
                f"{result['predictions']:>10,}"
                f"{result['precision'] * 100:>11.2f}%"
                f"{result['coverage'] * 100:>11.2f}%"
                f"{result['gpu_hits']:>12,}"
                f"{result['disk_fetches']:>10,}"
                f"{result['average_latency_ms']:>13.4f}"
            )

    print("=" * 160)

    # ==========================================================
    # BASIC WORKLOAD CHARACTERISTICS
    # ==========================================================

    print()
    print("=" * 100)
    print(
        "TRACE SIZE SUMMARY"
    )
    print("=" * 100)

    print(
        f"{'Workload':<15}"
        f"{'Routing Records':>20}"
        f"{'Expert Selections':>22}"
    )

    print("-" * 100)

    for workload_name, trace_path in TRACE_FILES.items():

        if not trace_path.exists():
            continue

        records = load_trace(
            trace_path
        )

        expert_selections = sum(
            len(record["selected_experts"])
            for record in records
        )

        print(
            f"{workload_name:<15}"
            f"{len(records):>20,}"
            f"{expert_selections:>22,}"
        )

    print("=" * 100)

    print()
    print(
        "PHASE 24 CROSS-WORKLOAD "
        "BENCHMARK COMPLETE"
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()