
import csv
import json
import random

from scripts import run_heldout_strategy_comparison as benchmark


PREFETCH_SIZES = [1, 2, 4, 8]
VALIDATION_SEED = 123

OUTPUT_DIR = (
    benchmark.PROJECT_ROOT
    / "results"
    / "experiments"
)


def subset_records(records, sample_ids):
    return [
        record
        for record in records
        if record["sample_id"] in sample_ids
    ]


def split_training_samples(records):
    """Split the original training samples into 64/16."""
    sample_ids = sorted({
        record["sample_id"] for record in records
    })

    if len(sample_ids) != 80:
        raise ValueError(
            f"Expected 80 training samples, found {len(sample_ids)}"
        )

    shuffled = sample_ids.copy()
    random.Random(VALIDATION_SEED).shuffle(shuffled)

    validation_ids = set(shuffled[:16])
    model_train_ids = set(shuffled[16:])

    if model_train_ids & validation_ids:
        raise RuntimeError("Training/validation leakage detected")

    return (
        subset_records(records, model_train_ids),
        subset_records(records, validation_ids),
        model_train_ids,
        validation_ids,
    )


def evaluate_transition(
    train_records,
    eval_records,
    prefetch_size,
):
    """Train on one split and evaluate on a separate split."""
    benchmark.PREFETCH_SIZE = prefetch_size

    predictor = benchmark.train_transition_model(
        train_records
    )

    result = benchmark.run_test(
        eval_records,
        "TRANSITION",
        trained_transition=predictor,
    )

    result["prefetch_size"] = prefetch_size
    return result


def print_result(label, result):
    precision = result["precision_pct"]
    precision_text = (
        f"{precision:.2f}%"
        if precision is not None
        else "N/A"
    )

    print(
        f"{label:<22}"
        f" size={result.get('prefetch_size', 0):<2}"
        f" precision={precision_text:>8}"
        f" coverage={result['coverage_pct']:6.2f}%"
        f" disk_misses={result['demand_disk_misses']:>7,}"
        f" cost/request="
        f"{result['modeled_latency_per_demand_request_ms']:.4f} ms"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("=" * 100)
    print("VALIDATION-BASED ADAPTIVE PREFETCH EVALUATION")
    print("Original training set: 80 samples")
    print("Model training: 64 | Validation: 16 | Final test: 20")
    print(f"Candidate prefetch sizes: {PREFETCH_SIZES}")
    print(
        f"GPU capacity={benchmark.GPU_CAPACITY}, "
        f"CPU capacity={benchmark.CPU_CAPACITY}"
    )
    print("=" * 100)

    for workload, prefix in benchmark.WORKLOADS.items():
        dataset_dir = (
            benchmark.PROJECT_ROOT / "results" / "dataset"
        )

        train_records = benchmark.load_jsonl(
            dataset_dir / f"{prefix}_train.jsonl"
        )
        test_records = benchmark.load_jsonl(
            dataset_dir / f"{prefix}_test.jsonl"
        )

        original_train_ids = {
            r["sample_id"] for r in train_records
        }
        test_ids = {
            r["sample_id"] for r in test_records
        }

        if original_train_ids & test_ids:
            raise RuntimeError(
                f"Original train/test leakage: {workload}"
            )

        (
            model_train_records,
            validation_records,
            model_train_ids,
            validation_ids,
        ) = split_training_samples(train_records)

        if model_train_ids & validation_ids:
            raise RuntimeError(
                f"Training/validation leakage: {workload}"
            )

        print(f"\n{'=' * 35} {workload} {'=' * 35}")
        print(
            f"Model-training samples={len(model_train_ids)}, "
            f"validation samples={len(validation_ids)}, "
            f"final test samples={len(test_ids)}"
        )

        # --------------------------------------------------
        # PHASE A: Select size using validation only.
        # --------------------------------------------------
        validation_results = []

        print("\nValidation sweep:")

        for size in PREFETCH_SIZES:
            result = evaluate_transition(
                model_train_records,
                validation_records,
                size,
            )

            result["workload"] = workload
            result["stage"] = "validation"
            all_results.append(result)
            validation_results.append(result)

            print_result(f"Validation", result)

        # Select by lowest modeled cost. Break ties using fewer
        # disk misses, then higher coverage.
        best = min(
            validation_results,
            key=lambda r: (
                r["modeled_latency_per_demand_request_ms"],
                r["demand_disk_misses"],
                -r["coverage_pct"],
            ),
        )

        selected_size = best["prefetch_size"]

        print(
            f"\nSelected size from validation: {selected_size}"
        )
        print(
            "Validation modeled cost/request: "
            f"{best['modeled_latency_per_demand_request_ms']:.4f} ms"
        )

        # --------------------------------------------------
        # PHASE B: Evaluate untouched test set.
        # Retrain transition counts on all 80 original train samples.
        # --------------------------------------------------
        print("\nFinal held-out test:")

        # LRU baseline.
        benchmark.PREFETCH_SIZE = 8
        lru_result = benchmark.run_test(
            test_records,
            "LRU",
        )
        lru_result["workload"] = workload
        lru_result["stage"] = "test"
        lru_result["prefetch_size"] = 0
        all_results.append(lru_result)

        print_result("LRU baseline", lru_result)

        # Fixed-size baseline (size 8).
        fixed_result = evaluate_transition(
            train_records,
            test_records,
            8,
        )
        fixed_result["workload"] = workload
        fixed_result["stage"] = "test"
        fixed_result["policy"] = "fixed_size_8"
        all_results.append(fixed_result)

        print_result("Fixed-size policy", fixed_result)

        # Adaptive size selected using validation.
        adaptive_result = evaluate_transition(
            train_records,
            test_records,
            selected_size,
        )
        adaptive_result["workload"] = workload
        adaptive_result["stage"] = "test"
        adaptive_result["policy"] = "validation_selected"
        all_results.append(adaptive_result)

        print_result("Adaptive policy", adaptive_result)

        fixed_cost = (
            fixed_result["modeled_latency_per_demand_request_ms"]
        )
        adaptive_cost = (
            adaptive_result["modeled_latency_per_demand_request_ms"]
        )

        change_pct = (
            100 * (adaptive_cost - fixed_cost) / fixed_cost
            if fixed_cost else 0.0
        )

        print(
            "Adaptive vs fixed-size-8 modeled cost change: "
            f"{change_pct:+.2f}%"
        )

    json_path = (
        OUTPUT_DIR / "adaptive_prefetch_evaluation.json"
    )
    csv_path = (
        OUTPUT_DIR / "adaptive_prefetch_evaluation.csv"
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = list(dict.fromkeys(
            key
            for result in all_results
            for key in result.keys()
        ))

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(all_results)


    print("\nEvaluation complete.")
    print(f"JSON results: {json_path}")
    print(f"CSV results:  {csv_path}")
    print(
        "All costs are simulated cache costs, "
        "not end-to-end OLMoE inference latency."
    )


if __name__ == "__main__":
    main()
