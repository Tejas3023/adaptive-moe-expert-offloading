import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(csv_path: Path) -> list[dict]:
    """
    Load experiment results from a CSV file.
    """

    results = []

    with open(
        csv_path,
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            results.append(
                {
                    "capacity": int(
                        row["capacity"]
                    ),
                    "hits": int(
                        row["hits"]
                    ),
                    "misses": int(
                        row["misses"]
                    ),
                    "evictions": int(
                        row["evictions"]
                    ),
                    "hit_rate_percent": float(
                        row["hit_rate_percent"]
                    ),
                }
            )

    return results


def main():

    print("=" * 80)
    print("CACHE CAPACITY EXPERIMENT PLOT")
    print("=" * 80)

    csv_path = Path(
        "results/experiments/"
        "cache_capacity_results.csv"
    )

    output_path = Path(
        "results/plots/"
        "cache_capacity_vs_hit_rate.png"
    )

    print("\nLoading experiment results...")
    print(f"Input file: {csv_path}")

    results = load_results(csv_path)

    capacities = [
        result["capacity"]
        for result in results
    ]

    hit_rates = [
        result["hit_rate_percent"]
        for result in results
    ]

    print(
        f"Results loaded: {len(results)}"
    )

    print("\nCreating graph...")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        capacities,
        hit_rates,
        marker="o",
    )

    plt.xlabel(
        "Cache Capacity (Number of Experts)"
    )

    plt.ylabel(
        "Cache Hit Rate (%)"
    )

    plt.title(
        "Effect of Cache Capacity on Expert Cache Hit Rate"
    )

    plt.xticks(capacities)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    print(
        f"\nGraph saved successfully."
    )

    print(
        f"Output file: {output_path}"
    )

    print("\n" + "=" * 80)
    print("PLOT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()