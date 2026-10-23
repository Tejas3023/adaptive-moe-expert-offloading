import json
from collections import defaultdict

path = "results/experiments/timing_sensitivity_analysis.json"

with open(path, encoding="utf-8") as f:
    data = json.load(f)

groups = defaultdict(list)
for row in data:
    groups[(row["workload"], row["compute_ms_per_record"])].append(row)

print(
    f"{'WORKLOAD':10} {'COMPUTE':>8} {'LRU ms/req':>12} "
    f"{'BEST METHOD':>14} {'BEST ms/req':>12} {'SAVING':>10}"
)
print("-" * 75)

for (workload, compute_ms), rows in sorted(groups.items()):
    lru = next(r for r in rows if r["method"] == "LRU")
    best = min(
        rows,
        key=lambda r: r["simulated_future_ms_per_demand_request"]
    )

    lru_cost = lru["simulated_future_ms_per_demand_request"]
    best_cost = best["simulated_future_ms_per_demand_request"]
    saving_pct = (lru_cost - best_cost) / lru_cost * 100

    print(
        f"{workload:10} {compute_ms:8.1f} {lru_cost:12.6f} "
        f"{best['method']:>14} {best_cost:12.6f} "
        f"{saving_pct:9.4f}%"
    )
