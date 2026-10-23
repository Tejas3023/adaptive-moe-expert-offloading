"""Analyze existing timing-aware MoE simulation results (no rerun required).

Run from the project root:
    python analyze_timing_results.py

Reads results/experiments/timing_aware_strategy_comparison.json and writes:
    results/experiments/timing_aware_test_analysis.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path.cwd()
JSON_PATH = ROOT / "results" / "experiments" / "timing_aware_strategy_comparison.json"
CSV_PATH = ROOT / "results" / "experiments" / "timing_aware_test_analysis.csv"

if not JSON_PATH.exists():
    raise SystemExit(
        f"Cannot find {JSON_PATH}\n"
        "Run this script from your project root, where results/ exists."
    )

with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Support either a top-level list or a dictionary wrapping result rows.
if isinstance(data, dict):
    rows = data.get("results", data.get("rows", []))
else:
    rows = data
if not isinstance(rows, list) or not rows:
    raise SystemExit("Could not find result rows in the JSON file.")

rows = [r for r in rows if isinstance(r, dict) and r.get("split") == "test"]
if not rows:
    raise SystemExit("No rows with split='test' found in the JSON file.")

# LRU is the per-workload reference for relative simulated wall time.
baselines = {}
for r in rows:
    if str(r.get("method", "")).upper() == "LRU":
        baselines[r.get("workload")] = float(r.get("simulated_future_wall_ms", 0.0))

columns = [
    "workload", "method", "prefetch_size", "selective_threshold",
    "simulated_future_wall_ms", "delta_vs_lru_ms", "delta_vs_lru_pct",
    "total_demand_stall_ms", "demand_disk_misses",
    "prefetch_transfers_scheduled", "prefetch_ready_before_demand",
    "prefetch_late_at_demand", "prefetch_not_ready_at_demand",
    "wasted_or_unconsumed_prefetches", "prefetch_utilization_pct",
    "prefetch_queue_wait_ms", "prefetch_queue_wait_count", "avg_prefetch_queue_wait_ms",
    "demand_queue_wait_ms", "demand_queue_wait_count", "avg_demand_queue_wait_ms",
    "demand_waiting_on_prefetch_ms", "demand_waiting_on_prefetch_count",
    "demand_waiting_on_demand_transfer_ms", "demand_waiting_on_demand_transfer_count",
    "total_transfer_work_ms", "prefetch_transfer_work_ms", "demand_transfer_work_ms",
]

analysis_rows = []
for r in rows:
    workload = r.get("workload", "UNKNOWN")
    wall = float(r.get("simulated_future_wall_ms", 0.0))
    baseline = baselines.get(workload)
    delta_ms = wall - baseline if baseline is not None else None
    delta_pct = (100.0 * delta_ms / baseline) if baseline not in (None, 0) else None
    out = {k: r.get(k) for k in columns if k not in ("delta_vs_lru_ms", "delta_vs_lru_pct")}
    out["delta_vs_lru_ms"] = delta_ms
    out["delta_vs_lru_pct"] = delta_pct
    analysis_rows.append(out)

# Consistent ordering: workload, then LRU followed by strategy names.
method_order = {"LRU": 0, "HISTORY": 1, "TRANSITION": 2, "SELECTIVE": 3, "EAM": 4}
analysis_rows.sort(key=lambda r: (str(r.get("workload")), method_order.get(str(r.get("method", "")).upper(), 99)))

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(analysis_rows)


def val(row, key, default=0.0):
    x = row.get(key)
    return default if x is None else x

print("\nHELD-OUT TEST: SIMULATED WALL TIME VS LRU")
print("Times are simulated under configured costs, not measured hardware latency.\n")
print(f"{'WORKLOAD':<10} {'METHOD':<11} {'TIME (ms)':>13} {'DELTA (ms)':>12} {'DELTA (%)':>10} {'DISK MISS':>11}")
print("-" * 74)
for r in analysis_rows:
    dms = r.get("delta_vs_lru_ms")
    dpct = r.get("delta_vs_lru_pct")
    print(f"{str(r.get('workload')):<10} {str(r.get('method')):<11} "
          f"{float(val(r, 'simulated_future_wall_ms')):>13.2f} "
          f"{(float(dms) if dms is not None else 0.0):>12.2f} "
          f"{(float(dpct) if dpct is not None else 0.0):>9.3f}% "
          f"{int(val(r, 'demand_disk_misses')):>11,}")

print("\nPREFETCH AND QUEUE DIAGNOSTICS (HELD-OUT TEST)")
print(f"{'WORKLOAD':<10} {'METHOD':<11} {'SCHED':>7} {'READY':>7} {'LATE':>6} {'WASTED':>7} "
      f"{'PF Q ms':>10} {'DEM Q ms':>10} {'WAIT PF ms':>11} {'WAIT DEM ms':>12}")
print("-" * 108)
for r in analysis_rows:
    print(f"{str(r.get('workload')):<10} {str(r.get('method')):<11} "
          f"{int(val(r, 'prefetch_transfers_scheduled')):>7,} "
          f"{int(val(r, 'prefetch_ready_before_demand')):>7,} "
          f"{int(val(r, 'prefetch_late_at_demand')):>6,} "
          f"{int(val(r, 'wasted_or_unconsumed_prefetches')):>7,} "
          f"{float(val(r, 'prefetch_queue_wait_ms')):>10.2f} "
          f"{float(val(r, 'demand_queue_wait_ms')):>10.2f} "
          f"{float(val(r, 'demand_waiting_on_prefetch_ms')):>11.2f} "
          f"{float(val(r, 'demand_waiting_on_demand_transfer_ms')):>12.2f}")

print("\nDEMANDS THAT JOINED AN ALREADY-RUNNING TRANSFER")
print(f"{'WORKLOAD':<10} {'METHOD':<11} {'PREFETCH JOINS':>15} {'PREFETCH WAIT ms':>17} "
      f"{'DEMAND JOINS':>13} {'DEMAND WAIT ms':>15}")
print("-" * 88)
for r in analysis_rows:
    print(f"{str(r.get('workload')):<10} {str(r.get('method')):<11} "
          f"{int(val(r, 'demand_waiting_on_prefetch_count')):>15,} "
          f"{float(val(r, 'demand_waiting_on_prefetch_ms')):>17.2f} "
          f"{int(val(r, 'demand_waiting_on_demand_transfer_count')):>13,} "
          f"{float(val(r, 'demand_waiting_on_demand_transfer_ms')):>15.2f}")

print(f"\nSaved analysis CSV: {CSV_PATH}")
print("Note: queue-wait totals overlap with simulated compute time and should not be added directly to wall time.")
print("Note: 'wasted/unconsumed' is the simulator's accounting category; inspect it alongside ready/late counts.")
