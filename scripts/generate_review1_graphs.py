"""
Review 1 graph generator for Adaptive MoE Expert Offloading.

IMPORTANT:
- Strategy/cache values below are PRELIMINARY 10-sample results.
- The 100-sample traces were collected and split 80/20, but the strategy
  comparison has not yet been recomputed on the expanded split.
- Do not label the preliminary strategy graphs as 100-sample results.

Copy to:
    scripts/generate_review1_graphs_updated.py

Run:
    python scripts/generate_review1_graphs_updated.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/graphs")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Expert locality
windows = [1, 2, 4, 8, 16, 32, 64]
hits = [0.27, 0.88, 3.39, 20.31, 41.17, 57.58, 74.77]

plt.figure(figsize=(8, 5))
plt.plot(windows, hits, marker="o")
plt.xlabel("Lookback / request window")
plt.ylabel("Hit rate (%)")
plt.title("Expert Locality vs Lookback Window (Preliminary WikiText)")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "01_locality_vs_window.png", dpi=200)
plt.close()

# 2. WikiText predictor precision vs coverage
names = ["History", "Transition", "Selective"]
precision = [31.83, 41.17, 85.07]
coverage = [31.80, 41.10, 1.41]

x = np.arange(len(names))
w = 0.36

plt.figure(figsize=(8, 5))
plt.bar(x - w/2, precision, w, label="Precision")
plt.bar(x + w/2, coverage, w, label="Coverage")
plt.xticks(x, names)
plt.ylabel("Percent (%)")
plt.ylim(0, 100)
plt.title("Prediction Precision vs Coverage — WikiText (10-sample)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "02_precision_vs_coverage_wikitext_10sample.png", dpi=200)
plt.close()

# 3. Selective threshold trade-off
thresholds = [0, 0.025, 0.05, 0.075, 0.10, 0.105]
sp = [41.17, 41.26, 56.07, 71.37, 83.68, 85.07]
sc = [41.10, 41.05, 20.85, 6.67, 2.11, 1.41]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, sp, marker="o", label="Precision")
plt.plot(thresholds, sc, marker="o", label="Coverage")
plt.xlabel("Relative transition confidence threshold")
plt.ylabel("Percent (%)")
plt.ylim(0, 100)
plt.title("Selective Prediction Trade-off — WikiText (10-sample)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "03_selective_threshold_tradeoff.png", dpi=200)
plt.close()

# 4. Disk fetches, including LRU baseline
workloads = ["WikiText", "GSM8K", "MBPP"]
lru_disk = [6675, 9047, 6561]
hist_disk = [5948, 7842, 5426]
trans_disk = [5664, 6591, 4571]
sel_disk = [6648, 9036, 6364]

x = np.arange(len(workloads))
w = 0.20

plt.figure(figsize=(9, 5))
plt.bar(x - 1.5*w, lru_disk, w, label="LRU")
plt.bar(x - 0.5*w, hist_disk, w, label="History")
plt.bar(x + 0.5*w, trans_disk, w, label="Transition")
plt.bar(x + 1.5*w, sel_disk, w, label="Selective")
plt.xticks(x, workloads)
plt.ylabel("Disk fetches")
plt.title("Disk Fetches by Strategy and Workload (10-sample)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "04_disk_fetches_all_strategies_10sample.png", dpi=200)
plt.close()

# 5. Average modeled latency
# This is NOT measured end-to-end OLMoE generation latency.
lru_lat = [0.4699, 1.2688, 2.8834]
hist_lat = [0.4870, 1.2798, 2.9228]
trans_lat = [0.5063, 1.2848, 2.8947]
sel_lat = [0.4701, 1.2688, 2.8838]

plt.figure(figsize=(9, 5))
plt.bar(x - 1.5*w, lru_lat, w, label="LRU")
plt.bar(x - 0.5*w, hist_lat, w, label="History")
plt.bar(x + 0.5*w, trans_lat, w, label="Transition")
plt.bar(x + 1.5*w, sel_lat, w, label="Selective")
plt.xticks(x, workloads)
plt.ylabel("Average modeled latency (ms)")
plt.title("Average Cache/Offloading Latency (10-sample)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "05_average_modeled_latency_10sample.png", dpi=200)
plt.close()

# 6. Precision vs coverage for each workload
strategy_names = ["History", "Transition", "Selective"]
precision_by_workload = {
    "WikiText": [31.83, 41.17, 85.07],
    "GSM8K": [31.87, 39.42, 76.84],
    "MBPP": [43.11, 55.58, 90.97],
}
coverage_by_workload = {
    "WikiText": [31.80, 41.10, 1.41],
    "GSM8K": [31.82, 39.26, 1.30],
    "MBPP": [42.87, 54.94, 19.51],
}

x = np.arange(len(strategy_names))
w = 0.25

for workload in workloads:
    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, precision_by_workload[workload], w, label="Precision")
    plt.bar(x + w/2, coverage_by_workload[workload], w, label="Coverage")
    plt.xticks(x, strategy_names)
    plt.ylabel("Percent (%)")
    plt.ylim(0, 100)
    plt.title(f"Precision vs Coverage — {workload} (10-sample)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT / f"06_precision_coverage_{workload.lower()}_10sample.png",
        dpi=200
    )
    plt.close()

# 7. Expanded 100-sample trace scale
trace_workloads = ["WikiText", "GSM8K", "MBPP"]
expert_selections = [1175040, 709888, 230400]

plt.figure(figsize=(8, 5))
plt.bar(trace_workloads, expert_selections)
plt.ylabel("Expert selections")
plt.title("Expanded Routing Traces: 100 Samples per Workload")
plt.tight_layout()
plt.savefig(OUT / "07_100sample_trace_scale.png", dpi=200)
plt.close()

# 8. Expanded 100-sample input-token counts
tokens = [9180, 5546, 1800]

plt.figure(figsize=(8, 5))
plt.bar(trace_workloads, tokens)
plt.ylabel("Input tokens")
plt.title("Expanded Routing Traces: Input Tokens")
plt.tight_layout()
plt.savefig(OUT / "08_100sample_input_tokens.png", dpi=200)
plt.close()

print("Generated Review 1 graphs in:", OUT.resolve())
for f in sorted(OUT.glob("*.png")):
    print(" ", f)
