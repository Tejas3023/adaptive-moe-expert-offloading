# Day 5 to 10 Progress — Adaptive MoE Expert Offloading

**Project:** Adaptive Mixture-of-Experts (MoE) Expert Offloading  
**Model:** `allenai/OLMoE-1B-7B-0924`  
**Development environment:** Windows + Python virtual environment + PyTorch + CUDA  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU

---

## 1. Overview

During Days 5–10, the project moved beyond the basic LRU-cache baseline developed earlier and started implementing **expert locality, history-based prefetching, latency modeling, multi-tier caching, and layer-aware prefetching**.

The main goal was to investigate whether the routing history of OLMoE can be used to predict which experts will be needed next, so those experts can be loaded into faster memory before they are requested.

The work completed during this phase can be summarized as:

```text
Real OLMoE routing traces
        ↓
Expert locality analysis
        ↓
History-based prediction
        ↓
Selective prefetching
        ↓
Layer-aware caching/prefetching
        ↓
Latency model
        ↓
GPU / CPU / Disk multi-tier cache
        ↓
Real memory-constrained evaluation
```

---

# 2. WikiText Workload and Larger Routing Trace

The project was extended from the small initial real OLMoE trace to a larger WikiText workload.

The evaluation trace used was:

```text
results/traces/wikitext_10_samples.jsonl
```

It contains routing information from:

```text
10 WikiText samples
16 OLMoE layers
```

The benchmark contained:

```text
159,360 total expert requests/selections
```

Since every token selects 8 experts, the trace represents a substantial number of expert-routing decisions compared with the initial 14-token validation experiment.

This trace became the main workload for evaluating caching and prefetching strategies.

---

# 3. Expert Locality Analysis

Before implementing a predictor, expert reuse was investigated.

The important observation was that expert routing has **temporal locality**.

For consecutive tokens, the average overlap was approximately:

```text
2.87 out of 8 experts
```

which corresponds to:

```text
35.83% overlap
```

When the previous 8 tokens were considered as a history window, the average coverage increased to approximately:

```text
6.04 out of 8 experts
```

or:

```text
75.46% potential coverage
```

This suggested that recent routing decisions contain useful information about future expert requests.

This became the motivation for the history-based prefetcher.

---

# 4. History-Based Prefetcher

A new module was created:

```text
src/prefetch/history_prefetcher.py
```

The `HistoryPrefetcher` maintains a recent history of token-level expert selections.

Conceptually:

```text
Previous token  → [experts]
Previous token  → [experts]
Previous token  → [experts]
        ...
        ↓
History window
        ↓
Count expert frequency
        ↓
Rank experts
        ↓
Predict next experts
```

The predictor uses a configurable:

```text
history_size
prefetch_size
```

For example:

```python
HistoryPrefetcher(
    history_size=8,
    prefetch_size=8
)
```

The predictor counts how often each expert appeared in the recent history and selects the most frequent experts.

A deterministic tie-breaking rule was also used by sorting by expert ID when frequencies are equal.

---

# 5. History Window Evaluation

The history predictor was tested with:

```text
Window sizes:
1
2
4
8
16
32
```

The results were:

| Window | Correct | Predicted | Actual | Coverage | Precision |
|---:|---:|---:|---:|---:|---:|
| 1 | 56,636 | 158,080 | 158,080 | 35.83% | 35.83% |
| 2 | 48,149 | 158,080 | 158,080 | 30.46% | 30.46% |
| 4 | 48,637 | 158,080 | 158,080 | 30.77% | 30.77% |
| 8 | 50,556 | 158,080 | 158,080 | 31.98% | 31.98% |
| 16 | 52,867 | 158,080 | 158,080 | 33.44% | 33.44% |
| 32 | 53,649 | 158,080 | 158,080 | 33.94% | 33.94% |

The 32-token window produced the highest coverage among the tested windows, but the 8-token window was retained as a practical configuration for later experiments.

An important result is that simply looking at recent expert frequency does **not** perfectly predict the next routing decision. This provides a useful baseline for more advanced prediction methods.

---

# 6. Prefetch-Aware Cache

The existing expert cache was extended to support prefetching.

The cache is layer-aware and identifies an expert using:

```text
(layer_id, expert_id)
```

This is important because:

```text
Expert 5 in Layer 0
```

and:

```text
Expert 5 in Layer 1
```

are different sets of parameters.

The cache now tracks statistics including:

```text
hits
misses
evictions
prefetches
prefetch_hits
prefetch_evictions
```

A basic cache test confirmed that predicted experts can be inserted before an actual request arrives.

---

# 7. Cache Capacity Meaning

The cache capacities used in the experiments were:

```text
8
16
32
64
```

These numbers represent the **number of layer-specific experts that can be resident in the cache at the same time**.

For example:

```text
Cache capacity = 8
```

means the cache can hold 8 `(layer_id, expert_id)` entries.

It does NOT mean:

```text
8 experts per layer
```

Instead, it means:

```text
8 total layer-aware expert entries
```

The same idea applies to capacities 16, 32 and 64.

This distinction became especially important once layer-aware caching was introduced.

---

# 8. Selective Prefetching

A benchmark was created to compare different numbers of predicted experts.

The configuration was:

```text
History window = 8

Prefetch sizes:
1
2
4
8

Cache capacities:
8
16
32
64
```

The prediction results were:

| Prefetch Size | Precision | Coverage |
|---:|---:|---:|
| 1 | 55.52% | 6.88% |
| 2 | 48.06% | 11.92% |
| 4 | 39.99% | 19.84% |
| 8 | 31.98% | 31.72% |

The main trade-off is clear:

```text
More predictions
      ↓
Higher coverage
      ↓
Lower precision
```

For example, predicting only one expert gives the highest precision:

```text
55.52%
```

while predicting eight experts increases coverage to:

```text
31.72%
```

but reduces precision to:

```text
31.98%
```

This demonstrates why selective prefetching is important: predicting too many experts can introduce unnecessary cache activity and evictions.

---

# 9. Baseline vs History Prefetching

The baseline LRU cache was compared with history-based prefetching.

Baseline results:

| Capacity | Hits | Misses | Hit Rate |
|---:|---:|---:|---:|
| 8 | 40,937 | 118,423 | 25.69% |
| 16 | 74,607 | 84,753 | 46.82% |
| 32 | 117,852 | 41,508 | 73.95% |
| 64 | 159,296 | 64 | 99.96% |

History-based prefetching with window 8 showed:

| Capacity | Hits | Misses | Hit Rate |
|---:|---:|---:|---:|
| 8 | 38,021 | 121,339 | 23.86% |
| 16 | 79,012 | 80,348 | 49.58% |
| 32 | 117,784 | 41,576 | 73.91% |
| 64 | 159,296 | 64 | 99.96% |

The important conclusion is that **history-based prefetching did not consistently improve cache hit rate**.

At small capacity, aggressive prefetching can actually make the cache worse because incorrectly predicted experts occupy space and cause useful experts to be evicted.

This is an important experimental result rather than a failure: it shows that a simple frequency-based predictor is not sufficient by itself.

---

# 10. Latency-Aware Cache

A latency-aware cache was implemented in:

```text
src/cache/latency_cache.py
```

A cost model was added in:

```text
src/cache/cost_model.py
```

The model distinguishes between:

```text
GPU hit
CPU → GPU fetch
Disk → GPU fetch
Prefetch
```

The initial simulation costs were:

```text
GPU hit:       0.05 ms
CPU → GPU:     1.00 ms
Disk → GPU:   10.00 ms
Prefetch:      1.00 ms
```

These values are simulation parameters used to compare cache strategies.

The latency-aware cache tracks:

```text
hits
misses
evictions
prefetches
prefetch_hits
total_latency_ms
```

---

# 11. Latency Benchmark

The latency benchmark compared the baseline LRU cache against history-based prefetching.

For the baseline:

| Cache | Hit Rate | Total Latency |
|---:|---:|---:|
| 8 | 25.63% | 120,561.05 ms |
| 16 | 46.59% | 88,832.95 ms |
| 32 | 73.00% | 48,837.00 ms |
| 64 | 95.48% | 14,804.20 ms |

For history prefetching with window 8 and prefetch size 1:

| Cache | Hit Rate | Total Latency |
|---:|---:|---:|
| 8 | 23.86% | 128,654.05 ms |
| 16 | 47.05% | 88,326.90 ms |
| 32 | 73.10% | 48,690.80 ms |
| 64 | 95.48% | 14,804.20 ms |

The result shows that prefetching has to be evaluated using **latency**, not only prediction accuracy.

A prediction can be correct but still be unhelpful if its movement into the cache causes other useful experts to be evicted.

---

# 12. GPU Transfer Benchmark

A hardware transfer benchmark was created:

```text
scripts/measure_gpu_transfer.py
```

The GPU was:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
```

Measured results:

| Transfer Size | Time | Bandwidth |
|---:|---:|---:|
| 1 MB | 0.164 ms | 5.971 GB/s |
| 10 MB | 1.618 ms | 6.037 GB/s |
| 50 MB | 7.969 ms | 6.127 GB/s |
| 100 MB | 15.935 ms | 6.128 GB/s |
| 250 MB | 41.171 ms | 5.930 GB/s |

The measured bandwidth was approximately:

```text
5.9–6.1 GB/s
```

This provides real hardware information that can later be used to improve the simulated transfer-cost model.

---

# 13. Real OLMoE Expert Size

A script was created:

```text
scripts/inspect_olmoe_expert_size.py
```

The real OLMoE model was loaded with offloading.

The inspection found:

```text
Experts per layer: 64
Parameters per expert: 6,291,456
Expert size: 12.00 MB
```

Therefore, one OLMoE layer contains approximately:

```text
64 × 12 MB = 768 MB
```

of expert parameters.

Since OLMoE contains 16 layers, there are:

```text
16 × 64 = 1,024
```

layer-specific experts in the model.

This measurement is important because cache capacity can now be related to actual memory consumption.

---

# 14. Multi-Tier Expert Cache

A multi-tier cache was implemented:

```text
src/cache/multi_tier_cache.py
```

The cache represents three storage levels:

```text
GPU
 ↓
CPU
 ↓
Disk
```

The idea is:

```text
Request expert
      ↓
Is it in GPU?
  ↓ yes → GPU hit
  ↓ no
Is it in CPU?
  ↓ yes → CPU hit + move to GPU
  ↓ no
Fetch from disk
```

This more closely represents the actual expert-offloading problem.

A test confirmed the expected behavior:

```text
GPU cache → fastest
CPU cache → slower
Disk → slowest
```

The test also confirmed that moving an expert from CPU to GPU can cause a GPU eviction.

---

# 15. Multi-Tier Benchmark

Several GPU/CPU cache configurations were evaluated.

| GPU Capacity | CPU Capacity | GPU Hits | CPU Hits | Disk Fetches | Avg Latency |
|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 38,019 | 59,944 | 61,397 | 4.2408 ms |
| 16 | 32 | 74,606 | 71,089 | 13,665 | 1.3270 ms |
| 32 | 32 | 118,099 | 40,840 | 421 | 0.3197 ms |
| 32 | 64 | 118,099 | 41,197 | 64 | 0.2996 ms |
| 64 | 64 | 159,296 | 0 | 64 | 0.0540 ms |

The main observation is:

```text
More GPU capacity
        ↓
Fewer CPU/disk accesses
        ↓
Lower average latency
```

The 64/64 configuration keeps all 64 currently used expert IDs available at the GPU level, leaving only compulsory initial misses.

---

# 16. Real Memory-Constrained Capacity

The measured expert size was combined with memory limits.

The configured limits were:

```text
GPU memory: 3072 MB
CPU memory: 2048 MB
Expert size: 12 MB
```

This gives approximately:

```text
GPU capacity = 3072 / 12 = 256 experts
CPU capacity = 2048 / 12 ≈ 170 experts
```

Therefore, the real-capacity benchmark used:

```text
GPU expert capacity: 256
CPU expert capacity: 170
```

This is much more realistic than choosing arbitrary cache capacities.

---

# 17. Real Memory-Constrained Benchmark

The real-capacity benchmark produced:

```text
GPU hits:       152,164
CPU hits:           519
Disk fetches:     6,677

Total requests: 159,360

Total latency: 74,897.20 ms
Average latency: 0.4700 ms
```

GPU hits dominate because a large number of experts can fit in the simulated GPU capacity.

However, CPU and disk accesses still occur because the cache is layer-aware and contains 1,024 possible `(layer, expert)` combinations while the GPU can hold only 256 of them.

This experiment connects the abstract cache simulation to the actual memory constraints of the laptop.

---

# 18. Layer-Aware History Prefetcher

A separate layer-aware predictor was implemented:

```text
src/prefetch/layer_aware_prefetcher.py
```

The main change is that routing history is maintained separately for each layer.

Conceptually:

```text
Layer 0
  ↓
History 0
  ↓
Predictions 0

Layer 1
  ↓
History 1
  ↓
Predictions 1

...
```

This is important because expert routing patterns can differ between transformer layers.

A test confirmed independent histories.

For example:

```text
Layer 0 history:
[5, 10, 20, 30]
[5, 10, 15, 20]

Layer 1 history:
[40, 50, 60, 70]
[40, 50, 80, 90]
```

The predictions were:

```text
Layer 0 → [5, 10, 20, 15]
Layer 1 → [40, 50, 60, 70]
```

This confirms that predictions are generated using the routing history belonging to the correct layer.

---

# 19. Layer-Aware Prefetch Benchmark

The layer-aware predictor was evaluated using:

```text
GPU capacity: 256
CPU capacity: 170
History window: 8
Prefetch sizes: 1, 2, 4, 8
```

Results:

| Prefetch Size | GPU Hits | CPU Hits | Disk | Avg Latency | Precision | Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 152,164 | 522 | 6,674 | 0.4698 ms | 55.52% | 6.88% |
| 2 | 152,164 | 522 | 6,674 | 0.4698 ms | 48.06% | 11.92% |
| 4 | 152,164 | 521 | 6,675 | 0.4699 ms | 39.99% | 19.84% |
| 8 | 152,164 | 520 | 6,676 | 0.4699 ms | 31.98% | 31.72% |

The latency improvement was very small.

The important conclusion is that **prediction accuracy and system-level performance are different metrics**.

A predictor may have useful precision/coverage but still fail to significantly reduce latency if the cache already has a high GPU hit rate or if prefetching introduces additional cache pressure.

---

# 20. Important Experimental Findings

Several useful findings were established during Days 5–10.

### Finding 1: Expert routing has locality

Recent routing history contains reusable expert information.

```text
Previous 8-token history
        ↓
Useful information about upcoming experts
```

However, the information is not sufficient for perfect prediction.

### Finding 2: Simple frequency prediction has limited accuracy

The history predictor achieved approximately:

```text
31.98% coverage
```

for an 8-token history and top-8 predictions.

Therefore, simply counting frequently used experts is not enough.

### Finding 3: Larger prefetch size increases coverage but reduces precision

```text
Prefetch 1 → 55.52% precision
Prefetch 2 → 48.06%
Prefetch 4 → 39.99%
Prefetch 8 → 31.98%
```

This demonstrates the precision/coverage trade-off.

### Finding 4: Prefetching can hurt a small cache

Incorrectly predicted experts consume cache space.

Therefore:

```text
Prediction
   ↓
Prefetch
   ↓
Cache insertion
   ↓
Possible eviction of useful expert
```

A prefetcher must therefore be evaluated together with the cache policy.

### Finding 5: Layer awareness is necessary

The cache and predictor must distinguish:

```text
(layer_id, expert_id)
```

rather than using only:

```text
expert_id
```

because Expert 5 in Layer 0 and Expert 5 in Layer 1 are different model parameters.

### Finding 6: Memory capacity strongly affects latency

The multi-tier experiments showed that increasing GPU capacity can dramatically reduce CPU and disk accesses.

### Finding 7: Real expert size allows realistic capacity modeling

The measured expert size is approximately:

```text
12 MB
```

This allowed the project to move from arbitrary cache sizes toward real memory-constrained experiments.

---

# 21. Current Project Architecture

The project now has the following conceptual architecture:

```text
                    ┌───────────────────┐
                    │   WikiText Data   │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │   Real OLMoE      │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Router Decisions  │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │  JSONL Traces     │
                    └─────────┬─────────┘
                              ↓
               ┌──────────────┴──────────────┐
               ↓                             ↓
       Locality Analysis              Prefetch Predictor
                                             ↓
                                  ┌──────────┴──────────┐
                                  ↓                     ↓
                           History Based          Layer Aware
                                  ↓                     ↓
                                  └──────────┬──────────┘
                                             ↓
                                    Multi-Tier Cache
                                             ↓
                              ┌──────────────┼──────────────┐
                              ↓              ↓              ↓
                             GPU            CPU            Disk
```

---

# 22. Important Files Added During This Phase

The main new source files and scripts created during this phase include:

```text
src/cache/cost_model.py
src/cache/latency_cache.py
src/cache/multi_tier_cache.py

src/prefetch/history_prefetcher.py
src/prefetch/layer_aware_prefetcher.py
```

Benchmark and analysis scripts include:

```text
scripts/analyse_expert_locality.py
scripts/analyse_token_locality.py
scripts/collect_wikitext_traces.py
scripts/evaluate_history_prefetcher.py
scripts/inspect_olmoe_expert_size.py
scripts/measure_gpu_transfer.py
scripts/run_latency_benchmark.py
scripts/run_layer_aware_prefetch_benchmark.py
scripts/run_multi_tier_benchmark.py
scripts/run_prefetch_benchmark.py
scripts/run_real_capacity_benchmark.py
scripts/run_selective_prefetch_benchmark.py
```

Testing scripts include:

```text
scripts/test_history_prefetcher.py
scripts/test_latency_cache.py
scripts/test_layer_aware_cache.py
scripts/test_layer_aware_prefetcher.py
scripts/test_multi_tier_cache.py
scripts/test_prefetch_cache.py
```

---

# 23. Current State of the Project

At the end of Day 10, the project has progressed from a basic LRU cache simulator to a substantially more complete experimental framework.

The current pipeline is:

```text
Real OLMoE
    ↓
Real expert routing traces
    ↓
Expert locality analysis
    ↓
LRU baseline
    ↓
History-based prediction
    ↓
Selective prefetching
    ↓
Layer-aware prediction
    ↓
Latency-aware caching
    ↓
GPU / CPU / Disk multi-tier simulation
    ↓
Real memory-constrained evaluation
```

The current real-capacity baseline is approximately:

```text
GPU hits:       152,164
CPU hits:           519
Disk fetches:     6,677
Average latency:  0.4700 ms
```

The current history-based predictor demonstrates measurable routing predictability, but it does not yet provide a major latency improvement.

That result is useful because it identifies a limitation of simple frequency-based history prediction.

---

# 24. Next Phase: Transition-Based Prefetching

The next planned component is:

```text
Transition-based prefetching
```

The idea is different from simply counting how often experts appeared.

Instead, the system will learn transitions such as:

```text
Current expert
      ↓
What experts tend to appear next?
```

For example:

```text
Expert 5
   ↓
Expert 10 → 40% probability
Expert 20 → 30% probability
Expert 15 → 20% probability
Expert 30 → 10% probability
```

The predictor can then use the current routing state to estimate the next expert requests.

Conceptually:

```text
Observed expert routing
        ↓
Build transition statistics
        ↓
Current expert(s)
        ↓
Look up likely next experts
        ↓
Rank predictions
        ↓
Prefetch selected experts
        ↓
Measure cache hits and latency
```

This is the natural next step because the current history-frequency method ignores **ordering**.

For example, these histories:

```text
A → B → C
```

and:

```text
C → B → A
```

contain the same experts but represent different sequences.

A transition model can distinguish them.

---

# 25. Overall Day 5–10 Conclusion

Days 5–10 established the core experimental infrastructure needed to study adaptive expert offloading.

The project now has:

- Real OLMoE routing traces.
- Expert locality measurements.
- An LRU cache baseline.
- A history-based prefetcher.
- Selective prefetching.
- Layer-aware prediction.
- Latency-aware cache simulation.
- Real GPU transfer measurements.
- Real expert memory-size measurements.
- GPU/CPU/disk multi-tier caching.
- Memory-constrained capacity evaluation.

The most important conclusion so far is:

```text
Expert routing contains useful locality,
but simple frequency-based prediction is not
accurate enough to provide a large performance gain.
```

This motivates the next phase:

```text
History Frequency
       ↓
Transition Modeling
       ↓
Better Expert Prediction
       ↓
More Accurate Prefetching
       ↓
Lower Cache Misses
       ↓
Lower CPU/Disk Fetches
       ↓
Lower End-to-End Latency
```

The project is therefore ready to begin implementing and evaluating the **transition-based prefetcher**.
