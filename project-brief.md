# Final Year Project Brief

## Predictive Expert Offloading for Mixture-of-Experts LLMs on Consumer Hardware, with Distributed Caching and Timing Side-Channel Analysis

---

## 1. One-paragraph summary

Mixture-of-Experts (MoE) language models activate only a small fraction of their total parameters per token, but the full model still has to be *stored* somewhere — and on a laptop GPU, it usually doesn't fit. This project builds a system that runs an open MoE model (OLMoE-1B-7B) on consumer laptop hardware by keeping frequently-used ("hot") experts on GPU and offloading rarely-used ("cold") experts to CPU/RAM, swapping them in and out as needed. The core contribution is making this swapping *predictive* instead of purely reactive — using logged patterns of which experts tend to follow which, so the system fetches the next likely expert *before* it's needed rather than stalling on it. We extend the single-laptop system across our team's machines (turning idle CPU/RAM on teammates' laptops into a shared expert cache over LAN), and we study whether the timing differences our own caching system introduces create a measurable information leak (a side-channel) about what the model is processing — a novel security angle that connects the systems work to cybersecurity.

---

## 2. Problem statement

**Why this problem exists:** Open-weight MoE models (Mixtral 8x7B, OLMoE, DeepSeek-MoE) offer strong performance-per-active-parameter, but their *total* parameter count often exceeds the VRAM available on consumer GPUs (8–12GB on typical gaming laptops). A dense model of equivalent total size would be unusable on this hardware; a MoE model theoretically doesn't need to be, since only 1–2 experts per layer activate per token. In practice, though, naive deployment still requires all experts resident in GPU memory, defeating the purpose.

**The gap:** Existing offloading systems (Mixtral-Offloading, MoE-Infinity, HOBBIT — see references) address this with reactive LRU/LFU caching, i.e., they swap in an expert only *after* the router decides it's needed, paying the transfer latency on the critical path. Prediction-based prefetching (fetching before it's needed) is mentioned as future work in the founding offloading paper (Eliseev & Mazur, 2023) and partially explored since (e.g., lookahead-gate methods), but a from-scratch, laptop-validated implementation with a rigorous ablation between heuristic and learned predictors — plus extension to a *distributed*, multi-machine cache and a *security* analysis of the resulting timing behavior — is not something the existing literature covers end-to-end. That combination is this project's contribution.

**Who this matters to, concretely:** Students, independent developers, small research labs, and NGOs who want to run capable open-weight models without cloud GPU budgets. As more labs release MoE-architecture open weights, "can I run this on my laptop" becomes a more common and more answerable question — this project is a real answer, not a simulated one.

---

## 3. Objectives

1. Deploy an open MoE model (OLMoE-1B-7B) for inference on a single consumer laptop GPU using CPU/RAM offloading for experts that don't fit in VRAM.
2. Build a reactive hot/cold expert caching layer (baseline) as the foundation.
3. Design and implement a **predictive prefetching** layer that anticipates which expert(s) will be needed next, based on logged routing history, and overlaps the fetch with ongoing computation.
4. Compare a simple heuristic predictor (n-gram/frequency-based over expert-ID sequences) against a small learned predictor (e.g., lightweight RNN/transformer over routing history) in a proper ablation.
5. Extend the caching system across multiple laptops on a LAN, adding a third storage tier (GPU → local CPU/RAM → networked peer) and evaluate whether pooling underused resources across machines is worth the network overhead.
6. Investigate whether the timing differences introduced by the caching/offloading system constitute a measurable side-channel — i.e., can an observer infer something about the query (its topic, its expert-activation pattern) purely from response timing — and, if so, prototype and evaluate a mitigation.
7. Produce a working, demoable system plus a rigorous empirical report (latency, throughput, cache-hit rate, and security findings) comparing all configurations against clean baselines.

---

## 4. What is *not* in scope (be explicit about this with your mentor)

- **Training a MoE model from scratch.** This project uses an existing, pretrained, open-weight MoE model. Training one from scratch would require pretraining-scale data and compute far beyond a laptop's or team's capacity, and — importantly — a from-scratch model trained on limited compute wouldn't develop the kind of realistic, non-trivial expert specialization this project needs to study in the first place.
- **Building a large custom labeled dataset.** All datasets used are existing public corpora (see Section 8). Anything resembling "training data" for the predictor is generated internally by logging the model's own routing decisions during inference — not hand-labeled.
- **Full-scale production deployment.** This is a research/systems prototype validated on 2–3 laptops, not a cloud-scale serving system. Comparisons to production systems (MoE-Infinity, vLLM) are for context/baseline purposes, not a claim of matching production engineering maturity.

---

## 5. System architecture (high level)

```
                     ┌─────────────────────────────┐
                     │        Inference Loop        │
                     │   (OLMoE forward pass, per   │
                     │    layer, per token)          │
                     └───────────────┬───────────────┘
                                     │ router selects top-k experts
                                     ▼
                     ┌─────────────────────────────┐
                     │      Expert Cache Manager     │
                     │  - tracks GPU-resident experts │
                     │  - eviction policy (baseline:  │
                     │    LRU/LFU)                    │
                     └───────┬───────────────┬───────┘
                             │ hit           │ miss
                             ▼               ▼
                    ┌────────────┐   ┌──────────────────────┐
                    │ Use expert  │   │ Fetch expert weights   │
                    │ from GPU    │   │ from:                  │
                    └────────────┘   │  1. Local CPU/RAM       │
                                     │  2. Peer laptop (LAN)   │
                                     └───────────┬────────────┘
                                                 │
                                                 ▼
                                     ┌──────────────────────┐
                                     │  Predictive Prefetcher │
                                     │  - logs routing history│
                                     │  - predicts next-needed│
                                     │    expert(s)            │
                                     │  - issues fetch ahead of│
                                     │    compute (overlap)    │
                                     └──────────────────────┘

                     ┌─────────────────────────────┐
                     │   Timing Instrumentation &    │
                     │   Side-Channel Analysis Module │
                     │  - logs per-token latency       │
                     │  - tests inferability of expert/│
                     │    topic identity from timing   │
                     │  - evaluates mitigation (e.g.,  │
                     │    constant-time padding)       │
                     └─────────────────────────────┘
```

---

## 6. Development stages / timeline (5–7 months)

### Month 1 — Foundations
- Literature review (see Section 10) — read and summarize the core offloading papers, MoE architecture papers, and side-channel papers.
- Set up environment: PyTorch, HuggingFace Transformers/vLLM, CUDA toolkit matching your GPU (Victus/ROG/Predator — confirm driver + CUDA version compatibility first).
- Get OLMoE-1B-7B loading and generating text with the *simplest possible* full-offload baseline (no caching, just correctness) on one laptop. This validates the whole premise before deeper investment.
- Deliverable: working baseline inference script + literature review document.

### Month 2 — Reactive caching baseline
- Implement the Expert Cache Manager: GPU-resident hot set + CPU/RAM cold set, LRU (and optionally LFU) eviction policy.
- Instrument routing decisions: log which expert(s) are selected at each layer, for each token, across a range of workloads.
- Benchmark against the naive full-offload baseline: latency, tokens/sec, cache hit rate.
- Deliverable: working reactive caching system + baseline benchmark numbers.

### Month 3 — Predictive prefetching (core contribution, part 1)
- Build the heuristic predictor: simple n-gram/frequency model over logged expert-ID sequences, predicting the likely next expert(s) per layer.
- Integrate prefetching into the cache manager: issue fetch requests ahead of need, overlapping I/O with compute.
- Benchmark against Month 2's reactive-only baseline.
- Deliverable: working heuristic prefetcher + comparative benchmarks.

### Month 4 — Predictive prefetching (core contribution, part 2) + workload analysis
- Build the learned predictor: a small RNN/transformer trained on your own logged routing traces to predict next-token expert activation.
- Run the heuristic-vs-learned ablation — this is your primary "is the complexity worth it" result.
- As part of this, test predictability across different workload domains (code, math, multilingual, casual text — using MBPP, GSM8K, FLORES, WikiText) to see if the prefetcher's effectiveness is domain-dependent.
- Deliverable: full predictor ablation report with domain breakdown.

### Month 5 — Distributed extension
- Extend the Expert Cache Manager with a third tier: a networked peer laptop over LAN, using sockets/gRPC to request cold experts from a teammate's machine instead of only local CPU/RAM.
- Benchmark three tiers (GPU-resident / local CPU-RAM / LAN peer) for latency, and determine under what conditions (network speed, expert size) pooling across laptops is worth it versus purely local offloading.
- Deliverable: working distributed caching demo across 2+ laptops + tier comparison report.

### Month 6 — Security side-channel study
- Using the timing data already being logged, test whether expert-swap timing (cache hit vs. miss, which tier an expert was fetched from) creates an observable signal that correlates with query content/topic/domain.
- If a signal is found, prototype at least one mitigation (e.g., constant-time padding of responses, dummy prefetches to mask real ones) and measure the latency cost of closing the leak.
- Deliverable: security analysis report — either a demonstrated side-channel + mitigation trade-off, or a rigorous negative result (no exploitable signal found, with the testing methodology documented).

### Month 7 — Integration, buffer, and writing
- Full end-to-end system integration and final benchmarking pass across all configurations.
- Buffer time for anything that slipped (very likely — build this in explicitly).
- Final report writing, poster/demo preparation, defense rehearsal.
- Deliverable: final report, working live demo, presentation materials.

---

## 7. What you need (hardware, software, accounts)

**Hardware**
- Your team's laptops (Victus / ROG / Predator) — confirm exact GPU model and VRAM on each before month 1, since this determines which model configuration is feasible per machine.
- A LAN connection between team laptops for the distributed extension (same Wi-Fi network is sufficient for prototyping; ethernet preferred for cleaner latency measurements).

**Software / tooling**
- Python 3.10+, PyTorch (CUDA-enabled build matching your GPU driver)
- HuggingFace Transformers and/or vLLM (OLMoE is integrated into both, plus llama.cpp and SGLang)
- CUDA toolkit + appropriate NVIDIA drivers for your specific GPU
- A quantization library if you choose to combine offloading with quantization (bitsandbytes, HQQ, or GPTQ — optional, not required for the core project)
- Networking: Python `socket` or `grpc` for the distributed cache tier
- Experiment tracking: Weights & Biases or simple CSV/logging — you'll want clean, reproducible benchmark numbers for your report
- Git/GitHub for version control across a multi-laptop team (essential given your setup)

**Accounts / access**
- HuggingFace account (to download OLMoE weights — free, no application needed since it's fully open)
- Google Colab (optional — useful for early prototyping/validation before moving to your own laptops, and as a fallback if a laptop's GPU underperforms)
- GitHub repository for the team

**No paid compute or API costs are required** — this is one of the project's practical strengths.

---

## 8. Datasets (all public, no dataset construction required)

| Purpose | Dataset | Notes |
|---|---|---|
| General inference workload | WikiText-103 | Standard language modeling benchmark, good for baseline latency/throughput testing |
| General inference workload | C4 (Colossal Clean Crawled Corpus) | Larger, more diverse text for stress-testing |
| Code-domain workload | MBPP, HumanEval | For testing domain-dependent routing predictability |
| Math-domain workload | GSM8K | Same purpose, math domain |
| Multilingual workload | FLORES-200 | Same purpose, language domain |
| Reasoning/QA workload | MMLU | Broad domain-labeled QA, useful for both workload testing and correctness sanity checks |

You are not building or labeling any of these — they are used purely as *inference input* to generate realistic workloads and routing traces. The "data" you generate yourself is telemetry (routing logs, timing logs) from running the existing model, not a labeled training set.

---

## 9. Evaluation metrics

- **Latency:** time-to-first-token and per-token generation latency, across all configurations (naive full-offload, reactive cache, heuristic prefetch, learned prefetch, distributed tiers)
- **Throughput:** tokens/second sustained generation
- **Cache hit rate:** percentage of expert requests served from GPU-resident cache vs. requiring a fetch
- **Prefetch accuracy:** how often the predicted next expert matches the actually-needed expert
- **Network overhead (distributed tier):** latency added by LAN fetch vs. local CPU/RAM fetch, and the bandwidth/expert-size threshold where LAN pooling becomes worthwhile
- **Side-channel signal strength:** classification accuracy of an observer model attempting to infer query domain/topic from timing traces alone; latency cost of any mitigation applied

---

## 10. Reference papers (10)

These span the foundational MoE architecture literature, the offloading/caching systems literature your project builds directly on, and the side-channel security literature relevant to the security extension. All are publicly available (arXiv, and in several cases also published at peer-reviewed venues as noted).

1. **Shazeer, N., et al. (2017).** *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538. — The foundational MoE paper; introduces sparse gating and the core idea your entire project depends on.

2. **Fedus, W., Zoph, B., & Shazeer, N. (2021/2022).** *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* arXiv:2101.03961; Journal of Machine Learning Research (JMLR), 2022. — Introduces top-1 routing and the load-balancing loss; essential background for understanding expert specialization and load imbalance, relevant to your routing-predictability analysis.

3. **Jiang, A. Q., et al. (2024).** *Mixtral of Experts.* arXiv:2401.04088. — The architecture your baseline offloading comparisons (Mixtral-Offloading) target; useful for understanding top-2 routing and natural expert-load balance without an explicit auxiliary loss.

4. **Eliseev, A., & Mazur, D. (2023).** *Fast Inference of Mixture-of-Experts Language Models with Offloading.* arXiv:2312.17238. — **Primary baseline paper.** Introduces the LRU-cache + speculative-loading offloading strategy your project builds on and directly extends with prediction; explicitly lists predictive expert prefetching as future work.

5. **Xue, L., Fu, Y., Lu, Z., Mai, L., & Marina, M. (2024).** *MoE-Infinity: Offloading-Efficient MoE Model Serving.* arXiv:2401.14361. — State-of-the-art offloading system using activation-aware, trace-based expert prefetching; the closest existing system to your Month 3–4 predictive prefetcher, and a strong comparison point for your evaluation.

6. **Muennighoff, N., et al. (2024).** *OLMoE: Open Mixture-of-Experts Language Models.* arXiv:2409.02060; ICLR 2025. — The paper for your chosen base model; includes the authors' own routing-specialization analysis, directly useful for your workload-adaptive prefetching study (Month 4).

7. **Lepikhin, D., et al. (2020).** *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.* arXiv:2006.16668. — Introduces expert sharding across devices at training time; conceptually relevant background for your distributed cache extension (Month 5), even though your setting is inference-time and consumer-hardware rather than training-time and datacenter-scale.

8. **Duddu, V., Samanta, D., Rao, D. V., & Balas, V. E. (2018).** *Stealing Neural Networks via Timing Side Channels.* arXiv:1812.11720. — Early, foundational work showing that neural network execution timing leaks architectural/behavioral information; the conceptual starting point for your Month 6 side-channel study.

9. **Wei, S., et al. (2022).** *Stealthy Inference Attack on DNN via Cache-based Side-Channel Attacks.* Design, Automation & Test in Europe Conference (DATE), 2022. — Demonstrates cache-timing side channels leaking label information from DNN inference; directly relevant methodology for testing whether your expert cache introduces an analogous leak.

10. **Zheng, T., et al. (2024).** *The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems.* arXiv:2409.20002. — Shows that caching optimizations (KV cache, semantic cache) in LLM serving systems create real, exploitable timing side channels; the most directly analogous prior work to your proposed expert-cache side-channel study, and a strong methodological template to adapt.

**Two optional additions if you want a fuller set (11–12):**

11. **Cai, W., et al. (2024) et al. — HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference.** arXiv:2411.01433. — A more recent offloading system combining quantization tiers with caching; useful additional comparison baseline and shows how offloading work has evolved since Eliseev & Mazur.

12. **[InputSnatch] — Stealing Input in LLM Services via Timing Side-Channel Attacks.** arXiv:2411.18191. — A recent, concrete demonstration of timing-based prompt-stealing attacks exploiting caching in LLM serving; strengthens the security-relevance framing of your Month 6 work with a very current example.

*(Note: verify exact author lists and publication venues against the arXiv listings directly when compiling your final bibliography — some of these papers have had author lists or venue placements updated between preprint and camera-ready versions.)*

---

## 11. Risks and mitigations (worth raising with your mentor directly)

| Risk | Mitigation |
|---|---|
| OLMoE doesn't run acceptably even at baseline on the weakest team laptop | Validate this in Month 1 before committing further; fall back to a smaller MoE checkpoint or Colab for that teammate's development work if needed |
| Predictive prefetching doesn't outperform reactive caching (routing turns out to be less predictable than hoped) | This is still a valid, reportable result — document it as a negative/mixed finding with the ablation data to back it up, rather than treating it as project failure |
| LAN distributed tier is slower than local CPU/RAM, making the extension "not worth it" | Also a valid finding — report the bandwidth/latency threshold at which it would become worthwhile, which is itself a useful contribution |
| No exploitable side-channel is found in Month 6 | Report this as a rigorous negative result with full methodology — absence of a suspected vulnerability, properly tested, is a legitimate and defensible finding |
| Timeline slippage | Month 7 is deliberately kept as buffer; if you're behind by Month 5, treat the distributed tier (Month 5) or the side-channel study (Month 6) as the droppable scope, not the core prefetching system |

---

## 12. Suggested team role split (adjust to your team's strengths)

- **Person A:** Core inference pipeline + reactive caching (Months 1–2), then predictor implementation (Months 3–4)
- **Person B:** Distributed/networking layer (Month 5 lead, but networking setup can start earlier in parallel), plus benchmarking infrastructure throughout
- **Person C:** Side-channel security study (Month 6 lead, but timing instrumentation should be built in from Month 2 onward so there's data to analyze later), plus report/documentation coordination

All three should be involved in Month 1 literature review and Month 7 integration/writing regardless of individual leads above.
