# Day 1 Progress Report — Adaptive MoE Expert Offloading

**Project:** Adaptive Mixture-of-Experts (MoE) Expert Offloading  
**Model:** `allenai/OLMoE-1B-7B-0924`  
**Development environment:** Windows + Python virtual environment + PyTorch + CUDA  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM  
**System RAM:** Approximately 7.34 GB  

---

# 1. Objective of the Work Completed

The goal of the first phase was to build the infrastructure required to study expert routing in a real Mixture-of-Experts model and simulate the effect of caching experts.

The following objectives were completed:

1. Set up the Python environment and Git repository.
2. Inspect the OLMoE configuration and architecture without loading the full model.
3. Understand how expert routing is implemented internally.
4. Build an expert-routing trace logger.
5. Store routing traces in JSONL format.
6. Build a trace analyzer for routing statistics.
7. Implement an LRU-based expert cache simulator.
8. Run cache-capacity experiments on synthetic traces.
9. Check system resources.
10. Download and run the real OLMoE model using GPU, CPU, and disk offloading.
11. Perform a successful real forward pass.
12. Collect real expert-routing traces from all OLMoE layers.
13. Run routing analysis and cache-capacity experiments on real OLMoE traces.

The project has now moved from a synthetic simulation stage to experiments using routing decisions produced by the real OLMoE model.

---

# 2. Git and Project Setup

A Git repository was initialized and connected to GitHub. The main branch was pushed successfully.

Project source files and scripts were committed during development. Large or unrelated files such as presentation files, images, and reference PDFs remained untracked. The `.venv/` directory was also not tracked, which is the correct approach because a virtual environment should normally be recreated from `requirements.txt`.

---

# 3. Python Environment and Dependencies

The following important packages were installed:

```text
transformers==4.49.0
accelerate==1.4.0
safetensors
huggingface_hub
```

The versions were verified successfully:

```text
Transformers: 4.49.0
Accelerate: 1.4.0
```

PyTorch with CUDA support was already available:

```text
torch 2.6.0+cu124
```

These packages provide the functionality required to load OLMoE, run inference, access router logits, and manage model offloading.

---

# 4. OLMoE Configuration Inspection

The model inspected was:

```text
allenai/OLMoE-1B-7B-0924
```

The important configuration values were:

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,304 |
| Hidden size | 2,048 |
| Intermediate size | 1,024 |
| Transformer layers | 16 |
| Attention heads | 16 |
| Maximum position embeddings | 4,096 |
| Total experts | 64 |
| Experts selected per token | 8 |
| Router auxiliary loss coefficient | 0.01 |
| Normalize top-k probabilities | False |

The key routing configuration is:

```text
64 total experts
8 experts selected for every token
```

Conceptually, routing works as:

```text
Token hidden state
        ↓
Router / Gate
        ↓
Scores all 64 experts
        ↓
Softmax
        ↓
Select top 8 experts
        ↓
Weighted expert computation
```

Expert selection is dynamic and can change depending on the token, its context, and the transformer layer.

---

# 5. OLMoE Architecture Inspection

The model architecture was inspected on the PyTorch `meta` device. This allowed inspection of the full architecture without downloading or allocating the actual model weights.

The top-level model class is:

```text
OlmoeForCausalLM
```

Its main components are:

```text
model: OlmoeModel
lm_head: Linear
```

The model contains 16 transformer layers. A layer contains:

```text
self_attn: OlmoeSdpaAttention
mlp: OlmoeSparseMoeBlock
input_layernorm: OlmoeRMSNorm
post_attention_layernorm: OlmoeRMSNorm
```

The main MoE component is:

```text
OlmoeSparseMoeBlock
```

It contains:

```text
gate: Linear
experts: ModuleList
```

The `gate` acts as the router. Its weight has shape:

```text
(64, 2048)
```

Each expert contains:

```text
gate_proj
up_proj
down_proj
act_fn: SiLU
```

The architecture inspection reported approximately:

```text
6,919,161,856 parameters
```

During this inspection, the parameters existed only as metadata on the `meta` device and no real weights were allocated.

---

# 6. Understanding OLMoE Routing

The implementation of `OlmoeSparseMoeBlock.forward()` was inspected.

## Step 1: Flatten hidden states

Input hidden states have shape:

```text
(batch_size, sequence_length, hidden_dimension)
```

They are reshaped to:

```text
(batch_size × sequence_length, hidden_dimension)
```

For a batch size of 1 and 14 tokens:

```text
(14, 2048)
```

## Step 2: Compute router logits

The router scores all 64 experts:

```python
router_logits = self.gate(hidden_states)
```

For 14 tokens:

```text
(14, 64)
```

Each row represents one token and each column represents one expert.

## Step 3: Apply softmax

The logits are converted into routing probabilities:

```python
routing_weights = F.softmax(
    router_logits,
    dim=1,
    dtype=torch.float,
)
```

## Step 4: Select top-8 experts

```python
routing_weights, selected_experts = torch.topk(
    routing_weights,
    self.top_k,
    dim=-1,
)
```

Since `top_k = 8`, every token selects exactly 8 experts.

## Step 5: Execute selected experts

Each selected expert processes the corresponding token state. Its output is multiplied by the routing weight and added to the final hidden state.

Therefore, routing weights determine how much each selected expert contributes to the final token representation.

---

# 7. Expert Routing Trace Logger

A custom tracing system was implemented to convert router logits into structured routing records.

The main data structure is:

```python
@dataclass
class ExpertRoutingTrace:
    layer_id: int
    token_position: int
    selected_experts: List[int]
    routing_weights: List[float]
```

Each trace records:

- Transformer layer.
- Token position.
- Eight selected expert IDs.
- Routing weights.

The `ExpertTraceLogger` uses:

```python
num_experts = 64
top_k = 8
norm_topk_prob = False
```

The logger reproduces the same routing procedure used by OLMoE:

1. Receive router logits.
2. Apply softmax.
3. Select the top 8 experts.
4. Record expert IDs.
5. Record routing weights.

This tracing system is important because it will later support analysis of expert popularity, temporal locality, cache hit rates, expert reuse, and prefetching strategies.

---

# 8. Synthetic Expert Trace Test

Before using the real model, the trace logger was tested using synthetic router logits.

The test used:

```text
5 tokens
64 experts
8 experts selected per token
```

Five routing traces were created successfully. For example:

```text
Token position: 0
Selected experts:
[45, 0, 61, 35, 23, 9, 16, 1]
```

This confirmed that:

- Router logits were processed correctly.
- Softmax was applied correctly.
- The top-8 experts were extracted.
- Routing weights were stored correctly.
- Trace objects were created successfully.

The synthetic test validated the tracing pipeline before integration with the real model.

---

# 9. Trace Writer

A trace-writing component was implemented to store routing events on disk.

The format used is:

```text
JSON Lines (.jsonl)
```

Each line represents one routing event, conceptually:

```json
{
  "layer_id": 0,
  "token_position": 0,
  "selected_experts": [45, 0, 61, 35, 23, 9, 16, 1],
  "routing_weights": [0.0738, 0.0607]
}
```

The `TraceWriter` provides:

```text
write()
read()
```

The write method accepts traces and can append if required.

The synthetic test confirmed that traces could be written to:

```text
results/traces/test_trace.jsonl
```

and then read back successfully.

---

# 10. Trace Analyzer

A `TraceAnalyzer` was implemented to calculate:

```text
Total routing events
Total expert selections
Unique experts used
Most frequently used experts
Expert usage percentages
Full routing summary
```

The analyzer works directly on a stored trace file, so the model does not need to remain loaded during repeated analysis.

---

# 11. Synthetic Trace Analysis

The synthetic trace file contained:

```text
Total routing events: 5
Total expert selections: 40
Unique experts used: 32
```

Because each event selects 8 experts:

```text
5 events × 8 experts = 40 selections
```

The most frequently selected expert was:

```text
Expert 0: 4 selections
```

This experiment was mainly used to validate the analysis pipeline.

---

# 12. LRU Expert Cache

An LRU-style expert cache simulator was implemented.

The policy is:

```text
Least Recently Used
```

When an expert is requested:

### Cache hit

If the expert is already cached:

```text
HIT
```

The expert becomes the most recently used item.

### Cache miss

If the expert is not cached:

```text
MISS
```

It is loaded into the cache.

If the cache is full, the least recently used expert is evicted.

---

# 13. LRU Cache Test

The cache was tested with:

```text
Cache capacity: 3
```

Request sequence:

```text
[1, 2, 3, 1, 4, 2, 5, 1]
```

The results were:

```text
Hits: 1
Misses: 7
Evictions: 4
Hit rate: 12.5%
```

This confirmed that the cache correctly maintained recency information and evicted the least recently used expert when full.

---

# 14. Cache Capacity Experiment on Synthetic Traces

The synthetic routing trace was replayed through the LRU cache using capacities:

```text
2, 4, 8, 16, 32, 64
```

The results were:

| Cache Capacity | Hits | Misses | Evictions | Hit Rate |
|---:|---:|---:|---:|---:|
| 2 | 0 | 40 | 38 | 0.00% |
| 4 | 0 | 40 | 36 | 0.00% |
| 8 | 2 | 38 | 30 | 5.00% |
| 16 | 5 | 35 | 19 | 12.50% |
| 32 | 8 | 32 | 0 | 20.00% |
| 64 | 8 | 32 | 0 | 20.00% |

The maximum hit rate was only 20% because there were 32 unique experts among 40 total requests.

This experiment validated the cache simulation but was not intended to represent realistic OLMoE behavior.

---

# 15. Benchmark and Result Infrastructure

Additional benchmark infrastructure was implemented, including:

```text
Cache benchmark
Result writer
Cache capacity experiment script
Cache capacity plotting script
```

Experiment results were saved to:

```text
results/experiments/cache_capacity_results.csv
```

The general workflow is:

```text
Routing traces
      ↓
Cache simulation
      ↓
Capacity experiments
      ↓
CSV results
      ↓
Plots and analysis
```

---

# 16. System Resource Check

A resource-checking script was created to inspect:

- Total RAM.
- Available RAM.
- Disk space.
- CUDA availability.
- GPU name.
- GPU VRAM.

The system used for the project has:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4.00 GB
Total RAM: approximately 7.34 GB
```

Available RAM varied depending on running applications.

After closing unnecessary applications, available RAM reached approximately:

```text
2.15 GB
```

The resource script reported:

```text
RAM status: LIMITED
Disk status: GOOD
```

Because the model is much larger than the available GPU memory and system RAM, direct full loading onto the GPU is not feasible.

---

# 17. Real OLMoE Model Loading

The real model:

```text
allenai/OLMoE-1B-7B-0924
```

was loaded using GPU, CPU, and disk offloading.

The loading limits were approximately:

```text
GPU memory limit: 3 GiB
CPU memory limit: 2 GiB
```

The remaining components were placed on disk.

The model was downloaded in three shards:

```text
model-00001-of-00003.safetensors → 5.00 GB
model-00002-of-00003.safetensors → 5.00 GB
model-00003-of-00003.safetensors → 3.84 GB
```

The downloaded checkpoint data therefore occupies roughly:

```text
13.84 GB
```

The model loaded successfully with a device map similar to:

```text
model.embed_tokens → GPU
model.layers.0 → GPU
model.layers.1 → GPU
model.layers.2 → CPU
model.layers.3 through model.layers.15 → Disk
model.norm → Disk
model.rotary_emb → Disk
lm_head → Disk
```

GPU memory after loading was approximately:

```text
Allocated GPU memory: 1.75 GB
Reserved GPU memory: 1.78 GB
```

This confirmed that the real model can run on the available laptop through offloading.

---

# 18. Real OLMoE Forward Pass

A successful real forward pass was performed using:

```text
The future of artificial intelligence is
```

The tokenizer produced:

```text
Input IDs shape: torch.Size([1, 6])
```

The output logits had shape:

```text
torch.Size([1, 6, 50304])
```

This represents:

- Batch size = 1.
- Sequence length = 6.
- Vocabulary size = 50,304.

The model returned router outputs from:

```text
16 MoE layers
```

Each router output had shape:

```text
(6, 64)
```

Therefore:

```text
6 tokens × 16 layers = 96 routing events
```

GPU memory changed only slightly:

```text
Before:
Allocated: 1.75 GB
Reserved: 1.78 GB

After:
Allocated: 1.76 GB
Reserved: 1.97 GB
```

This confirmed that the real model could perform inference and expose:

```text
outputs.router_logits
```

---

# 19. Real Expert Routing Trace Collection

The real trace collector used the prompt:

```text
The future of artificial intelligence is closely connected to efficient machine learning systems.
```

The tokenizer produced:

```text
Input IDs shape: torch.Size([1, 14])
```

The model was executed with:

```python
output_router_logits=True
```

The model returned:

```text
16 router outputs
```

Each layer produced:

```text
(14, 64)
```

Therefore:

```text
14 tokens × 16 layers = 224 routing events
```

The `ExpertTraceLogger` processed all layers successfully.

The traces were saved to:

```text
results/traces/real_olmoe_trace.jsonl
```

The summary was:

```text
Layers processed: 16
Tokens per layer: 14
Experts selected per token: 8
Expected routing events: 224
Actual routing events: 224
```

The expected and actual event counts matched exactly.

---

# 20. Example Real Routing Decisions

The first token in layer 0 selected:

```text
[5, 14, 61, 18, 6, 38, 19, 41]
```

with routing weights approximately:

```text
[0.169105, 0.096636, 0.071603, 0.051273,
 0.040107, 0.039274, 0.027271, 0.023879]
```

Another token selected:

```text
[5, 14, 18, 6, 41, 39, 15, 61]
```

This demonstrates that expert routing varies between tokens, although some experts can appear repeatedly.

---

# 21. Real OLMoE Trace Analysis

The real trace file produced:

```text
Total routing events: 224
Total expert selections: 1792
Unique experts used: 64
```

The total expert selections match:

```text
224 routing events × 8 experts = 1792 selections
```

All 64 experts were selected at least once.

The ten most frequently selected experts were:

| Expert | Selections |
|---:|---:|
| 18 | 54 |
| 1 | 44 |
| 48 | 43 |
| 33 | 41 |
| 49 | 40 |
| 13 | 40 |
| 36 | 39 |
| 47 | 39 |
| 50 | 38 |
| 29 | 38 |

The most frequently selected expert was:

```text
Expert 18
```

It was selected 54 times, representing approximately:

```text
3.01%
```

of all expert selections.

This unequal expert usage is important because frequently reused experts are more likely to benefit from caching.

---

# 22. Cache Capacity Experiment on Real OLMoE Traces

The real routing trace was replayed through the LRU cache simulator.

Results:

| Capacity | Hits | Misses | Evictions | Hit Rate |
|---:|---:|---:|---:|---:|
| 2 | 11 | 1781 | 1779 | 0.61% |
| 4 | 52 | 1740 | 1736 | 2.90% |
| 8 | 376 | 1416 | 1408 | 20.98% |
| 16 | 718 | 1074 | 1058 | 40.07% |
| 32 | 1130 | 662 | 630 | 63.06% |
| 64 | 1728 | 64 | 0 | 96.43% |

## Interpretation

### Capacity 2

```text
Hit rate: 0.61%
```

The cache is far too small compared with the diversity of expert requests.

### Capacity 4

```text
Hit rate: 2.90%
```

The cache improves slightly but still experiences frequent evictions.

### Capacity 8

```text
Hit rate: 20.98%
```

The significant improvement suggests that real routing has some short-term reuse and locality.

### Capacity 16

```text
Hit rate: 40.07%
```

The cache can retain a much larger portion of recently used experts.

### Capacity 32

```text
Hit rate: 63.06%
```

More than half of all expert requests are served as cache hits.

### Capacity 64

```text
Hit rate: 96.43%
Evictions: 0
```

All 64 experts fit in the cache.

The result is not 100% because the cache starts empty. The first request to each expert is a compulsory miss.

Since all 64 experts were eventually used:

```text
64 misses
1728 hits
1792 total requests
```

Therefore:

```text
1728 / 1792 × 100 = 96.43%
```

---

# 23. Important Technical Interpretation

The experiment demonstrates:

```text
Larger cache
    ↓
Fewer evictions
    ↓
More expert reuse
    ↓
Higher cache hit rate
```

The real results establish an important baseline:

```text
Cache size 2  → 0.61% hit rate
Cache size 8  → 20.98% hit rate
Cache size 16 → 40.07% hit rate
Cache size 32 → 63.06% hit rate
```

At this stage, the policy is only:

```text
LRU
```

No prediction or prefetching has been implemented yet.

These measurements will therefore act as a baseline for future adaptive policies.

---

# 24. Current Project Pipeline

The current system is:

```text
Input Prompt
     ↓
Tokenizer
     ↓
Real OLMoE Model
     ↓
Forward Pass
     ↓
Router Logits from 16 MoE Layers
     ↓
ExpertTraceLogger
     ↓
JSONL Routing Trace
     ↓
TraceAnalyzer
     ↓
Expert Usage Statistics
     ↓
LRU Expert Cache Simulator
     ↓
Cache Capacity Experiment
     ↓
CSV Results / Plots
```

The model execution and trace analysis are separated.

The expensive stage is:

```text
Loading and running OLMoE
```

After routing traces are stored, repeated cache experiments can be performed without loading the model again.

---

# 25. Important Files Created or Modified

## Configuration

```text
src/config.py
```

## Model Loading

```text
src/modeling/
```

Important functions include:

```text
load_tokenizer()
load_olmoe_model()
```

## Tracing

```text
src/tracing/expert_trace_logger.py
src/tracing/trace_writer.py
```

These handle routing extraction and trace storage.

## Benchmarking

```text
src/benchmark/trace_analyzer.py
src/benchmark/cache_benchmark.py
src/benchmark/results_writer.py
```

## Scripts

Important scripts include:

```text
scripts/inspect_olmoe_config.py
scripts/inspect_olmoe_architecture.py
scripts/inspect_olmoe_routing.py
scripts/test_expert_trace_logger.py
scripts/test_trace_writer.py
scripts/test_trace_analyzer.py
scripts/test_expert_cache.py
scripts/test_cache_benchmark.py
scripts/run_cache_capacity_experiment.py
scripts/plot_cache_capacity_experiment.py
scripts/check_system_resources.py
scripts/test_olmoe_loading.py
scripts/test_real_olmoe_forward.py
scripts/collect_real_olmoe_traces.py
scripts/trace_real_olmoe.py
```

---

# 26. Current Generated Results

Important output files include:

```text
results/traces/test_trace.jsonl
results/traces/real_olmoe_trace.jsonl
results/experiments/cache_capacity_results.csv
```

The most important current output is:

```text
results/traces/real_olmoe_trace.jsonl
```

because it contains expert-routing decisions generated by the real OLMoE model.

---

# 27. Current Limitations

## 1. Small workload

The real experiment currently uses one 14-token prompt:

```text
224 routing events
1792 expert selections
```

This is enough to validate the complete pipeline but is too small for final research conclusions.

## 2. No actual datasets have been processed yet

The broader project datasets have not yet been downloaded or used.

The planned datasets include:

```text
WikiText-103
C4
MBPP
HumanEval
GSM8K
FLORES-200
```

The general workflow will be:

```text
Dataset
   ↓
Select text / prompts / samples
   ↓
Tokenize each input
   ↓
Run OLMoE
   ↓
Collect router decisions
   ↓
Store routing traces
   ↓
Analyze cache behavior
```

Different datasets are expected to represent different workloads:

```text
General text       → WikiText / C4
Code               → MBPP / HumanEval
Mathematics        → GSM8K
Multilingual text  → FLORES-200
```

## 3. Only LRU is implemented

The current cache policy is:

```text
LRU
```

Future work can compare:

```text
Reactive caching
Heuristic prefetching
Learned prefetching
Different cache capacities
Potential distributed tiers
```

## 4. Expert-weight movement is not yet being physically simulated

The current cache experiment determines whether a requested expert would be a:

```text
HIT
or
MISS
```

The project is not yet physically moving individual expert weights between:

```text
Disk
CPU RAM
GPU VRAM
```

That is a later stage required for measuring real offloading overhead.

---

# 28. Planned Next Steps

The next phase should move from a manually written prompt to larger, dataset-driven workloads.

Recommended progression:

## Step 1: Download and prepare datasets

Start with manageable subsets rather than immediately processing complete large datasets.

Possible workloads:

```text
WikiText-103 subset
GSM8K subset
MBPP or HumanEval subset
FLORES-200 subset
```

## Step 2: Build a dataset workload runner

The runner should:

```text
Load a dataset
      ↓
Select samples
      ↓
Run OLMoE on each sample
      ↓
Collect router logits
      ↓
Write traces incrementally
```

Incremental writing is important because long experiments should not keep all routing traces in RAM.

## Step 3: Collect domain-specific traces

Possible outputs:

```text
results/traces/wikitext_trace.jsonl
results/traces/gsm8k_trace.jsonl
results/traces/code_trace.jsonl
results/traces/flores_trace.jsonl
```

## Step 4: Compare routing distributions

Analyze:

- Most-used experts.
- Expert usage distributions.
- Unique experts.
- Expert concentration.
- Reuse patterns.
- Temporal locality.

## Step 5: Run cache experiments

Replay each trace through different cache capacities and compare:

```text
Cache capacity
Cache hit rate
Miss rate
Evictions
Expert reuse
```

## Step 6: Implement prefetching

Possible approaches include:

```text
Frequency-based heuristic
Recent-history heuristic
Transition-based prediction
Learned predictor
```

## Step 7: Compare against the LRU baseline

The current LRU results will act as the baseline.

Future experiments can answer:

```text
Does prefetching improve hit rate?
Does it reduce misses?
Does the benefit vary by workload?
What cache capacity provides the best trade-off?
```

---

# 29. Final Day 1 Summary

Day 1 successfully established the complete baseline pipeline required for routing-based expert cache research.

The most important milestone is that the project is now using the **real OLMoE model**, rather than synthetic routing data.

The pipeline has been validated end to end:

```text
Real OLMoE
    ↓
Real router logits
    ↓
Real expert selections
    ↓
Stored routing traces
    ↓
Routing analysis
    ↓
LRU cache simulation
    ↓
Cache-capacity results
```

The real experiment produced:

```text
224 routing events
1792 expert selections
64 unique experts used
```

The cache baseline was:

```text
2 experts cached  → 0.61% hit rate
4 experts cached  → 2.90% hit rate
8 experts cached  → 20.98% hit rate
16 experts cached → 40.07% hit rate
32 experts cached → 63.06% hit rate
64 experts cached → 96.43% hit rate
```

These results establish the initial experimental baseline.

The next major milestone is to generate larger and more diverse routing workloads using the planned datasets. This will allow the project to study whether routing locality and cache performance differ across:

```text
General language
Code
Mathematics
Multilingual text
```

The project is ready to move into the dataset-driven experimental phase.
