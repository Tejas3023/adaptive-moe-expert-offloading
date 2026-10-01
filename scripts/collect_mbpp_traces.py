import sys
from pathlib import Path

import torch
from datasets import load_dataset


# ==============================================================
# PROJECT ROOT
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from src.modeling.olmoe_loader import (
    load_olmoe_model,
    load_tokenizer,
)

from src.tracing.expert_trace_logger import ExpertTraceLogger
from src.tracing.trace_writer import TraceWriter


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

DATASET_NAME = "google-research-datasets/mbpp"
DATASET_CONFIG = "sanitized"
DATASET_SPLIT = "test"

NUM_SAMPLES = 100
MAX_LENGTH = 256

OUTPUT_PATH = Path(
    "results/traces/mbpp_100_samples.jsonl"
)


# ==============================================================
# HEADER
# ==============================================================

print("=" * 80)
print("MBPP → REAL OLMOE EXPERT TRACE COLLECTION")
print("=" * 80)


# ==============================================================
# LOAD MBPP DATASET
# ==============================================================

print()
print("=" * 80)
print("LOADING MBPP")
print("=" * 80)

dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=DATASET_SPLIT,
)


# --------------------------------------------------------------
# Select non-empty prompts
# --------------------------------------------------------------

questions = []

for example in dataset:

    prompt = example.get(
        "prompt",
        ""
    ).strip()

    if prompt:

        questions.append(prompt)

    if len(questions) >= NUM_SAMPLES:
        break


print()
print(f"Dataset: {DATASET_NAME}")
print(f"Configuration: {DATASET_CONFIG}")
print(f"Split: {DATASET_SPLIT}")
print(f"Total dataset examples: {len(dataset)}")
print(
    f"Non-empty prompts selected: "
    f"{len(questions)}"
)


# ==============================================================
# LOAD OLMOE TOKENIZER
# ==============================================================

print()
print("=" * 80)
print("LOADING OLMOE TOKENIZER")
print("=" * 80)

print()
print(f"Model: {MODEL_NAME}")

# Your existing loader takes no arguments.
tokenizer = load_tokenizer()

print("Tokenizer loaded successfully.")


# ==============================================================
# LOAD REAL OLMOE MODEL
# ==============================================================

print()
print("=" * 80)
print("LOADING REAL OLMOE MODEL")
print("=" * 80)

print()
print(f"Model: {MODEL_NAME}")

# Your existing loader takes no arguments.
model = load_olmoe_model()

model.eval()

print("Model loaded successfully.")


# ==============================================================
# CREATE TRACE LOGGER
# ==============================================================

logger = ExpertTraceLogger(
    num_experts=64,
    top_k=8,
    norm_topk_prob=False,
)


# ==============================================================
# OUTPUT DIRECTORY
# ==============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==============================================================
# PROCESS MBPP PROMPTS
# ==============================================================

for sample_id, prompt in enumerate(questions):

    print()
    print("-" * 80)
    print(
        f"PROCESSING MBPP PROMPT "
        f"{sample_id + 1}/{len(questions)}"
    )
    print("-" * 80)

    # ----------------------------------------------------------
    # Show prompt
    # ----------------------------------------------------------

    print()
    print("Prompt preview:")
    print(prompt[:500])

    # ----------------------------------------------------------
    # Tokenize
    # ----------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    token_count = inputs["input_ids"].shape[1]

    print()
    print(f"Token count: {token_count}")

    # ----------------------------------------------------------
    # Move inputs to model device
    # ----------------------------------------------------------

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # ----------------------------------------------------------
    # REAL OLMOE FORWARD PASS
    # ----------------------------------------------------------

    print()
    print("Running real OLMoE forward pass...")

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_router_logits=True,
        )

    router_outputs = outputs.router_logits

    print(
        f"Router outputs: "
        f"{len(router_outputs)} layers"
    )

    # ----------------------------------------------------------
    # EXTRACT EXPERT ROUTING
    # ----------------------------------------------------------

    for layer_id, layer_router_logits in enumerate(
        router_outputs
    ):

        logger.process_router_logits(
            router_logits=layer_router_logits,
            layer_id=layer_id,
            sample_id=sample_id,
        )

    print(
        f"Routing events collected so far: "
        f"{len(logger)}"
    )

    # ==========================================================
    # INCREMENTAL SAVE
    # ==========================================================

    #
    # Save after every successfully processed prompt.
    #
    # This protects the experiment from losing all previous
    # traces if a later OLMoE forward pass crashes.
    #

    current_traces = logger.get_traces()

    progress_writer = TraceWriter(
        OUTPUT_PATH
    )

    written = progress_writer.write(
        current_traces
    )

    print(
        f"Progress saved: "
        f"{written} traces"
    )

    # ----------------------------------------------------------
    # RELEASE TENSORS BEFORE NEXT PROMPT
    # ----------------------------------------------------------

    del outputs
    del router_outputs
    del inputs

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ==============================================================
# FINAL TRACE COLLECTION
# ==============================================================

traces = logger.get_traces()


# ==============================================================
# WRITE FINAL TRACE FILE
# ==============================================================

print()
print("=" * 80)
print("WRITING FINAL MBPP TRACES")
print("=" * 80)

writer = TraceWriter(
    OUTPUT_PATH
)

written = writer.write(
    traces
)


# ==============================================================
# CALCULATE SUMMARY
# ==============================================================

total_routing_events = len(logger)

total_expert_selections = (
    total_routing_events * 8
)


# ==============================================================
# SUMMARY
# ==============================================================

print()
print("=" * 80)
print("MBPP TRACE COLLECTION SUMMARY")
print("=" * 80)

print()
print(
    f"Dataset samples processed: "
    f"{len(questions)}"
)

print(
    f"Total routing events: "
    f"{total_routing_events}"
)

print(
    "Transformer layers: 16"
)

print(
    "Experts per MoE layer: 64"
)

print(
    "Experts selected per token: 8"
)

print(
    f"Total expert selections: "
    f"{total_expert_selections}"
)

print(
    f"Traces written: "
    f"{written}"
)

print()
print(
    f"Output file: "
    f"{OUTPUT_PATH}"
)


# ==============================================================
# FIRST 5 ROUTING TRACES
# ==============================================================

print()
print("=" * 80)
print("FIRST 5 MBPP ROUTING TRACES")
print("=" * 80)

for record in traces[:5]:

    print()
    print("Layer:", record.layer_id)
    print("Sample ID:", record.sample_id)
    print("Token position:", record.token_position)
    print("Selected experts:", record.selected_experts)
    print("Routing weights:", record.routing_weights)


# ==============================================================
# COMPLETE
# ==============================================================

print()
print("=" * 80)
print("MBPP TRACE COLLECTION COMPLETE")
print("=" * 80)