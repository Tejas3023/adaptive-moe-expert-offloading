from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================
# OLMOE MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

# The model should not be loaded into normal full precision.
# float16 reduces memory usage and works with the RTX 3050.
MODEL_DTYPE = "float16"


# ============================================================
# OFFLOADING CONFIGURATION
# ============================================================

# Directory where model components can be offloaded to disk.
OFFLOAD_DIR = MODELS_DIR / "offload"

# Approximate memory limits for automatic device placement.
#
# Your RTX 3050 has 4 GB VRAM, but we should not allocate all
# of it to the model.
GPU_MEMORY_LIMIT = "3GiB"

# We intentionally keep this conservative because the system
# has 8 GB total RAM and Windows needs a significant amount.
CPU_MEMORY_LIMIT = "2GiB"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

DEFAULT_MAX_NEW_TOKENS = 5

# OLMoE selects 8 experts out of 64 per token.
NUM_EXPERTS = 64
TOP_K_EXPERTS = 8
NUM_LAYERS = 16