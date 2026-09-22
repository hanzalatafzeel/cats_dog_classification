"""
Global configuration for the Cats vs Dogs CNN classifier.

Centralizes file paths, hyperparameters, and reproducibility seeds so that
scripts, the notebook, and the API all read from a single source of truth.
All paths are resolved relative to the project root.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CATS_DIR = os.path.join(DATA_DIR, "cats")
DOGS_DIR = os.path.join(DATA_DIR, "dogs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")

MODEL_PATH = os.path.join(MODELS_DIR, "cat_dog_cnn.pt")
LABELS = ["cat", "dog"]
CLASS_TO_IDX = {"cat": 0, "dog": 1}

# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------
# Real dataset: Microsoft "Dogs vs Cats" on Hugging Face
HF_DATASET = "microsoft/cats_vs_dogs"
IMAGES_PER_CLASS = 1200          # downloaded by download_data.py
IMAGE_SIZE = 128                 # square input resolution
TRAIN_SPLIT = 0.80               # 80% train / 20% validation
SEED = 42

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# Model architecture (matches CatDogCNN + the Kaggle-trained checkpoint:
# 4 double-conv blocks with batch norm, global average pooling, 2 dense layers)
ARCH = {
    "conv_dims": [32, 64, 128, 256],  # channels per (double-conv) block
    "fc_units": 128,
    "dropout": 0.5,
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN = {
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_min": 1e-5,           # cosine annealing floor
    "aug_rotation": 5,        # gentle rotation (degrees)
    "aug_brightness": 0.05,   # gentle color jitter
    "aug_contrast": 0.05,
}

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000


def set_seed(seed: int = SEED) -> None:
    """Seed everything for reproducible results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Create all output directories if missing."""
    for d in (MODELS_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR):
        os.makedirs(d, exist_ok=True)