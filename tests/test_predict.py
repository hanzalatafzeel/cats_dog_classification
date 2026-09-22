"""
tests/test_predict.py - smoke tests for the prediction pipeline.

Run:  pytest -q
"""

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import CatDogCNN
from src.predict import load_model, predict_image


@pytest.fixture(scope="module")
def model():
    m, device, norm = load_model()
    return m, device, norm


def test_model_architecture():
    cnn = CatDogCNN(image_size=config.IMAGE_SIZE, arch=config.ARCH)
    assert cnn.num_params() > 100_000, "CNN should have learnable parameters"


def test_predict_requires_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.pt"))


def test_predict_image_pipeline(model, tmp_path):
    # Synthetic gray image: pipeline (load -> preprocess -> predict) must work
    img = Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8))
    p = tmp_path / "test.jpg"
    img.save(p)
    m, device, norm = model
    label, conf = predict_image(str(p), model=m, device=device, norm=norm)
    assert label in ("cat", "dog")
    assert 0.0 <= conf <= 100.0


def test_labels_match_config(model):
    # model checkpoint should carry the label mapping
    import torch
    ckpt_path = config.MODEL_PATH
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["labels"] == config.LABELS