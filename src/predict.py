"""src/predict.py - single-image and batch inference helpers + CLI."""

import os
import sys

import numpy as np
from PIL import Image

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import CatDogCNN


def to_device(model: torch.nn.Module) -> torch.nn.Module:
    """Move model to CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device


def load_model(model_path: str = config.MODEL_PATH) -> tuple:
    """Load a trained CatDogCNN from a checkpoint file.

    Returns:
        (model, device, norm) where norm is the dict
        {"mean": [...], "std": [...]} stored with the checkpoint
        (falls back to ImageNet stats if absent).
    """
    model = CatDogCNN(image_size=config.IMAGE_SIZE, arch=config.ARCH)
    model, device = to_device(model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    model.eval()

    mean = state.get("mean", [0.485, 0.456, 0.406])
    std = state.get("std", [0.229, 0.224, 0.225])
    norm = {"mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32)}
    return model, device, norm


def preprocess_image(path: str, image_size: int = config.IMAGE_SIZE,
                     norm: dict | None = None) -> torch.Tensor:
    """Load, resize and normalize an image to a (1, 3, H, W) tensor."""
    if norm is None:
        norm = {"mean": np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
                "std": np.asarray([0.229, 0.224, 0.225], dtype=np.float32)}
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [0,1]
    arr = (arr - norm["mean"]) / norm["std"]
    arr = np.transpose(arr, (2, 0, 1))                        # CHW
    return torch.from_numpy(arr.copy()).unsqueeze(0)


def predict_image(
    path: str,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    norm: dict | None = None,
    image_size: int = config.IMAGE_SIZE,
) -> tuple[str, float]:
    """
    Classify a single image.

    Returns:
        (label, confidence) where label is 'cat' or 'dog' and confidence
        is the probability (0-100) of the predicted class.
    """
    if model is None:
        model, device, norm = load_model()
    assert model is not None, "model failed to load"
    device = device or next(model.parameters()).device
    x = preprocess_image(path, image_size=image_size, norm=norm).to(device)
    model.to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()   # P(dog)
    label = "dog" if prob >= 0.5 else "cat"
    confidence = prob * 100 if label == "dog" else (1 - prob) * 100
    return label, confidence


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        print("Usage: python src/predict.py /path/to/image.jpg")
        return 1
    path = argv[0]
    model, device, norm = load_model()
    label, conf = predict_image(path, model=model, device=device, norm=norm)
    print("=" * 44)
    print(" CAT vs DOG CLASSIFICATION RESULT")
    print("=" * 44)
    print(f"  Image Path  : {path}")
    print(f"  Prediction  : {label.upper()}")
    print(f"  Confidence  : {conf:.2f}%")
    print("=" * 44)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())