"""src/data.py - dataset loading + leak-free train/val split."""

import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ImageNet-ish normalization applied to the [0,1] float images
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZE = transforms.Normalize(mean=MEAN, std=STD)


def load_images(dataset_dir: str = config.DATA_DIR,
                image_size: int = config.IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from data/cats (label 0) and data/dogs (label 1).

    Returns:
        X: (N, H, W, 3) float32 array in [0, 1]
        y: (N,) int array with 0=cat, 1=dog
    """
    cats_dir = os.path.join(dataset_dir, "cats")
    dogs_dir = os.path.join(dataset_dir, "dogs")

    if not (os.path.isdir(cats_dir) and os.path.isdir(dogs_dir)):
        raise FileNotFoundError(
            f"Dataset folders not found in {dataset_dir}. "
            "Run `python download_data.py` first."
        )

    images, labels = [], []
    for cls_dir, label in ((cats_dir, 0), (dogs_dir, 1)):
        files = sorted(
            p for p in os.listdir(cls_dir)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for name in files:
            path = os.path.join(cls_dir, name)
            try:
                img = Image.open(path).convert("RGB").resize((image_size, image_size))
                images.append(np.asarray(img, dtype=np.float32) / 255.0)
                labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

    if not images:
        raise RuntimeError("No images could be loaded from the dataset folders.")

    X = np.stack(images, axis=0).astype(np.float32)   # (N,H,W,C)
    y = np.array(labels, dtype=np.int64)
    return X, y


def load_train_val(dataset_dir: str = config.DATA_DIR,
                   image_size: int = config.IMAGE_SIZE,
                   seed: int = config.SEED,
                   split: float = config.TRAIN_SPLIT) -> tuple:
    """
    Load the dataset and perform a stratified, leak-free train/val split.

    Returns:
        (X_train, y_train, X_val, y_val) where X are (N, C, H, W) float tensors
        normalized to ImageNet stats and y are long tensors (0=cat, 1=dog).
    """
    X, y = load_images(dataset_dir, image_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1.0 - split, random_state=seed, stratify=y, shuffle=True
    )

    def _to_tensor(imgs, labels):
        t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()  # NCWH
        return NORMALIZE(t), torch.from_numpy(labels).long()

    Xt, yt = _to_tensor(X_train, y_train)
    Xv, yv = _to_tensor(X_val, y_val)
    return Xt, yt, Xv, yv