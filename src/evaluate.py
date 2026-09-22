"""
src/evaluate.py - evaluate a trained checkpoint on the validation split and
export diagnostics plots to config.OUTPUTS_DIR.

Important: preprocessing uses the mean/std stored in the checkpoint (the
Kaggle-trained model was normalized with dataset statistics, not ImageNet).

Run:  python src/evaluate.py [--model path/to/cat_dog_cnn.pt]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import load_images
from src.model import CatDogCNN
from src.predict import to_device


def val_loader(dataset_dir, split, seed, image_size, norm, batch_size=32):
    """Build a validation DataLoader using the SAME split strategy as
    src/train.py but the checkpoint's normalization stats."""
    X, y = load_images(dataset_dir, image_size)
    _Xtr, X_val, _ytr, y_val = train_test_split(
        X, y, test_size=1.0 - split, random_state=seed, stratify=y, shuffle=True
    )
    t = (torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
         - torch.from_numpy(norm["mean"]).view(1, 3, 1, 1))
    t = t / torch.from_numpy(norm["std"]).view(1, 3, 1, 1)
    ds = torch.utils.data.TensorDataset(t, torch.from_numpy(y_val).long())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def batch_predict(model, loader, device):
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            p = torch.sigmoid(logits).squeeze(-1)
            probs.extend(p.tolist())
            preds.extend((p >= 0.5).long().tolist())
            labels.extend(yb.tolist())
    return np.asarray(preds), np.asarray(probs), np.asarray(labels)


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], config.LABELS)
    ax.set_yticks([0, 1], config.LABELS)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_sample_predictions(model, device, norm, count=6):
    """Show count/2 real images of each class next to predictions."""
    from src.predict import predict_image

    picks = []
    for label_name, label_dir in (("cat", config.CATS_DIR), ("dog", config.DOGS_DIR)):
        files = sorted(os.listdir(label_dir))[::11][: count // 2]
        picks.extend((label_name, os.path.join(label_dir, f)) for f in files)

    if not picks:
        print("No data - skipping sample predictions plot.")
        return

    cols = len(picks)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
    if cols == 1:
        axes = [axes]
    for ax, (truth, path) in zip(axes, picks):
        pred, conf = predict_image(path, model=model, device=device, norm=norm)
        img = Image.open(path).convert("RGB").resize((128, 128))
        ax.imshow(img)
        ok = pred == truth
        ax.set_title(f"truth={truth}\npred={pred} ({conf:.0f}%)", fontsize=9,
                     color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle("Sample Predictions (Kaggle-trained model)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUTS_DIR, "sample_predictions.png"), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--model", default=config.MODEL_PATH)
    args = parser.parse_args()

    config.ensure_dirs()
    assert os.path.exists(args.model), f"Model not found: {args.model}"

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    norm = {
        "mean": np.asarray(state.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32),
        "std": np.asarray(state.get("std", [0.229, 0.224, 0.225]), dtype=np.float32),
    }

    model = CatDogCNN(image_size=config.IMAGE_SIZE, arch=config.ARCH)
    model.load_state_dict(state["state_dict"])
    model, device = to_device(model)

    loader = val_loader(config.DATA_DIR, config.TRAIN_SPLIT, config.SEED,
                        config.IMAGE_SIZE, norm)
    print(f"Evaluating on {len(loader.dataset)} validation images...")
    preds, probs, labels = batch_predict(model, loader, device)

    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds):.4f}")
    print(f"Recall   : {recall_score(labels, preds):.4f}")
    print(f"F1       : {f1_score(labels, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(labels, probs):.4f}")

    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)
    print(f"Saved {cm_path}")

    plot_sample_predictions(model, device, norm)
    print(f"Saved sample_predictions.png")


if __name__ == "__main__":
    main()