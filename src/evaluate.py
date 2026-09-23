"""
src/evaluate.py - evaluate a trained checkpoint on the validation split and
export diagnostics plots to config.OUTPUTS_DIR.

Important: preprocessing uses the mean/std stored in the checkpoint (the
Kaggle-trained model was normalized with dataset statistics, not ImageNet).

Run:  python src/evaluate.py [--model path/to/cat_dog_cnn.pt]
"""

import argparse
import datetime
import json
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
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


def dataset_counts(dataset_dir=config.DATA_DIR):
    cats = sum(1 for f in os.listdir(os.path.join(dataset_dir, "cats"))
               if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
        if os.path.isdir(os.path.join(dataset_dir, "cats")) else 0
    dogs = sum(1 for f in os.listdir(os.path.join(dataset_dir, "dogs"))
               if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
        if os.path.isdir(os.path.join(dataset_dir, "dogs")) else 0
    return cats, dogs


def build_metrics_dict(model, state, file_size, y_true, preds, probs,
                       val_size, cats, dogs):
    """Assemble every metric we can compute into a plain-JSON-friendly dict."""
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, probs)
    metric_dict = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds)),
        "recall": float(recall_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }

    # per-class: row = actual, col = predicted
    cat_recall = tn / (tn + fp + 1e-9)
    dog_recall = tp / (tp + fn + 1e-9)
    cat_precision = tn / (tn + fn + 1e-9)
    dog_precision = tp / (tp + fp + 1e-9)

    return {
        "model": {
            "name": "CatDogCNN",
            "params": model.num_params(),
            "image_size": int(state.get("image_size", config.IMAGE_SIZE)),
            "conv_dims": config.ARCH["conv_dims"],
            "fc_units": config.ARCH["fc_units"],
            "dropout": config.ARCH["dropout"],
            "labels": list(state.get("labels", config.LABELS)),
            "mean": [float(v) for v in state.get("mean", [])],
            "std": [float(v) for v in state.get("std", [])],
            "file_size_bytes": file_size,
        },
        "dataset": {
            "cats": cats,
            "dogs": dogs,
            "total": cats + dogs,
            "train": int((cats + dogs) * config.TRAIN_SPLIT),
            "val": val_size,
            "split": config.TRAIN_SPLIT,
        },
        "metrics": metric_dict,
        "checkpoint_metrics": {
            k: float(v) for k, v in state.get("metrics", {}).items()
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "per_class": {
            "cat": {"precision": float(cat_precision), "recall": float(cat_recall)},
            "dog": {"precision": float(dog_precision), "recall": float(dog_recall)},
        },
        "roc": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            "auc": float(auc(fpr, tpr)),
        },
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def generate_metrics_json(output_path, model_path=config.MODEL_PATH):
    """Run a full evaluation and write outputs/metrics.json (no plots)."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    norm = {
        "mean": np.asarray(state.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32),
        "std": np.asarray(state.get("std", [0.229, 0.224, 0.225]), dtype=np.float32),
    }
    model = CatDogCNN(image_size=config.IMAGE_SIZE, arch=config.ARCH)
    model.load_state_dict(state["state_dict"])
    model, device = to_device(model)

    loader = val_loader(config.DATA_DIR, config.TRAIN_SPLIT, config.SEED,
                        config.IMAGE_SIZE, norm)
    preds, probs, labels = batch_predict(model, loader, device)
    cats, dogs = dataset_counts()

    data = build_metrics_dict(
        model, state, os.path.getsize(model_path),
        labels, preds, probs, len(loader.dataset), cats, dogs,
    )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--no-json", action="store_true",
                        help="skip writing outputs/metrics.json")
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

    if not args.no_json:
        cats, dogs = dataset_counts()
        payload = build_metrics_dict(
            model, state, os.path.getsize(args.model),
            labels, preds, probs, len(loader.dataset), cats, dogs,
        )
        json_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Exported {json_path}")

    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, cm_path)
    print(f"Saved {cm_path}")

    plot_sample_predictions(model, device, norm)
    print(f"Saved sample_predictions.png")


if __name__ == "__main__":
    main()