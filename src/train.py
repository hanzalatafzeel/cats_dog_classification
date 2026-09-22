"""
src/train.py - end-to-end training pipeline for the from-scratch CNN.

Steps:
    1. Load dataset (data/cats, data/dogs) with stratified train/val split
    2. Build CatDogCNN (from scratch)
    3. Train with data augmentation + BCEWithLogitsLoss + Adam
    4. Save model checkpoint to models/cat_dog_cnn.pt
    5. Plot training curves + confusion matrix to outputs/
    6. Print Accuracy / Precision / Recall / F1

Usage:
    python src/train.py [--epochs 15] [--batch-size 32] [--lr 0.001]
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import load_train_val
from src.model import CatDogCNN
from src.predict import to_device


def build_loaders(X_train, y_train, X_val, y_val,
                  batch_size: int, apply_aug=True) -> tuple[DataLoader, DataLoader]:
    """Build torch DataLoaders with optional training-time augmentation."""
    from torchvision import transforms

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    # Real-time data augmentation (applied per-batch on GPU/cpu tensors)
    if apply_aug:
        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=config.TRAIN["aug_rotation"]),
            transforms.ColorJitter(brightness=config.TRAIN["aug_brightness"],
                                   contrast=config.TRAIN["aug_contrast"]),
        ])

        def _aug(t):
            return aug(t)

        class AugDS(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, i):
                x, y = self.ds[i]
                return _aug(x), y

        train_ds = AugDS(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        probs = torch.sigmoid(model(xb).squeeze(1))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(yb.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    return probs, labels, preds


def plot_training_history(history: dict, output_dir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_acc"], label="Training Accuracy", color="#1f77b4", linewidth=2.5)
    ax1.plot(history["val_acc"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax1.set_title("Model Accuracy vs Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12); ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(history["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2.5)
    ax2.plot(history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax2.set_title("Model Loss vs Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12); ax2.set_ylabel("BCE Loss", fontsize=12)
    ax2.legend(); ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, output_dir: str) -> str:
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"]).plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def train_and_evaluate(epochs=config.TRAIN["epochs"],
                       batch_size=config.TRAIN["batch_size"],
                       learning_rate=config.TRAIN["learning_rate"],
                       model_path=config.MODEL_PATH) -> dict:
    config.ensure_dirs()
    config.set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] Device: {device}")

    # 1. Load data
    X_train, y_train, X_val, y_val = load_train_val()
    print(f"[DATA] train={len(X_train)} val={len(X_val)} "
          f"(cats: {int((y_train == 0).sum()) + int((y_val == 0).sum())}, "
          f"dogs: {int((y_train == 1).sum()) + int((y_val == 1).sum())})")

    # 2. Build model
    model = CatDogCNN(image_size=config.IMAGE_SIZE, arch=config.ARCH)
    model, device = to_device(model)
    print(f"[MODEL] Parameters: {model.num_params():,}")

    # 3. Training setup
    train_loader, val_loader = build_loaders(X_train, y_train, X_val, y_val, batch_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=config.TRAIN["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.TRAIN["lr_min"])

    # 4. Training loop
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    best_val_acc, best_state = 0.0, None
    start = time.time()
    print(f"\n[TRAIN] {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    for epoch in range(1, epochs + 1):
        tloss, tacc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # validation loss
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb.float())
                vloss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                vcorrect += (preds == yb).sum().item()
                vtotal += xb.size(0)
        vloss, vacc = vloss / vtotal, vcorrect / vtotal

        history["train_loss"].append(tloss)
        history["train_acc"].append(tacc)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)

        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:>2}/{epochs} | loss={tloss:.4f} acc={tacc:.4f} | "
              f"val_loss={vloss:.4f} val_acc={vacc:.4f}")
        scheduler.step()

    print(f"[TRAIN] Finished in {time.time() - start:.1f}s")

    # 5. Restore best weights and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_val_np, val_preds = evaluate(model, val_loader, device)
    with torch.no_grad():
        probs_val = torch.sigmoid(model(X_val.to(device)).squeeze(1)).cpu().numpy()

    acc = accuracy_score(y_val_np, val_preds)
    prec = precision_score(y_val_np, val_preds, zero_division=0)
    rec = recall_score(y_val_np, val_preds, zero_division=0)
    f1 = f1_score(y_val_np, val_preds, zero_division=0)

    # 6. Save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": config.ARCH,
        "image_size": config.IMAGE_SIZE,
        "labels": config.LABELS,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
    }
    torch.save(checkpoint, model_path)
    print(f"[SAVE] Model saved to {model_path}")

    # 7. Plots
    hist_plot = plot_training_history(history, config.OUTPUTS_DIR)
    cm_plot = plot_confusion_matrix(y_val_np, val_preds, config.OUTPUTS_DIR)
    print(f"[PLOT] {hist_plot}")
    print(f"[PLOT] {cm_plot}")

    # 8. Metrics report
    print("\n" + "=" * 58)
    print(" MODEL EVALUATION METRICS ON VALIDATION SET")
    print("=" * 58)
    print(f"  * Accuracy        : {acc*100:.2f}%")
    print(f"  * Precision       : {prec*100:.2f}%")
    print(f"  * Recall (Sens.)  : {rec*100:.2f}%")
    print(f"  * F1-Score        : {f1*100:.2f}%")
    print("=" * 58)

    return {"history": history, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "val_probs": probs_val, "y_val": y_val_np}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CatDogCNN")
    parser.add_argument("--epochs", type=int, default=config.TRAIN["epochs"])
    parser.add_argument("--batch-size", type=int, default=config.TRAIN["batch_size"])
    parser.add_argument("--lr", type=float, default=config.TRAIN["learning_rate"])
    args = parser.parse_args(argv)
    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())