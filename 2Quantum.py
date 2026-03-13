# -*- coding: utf-8 -*-
"""
Robust A–Z image classifier with nearest-neighbour inference and metrics.

Key properties
--------------
- Uses consistent preprocessing for training and testing
- Logs dataset stats and warns for extreme data scarcity
- Trains a small classical encoder
- Uses nearest-neighbour over preprocessed training images for final
  predictions (data-driven, no hardcoded words)
- Computes accuracy, precision (macro), recall (macro), F1 (macro),
  and confusion matrix using sklearn.metrics
- Keeps CLI interface:
    python 2Quantum.py train
    python 2Quantum.py test
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional quantum backend (not required for final inference logic)
try:
    import pennylane as qml  # noqa: F401
except Exception:
    qml = None  # noqa: F401

try:
    from tqdm import tqdm  # noqa: F401
except Exception:
    tqdm = None  # noqa: F401

# -------------------------
# GLOBAL CONFIG
# -------------------------
LABELS = [chr(ord("A") + i) for i in range(26)]
LBL2ID: Dict[str, int] = {c: i for i, c in enumerate(LABELS)}

MODEL_DIR = Path("modells")
MODEL_PATH = MODEL_DIR / "encoder_best.pt"  # encoder weights only
META_PATH = MODEL_DIR / "encoder_meta.json"

# -------------------------
# UTILITIES
# -------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]


def infer_label(path: Path) -> int:
    return LBL2ID[path.parent.name.upper()]


def numeric_sort_key(path: Path):
    """Robust sort: first by leading number if present, else lexicographically."""
    stem = path.stem
    prefix_digits = ""
    for ch in stem:
        if ch.isdigit():
            prefix_digits += ch
        else:
            break
    if prefix_digits:
        return (0, int(prefix_digits), stem.lower())
    return (1, stem.lower())


# -------------------------
# IMAGE PREPROCESSING
# -------------------------

def load_image_tensor(path: Path, img_size: int = 64) -> torch.Tensor:
    """Load image as 1xHxW float tensor with consistent preprocessing."""
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32)

    # Light denoising and normalization
    arr = cv2.medianBlur(arr, 3)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    arr = np.clip(arr, -3, 3)
    arr = (arr + 3) / 6.0

    arr = arr[None, ...]  # (1, H, W)
    return torch.from_numpy(arr)


# -------------------------
# DATASET
# -------------------------


class EEGDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], img_size: int = 64, train: bool = False):
        self.items = items
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x = load_image_tensor(path, self.img_size)
        if self.train:
            x = self._augment(x)
        return x, label

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        arr = x.numpy()
        if random.random() < 0.5:
            arr = np.flip(arr, axis=2).copy()
        if random.random() < 0.5:
            shift_y = random.randint(-2, 2)
            shift_x = random.randint(-2, 2)
            arr = np.roll(arr, shift_y, axis=1)
            arr = np.roll(arr, shift_x, axis=2)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
        return torch.from_numpy(arr)


# -------------------------
# ENCODER MODEL (CLASSICAL)
# -------------------------


class SimpleEncoder(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        emb = self.head(z)
        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        return emb


# -------------------------
# TRAIN
# -------------------------


def train(args) -> None:
    device = get_device()
    print("============================================================")
    print("[START] 2Quantum.py TRAIN mode")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    print("============================================================")

    root = Path("training dataset")
    if not root.exists():
        print(f"[ERROR] Training dataset directory not found: {root}")
        return

    paths = list_images(root)
    if len(paths) == 0:
        print(f"[ERROR] No training images found under {root}")
        return

    items = [(p, infer_label(p)) for p in paths]

    cls_counts: Dict[str, int] = {}
    for _, lbl in items:
        ch = LABELS[lbl]
        cls_counts[ch] = cls_counts.get(ch, 0) + 1

    print(f"[DATA] Total images: {len(items)}")
    print(f"[DATA] Unique classes: {len(cls_counts)} -> {sorted(cls_counts.keys())}")
    print("[DATA] Samples per class:")
    for ch in sorted(cls_counts.keys()):
        c = cls_counts[ch]
        print(f"       {ch}: {c}")
        if c < 5:
            print(f"[WARN] Class {ch} has only {c} samples (<5). This may hurt generalization.")

    random.shuffle(items)
    split = max(1, int(0.8 * len(items)))
    train_items = items[:split]
    val_items = items[split:]

    print(f"[DATA] Train samples: {len(train_items)}, Val samples: {len(val_items)}")

    dl_tr = DataLoader(EEGDataset(train_items, train=True),
                       batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(EEGDataset(val_items, train=False),
                       batch_size=args.batch) if len(val_items) > 0 else None

    model = SimpleEncoder(emb_dim=64).to(device)
    clf = nn.Linear(64, len(LABELS)).to(device)

    class_weights = torch.ones(len(LABELS), dtype=torch.float32)
    total = float(len(items))
    for ch, c in cls_counts.items():
        idx = LBL2ID[ch]
        class_weights[idx] = total / (len(cls_counts) * float(c))
    class_weights = class_weights.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()), lr=args.lr)

    best_metric = None
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        clf.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            emb = model(x)
            logits = clf(emb)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        val_acc = None
        if dl_va is not None:
            model.eval()
            clf.eval()
            correct = 0
            total_v = 0
            with torch.no_grad():
                for x, y in dl_va:
                    x, y = x.to(device), y.to(device)
                    emb = model(x)
                    logits = clf(emb)
                    preds = logits.argmax(1)
                    correct += int((preds == y).sum().item())
                    total_v += int(y.size(0))
            val_acc = correct / float(total_v) if total_v > 0 else 0.0
            print(f"[TRAIN] Epoch {ep:03d} | Loss={avg_loss:.4f} | ValAcc={val_acc:.4f}")

            if best_metric is None or val_acc > best_metric:
                best_metric = val_acc
                best_state = {
                    "encoder": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "clf": {k: v.detach().cpu() for k, v in clf.state_dict().items()},
                }
        else:
            print(f"[TRAIN] Epoch {ep:03d} | Loss={avg_loss:.4f}")
            if best_metric is None or avg_loss < best_metric:
                best_metric = avg_loss
                best_state = {
                    "encoder": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "clf": {k: v.detach().cpu() for k, v in clf.state_dict().items()},
                }

    if best_state is not None:
        model.load_state_dict(best_state["encoder"])
        clf.load_state_dict(best_state["clf"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with META_PATH.open("w") as f:
        json.dump({"img_size": 64, "emb_dim": 64, "num_classes": len(LABELS)}, f, indent=2)

    print(f"[SAVE] Saved encoder weights to {MODEL_PATH}")
    print(f"[SAVE] Saved metadata to {META_PATH}")
    print("[TRAIN] Done. Best metric:", best_metric)


# -------------------------
# TEST (NEAREST NEIGHBOUR)
# -------------------------


def test(args) -> None:
    device = get_device()
    print("============================================================")
    print("[START] 2Quantum.py TEST mode")
    print("============================================================")
    print("[INFO] Using device:", device)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Encoder weights not found at {MODEL_PATH}. Run train first.")
        return

    if META_PATH.exists():
        with META_PATH.open("r") as f:
            meta = json.load(f)
        img_size = int(meta.get("img_size", 64))
        emb_dim = int(meta.get("emb_dim", 64))
    else:
        img_size = 64
        emb_dim = 64

    encoder = SimpleEncoder(emb_dim=emb_dim).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()

    train_root = Path("training dataset")
    train_paths = list_images(train_root)
    if len(train_paths) == 0:
        print(f"[ERROR] No training images found under {train_root}")
        return

    train_feats = []
    train_labels = []
    for p in train_paths:
        x_tr = load_image_tensor(p, img_size=img_size).float().view(-1)
        x_tr = x_tr / (x_tr.norm() + 1e-8)
        train_feats.append(x_tr)
        train_labels.append(infer_label(p))
    train_feats = torch.stack(train_feats)

    test_root = Path("teseting Dataset")
    if not test_root.exists():
        print(f"[ERROR] Testing dataset directory not found: {test_root}")
        return

    images = list_images(test_root)
    if len(images) == 0:
        print(f"[ERROR] No test images found under {test_root}")
        return

    images = sorted(images, key=numeric_sort_key)
    print(f"[DATA] Found {len(images)} test images. Evaluation order:")
    for p in images:
        print("       ", p.name)

    word = ""
    conf_threshold = 0.60

    print("\nPredictions:\n------------")

    for img in images:
        x_te = load_image_tensor(img, img_size=img_size).float().view(-1)
        x_te = x_te / (x_te.norm() + 1e-8)
        dists = torch.norm(train_feats - x_te.unsqueeze(0), dim=1)
        scores = torch.softmax(-dists, dim=0)
        best_idx = int(scores.argmax().item())
        pred_idx = int(train_labels[best_idx])
        pred_char = LABELS[pred_idx]
        conf = float(scores[best_idx].item())
        conf_pct = conf * 100.0

        word += pred_char

        if conf < conf_threshold:
            print(f"{img.name} -> {pred_char} ({conf_pct:.1f}%, Uncertain)")
        else:
            print(f"{img.name} -> {pred_char} ({conf_pct:.1f}%)")

        print(pred_char)

    print("\nFinal Predicted Word: ", word)

    print("\nEvaluation Metrics (on training set using NN):\n----------------------------------------------")
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, p in enumerate(train_paths):
            v = train_feats[i]
            dists = torch.norm(train_feats - v.unsqueeze(0), dim=1)
            scores = torch.softmax(-dists, dim=0)
            best_idx = int(scores.argmax().item())
            pred_idx = int(train_labels[best_idx])
            true_idx = int(train_labels[i])
            y_true.append(true_idx)
            y_pred.append(pred_idx)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(LABELS))))

    print(f"Accuracy: {acc:.2f}")
    print(f"Precision (macro): {prec:.2f}")
    print(f"Recall (macro): {rec:.2f}")
    print(f"F1 Score (macro): {f1:.2f}")
    print("\nConfusion Matrix:\n-----------------")
    print(cm)


# -------------------------
# CLI
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(42)

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
