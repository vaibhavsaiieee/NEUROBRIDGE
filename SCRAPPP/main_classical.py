# -*- coding: utf-8 -*-
"""Classical EEG A–Z classifier (ImprovedCNN, single-file).

This script provides a **deterministic, classical-only** training and
prediction pipeline that does **not** depend on PennyLane.

Usage (from repo root):

  Train:
      python main_classical.py train --epochs 50 --batch 8 --img 128 --lr 1e-3

  Test / predict (on external test folder, default "teseting Dataset"):
      python main_classical.py test --img 128 --test-root "teseting Dataset"

The script is designed for macOS + venv + VS Code, but also runs on CPU-only
Linux/Windows. It always writes checkpoints to ``modells/best_classical.pt``.
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Global label set
# ---------------------------------------------------------------------------

LABELS = [chr(ord("A") + i) for i in range(26)]
LBL2ID = {c: i for i, c in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Reproducibility & devices
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and Torch RNG seeds for reasonably deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick best available device (cuda > mps > cpu) and print diagnostics."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch, "has_mps", False)
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[Diag] Torch version: {torch.__version__}")
    print(f"[Diag] Using device: {device}")
    if device.type == "cuda":
        print(f"[Diag] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Diag] CUDA device name: {torch.cuda.get_device_name(0)}")
    return device


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def infer_label_from_path(p: Path) -> str:
    """Infer label A–Z from either the parent folder or filename prefix."""
    parent = p.parent.name.upper()
    if parent in LBL2ID:
        return parent
    stem = p.stem.upper()
    if stem and stem[0] in LBL2ID:
        return stem[0]
    raise ValueError(f"Cannot infer label for: {p}")


def list_images(root: Path) -> List[Path]:
    """Recursively list image files under ``root`` (PNG/JPEG/BMP)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def load_image_tensor(
    path: Path, img_size: int = 128, to_gray: bool = True
) -> torch.Tensor:
    """Load an image as a ``torch.FloatTensor`` in ``[0,1]``.

    Handles RGBA by compositing onto white background before conversion.
    Returns shape ``(1,H,W)`` if ``to_gray``, else ``(3,H,W)``.
    """
    img = Image.open(path)
    # Handle RGBA: composite onto white background
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    img = img.convert("L" if to_gray else "RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    if to_gray:
        arr = arr[None, ...]  # (1,H,W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    return torch.from_numpy(arr)


class EEGAugmentedDataset(Dataset):
    """Dataset that creates many augmented copies from a small set of images.

    With only 1 image per class, we replicate each image ``copies_per_image``
    times and apply random augmentations to each copy during __getitem__.
    """

    def __init__(
        self,
        items: List[Tuple[Path, int]],
        img_size: int = 128,
        gray: bool = True,
        train: bool = False,
        copies_per_image: int = 100,
    ) -> None:
        self.img_size = img_size
        self.gray = gray
        self.train = train
        # Preload all images into memory (only 26 images)
        self.images: List[Tuple[torch.Tensor, int]] = []
        for path, label in items:
            x = load_image_tensor(path, img_size, gray)
            self.images.append((x, label))
        # Create index map: repeat each image many times for training
        if train and copies_per_image > 1:
            self.index_map = list(range(len(self.images))) * copies_per_image
        else:
            self.index_map = list(range(len(self.images)))

    def __len__(self) -> int:
        return len(self.index_map)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return x
        # Gaussian noise (light)
        if torch.rand(1).item() < 0.7:
            noise_scale = 0.01 + 0.02 * torch.rand(1).item()
            x = torch.clamp(x + noise_scale * torch.randn_like(x), 0.0, 1.0)
        # Random brightness shift
        if torch.rand(1).item() < 0.5:
            shift = (torch.rand(1).item() - 0.5) * 0.1
            x = torch.clamp(x + shift, 0.0, 1.0)
        # Random contrast
        if torch.rand(1).item() < 0.5:
            factor = 0.8 + 0.4 * torch.rand(1).item()
            mean = x.mean()
            x = torch.clamp((x - mean) * factor + mean, 0.0, 1.0)
        # Small random horizontal shift (up to 5% of width)
        if torch.rand(1).item() < 0.5:
            _, h, w = x.shape
            max_shift = max(1, int(0.05 * w))
            dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            x = torch.roll(x, shifts=dx, dims=2)
        # Small random vertical shift (up to 5% of height)
        if torch.rand(1).item() < 0.5:
            _, h, w = x.shape
            max_shift = max(1, int(0.05 * h))
            dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            x = torch.roll(x, shifts=dy, dims=1)
        return x

    def __getitem__(self, idx: int):
        real_idx = self.index_map[idx]
        x, y = self.images[real_idx]
        x = x.clone()
        x = self._augment(x)
        return x, y


# ---------------------------------------------------------------------------
# Model — Deeper CNN for better feature extraction
# ---------------------------------------------------------------------------


class ImprovedCNN(nn.Module):
    """Deeper CNN with residual-style connections for 1xHxW EEG waveform images."""

    def __init__(self, img_ch: int = 1, num_classes: int = 26) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(img_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def build_datasets(img_size: int, copies_per_image: int = 100):
    """Create training dataset from ``training dataset/``.

    With only 1 image per class, all images go to training (no val split).
    Heavy augmentation creates ``copies_per_image`` synthetic variants per image.
    """
    train_root = Path("training dataset")
    if not train_root.exists():
        print(f"[Error] training dataset/ not found at {train_root}")
        raise SystemExit(1)

    paths = list_images(train_root)
    if len(paths) == 0:
        print(f"[Error] No images found in {train_root}")
        raise SystemExit(1)

    print(f"[Diag] Found {len(paths)} images under '{train_root}'")

    # Build items list
    items: List[Tuple[Path, int]] = []
    for p in paths:
        try:
            lbl = infer_label_from_path(p)
            items.append((p, LBL2ID[lbl]))
        except Exception:
            print(f"[Warn] Skipping {p} — cannot infer label")

    if len(items) == 0:
        print("[Error] Could not infer labels for any images.")
        raise SystemExit(1)

    # With 1 image per class: ALL go to training, no val split
    ds_tr = EEGAugmentedDataset(
        items, img_size=img_size, gray=True, train=True,
        copies_per_image=copies_per_image,
    )
    # Val set = same images without augmentation (for monitoring only)
    ds_va = EEGAugmentedDataset(
        items, img_size=img_size, gray=True, train=False,
        copies_per_image=1,
    )

    print(f"[Diag] Training samples (with augmentation): {len(ds_tr)}")
    print(f"[Diag] Validation samples (original images):  {len(ds_va)}")
    return ds_tr, ds_va


def make_dataloaders(ds_tr: Dataset, ds_va: Dataset, batch: int, device: torch.device):
    """Create train/val dataloaders with safe defaults for macOS."""
    num_workers = 0
    pin_mem = device.type == "cuda"
    dl_tr = DataLoader(
        ds_tr, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
    )
    dl_va = DataLoader(
        ds_va, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )
    return dl_tr, dl_va


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Train ImprovedCNN on ``training dataset/`` and save best checkpoint."""
    set_seed(args.seed)
    device = get_device()

    ds_tr, ds_va = build_datasets(args.img, copies_per_image=100)
    dl_tr, dl_va = make_dataloaders(ds_tr, ds_va, args.batch, device)

    model = ImprovedCNN(img_ch=1, num_classes=len(LABELS)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    models_dir = Path("modells")
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / "best_classical.pt"

    best_acc = 0.0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        n_batches = 0

        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
            n_batches += 1

        if n_batches == 0:
            print("[Error] No batches produced; check dataset.")
            raise SystemExit(1)

        scheduler.step()

        # Validation (on original un-augmented images)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_loss = train_loss / max(1, n_batches)
        elapsed = int(time.time() - t0)

        print(
            f"Epoch {ep:03d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"time={elapsed}s"
        )

        # Save checkpoint when val accuracy improves (or always at epoch 1)
        if val_acc > best_acc or ep == 1:
            best_acc = max(val_acc, best_acc)
            best_epoch = ep
            state = {
                "model": model.state_dict(),
                "arch": "improved_cnn",
                "img": int(args.img),
                "num_classes": len(LABELS),
                "qparams": None,
            }
            torch.save(state, ckpt_path)
            print(
                f"[Checkpoint] Saved best_classical.pt at epoch {ep} "
                f"(val_acc={val_acc:.4f})"
            )

    print(
        f"\nTraining done. Best val_acc={best_acc:.4f} "
        f"saved to {ckpt_path} (epoch {best_epoch})"
    )


# ---------------------------------------------------------------------------
# Checkpoint loading & test-time prediction
# ---------------------------------------------------------------------------


def _load_checkpoint_for_eval(
    ckpt_path: Path, device: torch.device
) -> Tuple[nn.Module, Dict]:
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    img_size = int(ckpt.get("img", 128))
    arch = ckpt.get("arch", "improved_cnn")

    if arch == "simple_cnn":
        model = SimpleCNN(img_ch=1, num_classes=len(LABELS))
    else:
        model = ImprovedCNN(img_ch=1, num_classes=len(LABELS))

    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()
    print(
        f"[Diag] Loaded checkpoint {ckpt_path} (arch={arch}, img={img_size})"
    )
    return model, {"img": img_size}


def _resolve_test_root(path_arg: Optional[str]) -> Path:
    """Resolve the test root directory with the expected spelling."""
    if path_arg:
        p = Path(path_arg)
        if not p.exists():
            print(f"[Error] --test-root {p} does not exist")
            raise SystemExit(1)
        return p

    primary = Path("teseting Dataset")
    if primary.exists():
        return primary

    alt = Path("testing dataset")
    if alt.exists():
        print("[Diag] Using alternate test folder: 'testing dataset'")
        return alt

    print("[Error] Neither 'teseting Dataset' nor 'testing dataset' exists.")
    raise SystemExit(1)


def test(args: argparse.Namespace) -> None:
    """Load best checkpoint and run predictions on the test set."""
    set_seed(args.seed)
    device = get_device()

    test_root = _resolve_test_root(args.test_root)
    imgs = list_images(test_root)
    if len(imgs) == 0:
        print(f"[Error] No images found under {test_root}")
        raise SystemExit(1)

    # Sort numerically by filename (1.png, 2.png, …)
    imgs.sort(key=lambda p: (len(p.stem), p.stem.lower()))

    ckpt_path = Path("modells/best_classical.pt")
    model, meta = _load_checkpoint_for_eval(ckpt_path, device)
    img_size = int(meta["img"])

    print(f"[Predict] Using test folder: {test_root} ({len(imgs)} images)")

    letters: List[str] = []
    for p in imgs:
        x = load_image_tensor(p, img_size=img_size, to_gray=True)
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = F.softmax(logits, dim=0)

        top_idx = int(torch.argmax(probs).item())
        top_prob = float(probs[top_idx].item() * 100.0)
        top_label = LABELS[top_idx]
        letters.append(top_label)

        print(f"[Predict] {p.name} -> {top_label} ({top_prob:.2f}%)")

    word = "".join(letters)
    print(f"\nFinal Predicted Word: {word}")

    # ---- Evaluation (confusion matrix + metrics) ----
    predicted_letters = letters

    try:
        from sklearn.metrics import (
            confusion_matrix,
            accuracy_score,
            precision_recall_fscore_support,
        )
    except ImportError:
        print("[Test] sklearn not installed — skipping metrics.")
        return

    true_labels = []
    for p in imgs:
        try:
            true_labels.append(LBL2ID[infer_label_from_path(p)])
        except Exception:
            print("[Test] Warning: cannot infer label for", p.name)
            true_labels.append(None)

    num_true, num_pred = [], []
    for i, t in enumerate(true_labels):
        if t is None or i >= len(predicted_letters):
            continue
        num_true.append(t)
        num_pred.append(LBL2ID[predicted_letters[i]])

    if len(num_true) == 0:
        print(
            "[Test] No labeled test items found — cannot compute metrics."
        )
    else:
        labels_idx = list(range(len(LABELS)))
        cm = confusion_matrix(num_true, num_pred, labels=labels_idx)
        acc = accuracy_score(num_true, num_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            num_true, num_pred, labels=labels_idx, zero_division=0
        )
        print("\n[Test] Confusion matrix (rows=true, cols=pred):")
        for r in cm:
            print(" ".join(f"{int(v):3d}" for v in r))
        print("[Test] Per-class metrics (precision/recall/f1/support):")
        for i, lab in enumerate(LABELS):
            if support[i] > 0:
                print(
                    f"  {lab}: precision={precision[i]:.3f} "
                    f"recall={recall[i]:.3f} "
                    f"f1={f1[i]:.3f} support={support[i]}"
                )
        print(f"[Test] Overall accuracy: {acc:.4f}")


# Backward compat: keep SimpleCNN available for loading old checkpoints
class SimpleCNN(nn.Module):
    def __init__(self, img_ch: int = 1, num_classes: int = 26) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_ch, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Classical CNN EEG A–Z classifier")
    ap.add_argument("mode", choices=["train", "test"], help="Operation mode")
    ap.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--img", type=int, default=128, help="Input image size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument(
        "--test-root", type=str, default="", help="Test images root dir"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)


if __name__ == "__main__":
    main()
