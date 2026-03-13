# -*- coding: utf-8 -*-
"""
Quantum-first EEG Waveform A–Z classifier (single-file, VS Code friendly).

Architecture
-----------
image -> SmallImageEncoder (tiny MLP) -> SafePennylaneLayer (TorchLayer QNode)
      -> classical head -> 26-way logits (A–Z).

This script is designed to be robust on macOS CPU (and optional GPU) and to be
used from VS Code's integrated terminal and debugger.

CLI
---
Train:
    python main_quantum.py train --epochs 10 --batch 4 --img 224 \
        --lr 3e-4 --n_qubits 4 --n_layers 2 --latent_dim 8 --q_output 16

Test:
    python main_quantum.py test --img 224

Checkpoints
-----------
Best model (quantum-first) is saved to:
    modells/quantum_best.pt
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Optional TensorBoard support
try:  # pragma: no cover - optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - environment-specific
    SummaryWriter = None

# Optional but recommended: torchvision for augmentations / backbones
try:  # pragma: no cover - optional dep
    import torchvision
    from torchvision import models as tv_models
    import torchvision.transforms.functional as TVF
except Exception:  # pragma: no cover - environment-specific
    torchvision = None
    tv_models = None
    TVF = None

# Optional but recommended
try:  # tqdm is nice but not required
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dep
    tqdm = None

# PennyLane / autoray diagnostics
try:
    import pennylane as qml
    _pennylane_exc = None
except Exception as e:  # pragma: no cover - environment-specific
    qml = None
    _pennylane_exc = e

try:
    import autoray as ar
except Exception as e:  # pragma: no cover - optional
    ar = None

# -------------------------
# Helpers & config
# -------------------------
LABELS = [chr(ord("A") + i) for i in range(26)]
LBL2ID = {c: i for i, c in enumerate(LABELS)}


def set_seed(seed: int = 42) -> None:
    import numpy as _np

    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism; may be limited by ops used
    try:  # pragma: no cover - backend specific
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:  # pragma: no cover - backend specific
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS Metal backend if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_label_from_path(p: Path) -> str:
    parent = p.parent.name.upper()
    if parent in LBL2ID:
        return parent
    stem = p.stem.upper()
    if stem and stem[0] in LBL2ID:
        return stem[0]
    raise ValueError(f"Cannot infer label for: {p}")


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def split_stratified(paths: List[Path], val_ratio=0.15, min_val_per_label=1):
    """Return (train_items, val_items) as list[(Path,label)].
    Robust stratified split that avoids empty train/val lists where possible.
    """
    by_lbl: Dict[str, List[Path]] = {c: [] for c in LABELS}
    for p in paths:
        try:
            lbl = infer_label_from_path(p)
            by_lbl[lbl].append(p)
        except Exception:
            pass

    train, val = [], []
    for c in LABELS:
        L = by_lbl[c]
        if not L:
            continue
        random.shuffle(L)
        n = len(L)
        n_val = max(0, int(n * val_ratio))
        if n >= (min_val_per_label + 1):
            n_val = max(n_val, min(min_val_per_label, n - 1))
        if n_val >= n:
            n_val = max(0, n - 1)
        val += [(p, LBL2ID[c]) for p in L[:n_val]]
        train += [(p, LBL2ID[c]) for p in L[n_val:]]
    # ensure neither split is empty if dataset has >1 items
    if len(train) == 0 and len(val) > 1:
        train.append(val.pop(0))
    if len(val) == 0 and len(train) > 1:
        val.append(train.pop(0))
    return train, val


def load_image_tensor(path: Path, img_size: int = 224, to_gray: bool = True) -> torch.Tensor:
    img = Image.open(path).convert("L" if to_gray else "RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    if to_gray:
        arr = arr[None, ...]  # (1,H,W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    x = torch.from_numpy(arr)
    return x


class EEGImageDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, int]],
        img_size: int = 224,
        gray: bool = True,
        train: bool = False,
        use_strong_augment: bool = False,
    ):
        self.items = items
        self.img_size = img_size
        self.gray = gray
        self.train = train
        self.use_strong_augment = use_strong_augment

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return x
        # light noise
        if torch.rand(1).item() < 0.7:
            x = torch.clamp(x + 0.03 * torch.randn_like(x), 0.0, 1.0)
        # random blackout patch (cutout-style)
        if torch.rand(1).item() < 0.5:
            _, h, w = x.shape
            ch = int(0.2 * h)
            cw = int(0.2 * w)
            y0 = torch.randint(0, max(1, h - ch), (1,)).item()
            x0 = torch.randint(0, max(1, w - cw), (1,)).item()
            x[:, y0 : y0 + ch, x0 : x0 + cw] = 0
        # optional stronger augmentations
        if self.use_strong_augment:
            # horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            # small random rotation via torchvision if available
            if TVF is not None and torch.rand(1).item() < 0.5:
                angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
                x = TVF.rotate(x, angle)
            # brightness / contrast jitter
            if torch.rand(1).item() < 0.5:
                factor = 0.2
                noise = (torch.rand_like(x) - 0.5) * factor
                x = torch.clamp(x + noise, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        x = load_image_tensor(path, self.img_size, self.gray)
        x = self._augment(x)
        return x, y


# -------------------------
# Quantum-first model pieces
# -------------------------
class SmallImageEncoder(nn.Module):
    """Tiny MLP encoder from image -> latent vector.

    We intentionally keep this very small to make the quantum part the bottleneck.
    """

    def __init__(self, img_ch: int = 1, img_size: int = 224, latent_dim: int = 8):
        super().__init__()
        in_dim = img_ch * img_size * img_size
        hidden = max(32, latent_dim * 4)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SafePennylaneLayer(nn.Module):
    """Quantum layer that prefers batched TorchLayer, with safe fallback.

    Behaviour:
      * If PennyLane is available, constructs a TorchLayer-backed QNode with
        full torch backprop enabled (diff_method="backprop").
      * On the first forward pass, it *attempts* a batched call. If that raises
        (e.g. due to autoray / interface quirks), it falls back to per-sample
        detached calls and will stay in fallback mode for the rest of the run.
      * If PennyLane is not importable, this becomes a purely-classical linear
        layer so that training still works.

    The script prints which path was selected so you can see whether full
    batched quantum gradients are active or the safe fallback is used.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_inputs: int = 8,
        q_output: int = 16,
        backend: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.q_output = q_output

        self.mode = "uninitialized"  # "batched" | "fallback" | "classical"
        self._batched_error = None

        if qml is None:
            # No PennyLane: use a small trainable MLP as a robust classical fallback
            hidden = max(n_inputs, q_output * 2)
            self.classical = nn.Sequential(
                nn.Linear(n_inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, q_output),
            )
            self.mode = "classical"
            print("[Quantum] PennyLane not available; using classical MLP fallback instead.")
            if _pennylane_exc is not None:
                print("[Quantum] PennyLane import error:", _pennylane_exc)
            return

        # True quantum path
        self.dev = qml.device(backend, wires=self.n_qubits)
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # inputs: shape (n_inputs,)
            # simple angle embedding into RX rotations
            for i in range(self.n_qubits):
                angle = inputs[i % inputs.shape[0]] if inputs.shape[0] > 0 else 0.0
                qml.RX(angle * math.pi, wires=i)
            # variational layers
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # measure PauliZ on each qubit as features
            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            return qml.stack(expvals)

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.post = nn.Linear(self.n_qubits, q_output) if q_output > self.n_qubits else None

    def _post_process(self, out: torch.Tensor) -> torch.Tensor:
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=0)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        if self.post is not None:
            out = torch.tanh(self.post(out))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_inputs)
        if self.mode == "classical":
            out = self.classical(x)
            return torch.tanh(out)

        B = x.shape[0]

        # First-time decision: try batched TorchLayer
        if self.mode == "uninitialized":
            try:
                _ = self._post_process(self.qlayer(x))
                self.mode = "batched"
                print("[Quantum] Using batched TorchLayer with full autodiff.")
            except Exception as e:  # pragma: no cover - environment-specific
                self.mode = "fallback"
                self._batched_error = e
                print("[Quantum] Batched TorchLayer failed, switching to per-sample fallback.")
                print("[Quantum] Batched error:", repr(e))

        if self.mode == "batched":
            try:
                out = self._post_process(self.qlayer(x))
                return out
            except Exception as e:  # pragma: no cover - rare runtime failure
                # If it ever fails at runtime, permanently fall back
                self.mode = "fallback"
                self._batched_error = e
                print("[Quantum] Runtime batched call failed, permanently using fallback.")
                print("[Quantum] Error:", repr(e))
                # fall through to fallback below

        # Fallback path: loop over batch dimension with detached inputs
        outs = []
        for i in range(B):
            single = x[i].detach()  # no grad through inputs; keeps this robust
            try:
                o = self.qlayer(single)
            except Exception:
                # last resort: move to CPU
                o = self.qlayer(single.cpu())
            outs.append(o.detach())
        stacked = torch.stack(outs, dim=0)
        stacked = self._post_process(stacked)
        if self.mode != "fallback":
            self.mode = "fallback"
        return stacked


class QuantumOnlyModel(nn.Module):
    """Full quantum-first classifier.

    image -> (SmallImageEncoder or ResNet18) -> SafePennylaneLayer -> head -> logits(26)
    """

    def __init__(
        self,
        num_classes: int = 26,
        img_ch: int = 1,
        img_size: int = 224,
        latent_dim: int = 8,
        n_qubits: int = 4,
        n_layers: int = 2,
        q_output: int = 16,
        use_resnet_backbone: bool = False,
    ) -> None:
        super().__init__()

        if use_resnet_backbone and tv_models is not None:
            # Simple ResNet18 backbone with 1-channel input; final projection to latent_dim
            resnet = tv_models.resnet18(weights=None)
            if resnet.conv1.in_channels != img_ch:
                resnet.conv1 = nn.Conv2d(img_ch, resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn_backbone = nn.Sequential(*(list(resnet.children())[:-1]))  # till global pool
            feat_dim = resnet.fc.in_features
            self.encoder_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, latent_dim),
                nn.Tanh(),
            )

            def resnet_encode(x: torch.Tensor) -> torch.Tensor:
                f = self.cnn_backbone(x)
                return self.encoder_head(f)

            self.encoder = resnet_encode  # callable
        else:
            self.cnn_backbone = None
            self.encoder_head = None
            self.encoder = SmallImageEncoder(img_ch=img_ch, img_size=img_size, latent_dim=latent_dim)

        self.q_layer = SafePennylaneLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_inputs=latent_dim,
            q_output=q_output,
        )
        self.head = nn.Sequential(
            nn.Linear(q_output, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        qf = self.q_layer(z)
        logits = self.head(qf)
        return logits


# -------------------------
# Train / Test helpers
# -------------------------
def build_dataloaders(img_size: int, batch: int, device: torch.device, use_strong_augment: bool):
    train_root = Path("training dataset")
    if not train_root.exists():
        print("[Error] training dataset/ not found at", train_root)
        raise SystemExit(1)

    paths = list_images(train_root)
    if len(paths) == 0:
        print("[Error] No images in training dataset/", train_root)
        raise SystemExit(1)

    train_items, val_items = split_stratified(paths, val_ratio=0.15)
    if len(val_items) == 0 and len(train_items) > 1:
        train_items, val_items = split_stratified(paths, val_ratio=0.05)

    ds_tr = EEGImageDataset(train_items, img_size=img_size, gray=True, train=True, use_strong_augment=use_strong_augment)
    ds_va = EEGImageDataset(val_items, img_size=img_size, gray=True, train=False, use_strong_augment=False)

    pin_mem = device.type == "cuda"
    num_workers = 2 if os.name != "nt" else 0
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return dl_tr, dl_va, len(paths)


def train(args) -> None:
    device = get_device()
    print(f"[Diag] Torch version: {torch.__version__}")
    print(f"[Diag] Using device: {device}")
    if qml is not None:
        print(f"[Diag] PennyLane version: {getattr(qml, '__version__', 'unknown')}")
        pennylane_ok = True
    else:
        pennylane_ok = False
        print("[Diag] PennyLane not importable.")
        if _pennylane_exc is not None:
            print("[Diag] PennyLane import error:", _pennylane_exc)
    if ar is not None:
        print(f"[Diag] autoray version: {getattr(ar, '__version__', 'unknown')}")
    else:
        print("[Diag] autoray not importable.")

    # experiment / logging IDs
    logs_dir = Path("logs")
    runs_dir = Path("runs")
    logs_dir.mkdir(exist_ok=True, parents=True)
    runs_dir.mkdir(exist_ok=True, parents=True)

    exp_id = args.exp_id or time.strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    metrics_path = logs_dir / f"{exp_id}.metrics.json"
    writer = SummaryWriter(log_dir=runs_dir / exp_id) if SummaryWriter is not None else None

    # dataloaders
    dl_tr, dl_va, n_imgs = build_dataloaders(args.img, args.batch, device, use_strong_augment=args.augment)

    # optional class weights
    class_weight_tensor = None
    if args.use_class_weight:
        counts = torch.zeros(len(LABELS), dtype=torch.float32)
        for _, y in getattr(dl_tr.dataset, "items", []):
            counts[y] += 1.0
        counts[counts == 0.0] = 1.0
        weights = 1.0 / counts
        weights = weights * (len(LABELS) / weights.sum())  # normalize
        class_weight_tensor = weights.to(device)

    model = QuantumOnlyModel(
        num_classes=26,
        img_ch=1,
        img_size=args.img,
        latent_dim=args.latent_dim,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        q_output=args.q_output,
        use_resnet_backbone=args.use_resnet_backbone,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - 1))
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir = Path(args.checkpoint_dir or "modells")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_ckpt_path = ckpt_dir / "quantum_best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"
    print(f"[Train] Found {n_imgs} images. Saving checkpoints under: {ckpt_dir}")

    best_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    patience = 500
    all_metrics = []

    def maybe_mixup(x, y):
        if not args.use_mixup:
            return x, y
        alpha = 0.2
        if alpha <= 0.0:
            return x, y
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return (mixed_x, (y_a, y_b, lam))

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        iterable = dl_tr
        if tqdm is not None:
            iterable = tqdm(dl_tr, desc=f"Epoch {ep}/{args.epochs} [train]", leave=False)

        for x, y in iterable:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x_mixed, mix_info = maybe_mixup(x, y)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x_mixed)
                    if isinstance(mix_info, tuple):
                        y_a, y_b, lam = mix_info
                        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    else:
                        loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x_mixed)
                if isinstance(mix_info, tuple):
                    y_a, y_b, lam = mix_info
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y)
                loss.backward()
                opt.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_train_loss = train_loss / max(1, n_batches)
        avg_val_loss = val_loss / max(1, len(dl_va))
        val_acc = correct / max(1, total)
        elapsed = int(time.time() - t0)

        print(
            f"Epoch {ep:03d}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  time={elapsed}s"
        )
        
        print(f"[Debug] bad_epochs={bad_epochs}, patience={patience}, best_acc={best_acc}")
        
        # log to TensorBoard (if available)
        if writer is not None:
            writer.add_scalar("loss/train", avg_train_loss, ep)
            writer.add_scalar("loss/val", avg_val_loss, ep)
            writer.add_scalar("acc/val", val_acc, ep)

        epoch_metrics = {
            "epoch": ep,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "val_acc": float(val_acc),
            "time_sec": int(elapsed),
        }
        all_metrics.append(epoch_metrics)
        with metrics_path.open("w") as f:
            json.dump(all_metrics, f, indent=2)

        # save last checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "qparams": {
                    "latent_dim": args.latent_dim,
                    "n_qubits": args.n_qubits,
                    "n_layers": args.n_layers,
                    "q_output": args.q_output,
                    "use_resnet_backbone": args.use_resnet_backbone,
                },
                "epoch": ep,
                "best_val_acc": best_acc,
            },
            last_ckpt_path,
        )

        # Save best checkpoint (and at least once after epoch 1).
        acc = val_acc
        if acc > best_acc or ep == 1:
            best_acc = max(acc, best_acc)
            best_epoch = ep
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "qparams": {
                        "latent_dim": args.latent_dim,
                        "n_qubits": args.n_qubits,
                        "n_layers": args.n_layers,
                        "q_output": args.q_output,
                        "use_resnet_backbone": args.use_resnet_backbone,
                    },
                    "epoch": ep,
                    "best_val_acc": best_acc,
                },
                best_ckpt_path,
            )
            print(f"[Checkpoint] Saved new best model (val_acc={best_acc:.4f}) to {best_ckpt_path}")
        else:
            bad_epochs += 1
            #if bad_epochs >= patience:
                #print("[Train] Early stopping: patience reached.")
                #break

    if writer is not None:
        writer.close()
    print(f"[Train] Done. Best val_acc={best_acc:.4f} at epoch {best_epoch}")


def test(args) -> None:
    device = get_device()
    print(f"[Diag] Torch version: {torch.__version__}")
    print(f"[Diag] Using device: {device}")
    if qml is not None:
        print(f"[Diag] PennyLane version: {getattr(qml, '__version__', 'unknown')}")
    else:
        print("[Diag] PennyLane not importable.")

    # resolve checkpoint
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_dir = Path(args.checkpoint_dir or "modells")
        ckpt_path = ckpt_dir / "quantum_best.pt"
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}. Run train first or specify --checkpoint-path.")
        raise SystemExit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    qparams = ckpt.get(
        "qparams",
        {"latent_dim": args.latent_dim, "n_qubits": args.n_qubits, "n_layers": args.n_layers, "q_output": args.q_output},
    )

    model = QuantumOnlyModel(
        num_classes=26,
        img_ch=1,
        img_size=args.img,
        latent_dim=qparams.get("latent_dim", args.latent_dim),
        n_qubits=qparams.get("n_qubits", args.n_qubits),
        n_layers=qparams.get("n_layers", args.n_layers),
        q_output=qparams.get("q_output", args.q_output),
        use_resnet_backbone=qparams.get("use_resnet_backbone", args.use_resnet_backbone),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    test_root = Path("teseting Dataset")
    if not test_root.exists():
        print("[Error] teseting Dataset/ not found at", test_root)
        raise SystemExit(1)

    test_imgs = list_images(test_root)
    test_imgs.sort(key=lambda p: p.name.lower())

    def predict_with_tta(x_batch: torch.Tensor) -> np.ndarray:
        # x_batch: (1, C, H, W)
        if not args.use_tta:
            with torch.no_grad():
                logits = model(x_batch)[0].detach().cpu().numpy()
            probs = np.exp(logits - logits.max())
            return probs / probs.sum()
        variants = [x_batch]
        if TVF is not None:
            variants.append(TVF.hflip(x_batch))
            variants.append(TVF.rotate(x_batch, 10))
            variants.append(TVF.rotate(x_batch, -10))
        else:
            variants.append(torch.flip(x_batch, dims=[3]))  # horizontal flip
        prob_sum = None
        with torch.no_grad():
            for xb in variants:
                logits = model(xb)[0].detach().cpu().numpy()
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                prob_sum = probs if prob_sum is None else (prob_sum + probs)
        return prob_sum / len(variants)

    predicted_letters: List[str] = []
    for p in test_imgs:
        x = load_image_tensor(p, img_size=args.img, to_gray=True).unsqueeze(0).to(device)
        probs = predict_with_tta(x)
        idx = int(np.argmax(probs))
        letter = LABELS[idx]
        conf = float(probs[idx] * 100.0)
        print(f"[Predict] {p} -> {letter} ({conf:.2f}%)")
        predicted_letters.append(letter)

    word = "".join(predicted_letters)
    print("\nFinal Predicted Word:", word)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantum-first EEG A–Z classifier (SmallImageEncoder + SafePennylaneLayer)",
    )
    ap.add_argument("mode", choices=["train", "test"], help="Operation mode: train or test")
    ap.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--img", type=int, default=224, help="Input image size (square)")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs")
    ap.add_argument("--n_qubits", type=int, default=4, help="Number of qubits in quantum circuit")
    ap.add_argument("--n_layers", type=int, default=2, help="Number of variational layers in circuit")
    ap.add_argument("--latent_dim", type=int, default=8, help="Latent dimension of classical encoder output")
    ap.add_argument("--q_output", type=int, default=16, help="Quantum feature vector size before head")
    ap.add_argument("--checkpoint-dir", type=str, default="modells", help="Directory to save/load checkpoints")
    ap.add_argument("--checkpoint-path", type=str, default="", help="Explicit checkpoint path for test mode")
    ap.add_argument("--augment", action="store_true", help="Enable stronger data augmentation during training")
    ap.add_argument("--use-class-weight", action="store_true", help="Use class-balanced loss weights")
    ap.add_argument("--use-resnet-backbone", action="store_true", help="Use ResNet18 encoder instead of SmallImageEncoder")
    ap.add_argument("--use-tta", action="store_true", help="Use simple test-time augmentation at inference")
    ap.add_argument("--use-mixup", action="store_true", help="Enable mixup regularization during training")
    ap.add_argument("--exp-id", type=str, default="", help="Optional experiment id for logging")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
    