# main_quantum.py
# -*- coding: utf-8 -*-
"""
Quantum-first EEG Waveform A–Z Classifier (single-file)
- Minimal classical encoder (MLP) -> PennyLane quantum circuit -> classical head
- Robust to PennyLane/autoray environment issues with a safe fallback path.
- Usage:
    ./venv/bin/python main_quantum.py train --epochs 60 --batch 8 --img 224 --n_qubits 4
    ./venv/bin/python main_quantum.py test  --img 224 --arch quantum
- Recommended installs (in venv):
    ./venv/bin/python -m pip install --upgrade pip
    ./venv/bin/python -m pip install "autoray==0.6.11" "pennylane==0.38.0" torch torchvision pillow numpy tqdm
"""
import os, argparse, random, math, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# labels & helpers
# -------------------------
LABELS = [chr(ord('A') + i) for i in range(26)]
LBL2ID = {c: i for i, c in enumerate(LABELS)}

def set_seed(seed=42):
    import numpy as _np, random as _random
    _random.seed(seed); _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def split_stratified(paths: List[Path], val_ratio=0.15):
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
        random.shuffle(L)
        n = len(L)
        if n == 0: continue
        n_val = max(1, int(n * val_ratio)) if n >= 2 else 0
        val += [(p, LBL2ID[c]) for p in L[:n_val]]
        train += [(p, LBL2ID[c]) for p in L[n_val:]]
    return train, val

def load_image_tensor(path: Path, img_size=224, to_gray=True):
    img = Image.open(path).convert("L" if to_gray else "RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    if to_gray:
        arr = arr[None, ...]          # (1,H,W)
    else:
        arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)
    return x

# -------------------------
# Dataset
# -------------------------
class EEGImageDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]], img_size=224, gray=True, train=False):
        self.items = items
        self.img_size = img_size
        self.gray = gray
        self.train = train

    def __len__(self): return len(self.items)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return x
        if torch.rand(1).item() < 0.6:
            x = torch.clamp(x + 0.02 * torch.randn_like(x), 0.0, 1.0)
        if torch.rand(1).item() < 0.4:
            _, h, w = x.shape
            ch = int(0.15*h); cw = int(0.15*w)
            y0 = torch.randint(0, max(1, h-ch), (1,)).item()
            x0 = torch.randint(0, max(1, w-cw), (1,)).item()
            x[:, y0:y0+ch, x0:x0+cw] = 0
        return x

    def __getitem__(self, idx):
        path, y = self.items[idx]
        x = load_image_tensor(path, self.img_size, self.gray)
        x = self._augment(x)
        return x, y

# -------------------------
# Minimal classical encoder (MLP) — NOT a CNN
# -------------------------
class SmallImageEncoder(nn.Module):
    """
    Compress image (1xHxW) -> small latent vector.
    This is intentionally simple (no conv stacks) to keep model classical portion minimal.
    """
    def __init__(self, img_size=224, in_ch=1, latent_dim=8):
        super().__init__()
        flat = in_ch * img_size * img_size
        hidden = max(128, latent_dim*16)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, latent_dim),
            nn.Tanh()
        )
    def forward(self, x): return self.net(x)

# -------------------------
# PennyLane quantum layer (wrapped for safety)
# -------------------------
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    _pennylane_exc = None
except Exception as e:
    qml = None
    _pennylane_exc = e

if qml is not None:
    class SafePennylaneLayer(nn.Module):
        """
        Torch module that wraps a PennyLane TorchLayer qnode with robust fallbacks.
        - First try: call TorchLayer with full batch (native autodiff).
        - Fallback: call per-sample with detached inputs and stack outputs (no gradient to encoder).
        """
        def __init__(self, n_qubits=4, n_layers=2, n_inputs=8, q_output=8, backend="default.qubit"):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_inputs = n_inputs
            self.q_output = q_output
            self.dev = qml.device(backend, wires=self.n_qubits)

            weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                # inputs is a 1D tensor
                # Map first n_qubits elements (or tile) to rotations
                for i in range(self.n_qubits):
                    angle = inputs[i % inputs.shape[0]]
                    qml.RX(angle * math.pi, wires=i)
                for l in range(self.n_layers):
                    for q in range(self.n_qubits):
                        qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
                    for q in range(self.n_qubits - 1):
                        qml.CNOT(wires=[q, q+1])
                # measure Z on each wire
                expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
                return qml.stack(expvals)

            # TorchLayer turns qnode into nn.Module that accepts torch input batches
            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
            self.post = nn.Linear(self.n_qubits, self.q_output) if self.q_output > self.n_qubits else None

        def forward(self, x: torch.Tensor):
            # x shape: (B, n_inputs)
            B = x.shape[0]
            try:
                # Try batched call — prefer native AD
                out = self.qlayer(x)   # expect (B, n_qubits)
                if isinstance(out, (list, tuple)):
                    out = torch.stack(out, dim=0)
                out = out.to(x.device)
                if self.post is not None:
                    out = torch.tanh(self.post(out))
                return out
            except Exception as e:
                # Fallback per-sample (robust)
                outs = []
                for i in range(B):
                    xi = x[i].detach()
                    try:
                        oi = self.qlayer(xi)
                    except Exception:
                        # try on cpu if pennylane expects cpu tensors
                        oi = self.qlayer(xi.cpu())
                    if isinstance(oi, (list, tuple)):
                        oi = torch.stack(oi, dim=0)
                    outs.append(oi.detach())
                stacked = torch.stack(outs, dim=0).to(x.device)
                if self.post is not None:
                    stacked = torch.tanh(self.post(stacked))
                return stacked
else:
    SafePennylaneLayer = None

# -------------------------
# Quantum-first model
# -------------------------
class QuantumOnlyModel(nn.Module):
    """
    Encoder (small MLP) -> Quantum layer -> classical head for 26 classes
    """
    def __init__(self, img_size=224, in_ch=1, latent_dim=8, n_qubits=4, n_layers=2, q_output=16):
        super().__init__()
        self.encoder = SmallImageEncoder(img_size=img_size, in_ch=in_ch, latent_dim=latent_dim)
        if SafePennylaneLayer is None:
            raise RuntimeError("PennyLane not available. Install pennylane to use QuantumOnlyModel.")
        self.q = SafePennylaneLayer(n_qubits=n_qubits, n_layers=n_layers, n_inputs=latent_dim, q_output=q_output)
        self.head = nn.Sequential(nn.Linear(q_output, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, len(LABELS)))

    def forward(self, x):
        z = self.encoder(x)            # (B, latent_dim)
        qf = self.q(z)                 # (B, q_output)
        logits = self.head(qf)
        return logits

# -------------------------
# Train & test loops
# -------------------------
def train(args):
    if qml is None:
        raise RuntimeError(f"PennyLane import failed: {_pennylane_exc}. Install pennylane to train quantum model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Quantum-first training. Qubits={args.n_qubits} layers={args.n_layers} latent={args.latent_dim}")

    train_root = Path("training dataset")
    assert train_root.exists(), "training dataset/ not found"
    paths = list_images(train_root)
    assert len(paths) > 0, "No images in training dataset"
    train_items, val_items = split_stratified(paths, val_ratio=0.15)
    if len(val_items) == 0 and len(train_items) > 1:
        train_items, val_items = split_stratified(paths, val_ratio=0.05)

    ds_tr = EEGImageDataset(train_items, img_size=args.img, gray=True, train=True)
    ds_va = EEGImageDataset(val_items, img_size=args.img, gray=True, train=False)

    num_workers = 2 if os.name != "nt" else 0
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=num_workers, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=num_workers, pin_memory=(device=="cuda"))

    model = QuantumOnlyModel(img_size=args.img, in_ch=1, latent_dim=args.latent_dim,
                             n_qubits=args.n_qubits, n_layers=args.n_layers, q_output=args.q_output).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,args.epochs-1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    models_dir = Path("modells"); models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / "quantum_best.pt"

    best_acc, bad, patience = 0.0, 0, 20
    print(f"Found {len(paths)} images. Train items: {len(ds_tr)} Val items: {len(ds_va)}")
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        batches = 0
        for x,y in dl_tr:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # use autocast on cuda only
            autocast_ctx = torch.cuda.amp.autocast if device=="cuda" else torch.cpu.amp.autocast
            with autocast_ctx(enabled=(device=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            batches += 1
        sched.step()

        # validation
        model.eval()
        tot, correct = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for x,y in dl_va:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc = correct / max(1, tot)
        avg_loss = total_loss / max(1, batches)
        avg_val_loss = val_loss / max(1, max(1, len(dl_va)))
        elapsed = int(time.time() - t0)
        print(f"Epoch {ep:03d}/{args.epochs} train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={acc:.4f} time={elapsed}s")

        if acc > best_acc:
            best_acc, bad = acc, 0
            torch.save({"model": model.state_dict(), "qparams": {"n_qubits":args.n_qubits,"n_layers":args.n_layers,"latent_dim":args.latent_dim,"q_output":args.q_output}, "arch":"quantum"}, ckpt_path)
            print(f"[Checkpoint saved] {ckpt_path} val_acc={best_acc:.4f}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
    print("Training finished. Best val_acc:", best_acc)

def predict_folder(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("modells/quantum_best.pt")
    assert ckpt_path.exists(), "Checkpoint not found. Train first."

    ckpt = torch.load(ckpt_path, map_location=device)
    qparams = ckpt.get("qparams", {"n_qubits":4,"n_layers":2,"latent_dim":8,"q_output":16})
    model = QuantumOnlyModel(img_size=args.img, in_ch=1, latent_dim=qparams["latent_dim"],
                             n_qubits=qparams["n_qubits"], n_layers=qparams["n_layers"], q_output=qparams["q_output"]).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)

    test_root = Path("teseting Dataset")
    assert test_root.exists(), "teseting Dataset/ not found"
    test_imgs = list_images(test_root)
    test_imgs.sort(key=lambda p: p.name.lower())

    predicted = []
    for p in test_imgs:
        x = load_image_tensor(p, img_size=args.img, to_gray=True).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0].cpu().numpy()
            probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
            idx = int(np.argmax(probs)); conf = float(probs[idx]*100.0)
            letter = LABELS[idx]
        print(f"[Predict] {p.name:30s} -> {letter} ({conf:.2f}%)")
        predicted.append(letter)
    print("\nFinal predicted word:", "".join(predicted))

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Quantum-first EEG A–Z classifier")
    ap.add_argument("mode", choices=["train","test"], help="train or test")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_qubits", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--q_output", type=int, default=16)
    return ap.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        predict_folder(args)

if __name__ == "__main__":
    main()
