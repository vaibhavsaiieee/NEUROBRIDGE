#!/usr/bin/env python3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import torch
from main_quantum import QuantumOnlyModel

qparams = {"latent_dim": 8, "n_qubits": 4, "n_layers": 2, "q_output": 16}
model = QuantumOnlyModel(
    num_classes=26,
    img_ch=1,
    img_size=224,
    latent_dim=qparams["latent_dim"],
    n_qubits=qparams["n_qubits"],
    n_layers=qparams["n_layers"],
    q_output=qparams["q_output"],
)
Path("modells").mkdir(parents=True, exist_ok=True)
ckpt = {"model": model.state_dict(), "qparams": qparams, "arch": "quantum"}
torch.save(ckpt, Path("modells/quantum_best.pt"))
print("Wrote modells/quantum_best.pt (dummy).")
