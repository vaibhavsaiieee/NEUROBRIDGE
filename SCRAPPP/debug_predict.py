# -*- coding: utf-8 -*-
"""Debug helper for classical CNN checkpoints.

Example:
    python debug_predict.py --img 224 --ckpt modells/best_classical.pt \
        --testdir "teseting Dataset" --topk 5

This prints per-image logits, top-k probabilities, entropy, and writes a CSV
under logs/predictions_debug.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from main_classical import LABELS, _resolve_testdir, list_images, load_image_tensor, _load_checkpoint_for_eval, get_device, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Debug predictions for classical CNN")
    ap.add_argument("--img", type=int, default=224, help="Input image size")
    ap.add_argument("--ckpt", type=str, default="", help="Checkpoint path")
    ap.add_argument("--testdir", type=str, default="", help="Test directory (defaults to teseting Dataset)")
    ap.add_argument("--topk", type=int, default=5, help="Top-k to display")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device()

    test_dir = _resolve_testdir(args.testdir)
    imgs = list_images(test_dir)
    if not imgs:
        print(f"[Error] No images found under {test_dir}")
        raise SystemExit(1)

    model, meta = _load_checkpoint_for_eval(args, device)
    img_size = int(meta["img"])

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / "predictions_debug.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "top1_label", "top1_prob", "entropy"])

        for p in imgs:
            x = load_image_tensor(p, img_size=img_size, to_gray=True).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)[0]
                probs = F.softmax(logits, dim=0)
                probs_np = probs.detach().cpu().numpy()

            entropy = float(-np.sum(probs_np * np.log(np.clip(probs_np, 1e-12, 1.0))))

            topk = min(args.topk, len(LABELS))
            topk_prob, topk_idx = torch.topk(probs, k=topk)

            print("====", p, "====")
            print("logits (first 10):", logits.detach().cpu().numpy()[:10])
            for rank, (idx, pv) in enumerate(zip(topk_idx, topk_prob), start=1):
                lbl = LABELS[int(idx.item())]
                print(f"  top{rank}: {lbl} prob={float(pv.item()):.4f}")
            print(f"entropy={entropy:.4f}")

            top1_idx = int(topk_idx[0].item())
            top1_prob = float(topk_prob[0].item())
            top1_label = LABELS[top1_idx]
            writer.writerow([str(p), top1_label, f"{top1_prob:.6f}", f"{entropy:.6f}"])

    print(f"[Debug] Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
