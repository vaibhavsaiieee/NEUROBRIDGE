#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$ROOT/training dataset"

MODE="check"
if [[ "${1:-}" == "--organize" ]]; then
  MODE="organize"
fi

python - << 'PY'
from pathlib import Path
import shutil
import string
import sys

root = Path(r"training dataset")
if not root.exists():
    print("[prepare_data] training dataset/ not found at", root)
    sys.exit(1)

label_dirs = {c: root / c for c in string.ascii_uppercase}

print("[prepare_data] Scanning", root)
counts = {c: 0 for c in string.ascii_uppercase}

for p in root.rglob("*"):
    if not p.is_file():
        continue
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        continue
    parent = p.parent.name.upper()
    stem = p.stem.upper()
    label = None
    if parent in counts:
        label = parent
    elif stem and stem[0] in counts:
        label = stem[0]
    if label is None:
        print("[prepare_data] Skipping (no label inferred):", p)
        continue
    counts[label] += 1

print("[prepare_data] Counts per label:")
for c in string.ascii_uppercase:
    print(f"  {c}: {counts[c]}")

mode = "check"
if len(sys.argv) > 1:
    mode = sys.argv[1]

if mode != "organize":
    print("[prepare_data] Run with --organize to move files into A/..Z/ folders.")
    sys.exit(0)

# organize: move files into training dataset/<LABEL>/
for c, d in label_dirs.items():
    d.mkdir(exist_ok=True, parents=True)

for p in list(root.rglob("*")):
    if not p.is_file():
        continue
    if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        continue
    parent = p.parent.name.upper()
    stem = p.stem.upper()
    label = None
    if parent in counts:
        label = parent
    elif stem and stem[0] in counts:
        label = stem[0]
    if label is None:
        continue
    dest = label_dirs[label] / p.name
    if dest == p:
        continue
    print(f"[prepare_data] Moving {p} -> {dest}")
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.move(str(p), str(dest))

print("[prepare_data] Done organizing.")
PY "$MODE"