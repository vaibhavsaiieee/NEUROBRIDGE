#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
  echo "[train.sh] venv not found at $ROOT/venv. Create it first."
  exit 1
fi

source "$ROOT/venv/bin/activate"
cd "$ROOT"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-8}"
IMG="${IMG:-224}"
LR="${LR:-0.0003}"
ARCH="${ARCH:-cnn}"

echo "[train.sh] Running classical training (logging to $LOG_FILE)" | tee "$LOG_FILE"
python main_classical.py train \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --img "$IMG" \
  --lr "$LR" \
  --arch "$ARCH" "$@" | tee -a "$LOG_FILE"
