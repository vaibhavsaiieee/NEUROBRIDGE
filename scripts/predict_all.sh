#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/predict.log"

if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
  echo "[predict_all.sh] venv not found at $ROOT/venv. Create it first."
  exit 1
fi

source "$ROOT/venv/bin/activate"
cd "$ROOT"

IMG="${IMG:-224}"

echo "[predict_all.sh] Running main_classical.py test on 'teseting Dataset' (logging to $LOG_FILE)" | tee "$LOG_FILE"
python main_classical.py test \
  --img "$IMG" \
  --test-root "teseting Dataset" "$@" | tee -a "$LOG_FILE"
