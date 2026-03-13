#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
CKPT_DIR="$ROOT/modells"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Parse args
EXP_ID=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_id|--exp-id)
      EXP_ID="$2"; shift 2;;
    *)
      EXTRA_ARGS+=("$1"); shift 1;;
  esac
done

if [[ -z "$EXP_ID" ]]; then
  EXP_ID="$(date +%Y%m%d_%H%M%S)_$$"
fi

LOG_FILE="$LOG_DIR/${EXP_ID}.log"
METRICS_FILE="$LOG_DIR/${EXP_ID}.metrics.json"

if [[ ! -f "$ROOT/venv/bin/python" ]]; then
  echo "[train_single] venv Python not found at $ROOT/venv/bin/python" >&2
  exit 1
fi

PYTHON="$ROOT/venv/bin/python"

cd "$ROOT"
START_TS=$(date +%s)

echo "[train_single] Starting experiment $EXP_ID" | tee "$LOG_FILE"
"$PYTHON" main_quantum.py train \
  --checkpoint-dir "$CKPT_DIR" \
  --exp-id "$EXP_ID" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

END_TS=$(date +%s)
ELAPSED=$((END_TS-START_TS))

SRC_CKPT="$CKPT_DIR/quantum_best.pt"
DEST_CKPT="$CKPT_DIR/exp_${EXP_ID}.pt"
if [[ -f "$SRC_CKPT" ]]; then
  cp "$SRC_CKPT" "$DEST_CKPT"
  echo "[train_single] Copied best checkpoint to $DEST_CKPT" | tee -a "$LOG_FILE"
else
  echo "[train_single] WARNING: expected checkpoint $SRC_CKPT not found" | tee -a "$LOG_FILE"
fi

# Ensure metrics file exists (main_quantum.py writes it as logs/<exp_id>.metrics.json)
if [[ -f "$METRICS_FILE" ]]; then
  echo "[train_single] Metrics written to $METRICS_FILE (elapsed ${ELAPSED}s)" | tee -a "$LOG_FILE"
else
  echo "[train_single] WARNING: metrics file $METRICS_FILE not found" | tee -a "$LOG_FILE"
fi

exit 0
