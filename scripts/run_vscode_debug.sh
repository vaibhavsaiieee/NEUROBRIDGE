#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (directory containing this script's parent)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

# Activate venv
if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
  echo "[run_vscode_debug] venv not found at $ROOT/venv. Create it first."
  exit 1
fi
source "$ROOT/venv/bin/activate"

MODE="${1:-train}"
shift || true

# Default hyperparameters (can be overridden via env vars or extra CLI args)
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-4}"
IMG="${IMG:-224}"
LR="${LR:-0.0003}"
N_QUBITS="${N_QUBITS:-4}"
N_LAYERS="${N_LAYERS:-2}"
LATENT_DIM="${LATENT_DIM:-8}"
Q_OUTPUT="${Q_OUTPUT:-16}"

cd "$ROOT"

case "$MODE" in
  classical-train)
    echo "[run_vscode_debug] Running CLASSICAL TRAIN (logging to $LOG_FILE)" | tee "$LOG_FILE"
    python main_classical.py train \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --img "$IMG" \
      --lr "$LR" "$@" | tee -a "$LOG_FILE"
    ;;
  classical-test)
    echo "[run_vscode_debug] Running CLASSICAL TEST (logging to $LOG_FILE)" | tee "$LOG_FILE"
    python main_classical.py test \
      --img "$IMG" \
      --test-root "teseting Dataset" "$@" | tee -a "$LOG_FILE"
    ;;
  train)
    echo "[run_vscode_debug] Running QUANTUM TRAIN (logging to $LOG_FILE)" | tee "$LOG_FILE"
    python main_quantum.py train \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --img "$IMG" \
      --lr "$LR" \
      --n_qubits "$N_QUBITS" \
      --n_layers "$N_LAYERS" \
      --latent_dim "$LATENT_DIM" \
      --q_output "$Q_OUTPUT" "$@" | tee -a "$LOG_FILE"
    ;;
  test|*)
    echo "[run_vscode_debug] Running QUANTUM TEST (logging to $LOG_FILE)" | tee "$LOG_FILE"
    python main_quantum.py test \
      --img "$IMG" \
      --n_qubits "$N_QUBITS" \
      --n_layers "$N_LAYERS" \
      --latent_dim "$LATENT_DIM" \
      --q_output "$Q_OUTPUT" "$@" | tee -a "$LOG_FILE"
    ;;
esac
