#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$ROOT/experiments"
LOG_DIR="$ROOT/logs"
CKPT_DIR="$ROOT/modells"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CKPT_DIR"

MAX_JOBS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-jobs)
      MAX_JOBS="$2"; shift 2;;
    *)
      echo "[run_experiments] Unknown arg: $1" >&2; shift 1;;
  esac
done

RESULTS_CSV="$RESULTS_DIR/results.csv"
if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "exp_id,latent_dim,q_output,n_qubits,lr,batch,seed,best_val_acc,best_epoch,ckpt_path,time_elapsed" > "$RESULTS_CSV"
fi

latent_dims=(8 16 32)
q_outputs=(8 16 32)
n_qubits_list=(2 4)
lrs=(0.0003 0.0001)
batches=(4 8)
seeds=(42 123)

EXP_INDEX=0

for ld in "${latent_dims[@]}"; do
  for qo in "${q_outputs[@]}"; do
    for nq in "${n_qubits_list[@]}"; do
      for lr in "${lrs[@]}"; do
        for bs in "${batches[@]}"; do
          for sd in "${seeds[@]}"; do
            EXP_INDEX=$((EXP_INDEX+1))
            EXP_ID="$(date +%Y%m%d_%H%M%S)_${EXP_INDEX}"
            echo "[run_experiments] Launching exp $EXP_ID (latent=$ld, q_out=$qo, n_qubits=$nq, lr=$lr, batch=$bs, seed=$sd)"

            START_TS=$(date +%s)
            bash "$ROOT/experiments/train_single.sh" \
              --exp_id "$EXP_ID" \
              --epochs 10 \
              --batch "$bs" \
              --img 224 \
              --lr "$lr" \
              --n_qubits "$nq" \
              --n_layers 2 \
              --latent_dim "$ld" \
              --q_output "$qo" \
              --seed "$sd"
            END_TS=$(date +%s)
            ELAPSED=$((END_TS-START_TS))

            METRICS_FILE="$LOG_DIR/${EXP_ID}.metrics.json"
            CKPT_PATH="$CKPT_DIR/exp_${EXP_ID}.pt"

            # Extract best_val_acc and best_epoch via a tiny Python helper
            BEST_ACC="0.0"
            BEST_EPOCH="0"
            if [[ -f "$METRICS_FILE" ]]; then
              read -r BEST_ACC BEST_EPOCH < <(
                python - "$METRICS_FILE" << 'PY'
import json, sys
p = sys.argv[1]
with open(p, 'r') as f:
    data = json.load(f)
if not data:
    print("0.0 0")
    raise SystemExit(0)
best = max(data, key=lambda d: d.get('val_acc', 0.0))
print(best.get('val_acc', 0.0), best.get('epoch', 0))
PY
              )
            fi

            echo "$EXP_ID,$ld,$qo,$nq,$lr,$bs,$sd,$BEST_ACC,$BEST_EPOCH,$CKPT_PATH,$ELAPSED" >> "$RESULTS_CSV"
          done
        done
      done
    done
  done
done

echo "[run_experiments] Done. Results in $RESULTS_CSV"