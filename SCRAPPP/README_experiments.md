# Quantum Experiments & Sweeps

This document describes how to run repeatable experiments, sweeps, and
diagnostics for the quantum-first EEG A–Z classifier.

Project root: `./` (same folder as `main_quantum.py`).

## 1. One-shot training: `experiments/train_single.sh`

Wrapper around `main_quantum.py train` that sets up logging and checkpoints.

Example (1-epoch smoke run):

```bash
source venv/bin/activate
bash experiments/train_single.sh \
  --exp_id smoke_1 \
  --epochs 1 \
  --batch 1 \
  --img 224 \
  --lr 3e-4 \
  --n_qubits 4 \
  --n_layers 2 \
  --latent_dim 8 \
  --q_output 16 \
  --augment
```

What it does:

- Uses `./venv/bin/python`.
- Runs `main_quantum.py train` with `--checkpoint-dir modells/` and
  `--exp-id <exp_id>`.
- Logs STDOUT/STDERR to `logs/<exp_id>.log`.
- Expects `main_quantum.py` to write metrics JSON to
  `logs/<exp_id>.metrics.json`.
- Copies `modells/quantum_best.pt` to `modells/exp_<exp_id>.pt`.

## 2. Grid sweeps: `experiments/run_experiments.sh`

Runs a fixed grid of hyperparameters and appends results to
`experiments/results.csv`.

Hyperparameter grid (hard-coded):

- `latent_dim`: 8, 16, 32
- `q_output`: 8, 16, 32
- `n_qubits`: 2, 4
- `lr`: 3e-4, 1e-4
- `batch`: 4, 8
- `seed`: 42, 123

Each combination launches `train_single.sh` with:

- `--epochs 10`
- `--img 224`
- plus the grid parameters.

Usage:

```bash
source venv/bin/activate
bash experiments/run_experiments.sh --max-jobs 1
```

Results are appended to `experiments/results.csv` with columns:

```text
exp_id,latent_dim,q_output,n_qubits,lr,batch,seed,best_val_acc,best_epoch,ckpt_path,time_elapsed
```

You can inspect them with:

```bash
cat experiments/results.csv | column -s, -t | sort -k8 -nr | head -n 10
```

## 3. Main training & testing (for reference)

```bash
# Train
./venv/bin/python main_quantum.py train \
  --epochs 10 \
  --batch 4 \
  --img 224 \
  --lr 3e-4 \
  --n_qubits 4 \
  --n_layers 2 \
  --latent_dim 8 \
  --q_output 16 \
  --checkpoint-dir modells \
  --augment \
  --use-class-weight

# Test
./venv/bin/python main_quantum.py test \
  --img 224 \
  --n_qubits 4 \
  --n_layers 2 \
  --latent_dim 8 \
  --q_output 16 \
  --checkpoint-dir modells \
  --use-tta
```

## 4. Dataset preparation and debugging

### Prepare / organize training data

Use `scripts/prepare_data.sh` to scan and optionally organize
`training dataset/`:

```bash
bash scripts/prepare_data.sh          # only reports counts per label
bash scripts/prepare_data.sh --organize  # moves files into A/..Z/ subfolders
```

### Dataset statistics

Use `utils/data_debug.py` to print per-letter counts and highlight missing
or low-count classes:

```bash
./venv/bin/python -m utils.data_debug "training dataset"
```

## 5. Evaluation utilities

The module `utils/eval_utils.py` contains helpers for post-hoc evaluation:

- `confusion_matrix(y_true, y_pred, num_classes=26)`
- `per_class_metrics(cm)`
- `topk_accuracy(logits, targets, k=5)`
- `save_confusion_matrix_csv(cm, out_csv, labels=None)`
- `save_confusion_matrix_png(cm, out_png, labels=None)` (requires matplotlib)

You can use these from a notebook or a small script after collecting
predictions and labels.

## 6. VS Code usage

The `.vscode/launch.json` file defines:

- **Quantum Train (main_quantum.py)** – runs training with typical defaults
  and `--augment --use-class-weight`.
- **Quantum Test (main_quantum.py)** – runs test with `--use-tta`.

Select a configuration in the **Run and Debug** panel and press **F5**.

## 7. Troubleshooting tips (accuracy)

If predictions tend to collapse to a single letter (e.g., always `P`):

- Check dataset balance via `utils/data_debug.py` – you likely have too few
  samples for some letters.
- Increase training epochs and use `--augment` for stronger data
  augmentation.
- Try `--use-resnet-backbone` for a more expressive encoder.
- Use `--use-class-weight` if some letters are rare.
- Run several seeds with `run_experiments.sh` and compare results in
  `experiments/results.csv`.
- Enable `--use-tta` in test mode for more stable predictions.
