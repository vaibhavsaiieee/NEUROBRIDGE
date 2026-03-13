# EEG A–Z Classifier (Classical-first)

This project now provides a **classical-first** CNN image classifier for EEG
waveform letters A–Z, with the quantum pipeline kept as optional legacy.

The primary entrypoint is `main_classical.py`. The older quantum experiment
script `main_quantum.py` is still available but not required for normal use.

## Environment

- Recommended: Python 3.9+ and a virtual environment at `./venv/`.
- Target OS: macOS (CPU) with optional GPU (CUDA or Apple Metal/MPS).

## Setup

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Classical Training

From the project root:

```bash
./venv/bin/python main_classical.py train \
  --epochs 20 \
  --batch 8 \
  --img 224 \
  --lr 3e-4
```

Requirements:
- Training images are under `training dataset/`.
- A validation split is created automatically in a stratified way, with a tiny
  fallback validation set if needed.

On success, the script will:
- Print diagnostics: Torch version and device.
- Save a checkpoint to `modells/best_classical.pt` after the first epoch and
  whenever validation accuracy improves.

## Classical Test / Prediction

After classical training has produced `modells/best_classical.pt`, run:

```bash
./venv/bin/python main_classical.py test \
  --test-root "teseting Dataset" \
  --img 224
```

If `teseting Dataset` does not exist but `testing dataset` does, the script will
fallback and log a diagnostic.

Each image line looks like:

```text
[Predict] path/to/img.png -> <LABEL> (<confidence>%)
Final Predicted Word: ABC...
```

## Quantum pipeline (legacy)

The previous quantum-first script remains available:

```bash
./venv/bin/python main_quantum.py train ...
./venv/bin/python main_quantum.py test --img 224
```

This path depends on PennyLane and autoray and is intended for experiments
rather than day-to-day use.

## VS Code integration

1. Open this folder in VS Code.
2. Ensure the Python extension is installed.
3. The file `.vscode/launch.json` defines four configurations:
   - **Classical Train (main_classical.py)**
   - **Classical Test (main_classical.py)**
   - **Quantum Train (main_quantum.py)**
   - **Quantum Test (main_quantum.py)**
4. The configurations use the interpreter at `./venv/bin/python`.

You can start debugging by selecting a configuration from the **Run and Debug**
panel and pressing **F5**.

### Helper scripts

The script `scripts/run_vscode_debug.sh` is provided for quick runs:

```bash
chmod +x scripts/run_vscode_debug.sh
./scripts/run_vscode_debug.sh classical-train   # classical train, logs to logs/run.log
./scripts/run_vscode_debug.sh classical-test    # classical test/predict, logs to logs/run.log
./scripts/run_vscode_debug.sh train             # quantum train (legacy)
./scripts/run_vscode_debug.sh test              # quantum test (legacy)
```

Additional helpers:

```bash
chmod +x scripts/train.sh scripts/predict.sh scripts/predict_all.sh
./scripts/train.sh          # classical train -> logs/train.log
./scripts/predict.sh        # classical test (configurable test root) -> logs/predict.log
./scripts/predict_all.sh    # test on "teseting Dataset" -> logs/predict.log
```

## Smoke tests & CI hook

Classical smoke test:

```bash
./venv/bin/python -m pytest tests/smoke_test_classical.py
```

Quantum smoke test (legacy):

```bash
./venv/bin/python -m pytest tests/smoke_test_quantum.py
```

The GitHub Actions workflow `.github/workflows/ci.yml` runs both smoke tests.

## Troubleshooting

### Missing dataset

- Ensure `training dataset/` exists for training and validation.
- Ensure `teseting Dataset/` (or `testing dataset/`) exists for prediction.
- Use `scripts/prepare_data.sh --organize` to arrange images into A/..Z/
  subfolders if needed.

### Missing checkpoint

- `main_classical.py test` will exit non-zero if the checkpoint is missing.
- Always run a short training first, e.g.:

  ```bash
  ./venv/bin/python main_classical.py train --epochs 1 --batch 1 --img 224 --lr 3e-4
  ```

### Python / venv issues

- Ensure you are using the workspace venv:

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- In VS Code, pick the `./venv/bin/python` interpreter.

### Torch install on macOS

If installing `torch==2.0.1` fails on macOS:

- Make sure you are using a compatible Python version (e.g., 3.9 or 3.10).
- Use the official PyTorch wheels for your platform by following the install
  command from the PyTorch website for macOS (CPU or Metal/MPS backend).

### PennyLane (legacy quantum path)

If you want to use `main_quantum.py`, ensure PennyLane and autoray are
installed via `requirements.txt` and refer to comments in that file for further
troubleshooting.

This project provides a **quantum-first** image classifier for EEG waveform
letters A–Z. Images are compressed by a tiny classical encoder (MLP), passed
through a PennyLane-based quantum layer (via `TorchLayer`), and finished by a
small classical head.

The main entrypoint for the quantum-first pipeline is `main_quantum.py`.

## Environment

- Recommended: Python 3.9+ and a virtual environment at `./venv/`.
- Target OS: macOS (CPU) with optional GPU (CUDA or Apple Metal/MPS).

## Setup

```bash
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Training

From the project root:

```bash
./venv/bin/python main_quantum.py train \
  --epochs 10 \
  --batch 4 \
  --img 224 \
  --lr 3e-4 \
  --n_qubits 4 \
  --n_layers 2 \
  --latent_dim 8 \
  --q_output 16
```

Requirements:
- Training images are under `training dataset/`.
- A validation split is created automatically in a stratified way.

On success, the script will:
- Print diagnostics: Torch, PennyLane, and `autoray` versions, device, and
  whether **batched TorchLayer** or **fallback per-sample** mode is used.
- Save a checkpoint to `modells/quantum_best.pt`.

## Testing / Prediction

After training, run:

```bash
./venv/bin/python main_quantum.py test --img 224
```

Requirements:
- Test images are under `teseting Dataset/` (note the original spelling).

The script will:
- Load `modells/quantum_best.pt`.
- Print per-image predictions.
- Print the final concatenated predicted word.

## VS Code integration

1. Open this folder in VS Code.
2. Ensure the Python extension is installed.
3. The file `.vscode/launch.json` defines two configurations:
   - **Quantum Train (main_quantum.py)**
   - **Quantum Test (main_quantum.py)**
4. The configurations use the interpreter at `./venv/bin/python`.

You can start debugging by selecting a configuration from the **Run and Debug**
panel and pressing **F5**.

### Helper script

The script `scripts/run_vscode_debug.sh` is provided for quick runs:

```bash
chmod +x scripts/run_vscode_debug.sh
./scripts/run_vscode_debug.sh train   # runs training and logs to logs/run.log
./scripts/run_vscode_debug.sh test    # runs test and logs to logs/run.log
```

Environment variables like `EPOCHS`, `BATCH`, `IMG`, `N_QUBITS`, etc. can be
used to override defaults.

## Smoke test & CI hook

A simple smoke test is provided in `tests/smoke_test_quantum.py`:

```bash
./venv/bin/python -m pytest tests/smoke_test_quantum.py
```

This test imports `QuantumOnlyModel`, runs a single forward pass with random
input, and asserts that the output shape is `(1, 26)`.

In CI you can add a job that:

1. Creates a venv.
2. Installs `requirements.txt`.
3. Runs `pytest tests/smoke_test_quantum.py`.

## Troubleshooting

### PennyLane import errors

If `main_quantum.py` prints that PennyLane is not importable:

- Ensure the venv is active: `source ./venv/bin/activate`.
- Reinstall with pinned versions:

  ```bash
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
  ```

When PennyLane is unavailable, `SafePennylaneLayer` automatically falls back
to a purely classical linear layer so that training can still proceed.

### Batched TorchLayer / autoray issues

The quantum layer uses `qml.qnn.TorchLayer` with `diff_method="backprop"`.
On the first forward pass it tries a **batched** call. If that fails due to
interface or `autoray` issues, it prints a message and permanently switches to
**per-sample detached** calls (no crash).

You will see one of the following messages in the logs:
- `[Quantum] Using batched TorchLayer with full autodiff.`
- `[Quantum] Batched TorchLayer failed, switching to per-sample fallback.`

### Torch install on macOS

If installing `torch==2.0.1` fails on macOS:

- Make sure you are using a compatible Python version (e.g., 3.9 or 3.10).
- Use the official PyTorch wheels for your platform by following the install
  command from the PyTorch website for macOS (CPU or Metal/MPS backend).

### Dataset layout

The code expects:
- Training: `training dataset/` containing subfolders or files that encode the
  label (folder name `A`–`Z` or filename starting with that letter).
- Testing: `teseting Dataset/` with images to classify.

If labels cannot be inferred, those files are silently skipped during training.

## Repro commands

```bash
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Train (example)
./venv/bin/python main_quantum.py train --epochs 10 --batch 4 --img 224 --n_qubits 4 --n_layers 2 --latent_dim 8 --q_output 16

# Test
./venv/bin/python main_quantum.py test --img 224
```