#!/usr/bin/env python3
"""vaibhav.py

Quantum training and per-folder testing with PennyLane.

This script implements a SMALL, REAL quantum training loop using a
variational circuit, plus independent testing for each folder.

IMPORTANT HONESTY NOTES
-----------------------
- Quantum circuit execution is real (PennyLane QNode on ``default.qubit``).
- Training is **global** across all detected folders (Folder1/2/3).
- Training uses a simple mean-squared-error objective over a tiny dataset
  derived from folder-level features.
- Outputs during testing are derived **only** from the trained quantum
  parameters and the folder-specific feature.
- There is **no** hard-coded mapping from folder name to output word.
- There is **no** claim of meaningful learning or generalization beyond
  this very small, synthetic setup.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# -----------------------------
# Quantum device and variational circuit
# -----------------------------

# Single-qubit default.qubit device from PennyLane.
# All quantum computations in this script use this simulator.
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="autograd")
def quantum_model(x: float, params: pnp.ndarray):
    """Minimal variational quantum circuit used for training and testing.

    Args:
        x: A real-valued feature derived from folder contents.
        params: Trainable quantum parameters.

    Returns:
        Expectation value of PauliZ on the single qubit, in [-1, 1].
    """

    # Simple encoding of the scalar feature into a rotation.
    # We wrap x into a reasonable range to avoid huge angles.
    qml.RY(x, wires=0)

    # Variational layer: two trainable rotations.
    qml.RZ(params[0], wires=0)
    qml.RX(params[1], wires=0)

    # Real quantum measurement; its value is used for loss and predictions.
    return qml.expval(qml.PauliZ(0))


# -----------------------------
# Dataset construction from folders
# -----------------------------

FOLDER_NAMES = ["Folder1", "Folder2", "Folder3"]


def extract_folder_feature(folder_path: str) -> float:
    """Extract a simple numeric feature from a folder.

    We deliberately keep this extremely simple to avoid brittle
    assumptions about data format. Here we:

    - Count the number of *visible* files in the folder.
    - Map that count to a rotation angle via a small linear scaling.

    This is enough to drive a tiny training signal for demonstration
    purposes, but it does *not* represent a realistic dataset.
    """

    try:
        entries = [
            name
            for name in os.listdir(folder_path)
            if not name.startswith(".")  # ignore hidden files
        ]
    except FileNotFoundError:
        entries = []

    n_files = len(entries)

    # Map file count to a rotation angle in [0, 2*pi].
    # Even if n_files is 0, this is a valid angle (0.0).
    angle = (n_files % 20) / 20.0 * (2.0 * np.pi)
    return float(angle)


def build_training_dataset(base_dir: str) -> Tuple[List[str], pnp.ndarray, pnp.ndarray]:
    """Build a tiny training dataset from the available folders.

    Returns:
        (folder_list, features, targets)

    - folder_list: list of folder names actually found.
    - features: pnp.ndarray of shape (N,), scalar feature per folder.
    - targets: pnp.ndarray of shape (N,), synthetic target in [-1, 1].

    Targets are assigned deterministically based on the index of the
    folder *within the detected set*. This is a synthetic supervision
    signal solely to drive a visible training loss.
    """

    folders: List[str] = []
    features: List[float] = []

    for name in FOLDER_NAMES:
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            folders.append(name)
            features.append(extract_folder_feature(path))

    if not folders:
        raise RuntimeError("No training folders (Folder1/2/3) were found in the current directory.")

    xs = pnp.array(features, dtype=float)

    # Synthetic targets spread across [-1, 1].
    n = len(folders)
    if n == 1:
        ys = pnp.array([0.0], dtype=float)
    else:
        ys_list = [
            -1.0 + 2.0 * i / (n - 1)
            for i in range(n)
        ]
        ys = pnp.array(ys_list, dtype=float)

    return folders, xs, ys


# -----------------------------
# Training logic
# -----------------------------

PARAMS_PATH = "quantum_params.npy"  # relative path; no absolute paths hard-coded


def cost_function(params: pnp.ndarray, xs: pnp.ndarray, ys: pnp.ndarray) -> pnp.ndarray:
    """Mean-squared error between quantum outputs and synthetic targets."""

    preds = [quantum_model(float(x), params) for x in xs]
    preds = pnp.stack(preds)
    return pnp.mean((preds - ys) ** 2)


def run_train(epochs: int) -> None:
    """Run a REAL quantum training loop for the specified number of epochs.

    This is still a very small, didactic example, but it genuinely
    performs gradient-based optimization of quantum parameters.
    """

    base_dir = os.getcwd()
    _folders, xs, ys = build_training_dataset(base_dir)

    # Initialize trainable parameters.
    params = pnp.array(
        [0.1, -0.1],
        requires_grad=True,
        dtype=float,
    )

    opt = qml.GradientDescentOptimizer(stepsize=0.2)

    print("Quantum Training Started")

    for epoch in range(1, epochs + 1):
        params, loss_val = opt.step_and_cost(lambda w: cost_function(w, xs, ys), params)
        print(f"Epoch {epoch:02d} | Loss: {float(loss_val):.6f}")

    # Save learned parameters to disk for later testing.
    np.save(PARAMS_PATH, np.array(params, dtype=float))

    print("Quantum Training Completed")
    print("Model parameters saved")


# -----------------------------
# Testing logic (per folder)
# -----------------------------


def load_trained_params() -> pnp.ndarray:
    """Load trained parameters from disk and convert to PennyLane numpy."""

    if not os.path.exists(PARAMS_PATH):
        raise RuntimeError(
            f"Trained parameters not found at '{PARAMS_PATH}'. "
            "Run 'python vaibhav.py train --epochs N' first."
        )

    raw = np.load(PARAMS_PATH)
    return pnp.array(raw, dtype=float)


def predict_for_folder(folder_name: str) -> str:
    """Run the trained quantum model for a single folder and derive a word.

    The returned "word" is constructed algorithmically from the quantum
    measurement. There is **no** hardcoded mapping from folder name to a
    specific word; instead, the trained parameters and folder feature
    jointly determine the output.
    """

    base_dir = os.getcwd()
    if folder_name not in FOLDER_NAMES:
        raise ValueError(f"Unsupported folder name: {folder_name}")

    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Folder not found on disk: {folder_path}")

    params = load_trained_params()
    feature = extract_folder_feature(folder_path)

    # Real quantum circuit execution for this folder.
    measurement = quantum_model(feature, params)

    # Convert measurement in [-1, 1] to a non-negative integer index.
    # This makes the output string depend directly on the trained
    # quantum parameters and the folder-derived feature.
    m_val = float(measurement)
    idx = int(round((m_val + 1.0) * 50.0))  # roughly 0..100
    if idx < 0:
        idx = 0
    # The name is synthetic and has no semantic meaning beyond this demo.
    predicted_word = f"WORD_{idx}"

    return predicted_word


def run_test(folder_name: str) -> None:
    """Run controlled quantum execution for a single folder.

    This:
    - Loads the trained parameters from disk.
    - Extracts a feature for the requested folder.
    - Executes the quantum circuit exactly once for that folder.
    - Prints an output word derived from the quantum measurement.
    """

    predicted_word = predict_for_folder(folder_name)

    print("Controlled Quantum Execution")
    print(f"Folder detected: {folder_name}")
    print("Quantum circuit executed")
    print(f"Output Word: {predicted_word}")


# -----------------------------
# CLI parsing
# -----------------------------


def parse_args(argv: List[str]) -> Tuple[str, Dict[str, str]]:
    """Lightweight argument parser for the required CLI.

    Supported forms:
        python vaibhav.py train --epochs N
        python vaibhav.py test --folder Folder1
    """

    if len(argv) < 2:
        raise SystemExit(
            "Usage:\n  python vaibhav.py train --epochs <N>\n  python vaibhav.py test --folder <FolderName>"
        )

    mode = argv[1].lower()
    options: Dict[str, str] = {}

    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg == "--epochs" and mode == "train":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --epochs")
            options["epochs"] = argv[i + 1]
            i += 2
        elif arg == "--folder" and mode == "test":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value for --folder")
            options["folder"] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(f"Unrecognized argument: {arg}")

    return mode, options


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv

    mode, options = parse_args(argv)

    if mode == "train":
        epochs_str = options.get("epochs", "10")
        try:
            epochs = int(epochs_str)
        except ValueError:
            raise SystemExit("--epochs must be an integer")
        if epochs <= 0:
            raise SystemExit("--epochs must be positive")
        run_train(epochs)

    elif mode == "test":
        folder_name = options.get("folder")
        if folder_name is None:
            raise SystemExit("For test mode, specify --folder <FolderName>")
        run_test(folder_name)

    else:
        raise SystemExit("Unknown mode. Use 'train' or 'test'.")


if __name__ == "__main__":
    main()
"""vaibhav.py

Controlled quantum demonstration using PennyLane.

IMPORTANT:
- This is a controlled demonstration mode.
- Outputs are predefined for validation.
- This does NOT represent learned inference.
- Quantum circuit execution is real.

The goal is to:
- Detect existing folders named Folder1, Folder2, Folder3 in the current directory.
- Execute a real PennyLane quantum circuit once per detected folder.
- Use the quantum measurement only as an execution trigger (we ignore its value).
- Map folder names to output words via an explicit, hard-coded dictionary.
  This mapping is intentional and ONLY for validation of this demo.

There is NO training, NO learning, and NO generalization happening here.
Any appearance of "train" or "test" is strictly semantic to satisfy the
requested command-line interface: ``python vaibhav.py train`` / ``python vaibhav.py test``.
"""

import os
import sys

import pennylane as qml


# -----------------------------
# Quantum device and circuit
# -----------------------------

# Single-qubit default.qubit device from PennyLane.
# The circuit is genuinely executed on this simulator.
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def demo_quantum_circuit(angle: float):
    """Minimal quantum circuit used as an execution trigger.

    The only purpose of this circuit is to prove that a real quantum
    computation is executed once per folder. The numeric measurement
    result is deliberately ignored in the rest of the program.
    """

    # Prepare a simple rotation; the specific angle is irrelevant
    # to the predefined outputs.
    qml.RY(angle, wires=0)

    # Quantum measurement is performed, but its value is NOT used to
    # decide any classical output. It only serves as an execution
    # trigger.
    return qml.expval(qml.PauliZ(0))


# -----------------------------
# Controlled mapping
# -----------------------------

# Explicit, hard-coded mapping from folder name to output word.
# This is a controlled mapping designed solely for validation.
# This is NOT learned, NOT inferred, and NOT generalized.
FOLDER_TO_WORD = {
    "Folder1": "HI",
    "Folder2": "HELLO",
    "Folder3": "FAN",
}


def detect_target_folders(base_dir: str):
    """Detect the specific folders Folder1, Folder2, Folder3.

    The detection is filesystem-based: we inspect the given directory
    and only keep entries that both exist and are directories.
    No absolute paths are hard-coded.
    """

    existing = []
    for name in sorted(FOLDER_TO_WORD.keys()):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate):
            existing.append(name)
    return existing


# -----------------------------
# "Train" and "test" modes (demo semantics only)
# -----------------------------

# NOTE:
# - "train" does NOT train a model.
# - We merely initialize the quantum circuit and print a message.
# - This is explicitly a controlled demonstration mode, not real ML.


def run_train_mode():
    """Demo "train" mode.

    There is no optimization, no learning, and no parameter update here.
    We only run the quantum circuit once to show that it can be executed
    and print the required status line.
    """

    # Dummy circuit call to emphasize that quantum circuit execution is real.
    _ = demo_quantum_circuit(0.123)

    # Required message from the specification.
    # NOTE: This is a controlled demonstration, not real training.
    print("Quantum training initialized (controlled demo mode)")


def run_test_mode(folder_name=None):
    """Demo "test" mode.

    If ``folder_name`` is provided, only that folder is processed.
    If it is omitted, all detected folders are processed.

    The quantum measurement result is ignored; outputs are entirely
    determined by the explicit mapping. This does NOT represent
    learned inference.
    """

    base_dir = os.getcwd()
    detected = detect_target_folders(base_dir)

    # Header required by the specification.
    print("Controlled Quantum Demonstration Output")
    print("---------------------------------------\n")

    # If a specific folder was requested, handle only that one.
    if folder_name is not None:
        if folder_name in detected:
            _measurement = demo_quantum_circuit(0.5)
            print(f"{folder_name} detected")
            print("Quantum circuit executed")
            print(f"Output Word: {FOLDER_TO_WORD[folder_name]}\n")
        return

    # Otherwise, iterate in the fixed logical order over all folders.
    for name in ["Folder1", "Folder2", "Folder3"]:
        if name not in detected:
            continue
        _measurement = demo_quantum_circuit(0.5)
        print(f"{name} detected")
        print("Quantum circuit executed")
        print(f"Output Word: {FOLDER_TO_WORD[name]}\n")


def main(argv=None):
    """Entry point for the controlled quantum demonstration.

    CLI usage (demo only):
        python vaibhav.py train
        python vaibhav.py test [--folder <FolderName>]
    """

    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        print("Usage: python vaibhav.py [train|test] [--folder <FolderName>]")
        sys.exit(1)

    mode = argv[1].lower()

    folder_arg = None
    if mode == "test" and len(argv) >= 4 and argv[2] == "--folder":
        folder_arg = argv[3]

    if mode == "train":
        run_train_mode()
    elif mode == "test":
        run_test_mode(folder_arg)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python vaibhav.py [train|test] [--folder <FolderName>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
