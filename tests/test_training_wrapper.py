import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "training dataset").exists(), reason="training dataset not available")
def test_train_single_smoke():
    """Run a 1-epoch training via train_single.sh and ensure a checkpoint is produced.

    This is a smoke test and may be slow if the dataset is large; it is intended
    mainly for local verification.
    """

    script = ROOT / "experiments" / "train_single.sh"
    assert script.exists(), f"Missing script: {script}"

    exp_id = "pytest_smoke"
    cmd = ["bash", str(script), "--exp_id", exp_id, "--epochs", "1", "--batch", "1", "--img", "64"]
    subprocess.check_call(cmd, cwd=ROOT)

    ckpt = ROOT / "modells" / f"exp_{exp_id}.pt"
    assert ckpt.exists(), f"Expected checkpoint not found: {ckpt}"
