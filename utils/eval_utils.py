from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

LABELS = [chr(ord("A") + i) for i in range(26)]


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 26) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> List[dict]:
    num_classes = cm.shape[0]
    metrics = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append({
            "class": LABELS[c] if c < len(LABELS) else str(c),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return metrics


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy for numpy arrays.

    logits: (N, C), targets: (N,)
    """

    topk = np.argsort(-logits, axis=1)[:, :k]
    correct = 0
    for i, t in enumerate(targets):
        if t in topk[i]:
            correct += 1
    return correct / max(1, len(targets))


def save_confusion_matrix_csv(cm: np.ndarray, out_csv: Path, labels: Iterable[str] | None = None) -> None:
    out_csv = Path(out_csv)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    labels = list(labels)
    with out_csv.open("w") as f:
        f.write(",".join(["label"] + labels) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([labels[i]] + [str(int(v)) for v in row]) + "\n")


def save_confusion_matrix_png(cm: np.ndarray, out_png: Path, labels: Iterable[str] | None = None) -> None:
    """Save a confusion matrix heatmap to PNG.

    Requires matplotlib; if unavailable, this function prints a warning instead.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional
        print("[eval_utils] matplotlib not available; cannot save confusion matrix PNG.")
        return

    out_png = Path(out_png)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    labels = list(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


__all__ = [
    "confusion_matrix",
    "per_class_metrics",
    "topk_accuracy",
    "save_confusion_matrix_csv",
    "save_confusion_matrix_png",
]
