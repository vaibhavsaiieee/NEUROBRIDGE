from pathlib import Path
from typing import Dict

LABELS = [chr(ord("A") + i) for i in range(26)]


def _infer_label(path: Path):
    parent = path.parent.name.upper()
    if parent in LABELS:
        return parent
    stem = path.stem.upper()
    if stem and stem[0] in LABELS:
        return stem[0]
    return None


def print_dataset_report(root: Path) -> None:
    """Print counts per letter and highlight missing/low-count classes.

    Usage:
        python -m utils.data_debug "training dataset"
    """

    root = Path(root)
    if not root.exists():
        print("[data_debug] Root not found:", root)
        return

    counts: Dict[str, int] = {c: 0 for c in LABELS}
    total = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        lbl = _infer_label(p)
        if lbl is None:
            print("[data_debug] Skipping (no label inferred):", p)
            continue
        counts[lbl] += 1
        total += 1

    print(f"[data_debug] Root: {root}  total_images={total}")
    print("[data_debug] Counts per label:")
    for c in LABELS:
        print(f"  {c}: {counts[c]}")

    missing = [c for c in LABELS if counts[c] == 0]
    low = [c for c in LABELS if 0 < counts[c] < 5]
    if missing:
        print("[data_debug] Missing labels:", ", ".join(missing))
    if low:
        print("[data_debug] Low-count labels (<5):", ", ".join(low))


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path("training dataset")
    print_dataset_report(root)
