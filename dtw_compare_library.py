"""
DTW comparison using dtw-python on saved *_dtw_features.npy files.
-------------------------------------------------------------------

This script avoids MediaPipe dependencies by operating only on saved
feature matrices produced by asl_landmark_timeseries_dtw.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from dtw import dtw as dtw_align
except ImportError as exc:
    raise RuntimeError("dtw-python is required. Install it with: pip install -r requirements-dtw.txt") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def feature_label(path: Path) -> str:
    label = path.stem
    if label.endswith("_dtw_features"):
        label = label[: -len("_dtw_features")]
    return label


def list_feature_files(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Target must be a .npy file: {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.rglob("*_dtw_features.npy"))
        if not files:
            raise RuntimeError(f"No *_dtw_features.npy files found in: {path}")
        return files

    raise FileNotFoundError(f"Target path does not exist: {path}")


def plot_dtw_alignment(
    cost_matrix: np.ndarray,
    path: np.ndarray,
    output_path: Path,
    reference_label: str,
    target_label: str,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it with: pip install -r requirements-dtw.txt")

    plt.figure(figsize=(7, 6))
    plt.imshow(cost_matrix.T, origin="lower", aspect="auto", cmap="magma")
    plt.plot(path[:, 0], path[:, 1], color="cyan", linewidth=1.0)
    plt.xlabel(f"Reference frames ({reference_label})")
    plt.ylabel(f"Target frames ({target_label})")
    plt.title(f"DTW alignment: {target_label} vs {reference_label}")
    plt.colorbar(label="Accumulated cost")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def append_rows(output_path: Path, rows: List[Dict[str, Optional[float]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    fieldnames = [
        "reference",
        "target",
        "dtw_distance",
        "normalized_distance",
        "reference_frames",
        "target_frames",
    ]

    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved DTW feature matrices using dtw-python."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=str,
        help="Reference *_dtw_features.npy file.",
    )
    parser.add_argument(
        "--targets",
        required=True,
        type=str,
        help="Target *_dtw_features.npy file or a folder containing them.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dtw_comparisons_dtw_python.csv",
        help="CSV file to append comparison results to.",
    )
    parser.add_argument(
        "--plot-dtw",
        action="store_true",
        help="Save DTW accumulated-cost plots with alignment paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_path = Path(args.reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference path does not exist: {reference_path}")
    if reference_path.suffix.lower() != ".npy":
        raise ValueError("Reference must be a *_dtw_features.npy file.")

    reference = np.load(reference_path)
    reference_label = feature_label(reference_path)

    target_paths = list_feature_files(Path(args.targets))
    output_path = Path(args.output)

    rows: List[Dict[str, Optional[float]]] = []
    for target_path in target_paths:
        target = np.load(target_path)
        alignment = dtw_align(
            reference,
            target,
            dist_method="euclidean",
            keep_internals=args.plot_dtw,
            distance_only=not args.plot_dtw,
        )

        normalized = getattr(alignment, "normalizedDistance", None)
        rows.append(
            {
                "reference": reference_label,
                "target": feature_label(target_path),
                "dtw_distance": float(alignment.distance),
                "normalized_distance": None if normalized is None else float(normalized),
                "reference_frames": reference.shape[0],
                "target_frames": target.shape[0],
            }
        )

        if args.plot_dtw:
            path = np.column_stack((alignment.index1, alignment.index2))
            plot_dtw_alignment(
                alignment.costMatrix,
                path,
                output_path.parent
                / f"{feature_label(target_path)}_vs_{reference_label}_dtw.png",
                reference_label=reference_label,
                target_label=feature_label(target_path),
            )

    append_rows(output_path, rows)
    print(f"Saved DTW comparisons to: {output_path}")


if __name__ == "__main__":
    main()
