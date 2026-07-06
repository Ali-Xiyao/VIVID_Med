"""Plot dataset embeddings using UMAP when available, otherwise PCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from case_study_modules_common import FINAL_DIR, write_csv_rows, write_md_sections


def read_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    labels: list[str] = []
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        matrix = np.asarray(data["embeddings"] if "embeddings" in data else data[next(iter(data.files))], dtype=np.float32)
        labels = [path.stem] * len(matrix)
        return matrix, labels
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(row.get("embedding", []))
                labels.append(str(row.get("dataset", path.stem)))
    return np.asarray(rows, dtype=np.float32), labels


def reduce_2d(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(matrix), "umap"
    except Exception:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return centered @ vh[:2].T, "pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="append", type=Path)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "dataset_embedding_projection.csv")
    parser.add_argument("--output-png", type=Path, default=FINAL_DIR / "dataset_embedding_projection.png")
    parser.add_argument("--report-md", type=Path, default=FINAL_DIR / "dataset_embedding_projection.md")
    parser.add_argument("--max-rows", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embedding:
        write_csv_rows(args.output_csv, [{"status": "missing_embeddings", "note": "No embedding files supplied."}], ["status", "note"])
        write_md_sections(args.report_md, "Dataset Embedding Projection", [("Boundary", "No embedding files were supplied; projection is a documented boundary, not evidence.")])
        return
    matrices = []
    labels: list[str] = []
    for path in args.embedding:
        matrix, row_labels = read_embeddings(path)
        matrices.append(matrix[: args.max_rows])
        labels.extend(row_labels[: args.max_rows])
    joined = np.vstack(matrices)
    coords, method = reduce_2d(joined)
    rows = [{"dataset": labels[i], "x": float(coords[i, 0]), "y": float(coords[i, 1]), "method": method} for i in range(len(coords))]
    write_csv_rows(args.output_csv, rows, ["dataset", "x", "y", "method"])
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        for label in sorted(set(labels)):
            points = coords[[i for i, item in enumerate(labels) if item == label]]
            plt.scatter(points[:, 0], points[:, 1], s=8, label=label, alpha=0.65)
        plt.legend()
        plt.title(f"Dataset embedding projection ({method})")
        args.output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_png, dpi=180)
    except Exception as exc:
        args.output_png = Path("")
        method = f"{method}; plot_failed={exc}"
    write_md_sections(args.report_md, "Dataset Embedding Projection", [("Projection", f"Wrote {len(rows)} projected rows using {method}. PNG: `{args.output_png}`")])


if __name__ == "__main__":
    main()
