"""Compute simple domain-shift statistics and MMD from embedding files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from case_study_modules_common import FINAL_DIR, write_csv_rows, write_md_table


def read_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "embeddings" in data:
            return np.asarray(data["embeddings"], dtype=np.float32)
        first = next(iter(data.files))
        return np.asarray(data[first], dtype=np.float32)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "embedding" in row:
                rows.append(row["embedding"])
    return np.asarray(rows, dtype=np.float32)


def pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    distances = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(distances, 0.0)


def rbf_kernel_mean(x: np.ndarray, y: np.ndarray, gamma: float, chunk_size: int = 512) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    total = 0.0
    count = 0
    for start in range(0, len(x), chunk_size):
        distances = pairwise_squared_distances(x[start : start + chunk_size], y)
        values = np.exp(-gamma * distances)
        total += float(values.sum())
        count += int(values.size)
    return total / max(count, 1)


def median_gamma(x: np.ndarray, y: np.ndarray, sample_rows: int = 512) -> float:
    joined = np.vstack([x[: min(len(x), sample_rows)], y[: min(len(y), sample_rows)]])
    dist = pairwise_squared_distances(joined, joined)
    positive = dist[dist > 0]
    median = np.median(positive) if positive.size else 1.0
    return 1.0 / max(float(median), 1e-6)


def squared_mmd(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if gamma is None:
        gamma = median_gamma(x, y)
    kxx = rbf_kernel_mean(x, x, gamma)
    kyy = rbf_kernel_mean(y, y, gamma)
    kxy = rbf_kernel_mean(x, y, gamma)
    return float(kxx + kyy - 2 * kxy)


def stats(name: str, matrix: np.ndarray) -> dict[str, Any]:
    return {
        "domain": name,
        "n": len(matrix),
        "dim": matrix.shape[1] if matrix.ndim == 2 else 0,
        "embedding_norm_mean": float(np.linalg.norm(matrix, axis=1).mean()) if matrix.size else "",
        "embedding_norm_std": float(np.linalg.norm(matrix, axis=1).std()) if matrix.size else "",
        "feature_variance_mean": float(matrix.var(axis=0).mean()) if matrix.size else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--source-name", default="source")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "domain_shift_mmd.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "domain_shift_mmd.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]]
    note = ""
    if not args.source or not args.target or not args.source.exists() or not args.target.exists():
        rows = [
            {
                "domain": "boundary",
                "n": "",
                "dim": "",
                "embedding_norm_mean": "",
                "embedding_norm_std": "",
                "feature_variance_mean": "",
                "mmd_vs_other": "",
                "note": "Embedding files were not supplied; generate/export embeddings before claiming domain-shift MMD.",
            }
        ]
    else:
        x = read_embeddings(args.source)[: args.max_rows]
        y = read_embeddings(args.target)[: args.max_rows]
        mmd = squared_mmd(x, y)
        rows = [stats(args.source_name, x), stats(args.target_name, y)]
        rows[0]["mmd_vs_other"] = mmd
        rows[1]["mmd_vs_other"] = mmd
        note = "MMD uses an RBF kernel with median-distance gamma on the sampled embeddings."
    columns = ["domain", "n", "dim", "embedding_norm_mean", "embedding_norm_std", "feature_variance_mean", "mmd_vs_other", "note"]
    write_csv_rows(args.output_csv, rows, columns)
    write_md_table(args.output_md, "Domain Shift MMD", rows, columns, note)


if __name__ == "__main__":
    main()
