"""Mine hard negative image pairs from embedding JSONL/NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl_embeddings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "embedding" not in row:
                continue
            row["embedding"] = np.asarray(row["embedding"], dtype=np.float32)
            rows.append(row)
    return rows


def read_npz_embeddings(path: Path, metadata_jsonl: Path | None) -> list[dict[str, Any]]:
    data = np.load(path)
    embeddings = data["embeddings"]
    if metadata_jsonl is not None:
        metadata = []
        with metadata_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    metadata.append(json.loads(line))
    else:
        metadata = [{"sample_id": str(index)} for index in range(len(embeddings))]
    rows = []
    for index, embedding in enumerate(embeddings):
        row = dict(metadata[index] if index < len(metadata) else {"sample_id": str(index)})
        row["embedding"] = np.asarray(embedding, dtype=np.float32)
        rows.append(row)
    return rows


def read_embeddings(path: Path, metadata_jsonl: Path | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".npz":
        return read_npz_embeddings(path, metadata_jsonl)
    return read_jsonl_embeddings(path)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return matrix / denom


def compatible_negative(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if str(anchor.get("sample_id")) == str(candidate.get("sample_id")):
        return False
    finding_a = str(anchor.get("finding") or "")
    finding_b = str(candidate.get("finding") or "")
    if finding_a and finding_b and finding_a != finding_b:
        return False
    state_a = str(anchor.get("state") or "")
    state_b = str(candidate.get("state") or "")
    answer_a = str(anchor.get("answer") or anchor.get("answer_short") or "")
    answer_b = str(candidate.get("answer") or candidate.get("answer_short") or "")
    if state_a and state_b and state_a != state_b:
        return True
    if answer_a and answer_b and answer_a != answer_b:
        return True
    location_a = str(anchor.get("location") or "")
    location_b = str(candidate.get("location") or "")
    return bool(location_a and location_b and location_a != location_b)


def mine(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    matrix = normalize_matrix(np.stack([row["embedding"] for row in rows]))
    scores = matrix @ matrix.T
    mined: list[dict[str, Any]] = []
    for index, anchor in enumerate(rows):
        order = np.argsort(-scores[index])
        found = 0
        for candidate_index in order:
            if int(candidate_index) == index:
                continue
            candidate = rows[int(candidate_index)]
            if not compatible_negative(anchor, candidate):
                continue
            mined.append(
                {
                    "instruction_id": anchor.get("instruction_id"),
                    "sample_id": anchor.get("sample_id"),
                    "image_path": anchor.get("image_path"),
                    "finding": anchor.get("finding"),
                    "state": anchor.get("state"),
                    "answer": anchor.get("answer") or anchor.get("answer_short"),
                    "answer_type": anchor.get("answer_type"),
                    "laterality": anchor.get("laterality"),
                    "location": anchor.get("location"),
                    "visual_dependency": anchor.get("visual_dependency"),
                    "negative_instruction_id": candidate.get("instruction_id"),
                    "negative_sample_id": candidate.get("sample_id"),
                    "negative_image_path": candidate.get("image_path"),
                    "negative_state": candidate.get("state"),
                    "negative_answer": candidate.get("answer") or candidate.get("answer_short"),
                    "negative_laterality": candidate.get("laterality"),
                    "negative_location": candidate.get("location"),
                    "cosine_similarity": float(scores[index, int(candidate_index)]),
                    "rank": found + 1,
                }
            )
            found += 1
            if found >= top_k:
                break
    return mined


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_embeddings(args.embeddings, args.metadata_jsonl)
    mined = mine(rows, args.top_k)
    write_jsonl(args.output_jsonl, mined)
    if args.output_csv:
        write_csv(args.output_csv, mined)
    print(json.dumps({"embeddings": str(args.embeddings), "rows": len(rows), "mined_pairs": len(mined)}, indent=2))


if __name__ == "__main__":
    main()
