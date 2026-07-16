"""Build the VSL-CXR D9 full dataset candidate.

D9 is a composite data package: instruction rows for VSL-4class/SAMEQ/K/HNMB
training plus companion CEQ target rows and CCSH statement-pair rows. The
companion files keep module supervision explicit instead of flattening every
label type into a text-answer row.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "instruction_data" / "vsl_cxr"
FINAL_DIR = ROOT / "outputs" / "final_tables"


def root_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows is not None and len(rows) >= max_rows:
                    break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[index] for index in indices[:limit]]


def normalize_instruction_rows(rows: list[dict[str, Any]], component: str, source_dataset_id: str, split: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        flags = list(item.get("quality_flags") or [])
        for flag in ["d9_vsl_full", component, source_dataset_id]:
            if flag not in flags:
                flags.append(flag)
        original_id = item.get("instruction_id") or item.get("sample_id") or f"{component}_{index}"
        item["source_instruction_id"] = original_id
        item["instruction_id"] = f"d9_{split}_{component}_{index:07d}"
        item["vsl_dataset_id"] = "D9"
        item["d9_component"] = component
        item["source_dataset_id"] = source_dataset_id
        item["source"] = item.get("source") or item.get("source_mode") or f"d9_{component}_source"
        item["quality_flags"] = flags
        item.setdefault("validation_status", "auto_vsl_full_candidate")
        normalized.append(item)
    return normalized


def normalize_companion_rows(rows: list[dict[str, Any]], component: str, source_dataset_id: str, split: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        original_id = item.get("target_id") or item.get("statement_id") or item.get("source_instruction_id") or f"{component}_{index}"
        item["source_companion_id"] = original_id
        item["d9_companion_id"] = f"d9_{split}_{component}_{index:07d}"
        item["vsl_dataset_id"] = "D9"
        item["d9_component"] = component
        item["source_dataset_id"] = source_dataset_id
        normalized.append(item)
    return normalized


def split_companion(rows: list[dict[str, Any]], val_count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled = sample_rows(rows, len(rows), seed)
    val = sampled[: min(val_count, len(sampled))]
    train = sampled[min(val_count, len(sampled)) :]
    return train, val


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def summarize_jsonl(path: Path, split: str, component: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    answer_types = Counter(str(row.get("answer_type") or "") for row in rows)
    sufficiency = Counter(str(row.get("sufficiency_label") or "") for row in rows)
    findings = Counter(str(row.get("finding") or "") for row in rows)
    return {
        "split": split,
        "component": component,
        "artifact": repo_rel(path),
        "rows": len(rows),
        "answer_types": json.dumps(dict(answer_types.most_common(8)), ensure_ascii=False, sort_keys=True),
        "sufficiency_labels": json.dumps(dict(sufficiency.most_common(8)), ensure_ascii=False, sort_keys=True),
        "top_findings": json.dumps(dict(findings.most_common(8)), ensure_ascii=False),
        "status": "materialized" if rows else "empty",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--d6-train", type=Path, default=ROOT / "outputs/instruction_data/vsl_cxr/d6_vsl_4class_train.jsonl")
    parser.add_argument("--d6-val", type=Path, default=ROOT / "outputs/instruction_data/vsl_cxr/d6_vsl_4class_val.jsonl")
    parser.add_argument("--sameq-train", type=Path, default=ROOT / "outputs/instruction_data/cvcp_ccsh/cvcp_v1_sameq_full_train.jsonl")
    parser.add_argument("--sameq-val", type=Path, default=ROOT / "outputs/instruction_data/next_stage/sameq_shuf_val.jsonl")
    parser.add_argument("--sameq-k-train", type=Path, default=ROOT / "outputs/instruction_data/cvcp_ccsh/cvcp_v2_shuf_k4_train.jsonl")
    parser.add_argument("--sameq-k-val", type=Path, default=ROOT / "outputs/instruction_data/next_stage/shuf_k4_val.jsonl")
    parser.add_argument("--hnmb-train", type=Path, default=ROOT / "outputs/instruction_data/next_stage/selfhard_shuf_train.jsonl")
    parser.add_argument("--hnmb-val", type=Path, default=ROOT / "outputs/instruction_data/next_stage/mined_shuf_val.jsonl")
    parser.add_argument("--ceq-targets", type=Path, default=ROOT / "outputs/instruction_data/cvcp_ccsh/ceq_targets_train.jsonl")
    parser.add_argument("--ccsh-pairs", type=Path, default=ROOT / "outputs/instruction_data/cvcp_ccsh/ccsh_statements_train.jsonl")
    parser.add_argument("--d6-train-limit", type=int, default=6000)
    parser.add_argument("--sameq-train-limit", type=int, default=4000)
    parser.add_argument("--sameq-k-train-limit", type=int, default=4000)
    parser.add_argument("--hnmb-train-limit", type=int, default=4000)
    parser.add_argument("--val-limit-per-component", type=int, default=500)
    parser.add_argument("--companion-val-count", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = root_path(args.out_dir)

    train_components = [
        ("vsl4", "D6", root_path(args.d6_train), args.d6_train_limit),
        ("sameq", "D2", root_path(args.sameq_train), args.sameq_train_limit),
        ("sameq_k4", "D4", root_path(args.sameq_k_train), args.sameq_k_train_limit),
        ("hnmb_selfhard", "D5", root_path(args.hnmb_train), args.hnmb_train_limit),
    ]
    val_components = [
        ("vsl4", "D6", root_path(args.d6_val), args.val_limit_per_component),
        ("sameq", "D2", root_path(args.sameq_val), args.val_limit_per_component),
        ("sameq_k4", "D4", root_path(args.sameq_k_val), args.val_limit_per_component),
        ("hnmb_mined", "D5", root_path(args.hnmb_val), args.val_limit_per_component),
    ]

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for offset, (component, source_id, path, limit) in enumerate(train_components):
        rows = sample_rows(read_jsonl(path), limit, args.seed + offset)
        train_rows.extend(normalize_instruction_rows(rows, component, source_id, "train"))
    for offset, (component, source_id, path, limit) in enumerate(val_components, start=100):
        rows = sample_rows(read_jsonl(path), limit, args.seed + offset)
        val_rows.extend(normalize_instruction_rows(rows, component, source_id, "val"))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    for index, row in enumerate(train_rows):
        row["instruction_id"] = f"d9_train_mixed_{index:07d}"
    for index, row in enumerate(val_rows):
        row["instruction_id"] = f"d9_val_mixed_{index:07d}"

    train_path = out_dir / "d9_vsl_full_train.jsonl"
    val_path = out_dir / "d9_vsl_full_val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    ceq_train, ceq_val = split_companion(read_jsonl(root_path(args.ceq_targets)), args.companion_val_count, args.seed + 200)
    ccsh_train, ccsh_val = split_companion(read_jsonl(root_path(args.ccsh_pairs)), args.companion_val_count, args.seed + 300)
    companion_paths = {
        "ceq_train": out_dir / "d9_ceq_targets_train.jsonl",
        "ceq_val": out_dir / "d9_ceq_targets_val.jsonl",
        "ccsh_train": out_dir / "d9_ccsh_pairs_train.jsonl",
        "ccsh_val": out_dir / "d9_ccsh_pairs_val.jsonl",
    }
    write_jsonl(companion_paths["ceq_train"], normalize_companion_rows(ceq_train, "ceq_targets", "D7", "train"))
    write_jsonl(companion_paths["ceq_val"], normalize_companion_rows(ceq_val, "ceq_targets", "D7", "val"))
    write_jsonl(companion_paths["ccsh_train"], normalize_companion_rows(ccsh_train, "ccsh_pairs", "D8", "train"))
    write_jsonl(companion_paths["ccsh_val"], normalize_companion_rows(ccsh_val, "ccsh_pairs", "D8", "val"))

    manifest_rows = [
        summarize_jsonl(train_path, "train", "instruction_mixture"),
        summarize_jsonl(val_path, "val", "instruction_mixture"),
    ]
    for key, path in companion_paths.items():
        split = "train" if key.endswith("train") else "val"
        component = key.rsplit("_", 1)[0]
        manifest_rows.append(
            {
                "split": split,
                "component": component,
                "artifact": repo_rel(path),
                "rows": count_jsonl(path),
                "answer_types": "",
                "sufficiency_labels": "",
                "top_findings": "",
                "status": "materialized" if count_jsonl(path) else "empty",
            }
        )
    columns = ["split", "component", "artifact", "rows", "answer_types", "sufficiency_labels", "top_findings", "status"]
    manifest_csv = FINAL_DIR / "vsl_cxr_d9_full_dataset_manifest.csv"
    manifest_md = FINAL_DIR / "vsl_cxr_d9_full_dataset_manifest.md"
    write_csv(manifest_csv, manifest_rows, columns)
    manifest_md.write_text(
        "# VSL-CXR D9 Full Dataset Manifest\n\n"
        "D9 is a composite package: mixed instruction rows plus CEQ/CCSH companion supervision files. "
        "This is a candidate package and still requires exact formal-run validation.\n\n"
        + md_table(manifest_rows, columns)
        + "\n",
        encoding="utf-8",
    )

    print(f"train_rows={len(train_rows)}")
    print(f"val_rows={len(val_rows)}")
    for name, path in {"train": train_path, "val": val_path, **companion_paths}.items():
        print(f"{name}={repo_rel(path)} rows={count_jsonl(path)}")
    print(f"manifest={repo_rel(manifest_csv)}")


if __name__ == "__main__":
    main()
