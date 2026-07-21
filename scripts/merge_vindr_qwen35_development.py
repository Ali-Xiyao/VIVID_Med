#!/usr/bin/env python
"""Merge and summarize the two locked VinDr Qwen3.5 development shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.localization_causality import summarize_audit_rows  # noqa: E402
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


DEFAULT_INPUT = ROOT / "local_runs/cxr_localization_causality/vindr_qwen35_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT / "merged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_results = [read_json(args.input_dir / f"gpu{index}/shard_result.json") for index in range(2)]
    if any(result.get("status") != "complete_nonformal" for result in shard_results):
        raise ValueError("VinDr development shard is incomplete")
    if any(result.get("exclusions") for result in shard_results):
        raise ValueError("VinDr development shard has fail-closed exclusions")
    lock_hashes = {result.get("input_lock_canonical_sha256") for result in shard_results}
    if len(lock_hashes) != 1:
        raise ValueError("VinDr development shard lock mismatch")
    rows = []
    for index in range(2):
        path = args.input_dir / f"gpu{index}/audit_rows.jsonl"
        if file_sha256(path) != shard_results[index].get("rows_sha256"):
            raise ValueError(f"VinDr GPU{index} row hash changed")
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: str(row["row_id"]))
    if len(rows) != 8 or len({row["row_id"] for row in rows}) != 8:
        raise ValueError("VinDr development merge requires eight unique operator rows")
    if any(row.get("patient_level_claim") is not False for row in rows):
        raise ValueError("VinDr development patient-level boundary changed")

    summary = relabel_summary_as_image_level(
        summarize_audit_rows(rows, bootstrap_replicates=1000, bootstrap_seed=20260719)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "audit_rows.jsonl"
    write_jsonl(rows_path, rows)
    result: dict[str, Any] = {
        "format_version": "vindr_qwen35_local_development_merged_v1",
        "status": "complete_nonformal_image_level",
        "formal_result": False,
        "test_opened": False,
        "patient_level_claim": False,
        "cluster_unit": "image_id",
        "source_samples": 4,
        "audit_rows": len(rows),
        "input_lock_canonical_sha256": next(iter(lock_hashes)),
        "shard_canonical_sha256": [result["canonical_sha256"] for result in shard_results],
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "interpretation_boundary": (
            "Prior-exposed VinDr-train development only. Public VinDr does not provide "
            "patient identifiers, so bootstrap units are image proxies and no patient-level "
            "or independent-primary claim is permitted."
        ),
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    write_json(args.output_dir / "merged_result.json", result)
    print(json.dumps({key: result[key] for key in (
        "status", "source_samples", "audit_rows", "rows_sha256", "canonical_sha256"
    )}, indent=2))
    return 0


def relabel_summary_as_image_level(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove patient terminology when the public release has no patient IDs."""

    output = dict(summary)
    output["image_units"] = int(output.pop("patients"))
    output["patient_level_claim"] = False
    output["cluster_unit"] = "image_id"
    groups = {}
    for key, value in output["groups"].items():
        item = dict(value)
        item["image_units"] = int(item.pop("patients"))
        item["image_cluster_bootstrap_95ci"] = item.pop(
            "patient_cluster_bootstrap_95ci"
        )
        groups[key] = item
    output["groups"] = groups
    return output


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
