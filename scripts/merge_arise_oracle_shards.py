#!/usr/bin/env python
"""Merge patient-disjoint ARISE oracle shards and recompute the gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arise_cxr.oracle_ceiling import evaluate_oracle_ceiling  # noqa: E402
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def normalized_identity(identity: dict[str, Any]) -> dict[str, Any]:
    ignored = {"device", "shard_index"}
    return {key: value for key, value in identity.items() if key not in ignored}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.shard_dir) < 2:
        raise ValueError("ARISE oracle merge requires at least two shards")

    results = [read_json(path / "result.json") for path in args.shard_dir]
    for path, result in zip(args.shard_dir, results, strict=True):
        if result.get("status") != "complete_development" or result.get("test_opened"):
            raise ValueError(f"ARISE oracle shard is incomplete or test-open: {path}")
        rows_path = path / "audit_rows.jsonl"
        if file_sha256(rows_path) != result.get("rows_sha256"):
            raise ValueError(f"ARISE oracle shard row hash changed: {path}")
    expected_shards = int(results[0]["identity"].get("num_shards", 0))
    indices = sorted(int(result["identity"].get("shard_index", -1)) for result in results)
    if expected_shards != len(results) or indices != list(range(expected_shards)):
        raise ValueError("ARISE oracle shard set is incomplete")
    reference_identity = normalized_identity(results[0]["identity"])
    if any(normalized_identity(result["identity"]) != reference_identity for result in results[1:]):
        raise ValueError("ARISE oracle shard identities differ")
    if any(result["model"] != results[0]["model"] for result in results[1:]):
        raise ValueError("ARISE oracle shard model identities differ")

    rows = [
        row
        for path in args.shard_dir
        for row in read_jsonl(path / "audit_rows.jsonl")
    ]
    row_ids = [str(row["row_id"]) for row in rows]
    if len(rows) != 198 or len(set(row_ids)) != len(row_ids):
        raise ValueError("merged ARISE oracle rows are incomplete or duplicated")
    pairs = {(str(row["image_id"]), str(row["pathology_id"])) for row in rows}
    if len(pairs) != 99:
        raise ValueError("merged ARISE oracle pair count changed")
    shard_patients = [
        {str(row["patient_id"]) for row in read_jsonl(path / "audit_rows.jsonl")}
        for path in args.shard_dir
    ]
    for left in range(len(shard_patients)):
        for right in range(left + 1, len(shard_patients)):
            if shard_patients[left] & shard_patients[right]:
                raise ValueError("ARISE oracle patient shards overlap")

    rows.sort(key=lambda row: str(row["row_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "audit_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    pathologies = sorted({str(row["pathology_id"]) for row in rows})
    operators = sorted({str(row["operator_id"]) for row in rows})
    gate = evaluate_oracle_ceiling(
        rows,
        required_pathologies=pathologies,
        required_operators=operators,
        minimum_passing_pathologies=3,
        bootstrap_replicates=2000,
        bootstrap_seed=20260722,
    )
    result = {
        "schema_version": "arise-dense-oracle-merged-result-v1",
        "status": "complete_development",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "new_model_scores_created": True,
        "identity": reference_identity,
        "model": results[0]["model"],
        "source_samples": len(pairs),
        "audit_rows": len(rows),
        "rows_sha256": file_sha256(rows_path),
        "source_shard_result_sha256": [result["canonical_sha256"] for result in results],
        "oracle_gate": gate,
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
