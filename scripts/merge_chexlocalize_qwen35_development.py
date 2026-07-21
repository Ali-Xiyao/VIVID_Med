#!/usr/bin/env python
"""Merge and summarize CheXlocalize validation Qwen3.5 development shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.localization_causality import summarize_audit_rows  # noqa: E402
from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402


DEFAULT_INPUT = ROOT / "local_runs/cxr_localization_causality/chexlocalize_qwen35_development"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT / "merged")
    args = parser.parse_args()
    shard_results = [json.loads((args.input_dir / f"gpu{i}/shard_result.json").read_text(encoding="utf-8")) for i in range(2)]
    if any(result.get("status") != "complete_nonformal" for result in shard_results):
        raise ValueError("CheXlocalize development shard is incomplete")
    lock_hashes = {result["input_lock_canonical_sha256"] for result in shard_results}
    if len(lock_hashes) != 1:
        raise ValueError("CheXlocalize shard lock mismatch")
    rows = []
    for index, result in enumerate(shard_results):
        path = args.input_dir / f"gpu{index}/audit_rows.jsonl"
        if file_sha256(path) != result["rows_sha256"]:
            raise ValueError("CheXlocalize shard row hash changed")
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    rows.sort(key=lambda row: row["row_id"])
    exclusions = [item for result in shard_results for item in result["exclusions"]]
    if len(rows) != 2 * (100 - len(exclusions)) or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("CheXlocalize merged row/exclusion accounting mismatch")
    if any(row.get("patient_level_claim") is not True for row in rows):
        raise ValueError("CheXlocalize patient-level boundary changed")
    summary = summarize_audit_rows(rows, bootstrap_replicates=2000, bootstrap_seed=20260719)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "audit_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    result = {
        "schema_version": "chexlocalize-qwen35-development-merged-v1",
        "status": "complete_nonformal_development",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "patient_level_claim": True,
        "cluster_unit": "patient_id_hash",
        "source_samples": 100,
        "audit_rows": len(rows),
        "exclusions": exclusions,
        "input_lock_canonical_sha256": next(iter(lock_hashes)),
        "shard_canonical_sha256": [item["canonical_sha256"] for item in shard_results],
        "rows_sha256": file_sha256(rows_path),
        "summary": summary,
        "interpretation_boundary": "Prior-exposed CheXlocalize validation development only; not independent or confirmatory evidence.",
    }
    result["canonical_sha256"] = canonical_json_sha256(result)
    (args.output_dir / "merged_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "source_samples", "audit_rows", "exclusions", "rows_sha256", "canonical_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
