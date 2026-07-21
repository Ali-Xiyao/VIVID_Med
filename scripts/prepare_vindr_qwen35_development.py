#!/usr/bin/env python
"""Prepare the score-free four-sample VinDr Qwen3.5 development lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.provenance import canonical_json_sha256, file_sha256  # noqa: E402
from bives_cxr.qwen35_localization_audit import QWEN35_2B_SNAPSHOT_SHA256  # noqa: E402
from bives_cxr.vindr_qwen35_development import (  # noqa: E402
    prepare_expert_masks,
    select_development_rows,
)


DEFAULT_SOURCE = ROOT / "local_runs/bives_cxr/vindr_rescue_dev/vindr_train_rescue_dev.jsonl"
DEFAULT_SOURCE_LOCK = ROOT / "local_runs/bives_cxr/vindr_rescue_dev/vindr_train_rescue_dev_lock.json"
DEFAULT_OPENING = ROOT / "audit/local_vindr_qwen35_development_opening_20260719.json"
DEFAULT_MODEL = Path("H:/Xiyao_Wang/001_models/Qwen3.5-2B")
DEFAULT_OUTPUT = ROOT / "local_runs/cxr_localization_causality/vindr_qwen35_development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--opening", type=Path, default=DEFAULT_OPENING)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.source_manifest)
    source_lock = read_json(args.source_lock)
    if source_lock.get("status") != "pass" or source_lock.get("patient_level_claim") is not False:
        raise ValueError("VinDr source lock boundary changed")
    if file_sha256(args.source_manifest) != source_lock.get("manifest_sha256"):
        raise ValueError("VinDr source manifest hash changed")
    if snapshot_model_files(args.model_path) != QWEN35_2B_SNAPSHOT_SHA256:
        raise ValueError("Qwen3.5-2B snapshot hash changed")

    selected = select_development_rows(source_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "expert_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    locked_rows = []
    for row in selected:
        expert, content, audit = prepare_expert_masks(row)
        mask_name = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest() + ".npz"
        mask_path = mask_dir / mask_name
        temporary = mask_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, expert_mask=expert, content_mask=content)
        temporary.replace(mask_path)
        locked_rows.append(
            {
                **row,
                "mask_file": mask_name,
                "mask_sha256": file_sha256(mask_path),
                "score_free_audit": audit,
            }
        )
    manifest_path = args.output_dir / "development_manifest.jsonl"
    write_jsonl(manifest_path, locked_rows)
    source_files = [
        args.source_manifest,
        args.source_lock,
        args.opening,
        ROOT / "bives_cxr/vindr_qwen35_development.py",
        ROOT / "bives_cxr/qwen35_localization_audit.py",
        ROOT / "bives_cxr/localization_causality.py",
        ROOT / "scripts/run_vindr_qwen35_development.py",
    ]
    lock: dict[str, Any] = {
        "format_version": "vindr_qwen35_local_development_lock_v1",
        "status": "score_free_ready",
        "formal_result": False,
        "test_opened": False,
        "patient_level_claim": False,
        "dataset_role": "supplemental_prior_exposed_development",
        "source_sha256": {str(path): file_sha256(path) for path in source_files},
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_canonical_sha256": canonical_json_sha256(locked_rows),
        "samples": len(locked_rows),
        "findings": sorted({row["canonical_statement_id"] for row in locked_rows}),
        "area_quartiles": sorted({int(row["box_area_quartile"]) for row in locked_rows}),
        "model_snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
        "model_path": args.model_path.as_posix(),
        "explanation": "local_mean_occlusion_4x4_stable_top1",
        "operators": ["local_mean_ring8", "masked_gaussian_blur_sigma8"],
        "shards": 2,
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    write_json(args.output_dir / "development_lock.json", lock)
    print(json.dumps(lock, indent=2, ensure_ascii=False))
    return 0


def snapshot_model_files(model_root: Path) -> str:
    names = {"config.json", "model.safetensors.index.json"}
    names.update(path.name for path in model_root.glob("*.safetensors"))
    files = {
        name: file_sha256(model_root / name)
        for name in sorted(names)
        if (model_root / name).is_file()
    }
    return canonical_json_sha256(files)


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
