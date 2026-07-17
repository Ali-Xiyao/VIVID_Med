"""Audit topology-matched translated controls on locked VinDr-train development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.pixel_interventions import (  # noqa: E402
    transform_mask_to_letterbox,
    union_box_mask,
)
from bives_cxr.rescue_protocol import (  # noqa: E402
    TOPOLOGY_CONTROL_VERSION,
    deterministic_translated_control_mask,
)


DEFAULT_INTAKE_DIR = Path("local_runs/bives_cxr/vindr_rescue_dev")
DEFAULT_OUTPUT_DIR = Path("local_runs/bives_cxr/vindr_rescue_geometry")
FORMAT_VERSION = "bives_vindr_rescue_geometry_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_INTAKE_DIR / "vindr_train_rescue_dev.jsonl"
    )
    parser.add_argument(
        "--data-lock", type=Path, default=DEFAULT_INTAKE_DIR / "vindr_train_rescue_dev_lock.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--minimum-feasibility", type=float, default=0.90)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def content_geometry(width: int, height: int, image_size: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    box = (left, top, left + resized_width, top + resized_height)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[top : top + resized_height, left : left + resized_width] = True
    return box, mask


def audit_geometry(
    manifest_path: Path,
    data_lock_path: Path,
    output_dir: Path,
    *,
    image_size: int = 448,
    minimum_feasibility: float = 0.90,
) -> dict[str, Any]:
    if image_size <= 0 or not 0.0 < minimum_feasibility <= 1.0:
        raise ValueError("invalid image_size or minimum_feasibility")
    data_lock = json.loads(data_lock_path.read_text(encoding="utf-8"))
    if data_lock.get("status") != "pass":
        raise ValueError("R001 data lock is not pass")
    if data_lock.get("source_split") != "train_only":
        raise ValueError("R002 requires a train-only data lock")
    if data_lock.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("R001 manifest does not match its lock")
    rows = read_jsonl(manifest_path)
    positives = [
        row
        for row in rows
        if row["rescue_split"] == "protocol_design" and int(row["binary_label"]) == 1
    ]
    if not positives:
        raise ValueError("protocol_design positive cohort is empty")

    audit_rows: list[dict[str, Any]] = []
    finding_totals: Counter[str] = Counter()
    finding_eligible: Counter[str] = Counter()
    for row in positives:
        if row.get("source_split") != "train" or "test" in Path(row["image_path"]).parts:
            raise ValueError(f"VinDr-test path is forbidden in R002: {row['image_path']}")
        finding = str(row["canonical_statement_id"])
        finding_totals[finding] += 1
        width = int(row["native_columns"])
        height = int(row["native_rows"])
        box, content_mask = content_geometry(width, height, image_size)
        target = transform_mask_to_letterbox(
            union_box_mask(width, height, row["bounding_boxes"]), box, image_size
        ) & content_mask
        try:
            _, geometry = deterministic_translated_control_mask(
                target,
                content_mask,
                seed_key=f"{row['sample_id']}:{TOPOLOGY_CONTROL_VERSION}",
            )
            target_geometry = geometry["target"]
            control_geometry = geometry["control"]
            contracts = {
                "area_equal": target_geometry["area_pixels"] == control_geometry["area_pixels"],
                "disjoint": bool(geometry["disjoint"]),
                "component_count_equal": (
                    target_geometry["component_count"] == control_geometry["component_count"]
                ),
                "perimeter_equal": (
                    target_geometry["perimeter_edges"] == control_geometry["perimeter_edges"]
                ),
                "compactness_equal": bool(
                    np.isclose(
                        target_geometry["compactness"],
                        control_geometry["compactness"],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                "bbox_aspect_ratio_equal": bool(
                    np.isclose(
                        target_geometry["bbox_aspect_ratio"],
                        control_geometry["bbox_aspect_ratio"],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                "vertical_band_equal": (
                    target_geometry["vertical_band"] == control_geometry["vertical_band"]
                ),
            }
            if not all(contracts.values()):
                raise AssertionError(f"translated control contract failure: {contracts}")
            finding_eligible[finding] += 1
            audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "unit_id": row["unit_id"],
                    "canonical_statement_id": finding,
                    "reader_consensus": row["reader_consensus"],
                    "box_area_quartile": row["box_area_quartile"],
                    "rescue_split": "protocol_design",
                    "feasible": True,
                    "exclusion_reason": None,
                    "control_geometry": geometry,
                    "contracts": contracts,
                    "formal_result": False,
                }
            )
        except ValueError as error:
            audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "unit_id": row["unit_id"],
                    "canonical_statement_id": finding,
                    "reader_consensus": row["reader_consensus"],
                    "box_area_quartile": row["box_area_quartile"],
                    "rescue_split": "protocol_design",
                    "feasible": False,
                    "exclusion_reason": str(error),
                    "control_geometry": None,
                    "contracts": None,
                    "formal_result": False,
                }
            )

    per_finding = {}
    for finding, total in sorted(finding_totals.items()):
        eligible = int(finding_eligible[finding])
        rate = eligible / total
        per_finding[finding] = {
            "total": int(total),
            "eligible": eligible,
            "excluded": int(total - eligible),
            "feasibility": float(rate),
            "pass": bool(rate >= minimum_feasibility),
        }
    total = len(positives)
    eligible = sum(finding_eligible.values())
    overall_feasibility = eligible / total
    status = (
        "pass"
        if overall_feasibility >= minimum_feasibility
        and all(value["pass"] for value in per_finding.values())
        else "fail"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "geometry_rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(rows_path)
    lock = {
        "format_version": FORMAT_VERSION,
        "status": status,
        "formal_result": False,
        "source_split": "vindr_train_protocol_design_only",
        "forbidden_test_path_accessed": False,
        "image_disjoint_only": True,
        "patient_level_claim": False,
        "model_loaded": False,
        "scores_accessed": False,
        "image_size": int(image_size),
        "minimum_feasibility": float(minimum_feasibility),
        "control_version": TOPOLOGY_CONTROL_VERSION,
        "data_lock_sha256": file_sha256(data_lock_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "geometry_rows_sha256": file_sha256(rows_path),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "protocol_module_sha256": file_sha256(ROOT / "bives_cxr" / "rescue_protocol.py"),
        "total": total,
        "eligible": int(eligible),
        "excluded": int(total - eligible),
        "overall_feasibility": float(overall_feasibility),
        "per_finding": per_finding,
    }
    lock_path = output_dir / "geometry_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": status,
        "format_version": FORMAT_VERSION,
        "geometry_rows": str(rows_path),
        "geometry_lock": str(lock_path),
        "geometry_rows_sha256": lock["geometry_rows_sha256"],
        "total": total,
        "eligible": int(eligible),
        "excluded": int(total - eligible),
        "overall_feasibility": float(overall_feasibility),
        "per_finding": per_finding,
        "model_loaded": False,
        "scores_accessed": False,
        "patient_level_claim": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    result = audit_geometry(
        args.manifest,
        args.data_lock,
        args.output_dir,
        image_size=args.image_size,
        minimum_feasibility=args.minimum_feasibility,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
