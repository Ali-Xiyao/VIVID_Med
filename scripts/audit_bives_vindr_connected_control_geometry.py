"""Audit the accepted coordinate-zone connected control on VinDr train only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    COORDINATE_ZONE_CONTROL_VERSION,
    deterministic_coordinate_zone_connected_control_mask,
)


DEFAULT_INTAKE_DIR = Path("local_runs/bives_cxr/vindr_rescue_dev")
DEFAULT_OUTPUT_DIR = Path("local_runs/bives_cxr/vindr_connected_control_geometry")
FORMAT_VERSION = "bives_vindr_coordinate_zone_connected_geometry_audit_v1"
IMAGE_SIZE = 448
MINIMUM_OVERALL_FINDING_FEASIBILITY = 0.95
MINIMUM_FINDING_AREA_QUARTILE_FEASIBILITY = 0.90
EXPECTED_R001_MANIFEST_SHA256 = (
    "bd84cd7ca5384afbcb6228c49331b028a9641dd3dd9011157c2cec75b1f6514f"
)
EXPECTED_R001_DATA_LOCK_SHA256 = (
    "4251027b3069b21fb6fb5acd6bc02bf003206fbcfffb6d045abd2289ea2ac409"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INTAKE_DIR / "vindr_train_rescue_dev.jsonl",
    )
    parser.add_argument(
        "--data-lock",
        type=Path,
        default=DEFAULT_INTAKE_DIR / "vindr_train_rescue_dev_lock.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
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
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def content_geometry(
    width: int,
    height: int,
    image_size: int,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    box = (left, top, left + resized_width, top + resized_height)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[top : top + resized_height, left : left + resized_width] = True
    return box, mask


def audit_one_positive(row: dict[str, Any], image_size: int) -> dict[str, Any]:
    if row.get("rescue_split") != "protocol_design" or int(row["binary_label"]) != 1:
        raise ValueError("C2 worker received a non-design or non-positive row")
    image_path = Path(str(row["image_path"]))
    if row.get("source_split") != "train" or "test" in {
        part.lower() for part in image_path.parts
    }:
        raise ValueError(f"VinDr-test path is forbidden in C2: {image_path}")
    width = int(row["native_columns"])
    height = int(row["native_rows"])
    letterbox, content_mask = content_geometry(width, height, image_size)
    target = transform_mask_to_letterbox(
        union_box_mask(width, height, row["bounding_boxes"]),
        letterbox,
        image_size,
    ) & content_mask
    finding = str(row["canonical_statement_id"])
    base = {
        "sample_id": row["sample_id"],
        "unit_id": row["unit_id"],
        "canonical_statement_id": finding,
        "reader_consensus": row["reader_consensus"],
        "box_area_quartile": int(row["box_area_quartile"]),
        "rescue_split": "protocol_design",
        "source_split": "train",
        "formal_result": False,
        "patient_level_claim": False,
        "model_loaded": False,
        "scores_accessed": False,
        "image_pixels_accessed": False,
    }
    try:
        control, geometry = deterministic_coordinate_zone_connected_control_mask(
            target,
            content_mask,
            seed_key=f"{row['sample_id']}:{COORDINATE_ZONE_CONTROL_VERSION}",
        )
        target_geometry = geometry["target"]
        control_geometry = geometry["control"]
        contracts = {
            "area_equal": (
                target_geometry["area_pixels"] == control_geometry["area_pixels"]
            ),
            "contained_in_content": not bool((control & ~content_mask).any()),
            "target_disjoint": not bool((control & target).any()),
            "single_4_connected_component": (
                control_geometry["component_count"] == 1
            ),
            "horizontal_zone_equal": (
                geometry["target_zone"]["horizontal"]
                == geometry["control_zone"]["horizontal"]
            ),
            "vertical_zone_equal": (
                geometry["target_zone"]["vertical"]
                == geometry["control_zone"]["vertical"]
            ),
            "not_true_anatomy_claim": not geometry["true_anatomy_segmentation"],
        }
        if not all(contracts.values()):
            raise AssertionError(f"connected-control contract failure: {contracts}")
        return {
            **base,
            "feasible": True,
            "exclusion_reason": None,
            "control_geometry": geometry,
            "contracts": contracts,
        }
    except ValueError as error:
        return {
            **base,
            "feasible": False,
            "exclusion_reason": str(error),
            "control_geometry": None,
            "contracts": None,
        }


def summarize_group(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    *,
    minimum: float | None,
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in key_fields)].append(row)
    summary: dict[str, Any] = {}
    for key, subset in sorted(groups.items()):
        eligible = sum(bool(row["feasible"]) for row in subset)
        feasibility = eligible / len(subset)
        label = "|".join(key)
        entry = {
            "total": len(subset),
            "eligible": eligible,
            "excluded": len(subset) - eligible,
            "feasibility": float(feasibility),
        }
        if minimum is not None:
            entry["minimum"] = float(minimum)
            entry["pass"] = bool(feasibility >= minimum)
        summary[label] = entry
    return summary


def audit_geometry(
    manifest_path: Path,
    data_lock_path: Path,
    output_dir: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    start = time.perf_counter()
    manifest_sha256 = file_sha256(manifest_path)
    data_lock_sha256 = file_sha256(data_lock_path)
    if manifest_sha256 != EXPECTED_R001_MANIFEST_SHA256:
        raise ValueError("C2 manifest is not the accepted R001 manifest")
    if data_lock_sha256 != EXPECTED_R001_DATA_LOCK_SHA256:
        raise ValueError("C2 data lock is not the accepted final R001 lock")
    data_lock = json.loads(data_lock_path.read_text(encoding="utf-8"))
    if data_lock.get("status") != "pass" or data_lock.get("source_split") != "train_only":
        raise ValueError("C2 requires the passing train-only R001 data lock")
    if data_lock.get("manifest_sha256") != manifest_sha256:
        raise ValueError("R001 manifest does not match its data lock")

    plan_path = ROOT / "refine-logs" / "CONNECTED_CONTROL_RESCUE_PLAN.md"
    versioned_plan_path = (
        ROOT / "refine-logs" / "CONNECTED_CONTROL_RESCUE_PLAN_20260718.md"
    )
    tracker_path = ROOT / "refine-logs" / "CONNECTED_CONTROL_RESCUE_TRACKER.md"
    versioned_tracker_path = (
        ROOT / "refine-logs" / "CONNECTED_CONTROL_RESCUE_TRACKER_20260718.md"
    )
    if plan_path.read_bytes() != versioned_plan_path.read_bytes():
        raise ValueError("connected-control plan aliases are not byte-identical")
    if tracker_path.read_bytes() != versioned_tracker_path.read_bytes():
        raise ValueError("connected-control tracker aliases are not byte-identical")

    all_rows = read_jsonl(manifest_path)
    positives = sorted(
        (
            row
            for row in all_rows
            if row.get("rescue_split") == "protocol_design"
            and int(row["binary_label"]) == 1
        ),
        key=lambda row: str(row["sample_id"]),
    )
    if len(positives) != 377:
        raise ValueError(f"C2 expected 377 protocol-design positives, found {len(positives)}")
    if any(row.get("source_split") != "train" for row in positives):
        raise ValueError("C2 selected a non-train row")

    audit_rows: list[dict[str, Any]] = []
    if workers == 1:
        for index, row in enumerate(positives, start=1):
            audit_rows.append(audit_one_positive(row, IMAGE_SIZE))
            if index % 10 == 0 or index == len(positives):
                print(f"C2_PROGRESS {index}/{len(positives)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(audit_one_positive, row, IMAGE_SIZE): row["sample_id"]
                for row in positives
            }
            for index, future in enumerate(as_completed(futures), start=1):
                audit_rows.append(future.result())
                if index % 10 == 0 or index == len(positives):
                    print(f"C2_PROGRESS {index}/{len(positives)}", flush=True)
    audit_rows.sort(key=lambda row: str(row["sample_id"]))

    invariant_failures = sum(
        row["feasible"] and not all(row["contracts"].values()) for row in audit_rows
    )
    overall_eligible = sum(bool(row["feasible"]) for row in audit_rows)
    overall_feasibility = overall_eligible / len(audit_rows)
    per_finding = summarize_group(
        audit_rows,
        ("canonical_statement_id",),
        minimum=MINIMUM_OVERALL_FINDING_FEASIBILITY,
    )
    per_finding_area_quartile = summarize_group(
        audit_rows,
        ("canonical_statement_id", "box_area_quartile"),
        minimum=MINIMUM_FINDING_AREA_QUARTILE_FEASIBILITY,
    )
    per_finding_consensus = summarize_group(
        audit_rows,
        ("canonical_statement_id", "reader_consensus"),
        minimum=None,
    )
    exclusion_reasons = Counter(
        str(row["exclusion_reason"]) for row in audit_rows if not row["feasible"]
    )
    status = (
        "pass"
        if overall_feasibility >= MINIMUM_OVERALL_FINDING_FEASIBILITY
        and all(entry["pass"] for entry in per_finding.values())
        and all(entry["pass"] for entry in per_finding_area_quartile.values())
        and invariant_failures == 0
        else "fail"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "connected_geometry_rows.jsonl"
    temporary = rows_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(rows_path)

    lock = {
        "format_version": FORMAT_VERSION,
        "status": status,
        "formal_result": False,
        "source_split": "vindr_train_protocol_design_positive_only",
        "forbidden_test_path_accessed": False,
        "rescue_confirm_rows_used": 0,
        "image_disjoint_only": True,
        "patient_level_claim": False,
        "model_loaded": False,
        "scores_accessed": False,
        "image_pixels_accessed": False,
        "image_size": IMAGE_SIZE,
        "minimum_overall_finding_feasibility": (
            MINIMUM_OVERALL_FINDING_FEASIBILITY
        ),
        "minimum_finding_area_quartile_feasibility": (
            MINIMUM_FINDING_AREA_QUARTILE_FEASIBILITY
        ),
        "control_version": COORDINATE_ZONE_CONTROL_VERSION,
        "source_base_commit": git_head(),
        "accepted_plan_sha256": file_sha256(plan_path),
        "accepted_tracker_sha256": file_sha256(tracker_path),
        "data_lock_sha256": data_lock_sha256,
        "source_manifest_sha256": manifest_sha256,
        "geometry_rows_sha256": file_sha256(rows_path),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "protocol_module_sha256": file_sha256(ROOT / "bives_cxr" / "rescue_protocol.py"),
        "total": len(audit_rows),
        "eligible": int(overall_eligible),
        "excluded": int(len(audit_rows) - overall_eligible),
        "overall_feasibility": float(overall_feasibility),
        "invariant_failures": int(invariant_failures),
        "per_finding": per_finding,
        "per_finding_area_quartile": per_finding_area_quartile,
        "per_finding_consensus": per_finding_consensus,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "workers": int(workers),
        "wall_seconds": float(time.perf_counter() - start),
    }
    lock_path = output_dir / "connected_geometry_lock.json"
    write_json(lock_path, lock)
    summary = {
        "status": status,
        "format_version": FORMAT_VERSION,
        "geometry_rows": str(rows_path),
        "geometry_lock": str(lock_path),
        "geometry_rows_sha256": lock["geometry_rows_sha256"],
        "total": lock["total"],
        "eligible": lock["eligible"],
        "excluded": lock["excluded"],
        "overall_feasibility": lock["overall_feasibility"],
        "invariant_failures": lock["invariant_failures"],
        "per_finding": per_finding,
        "per_finding_area_quartile": per_finding_area_quartile,
        "per_finding_consensus": per_finding_consensus,
        "exclusion_reasons": lock["exclusion_reasons"],
        "model_loaded": False,
        "scores_accessed": False,
        "image_pixels_accessed": False,
        "patient_level_claim": False,
        "wall_seconds": lock["wall_seconds"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    result = audit_geometry(
        args.manifest,
        args.data_lock,
        args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
