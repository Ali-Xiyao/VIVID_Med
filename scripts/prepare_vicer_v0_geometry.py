"""Freeze score-free expert and translated-control masks for VICER V0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bives_cxr.pixel_interventions import transform_mask_to_letterbox, union_box_mask  # noqa: E402
from bives_cxr.qwen35_preprocessing import letterbox_image  # noqa: E402
from bives_cxr.rescue_protocol import deterministic_translated_control_mask, mask_geometry  # noqa: E402
from scripts.cache_qwen35_patch_tokens import read_image  # noqa: E402
from vicer_cxr.validity import canonical_sha256, file_sha256, validate_v0_manifest  # noqa: E402
from vicer_cxr.matched_controls import deterministic_connected_statistics_control  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    validate_v0_manifest(rows)
    data_lock = json.loads(args.data_lock.read_text(encoding="utf-8"))
    if data_lock.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("VICER V0 geometry data-lock mismatch")
    eval_rows = [row for row in rows if row["v0_role"] == "validity_eval"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    geometry_rows = []
    for row in eval_rows:
        source = Path(row["image_path"])
        if file_sha256(source) != row["image_sha256"]:
            raise ValueError("VICER V0 geometry source hash changed")
        image = read_image(source)
        prepared, content_box = letterbox_image(image, args.image_size)
        native = union_box_mask(
            int(row["native_columns"]), int(row["native_rows"]), row["roi_boxes"]
        )
        target = transform_mask_to_letterbox(native, content_box, args.image_size)
        content = np.zeros((args.image_size, args.image_size), dtype=bool)
        left, top, right, bottom = content_box
        content[top:bottom, left:right] = True
        try:
            control, certificate = deterministic_translated_control_mask(
                target,
                content,
                seed_key=f"vicer-v0:{row['sample_id']}",
            )
            control_family = "exact_target_shape_translated_same_vertical_band"
        except ValueError as error:
            grayscale = np.asarray(prepared.convert("L"), dtype=np.float64) / 255.0
            control, certificate = deterministic_connected_statistics_control(
                grayscale,
                target,
                content,
                seed_key=f"vicer-v0:{row['sample_id']}:fallback",
            )
            certificate["translated_control_failure"] = str(error)
            control_family = "exact_area_connected_statistics_fallback"
        if bool((target & control).any()) or int(target.sum()) != int(control.sum()):
            raise AssertionError("VICER V0 control geometry contract failed")
        target_geometry = mask_geometry(target, content)
        control_geometry = mask_geometry(control, content)
        file_name = f"{row['sample_id'].replace(':', '_')}.npz"
        mask_path = masks_dir / file_name
        np.savez_compressed(
            mask_path,
            target=target.astype(np.uint8),
            control=control.astype(np.uint8),
            content=content.astype(np.uint8),
        )
        geometry_rows.append(
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "canonical_statement_id": row["canonical_statement_id"],
                "mask_file": str(mask_path.relative_to(args.output_dir).as_posix()),
                "mask_file_sha256": file_sha256(mask_path),
                "target_geometry": target_geometry,
                "control_geometry": control_geometry,
                "control_certificate": certificate,
                "control_family": control_family,
                "exact_area": True,
                "disjoint": True,
                "same_vertical_band": bool(control_family.startswith("exact_target_shape")),
                "target_shape_translated": bool(control_family.startswith("exact_target_shape")),
                "model_score_used": False,
            }
        )
    rows_path = args.output_dir / "geometry_rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in geometry_rows),
        encoding="utf-8",
        newline="\n",
    )
    lock = {
        "schema_version": "vicer-v0-score-free-geometry-lock-v1",
        "status": "complete",
        "records": len(geometry_rows),
        "manifest_sha256": file_sha256(args.manifest),
        "data_lock_canonical_sha256": data_lock["canonical_sha256"],
        "rows_sha256": file_sha256(rows_path),
        "control_family": "translated exact-shape primary; connected exact-area/statistics fallback",
        "true_anatomy_matching": False,
        "local_statistics_matching": "fallback only",
        "model_or_score_opened": False,
        "chexlocalize_test_opened": False,
    }
    lock["canonical_sha256"] = canonical_sha256(lock)
    (args.output_dir / "geometry_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
