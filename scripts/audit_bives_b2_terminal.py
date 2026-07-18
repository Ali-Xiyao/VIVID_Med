#!/usr/bin/env python
"""Read-only terminal audit of frozen C5 and C6I BiVES B2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.pixel_interventions import (  # noqa: E402
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)
from bives_cxr.qwen35_preprocessing import letterbox_image  # noqa: E402
from bives_cxr.terminal_audit import (  # noqa: E402
    OPERATORS,
    flatten_effect_rows,
    image_perturbation_metrics,
    summarize_effect_rows,
    summarize_image_audit,
)


DEFAULT_C5 = ROOT / "local_runs/bives_cxr/connected_control_c5_confirmation"
DEFAULT_C6I_GEOMETRY = ROOT / "local_runs/bives_cxr/c6i_ms_cxr_actual_input_geometry"
DEFAULT_C6I_EVALUATION = ROOT / "local_runs/bives_cxr/c6i_ms_cxr_replacement_one_time/evaluation"
DEFAULT_C6_MANIFEST = ROOT / "local_runs/bives_cxr/c6_ms_cxr_postc5/ms_cxr_postc5_manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "local_runs/bives_cxr/b2_terminal_read_only_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c5-dir", type=Path, default=DEFAULT_C5)
    parser.add_argument("--c6i-geometry-dir", type=Path, default=DEFAULT_C6I_GEOMETRY)
    parser.add_argument("--c6i-evaluation-dir", type=Path, default=DEFAULT_C6I_EVALUATION)
    parser.add_argument("--c6-manifest", type=Path, default=DEFAULT_C6_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c5_rows_path = args.c5_dir / "confirmation_rows.jsonl"
    c5_geometry_path = args.c5_dir / "geometry_rows.jsonl"
    c5_metrics_path = args.c5_dir / "metrics_final.json"
    c6i_rows_path = args.c6i_evaluation_dir / "evaluation_rows.jsonl"
    c6i_metrics_path = args.c6i_evaluation_dir / "metrics_final.json"
    c6i_geometry_path = args.c6i_geometry_dir / "c6i_geometry_rows.jsonl"
    inputs = [
        c5_rows_path,
        c5_geometry_path,
        c5_metrics_path,
        c6i_rows_path,
        c6i_metrics_path,
        c6i_geometry_path,
        args.c6_manifest,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    c5_rows = read_jsonl(c5_rows_path)
    c5_geometry = index_by_sample(read_jsonl(c5_geometry_path))
    c6i_rows = read_jsonl(c6i_rows_path)
    c6i_geometry = index_by_sample(read_jsonl(c6i_geometry_path))
    manifest = index_by_sample(read_jsonl(args.c6_manifest))
    c5_metrics = read_json(c5_metrics_path)
    c6i_metrics = read_json(c6i_metrics_path)
    if file_sha256(c6i_rows_path) != c6i_metrics["evaluation_rows_sha256"]:
        raise ValueError("frozen C6I evaluation row hash changed")
    if file_sha256(c5_rows_path) != c5_metrics["confirmation_rows_sha256"]:
        raise ValueError("frozen C5 confirmation row hash changed")
    if c6i_metrics["status"] != "fail_final_stop":
        raise ValueError("C6I is not frozen as fail_final_stop")
    if c5_metrics["complete_c5_gate_pass"] is not False:
        raise ValueError("C5 final-stop identity changed")

    effect_rows = flatten_effect_rows(
        c5_rows,
        source="c5",
        geometry_by_sample=c5_geometry,
    ) + flatten_effect_rows(
        c6i_rows,
        source="c6i",
        geometry_by_sample=c6i_geometry,
        manifest_by_sample=manifest,
    )
    image_rows = run_c6i_image_audit(
        c6i_rows,
        c6i_geometry,
        manifest,
        args.c6i_geometry_dir / "geometry_masks",
    )
    summary = {
        "format_version": "bives_b2_terminal_read_only_audit_v1",
        "status": "complete_terminal_negative_no_model_access",
        "model_loaded": False,
        "scores_computed": False,
        "training_performed": False,
        "new_experiment_stage_created": False,
        "c6j_created": False,
        "qwen35_4b_9b_authorized": False,
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in inputs},
        "stage_summary": build_stage_summary(c5_metrics, c6i_metrics),
        "effect_decomposition": summarize_effect_rows(effect_rows),
        "image_space_perturbation": summarize_image_audit(image_rows),
        "interpretation_boundary": (
            "Descriptive post-stop audit only. Associations and image-space differences "
            "do not establish causality, clinical validity, or a rescued BiVES mechanism."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "effect_rows.jsonl", effect_rows)
    write_jsonl(args.output_dir / "image_space_rows.jsonl", image_rows)
    write_json(args.output_dir / "audit_summary.json", summary)
    plot_effects(effect_rows, args.output_dir / "paired_effect_scatter.png")
    print(json.dumps({
        "status": summary["status"],
        "effect_rows": len(effect_rows),
        "image_space_rows": len(image_rows),
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


def run_c6i_image_audit(
    score_rows: list[dict[str, Any]],
    geometry: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    mask_dir: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in score_rows:
        sample_id = str(row["sample_id"])
        geometry_row = geometry[sample_id]
        manifest_row = manifest[sample_id]
        image_path = Path(str(manifest_row["image_path"]))
        if file_sha256(image_path) != manifest_row["official_image_sha256"]:
            raise ValueError(f"bound C6I image hash changed: {sample_id}")
        mask_path = mask_dir / str(geometry_row["mask_file"])
        if file_sha256(mask_path) != geometry_row["mask_sha256"]:
            raise ValueError(f"bound C6I mask hash changed: {sample_id}")
        with Image.open(image_path) as handle:
            source = handle.convert("RGB")
        if source.size != (224, 224):
            raise ValueError(f"bound C6I image geometry changed: {sample_id}")
        letterboxed, _ = letterbox_image(source, 448)
        with np.load(mask_path, allow_pickle=False) as payload:
            target = payload["target_mask"].astype(bool)
            control = payload["control_mask"].astype(bool)
            content = payload["content_mask"].astype(bool)
        original = np.asarray(letterboxed)
        for operator in OPERATORS:
            if operator == "local_mean":
                target_image = replace_with_local_ring_mean(letterboxed, target, content)
                control_image = replace_with_local_ring_mean(letterboxed, control, content)
            else:
                target_image = replace_with_masked_gaussian_blur(letterboxed, target, content)
                control_image = replace_with_masked_gaussian_blur(letterboxed, control, content)
            output.append(
                {
                    "sample_id": sample_id,
                    "canonical_statement_id": row["canonical_statement_id"],
                    "operator": operator,
                    "model_score_accessed": False,
                    "target": image_perturbation_metrics(
                        original, np.asarray(target_image), target, content
                    ),
                    "control": image_perturbation_metrics(
                        original, np.asarray(control_image), control, content
                    ),
                }
            )
    return output


def build_stage_summary(c5: dict[str, Any], c6i: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage": "C4",
            "dataset_role": "VinDr-train protocol-design positives",
            "mechanism_gate": "pass",
            "polarity_gate": "not_opened",
            "terminal_decision": "advance_once_to_C5",
        },
        {
            "stage": "C5",
            "dataset_role": "VinDr-train disjoint confirmation",
            "mechanism_gate": c5["mechanism_gate"]["status"],
            "polarity_gate": c5["polarity_gate"]["status"],
            "terminal_decision": "fail_final_stop",
        },
        {
            "stage": "C6I",
            "dataset_role": "independent MS-CXR publisher test positive-only",
            "mechanism_gate": c6i["survival_gate"]["status"],
            "polarity_gate": "not_computed_positive_only",
            "terminal_decision": c6i["status"],
        },
    ]


def plot_effects(rows: list[dict[str, Any]], output: Path) -> None:
    sources = ("c5", "c6i")
    findings = ("consolidation", "pleural_effusion")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False)
    for source_index, source in enumerate(sources):
        for operator_index, operator in enumerate(OPERATORS):
            for finding_index, finding in enumerate(findings):
                column = operator_index * 2 + finding_index
                ax = axes[source_index, column]
                subset = [
                    row for row in rows
                    if row["source"] == source
                    and row["operator"] == operator
                    and row["canonical_statement_id"] == finding
                ]
                x = [row["control_effect"] for row in subset]
                y = [row["target_effect"] for row in subset]
                ax.scatter(x, y, s=20, alpha=0.7)
                bounds = x + y + [0.0]
                low, high = min(bounds), max(bounds)
                padding = max(0.005, (high - low) * 0.08)
                ax.plot([low - padding, high + padding], [low - padding, high + padding], "--", color="gray")
                ax.axhline(0.0, color="black", linewidth=0.6)
                ax.axvline(0.0, color="black", linewidth=0.6)
                ax.set_title(f"{source.upper()} | {operator}\n{finding}")
                ax.set_xlabel("control effect")
                ax.set_ylabel("target effect")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def index_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate sample_id in frozen input")
    return indexed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
