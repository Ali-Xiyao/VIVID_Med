#!/usr/bin/env python
"""Build the aggregate-only VinDr C5 plus MS-CXR C6I retrospective."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.frozen_retrospective import (  # noqa: E402
    aggregate_frozen_rows,
    build_retrospective_summary,
    normalize_frozen_rows,
)


DEFAULT_C5 = ROOT / "local_runs/bives_cxr/connected_control_c5_confirmation"
DEFAULT_C6I = ROOT / "local_runs/bives_cxr/c6i_ms_cxr_replacement_one_time/evaluation"
DEFAULT_OUTPUT = ROOT / "local_runs/cxr_localization_causality/frozen_existing_data_retrospective"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c5-dir", type=Path, default=DEFAULT_C5)
    parser.add_argument("--c6i-dir", type=Path, default=DEFAULT_C6I)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c5_rows_path = args.c5_dir / "confirmation_rows.jsonl"
    c5_metrics_path = args.c5_dir / "metrics_final.json"
    c6i_rows_path = args.c6i_dir / "evaluation_rows.jsonl"
    c6i_metrics_path = args.c6i_dir / "metrics_final.json"
    for path in (c5_rows_path, c5_metrics_path, c6i_rows_path, c6i_metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    c5_rows = read_jsonl(c5_rows_path)
    c6i_rows = read_jsonl(c6i_rows_path)
    c5_metrics = read_json(c5_metrics_path)
    c6i_metrics = read_json(c6i_metrics_path)
    validate_frozen_identity(
        c5_rows_path=c5_rows_path,
        c5_metrics=c5_metrics,
        c5_rows=c5_rows,
        c6i_rows_path=c6i_rows_path,
        c6i_metrics=c6i_metrics,
        c6i_rows=c6i_rows,
    )

    normalized = normalize_frozen_rows(c5_rows, source="c5")
    normalized.extend(normalize_frozen_rows(c6i_rows, source="c6i"))
    aggregate = aggregate_frozen_rows(normalized)
    source_paths = (c5_rows_path, c5_metrics_path, c6i_rows_path, c6i_metrics_path)
    summary = build_retrospective_summary(
        aggregate,
        source_sha256={
            str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
        },
        source_record_counts={"c5": len(c5_rows), "c6i": len(c6i_rows)},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "aggregate_rows.jsonl", aggregate)
    write_json(args.output_dir / "retrospective_summary.json", summary)
    (args.output_dir / "retrospective_result.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "aggregate_cells": len(aggregate),
                "normalized_operator_rows": len(normalized),
                "canonical_artifact_sha256": summary["canonical_artifact_sha256"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


def validate_frozen_identity(
    *,
    c5_rows_path: Path,
    c5_metrics: dict[str, Any],
    c5_rows: list[dict[str, Any]],
    c6i_rows_path: Path,
    c6i_metrics: dict[str, Any],
    c6i_rows: list[dict[str, Any]],
) -> None:
    if c5_metrics.get("status") != "fail_final_stop":
        raise ValueError("C5 terminal status changed")
    if c5_metrics.get("complete_c5_gate_pass") is not False:
        raise ValueError("C5 final-stop gate changed")
    if c5_metrics.get("patient_level_claim") is not False:
        raise ValueError("C5 patient-level limitation changed")
    if len(c5_rows) != 756 or file_sha256(c5_rows_path) != c5_metrics.get(
        "confirmation_rows_sha256"
    ):
        raise ValueError("C5 frozen row identity changed")
    if c6i_metrics.get("status") != "fail_final_stop":
        raise ValueError("C6I terminal status changed")
    if len(c6i_rows) != 29 or file_sha256(c6i_rows_path) != c6i_metrics.get(
        "evaluation_rows_sha256"
    ):
        raise ValueError("C6I frozen row identity changed")
    if c6i_metrics.get("classification_metrics_computed") is not False:
        raise ValueError("C6I positive-only classification boundary changed")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen existing-data retrospective",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "This is descriptive reuse of frozen scores. No model was loaded, no GPU was used, and no new score or test opening occurred.",
        "",
        "| Dataset | Role | Unit | Operator | Finding | N | Mean localization gain | Mean TCIG | Spearman rho |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregate_rows"]:
        rho = row["localization_tcig_spearman_rho"]
        lines.append(
            "| {dataset} | {role} | {unit} | {operator} | {finding} | {n} | {loc:.6f} | {tcig:.6f} | {rho} |".format(
                dataset=row["dataset"],
                role=row["dataset_role"],
                unit=row["claim_unit"],
                operator=row["operator"],
                finding=row["canonical_statement_id"],
                n=row["records"],
                loc=row["mean_localization_gain"],
                tcig=row["mean_tcig"],
                rho="NA" if rho is None else f"{rho:.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            summary["interpretation_boundary"],
            "",
            "The legacy rows lack separate explanation-region and expert-region controls, so this artifact is not compatible with the active primary schema.",
            "",
            f"Canonical artifact SHA-256: `{summary['canonical_artifact_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
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
