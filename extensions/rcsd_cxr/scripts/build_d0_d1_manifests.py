"""Freeze hard-UMS, three-source, and D1 reliability manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rcsd_cxr.d0_d1_contract import (
    entropy_agreement_weight,
    render_hard_ums_target,
)
from rcsd_cxr.gold_mapping import FINDINGS


RAW_TO_STATE = {"1": "present", "1.0": "present", "0": "absent",
                "0.0": "absent", "-1": "uncertain",
                "-1.0": "uncertain", "": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


def normalize(value: object) -> str | None:
    text = str(value or "").strip()
    if text not in RAW_TO_STATE:
        raise ValueError(f"unsupported state: {value}")
    return RAW_TO_STATE[text]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def stable_key(patient_id: str, study_id: str) -> str:
    return hashlib.sha256(f"{patient_id}|{study_id}".encode()).hexdigest()


def build_rows(
    chexbert_rows: list[dict[str, str]],
    official_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    official_by_study = {row["study_id"]: row for row in official_rows}
    if len(official_by_study) != len(official_rows):
        raise ValueError("official source manifest has duplicate studies")
    hard_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    finding_weights: list[float] = []
    observed_sources: Counter[int] = Counter()
    state_counts = {finding: Counter() for finding in FINDINGS}
    for row in chexbert_rows:
        study_id = row["study_id"]
        official = official_by_study.get(study_id)
        if official is None:
            raise ValueError(f"missing official source row for study {study_id}")
        patient_id = row["patient_id"]
        row_id = stable_key(patient_id, study_id)
        states: dict[str, str | None] = {}
        sources: dict[str, dict[str, str | None]] = {}
        weights: dict[str, float] = {}
        for finding in FINDINGS:
            chexbert = normalize(row.get(finding))
            states[finding] = chexbert
            if chexbert is None:
                continue
            source_states = {
                "chexpert": normalize(
                    official.get(f"chexpert__{slug(finding)}")
                ),
                "negbio": normalize(
                    official.get(f"negbio__{slug(finding)}")
                ),
                "chexbert": chexbert,
            }
            agreement = entropy_agreement_weight(source_states)
            sources[finding] = source_states
            weights[finding] = agreement.weight
            finding_weights.append(agreement.weight)
            observed_sources[agreement.observed_sources] += 1
            state_counts[finding][chexbert] += 1
        if not weights:
            continue
        target = render_hard_ums_target(states)
        identity = {
            "row_id": row_id,
            "patient_id": patient_id,
            "study_id": study_id,
            "split": row["split"],
            "image_path": row["image_path"],
        }
        hard_rows.append({**identity, "target": target})
        source_rows.append({"row_id": row_id, "sources": sources})
        reliability_rows.append(
            {
                "row_id": row_id,
                "finding_weights": weights,
                "mean_weight": float(np.mean(list(weights.values()))),
            }
        )
    if not hard_rows:
        raise ValueError("no supervised D0/D1 rows")
    if len(hard_rows) != len(source_rows) or len(hard_rows) != len(reliability_rows):
        raise AssertionError("D0/D1 manifest row counts diverged")
    train_values = [
        weight
        for hard, reliability in zip(hard_rows, reliability_rows, strict=True)
        if hard["split"] == "train"
        for weight in reliability["finding_weights"].values()
    ]
    cuts = np.quantile(train_values, [0.25, 0.5, 0.75]).tolist()
    for row in reliability_rows:
        value = float(row["mean_weight"])
        row["quartile"] = 1 + sum(value > cut for cut in cuts)
    audit = {
        "rows": len(hard_rows),
        "split_counts": dict(Counter(row["split"] for row in hard_rows)),
        "patients": len({row["patient_id"] for row in hard_rows}),
        "weights": {
            "count": len(finding_weights),
            "minimum": min(finding_weights),
            "maximum": max(finding_weights),
            "mean": float(np.mean(finding_weights)),
            "quartile_cuts": cuts,
            "nonunit_count": int(sum(
                not np.isclose(weight, 1.0) for weight in finding_weights
            )),
            "near_zero_count": int(sum(
                np.isclose(weight, 0.0) for weight in finding_weights
            )),
            "unique_rounded": sorted(
                {round(weight, 12) for weight in finding_weights}
            ),
            "quartiles_degenerate": len(set(cuts)) < 3,
        },
        "observed_sources": dict(observed_sources),
        "state_counts": {
            finding: dict(counts) for finding, counts in state_counts.items()
        },
    }
    return hard_rows, source_rows, reliability_rows, audit


def select_overfit_ids(
    hard_rows: list[dict[str, object]],
    reliability_rows: list[dict[str, object]],
    count: int = 256,
) -> list[str]:
    reliability = {row["row_id"]: row for row in reliability_rows}
    candidates = [
        row
        for row in hard_rows
        if row["split"] == "train"
        and len(reliability[row["row_id"]]["finding_weights"]) >= 2
    ]
    candidates.sort(key=lambda row: row["row_id"])
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} overfit candidates")
    return [str(row["row_id"]) for row in candidates[:count]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chexbert-manifest", required=True, type=Path)
    parser.add_argument("--official-source-manifest", required=True, type=Path)
    parser.add_argument("--hard-output", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--reliability-output", required=True, type=Path)
    parser.add_argument("--overfit-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    hard, sources, reliability, audit = build_rows(
        load_csv(args.chexbert_manifest),
        load_csv(args.official_source_manifest),
    )
    write_jsonl(args.hard_output, hard)
    write_jsonl(args.source_output, sources)
    write_jsonl(args.reliability_output, reliability)
    overfit_ids = select_overfit_ids(hard, reliability)
    args.overfit_output.parent.mkdir(parents=True, exist_ok=True)
    args.overfit_output.write_text(
        json.dumps({"row_ids": overfit_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "artifact": "d0_d1_manifest_lock",
        "pass": True,
        **audit,
        "overfit_rows": len(overfit_ids),
        "hashes": {
            "chexbert_manifest": sha256_file(args.chexbert_manifest),
            "official_source_manifest": sha256_file(
                args.official_source_manifest
            ),
            "hard_ums_manifest": sha256_file(args.hard_output),
            "source_manifest": sha256_file(args.source_output),
            "reliability_manifest": sha256_file(args.reliability_output),
            "overfit_ids": sha256_file(args.overfit_output),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
