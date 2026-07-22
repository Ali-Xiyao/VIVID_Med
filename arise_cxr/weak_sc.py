"""Configurable weak explicit S/C manifest preparation for ARISE development."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from bives_cxr.provenance import canonical_json_sha256, file_sha256


ARISE_WEAK_SC_VERSION = "arise_weak_sc_explicit_v1"
STATE_LABELS = {"support": 1, "contradict": 0}


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _deduplicate(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["patient_id"]),
            str(row["canonical_statement_id"]),
            str(row["parser_state_candidate"]),
        )
        previous = selected.get(key)
        if previous is None or _stable_key(seed, str(row["candidate_id"])) < _stable_key(
            seed, str(previous["candidate_id"])
        ):
            selected[key] = row
    return list(selected.values())


def _choose_validation_patients(
    rows: list[dict[str, Any]],
    findings: tuple[str, ...],
    *,
    validation_fraction: float,
    seed: int,
) -> set[str]:
    patients = sorted({str(row["patient_id"]) for row in rows})
    validation_size = max(1, min(len(patients) - 1, round(len(patients) * validation_fraction)))
    strata = [(finding, state) for finding in findings for state in sorted(STATE_LABELS)]
    best: tuple[tuple[int, int], set[str]] | None = None
    for attempt in range(256):
        ordered = sorted(patients, key=lambda value: _stable_key(seed + attempt, value))
        validation = set(ordered[:validation_size])
        train_counts: Counter[tuple[str, str]] = Counter()
        val_counts: Counter[tuple[str, str]] = Counter()
        for row in rows:
            key = (str(row["canonical_statement_id"]), str(row["parser_state_candidate"]))
            (val_counts if str(row["patient_id"]) in validation else train_counts)[key] += 1
        minimum = min(
            *(train_counts[key] for key in strata),
            *(val_counts[key] for key in strata),
        )
        balanced = sum(
            min(train_counts[(finding, "support")], train_counts[(finding, "contradict")])
            + min(val_counts[(finding, "support")], val_counts[(finding, "contradict")])
            for finding in findings
        )
        score = (minimum, balanced)
        if best is None or score > best[0]:
            best = (score, validation)
    if best is None or best[0][0] == 0:
        raise ValueError("could not create patient-disjoint train/val with all S/C strata")
    return best[1]


def _balance_split(
    rows: list[dict[str, Any]],
    findings: tuple[str, ...],
    *,
    split: str,
    seed: int,
    verify_images: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for finding in findings:
        by_state = {
            state: sorted(
                [
                    row
                    for row in rows
                    if row["canonical_statement_id"] == finding
                    and row["parser_state_candidate"] == state
                ],
                key=lambda row: _stable_key(seed, str(row["candidate_id"])),
            )
            for state in STATE_LABELS
        }
        target = min(len(by_state["support"]), len(by_state["contradict"]))
        if target == 0:
            raise ValueError(f"{split}/{finding} has no balanced S/C records")
        for state in ("support", "contradict"):
            for source in by_state[state][:target]:
                image_path = Path(source["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                output.append(
                    {
                        "sample_id": f"arise-weak-sc::{split}::{source['candidate_id']}",
                        "unit_id": str(source["image_id"]),
                        "patient_id": str(source["patient_id"]),
                        "study_id": str(source["study_id"]),
                        "image_id": str(source["image_id"]),
                        "image_path": str(image_path),
                        "image_sha256": file_sha256(image_path) if verify_images else None,
                        "canonical_statement_id": finding,
                        "statement_text": str(source["statement_text"]),
                        "state": state,
                        "binary_label": STATE_LABELS[state],
                        "split": split,
                        "source_candidate_id": str(source["candidate_id"]),
                        "source_report_sha256": str(source["report_sha256"]),
                        "parser_version": str(source["parser_version"]),
                        "parser_rules_sha256": str(source["parser_rules_sha256"]),
                        "parser_cue": str(source["parser_cue"]),
                        "label_source": "explicit_report_cue_rule_parser",
                        "weak_label_claim": "weak_report_supervision_not_clinical_ground_truth",
                        "four_state_claim": False,
                        "bounding_boxes": [],
                    }
                )
    return sorted(output, key=lambda row: row["sample_id"])


def prepare_arise_weak_sc(
    candidates_path: Path,
    output_dir: Path,
    *,
    findings: Iterable[str],
    validation_fraction: float = 0.2,
    seed: int = 20260722,
    verify_images: bool = True,
) -> dict[str, Any]:
    finding_tuple = tuple(sorted({str(value).strip() for value in findings if str(value).strip()}))
    if len(finding_tuple) < 2:
        raise ValueError("ARISE weak S/C preparation requires at least two findings")
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5]")
    candidates = _read_jsonl(candidates_path)
    eligible = _deduplicate(
        (
            row
            for row in candidates
            if str(row.get("canonical_statement_id")) in finding_tuple
            and str(row.get("parser_status")) == "candidate"
            and str(row.get("parser_state_candidate")) in STATE_LABELS
            and str(row.get("review_track")) == "p0_1_explicit_positive_negative"
        ),
        seed,
    )
    validation_patients = _choose_validation_patients(
        eligible,
        finding_tuple,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train = _balance_split(
        [row for row in eligible if str(row["patient_id"]) not in validation_patients],
        finding_tuple,
        split="train",
        seed=seed,
        verify_images=verify_images,
    )
    val = _balance_split(
        [row for row in eligible if str(row["patient_id"]) in validation_patients],
        finding_tuple,
        split="val",
        seed=seed + 1,
        verify_images=verify_images,
    )
    train_patients = {row["patient_id"] for row in train}
    val_patients = {row["patient_id"] for row in val}
    if train_patients & val_patients:
        raise RuntimeError("ARISE weak S/C patient leakage")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "weak_sc_train.jsonl",
        "val": output_dir / "weak_sc_val.jsonl",
    }
    for split, rows in (("train", train), ("val", val)):
        paths[split].write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    f"{row['canonical_statement_id']}|{row['state']}" for row in rows
                ).items()
            )
        )

    lock = {
        "schema_version": ARISE_WEAK_SC_VERSION,
        "status": "ready_for_token_cache" if verify_images else "feasibility_only_unhashed",
        "formal_result": False,
        "confirmatory_evidence": False,
        "test_opened": False,
        "findings": list(finding_tuple),
        "source_candidates_sha256": file_sha256(candidates_path),
        "selection_seed": seed,
        "validation_fraction": validation_fraction,
        "verify_images": verify_images,
        "train_records": len(train),
        "val_records": len(val),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "patient_overlap": 0,
        "train_counts": counts(train),
        "val_counts": counts(val),
        "train_manifest_sha256": file_sha256(paths["train"]),
        "val_manifest_sha256": file_sha256(paths["val"]),
        "claim_boundary": "weak explicit report S/C supervision only",
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    (output_dir / "weak_sc_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock
