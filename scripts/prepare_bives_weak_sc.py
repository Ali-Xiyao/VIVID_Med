"""Build patient-disjoint balanced weak MIMIC S/C manifests from explicit cues."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "local_runs/bives_cxr/p0_intake/mimic_parser_candidates_5k.jsonl"
DEFAULT_OUTPUT = ROOT / "local_runs/bives_cxr/weak_sc_v1"
FINDINGS = {"pleural_effusion", "consolidation"}
STATES = {"support": 1, "contradict": 0}
FORMAT_VERSION = "bives_weak_sc_explicit_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eligible(row: dict[str, Any]) -> bool:
    return (
        str(row.get("canonical_statement_id")) in FINDINGS
        and str(row.get("parser_status")) == "candidate"
        and str(row.get("parser_state_candidate")) in STATES
        and str(row.get("review_track")) == "p0_1_explicit_positive_negative"
    )


def _deduplicate_patient_strata(
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
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
    validation_fraction: float,
    seed: int,
) -> set[str]:
    patients = sorted({str(row["patient_id"]) for row in rows})
    validation_size = max(1, min(len(patients) - 1, round(len(patients) * validation_fraction)))
    best: tuple[tuple[int, int], set[str]] | None = None
    for attempt in range(256):
        ordered = sorted(patients, key=lambda value: _stable_key(seed + attempt, value))
        validation = set(ordered[:validation_size])
        train_counts: Counter[tuple[str, str]] = Counter()
        val_counts: Counter[tuple[str, str]] = Counter()
        for row in rows:
            key = (
                str(row["canonical_statement_id"]),
                str(row["parser_state_candidate"]),
            )
            (val_counts if str(row["patient_id"]) in validation else train_counts)[key] += 1
        strata = [(finding, state) for finding in sorted(FINDINGS) for state in sorted(STATES)]
        minimum = min(*(train_counts[key] for key in strata), *(val_counts[key] for key in strata))
        total_balanced = sum(
            min(train_counts[(finding, "support")], train_counts[(finding, "contradict")])
            + min(val_counts[(finding, "support")], val_counts[(finding, "contradict")])
            for finding in FINDINGS
        )
        score = (minimum, total_balanced)
        if best is None or score > best[0]:
            best = (score, validation)
    assert best is not None
    if best[0][0] == 0:
        raise ValueError("could not create patient-disjoint train/val with all S/C strata")
    return best[1]


def _balance_split(
    rows: list[dict[str, Any]],
    split: str,
    seed: int,
    verify_images: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for finding in sorted(FINDINGS):
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
            for state in STATES
        }
        target = min(len(by_state["support"]), len(by_state["contradict"]))
        if target == 0:
            raise ValueError(f"{split}/{finding} has no balanced S/C records")
        for state in ("support", "contradict"):
            for source in by_state[state][:target]:
                image_path = Path(source["image_path"])
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image_hash = file_sha256(image_path) if verify_images else None
                output.append(
                    {
                        "sample_id": f"weak-sc::{split}::{source['candidate_id']}",
                        "unit_id": str(source["image_id"]),
                        "patient_id": str(source["patient_id"]),
                        "study_id": str(source["study_id"]),
                        "image_id": str(source["image_id"]),
                        "image_path": str(image_path),
                        "image_sha256": image_hash,
                        "canonical_statement_id": finding,
                        "statement_text": str(source["statement_text"]),
                        "state": state,
                        "binary_label": STATES[state],
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


def prepare_weak_sc(
    candidates_path: Path,
    output_dir: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 17,
    verify_images: bool = True,
) -> dict[str, Any]:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5]")
    candidates = _read_jsonl(candidates_path)
    eligible = _deduplicate_patient_strata(
        [row for row in candidates if _eligible(row)],
        seed,
    )
    validation_patients = _choose_validation_patients(
        eligible,
        validation_fraction,
        seed,
    )
    train_source = [row for row in eligible if str(row["patient_id"]) not in validation_patients]
    val_source = [row for row in eligible if str(row["patient_id"]) in validation_patients]
    train = _balance_split(train_source, "train", seed, verify_images)
    val = _balance_split(val_source, "val", seed + 1, verify_images)
    train_patients = {row["patient_id"] for row in train}
    val_patients = {row["patient_id"] for row in val}
    if train_patients & val_patients:
        raise RuntimeError("weak S/C patient leakage")

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
        counter = Counter(
            f"{row['canonical_statement_id']}|{row['state']}" for row in rows
        )
        return dict(sorted(counter.items()))

    lock = {
        "format_version": FORMAT_VERSION,
        "status": "ready_for_token_cache",
        "source_candidates": str(candidates_path.resolve()),
        "source_candidates_sha256": file_sha256(candidates_path),
        "selection_seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "verify_images": bool(verify_images),
        "eligible_definition": {
            "findings": sorted(FINDINGS),
            "parser_status": "candidate",
            "states": sorted(STATES),
            "review_track": "p0_1_explicit_positive_negative",
            "one_record_per_patient_finding_state": True,
            "report_omission_negative": False,
            "parser_uncertain": False,
            "synthetic_insufficient": False,
        },
        "train_records": len(train),
        "val_records": len(val),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "patient_overlap": 0,
        "train_counts": counts(train),
        "val_counts": counts(val),
        "train_manifest_sha256": file_sha256(paths["train"]),
        "val_manifest_sha256": file_sha256(paths["val"]),
        "formal_result": False,
        "claim_boundary": "weak explicit report S/C supervision only",
    }
    lock_path = output_dir / "weak_sc_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**lock, "lock": str(lock_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-image-sha256", action="store_true")
    args = parser.parse_args()
    result = prepare_weak_sc(
        args.candidates,
        args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        verify_images=not args.skip_image_sha256,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
