"""Build deterministic nonclinical BiVES proxy-P0 train/validation quartets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.audit import audit_manifests
from bives_cxr.provenance import canonical_json_sha256


PROXY_VERSION = "bives_weak_proxy_p0_v1"
MATCHING_VERSION = "bives_weak_proxy_match_v1"
DEFAULT_FINDINGS = ("atelectasis", "consolidation", "pulmonary_edema")
STATES = ("support", "contradict", "uncertain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--findings", nargs="+", default=list(DEFAULT_FINDINGS))
    parser.add_argument("--train-groups-per-finding", type=int, default=2)
    parser.add_argument("--val-groups-per-finding", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: dict[str, Any], seed: int) -> str:
    token = f"{seed}|{row['candidate_id']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def read_candidates(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty parser candidate table: {path}")
    rules = {str(row.get("parser_rules_sha256", "")) for row in rows}
    versions = {str(row.get("parser_version", "")) for row in rows}
    if len(rules) != 1 or "" in rules or len(versions) != 1 or "" in versions:
        raise ValueError("proxy candidates must share one non-empty parser version and rules hash")
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    if "" in candidate_ids:
        raise ValueError("proxy candidates must have non-empty candidate_id values")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("proxy candidate_id values must be globally unique across findings")
    return rows


def make_proxy_insufficient(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("L")
        image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        width, height = image.size
        low = image.resize((16, 16), Image.Resampling.BILINEAR)
        removed = low.resize((width, height), Image.Resampling.BILINEAR)
        removed = removed.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(width, height) / 24.0)))
        median = int(ImageStat.Stat(image).median[0])
        removed = Image.blend(removed, Image.new("L", image.size, median), 0.80)
        extrema = ImageStat.Stat(removed).extrema[0]
        if extrema[0] == extrema[1]:
            raise ValueError(f"synthetic evidence removal became constant for {source}")
        removed.save(destination, format="PNG", optimize=True)
    return {
        "kind": "downsample16_blur_median_blend",
        "max_size": 1024,
        "downsample_size": [16, 16],
        "median_blend": 0.80,
    }


def _select_group(
    pools: dict[str, list[dict[str, Any]]],
    conflict_pool: list[dict[str, Any]],
    split: str,
    patient_split: dict[str, str],
    used_candidates: set[str],
) -> dict[str, dict[str, Any]]:
    candidates_by_state = {**pools, "insufficient": conflict_pool}
    order = ("uncertain", "contradict", "support", "insufficient")

    def search(index: int, selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        if index == len(order):
            return selected
        state = order[index]
        selected_patients = {str(row["patient_id"]) for row in selected.values()}
        selected_studies = {str(row["study_id"]) for row in selected.values()}
        selected_images = {str(row["image_path"]) for row in selected.values()}
        for row in candidates_by_state[state]:
            candidate_id = str(row["candidate_id"])
            patient = str(row["patient_id"])
            if candidate_id in used_candidates or patient_split.get(patient, split) != split:
                continue
            if patient in selected_patients or str(row["study_id"]) in selected_studies or str(row["image_path"]) in selected_images:
                continue
            result = search(index + 1, {**selected, state: row})
            if result is not None:
                return result
        return None

    result = search(0, {})
    if result is None:
        raise RuntimeError(f"cannot form another patient/image/study-disjoint proxy quartet for split={split}")
    for row in result.values():
        used_candidates.add(str(row["candidate_id"]))
        patient_split[str(row["patient_id"])] = split
    return result


def build_proxy_manifests(
    parsed_candidates: Path,
    output_dir: Path,
    findings: list[str],
    train_groups_per_finding: int,
    val_groups_per_finding: int,
    seed: int,
) -> dict[str, Any]:
    if train_groups_per_finding < 1 or val_groups_per_finding < 1:
        raise ValueError("proxy P0 requires at least one train and one validation group per finding")
    rows = read_candidates(parsed_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "synthetic_insufficient"
    by_finding_state: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        finding = str(row.get("canonical_statement_id", ""))
        state = row.get("parser_state_candidate")
        status = str(row.get("parser_status", ""))
        if finding not in findings:
            continue
        bucket = "conflict" if status == "requires_review_conflict" else str(state)
        if bucket in {*STATES, "conflict"}:
            by_finding_state[finding][bucket].append(row)
    for finding in findings:
        for bucket in (*STATES, "conflict"):
            by_finding_state[finding][bucket].sort(key=lambda row: stable_key(row, seed))
            distinct = {str(row["patient_id"]) for row in by_finding_state[finding][bucket]}
            needed = train_groups_per_finding + val_groups_per_finding
            if len(distinct) < needed:
                raise ValueError(f"{finding}/{bucket} has {len(distinct)} patients; needs {needed}")

    patient_split: dict[str, str] = {}
    used_candidates: set[str] = set()
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    groups_per_split = {"val": val_groups_per_finding, "train": train_groups_per_finding}
    group_index = 0
    for split in ("val", "train"):
        for finding in findings:
            pools = {state: by_finding_state[finding][state] for state in STATES}
            conflicts = by_finding_state[finding]["conflict"]
            for _ in range(groups_per_split[split]):
                selected = _select_group(pools, conflicts, split, patient_split, used_candidates)
                group_id = f"proxy-{split}-{finding}-{group_index:04d}"
                group_index += 1
                statement_text = str(selected["support"]["statement_text"])
                stratum = {
                    "source_dataset": "MIMIC-CXR-JPG",
                    "proxy_version": PROXY_VERSION,
                    "finding": finding,
                }
                for state in ("support", "contradict", "uncertain", "insufficient"):
                    source = selected[state]
                    image_path = Path(str(source["image_path"]))
                    row_image_path = image_path
                    transform: dict[str, Any] | None = None
                    if state == "insufficient":
                        row_image_path = image_dir / f"{group_id}-{source['image_id']}.png"
                        transform = make_proxy_insufficient(image_path, row_image_path)
                    manifest_row: dict[str, Any] = {
                        "sample_id": f"{group_id}-{state}",
                        "patient_id": str(source["patient_id"]),
                        "study_id": str(source["study_id"]) + ("-proxy-i" if state == "insufficient" else ""),
                        "image_id": str(source["image_id"]) + ("-proxy-i" if state == "insufficient" else ""),
                        "image_path": str(row_image_path),
                        "image_sha256": file_sha256(row_image_path),
                        "group_id": group_id,
                        "canonical_statement_id": finding,
                        "statement_text": statement_text,
                        "state": state,
                        "label_source": "synthetic_evidence_removal_v1" if state == "insufficient" else "rule_parser_v1",
                        "annotation_status": "weak_proxy_unreviewed",
                        "matching_protocol_version": MATCHING_VERSION,
                        "matching_stratum": stratum,
                        "source_dataset": "MIMIC-CXR-JPG",
                        "finding": finding,
                        "split": split,
                        "proxy_version": PROXY_VERSION,
                        "weak_label_claim": "proxy_only_not_clinical_ground_truth",
                        "source_candidate_id": str(source["candidate_id"]),
                        "source_image_candidate_id": str(
                            source.get("source_image_candidate_id", source["candidate_id"])
                        ),
                        "source_report_sha256": str(source["report_sha256"]),
                        "parser_version": str(source["parser_version"]),
                        "parser_rules_sha256": str(source["parser_rules_sha256"]),
                    }
                    if state == "insufficient":
                        manifest_row.update(
                            {
                                "insufficient_kind": "synthetic",
                                "source_image_path": str(image_path),
                                "source_image_sha256": file_sha256(image_path),
                                "synthetic_transform": transform,
                            }
                        )
                    manifests[split].append(manifest_row)

    manifest_paths: dict[str, str] = {}
    for split, split_rows in manifests.items():
        path = output_dir / f"{split}_proxy.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest_paths[split] = str(path)
    audit_options = {
        "check_images": True,
        "require_complete_statements": True,
        "check_decodable": True,
        "reject_constant_images": True,
        "require_provenance": True,
        "verify_image_sha256": True,
        "require_matching_protocol": True,
        "require_both_insufficient_kinds": False,
    }
    audit = audit_manifests(
        {split: Path(path) for split, path in manifest_paths.items()},
        data_root=Path("."),
        **audit_options,
    )
    if audit["status"] != "pass":
        raise RuntimeError("proxy manifest audit failed: " + "; ".join(audit["errors"][:5]))
    lock = {
        "format_version": 1,
        "kind": "bives_weak_proxy_dataset_lock",
        "status": "pass",
        "formal_result": False,
        "clinical_ground_truth": False,
        "proxy_version": PROXY_VERSION,
        "manifest_sha256": {
            split: file_sha256(Path(path)) for split, path in manifest_paths.items()
        },
        "split_set_sha256": {
            split: {
                "patient_ids": canonical_json_sha256(sorted({row["patient_id"] for row in split_rows})),
                "study_ids": canonical_json_sha256(sorted({row["study_id"] for row in split_rows})),
                "image_hashes": canonical_json_sha256(sorted({row["image_sha256"] for row in split_rows})),
                "group_ids": canonical_json_sha256(sorted({row["group_id"] for row in split_rows})),
            }
            for split, split_rows in manifests.items()
        },
        "audit_options": audit_options,
        "audit_report_sha256": canonical_json_sha256(audit),
    }
    lock_path = output_dir / "proxy_dataset_lock.json"
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = output_dir / "proxy_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "proxy_ready_for_audit",
        "formal_result": False,
        "clinical_ground_truth": False,
        "proxy_version": PROXY_VERSION,
        "parser_candidate_sha256": file_sha256(parsed_candidates),
        "findings": findings,
        "groups": {split: len(rows) // 4 for split, rows in manifests.items()},
        "records": {split: len(rows) for split, rows in manifests.items()},
        "patients": {split: len({row['patient_id'] for row in rows}) for split, rows in manifests.items()},
        "manifests": manifest_paths,
        "audit": str(audit_path),
        "dataset_lock": str(lock_path),
        "dataset_lock_sha256": canonical_json_sha256(lock),
    }
    (output_dir / "proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_proxy_manifests(
        args.parsed_candidates,
        args.output_dir,
        list(args.findings),
        args.train_groups_per_finding,
        args.val_groups_per_finding,
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
