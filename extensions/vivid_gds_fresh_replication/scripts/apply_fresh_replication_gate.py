"""Apply the frozen three-seed patient-bootstrap replication gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ARMS = ("A0_direct", "A2_ums", "A3_gds")
COMPARISONS = {
    "A3_minus_A0": ("A3_gds", "A0_direct"),
    "A3_minus_A2": ("A3_gds", "A2_ums"),
}


def load_predictions(
    path: Path, findings: list[str]
) -> dict[str, np.ndarray | list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "patients": [str(row["patient_id"]) for row in rows],
        "paths": [str(row["image_path"]) for row in rows],
        "labels": np.asarray(
            [
                [
                    float(row[f"{finding}_label"])
                    if str(row[f"{finding}_label"]).strip()
                    else np.nan
                    for finding in findings
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "scores": np.asarray(
            [
                [float(row[f"{finding}_score"]) for finding in findings]
                for row in rows
            ],
            dtype=np.float64,
        ),
    }


def metrics(
    labels: np.ndarray, scores: np.ndarray, findings: list[str]
) -> dict[str, object]:
    per_finding: dict[str, dict[str, float]] = {}
    aucs: list[float] = []
    auprcs: list[float] = []
    for index, finding in enumerate(findings):
        valid = np.isfinite(labels[:, index])
        truth = labels[valid, index]
        probability = scores[valid, index]
        if len(np.unique(truth)) != 2:
            raise ValueError(f"degenerate bootstrap labels for {finding}")
        auc = float(roc_auc_score(truth, probability))
        auprc = float(average_precision_score(truth, probability))
        per_finding[finding] = {"auroc": auc, "auprc": auprc}
        aucs.append(auc)
        auprcs.append(auprc)
    return {
        "macro_auroc": float(np.mean(aucs)),
        "macro_auprc": float(np.mean(auprcs)),
        "per_finding": per_finding,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    findings = list(lock["data"]["findings"])
    seeds = [int(seed) for seed in lock["pretraining"]["seeds"]]
    payloads: dict[tuple[int, str], dict[str, object]] = {}
    summaries: dict[tuple[int, str], dict[str, object]] = {}
    for seed in seeds:
        for arm in ARMS:
            root = args.probe_root / f"seed{seed}" / arm
            payloads[(seed, arm)] = load_predictions(
                root / "fresh_predictions.csv", findings
            )
            summaries[(seed, arm)] = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )

    reference = payloads[(seeds[0], ARMS[0])]
    reference_patients = list(reference["patients"])
    reference_paths = list(reference["paths"])
    reference_labels = np.asarray(reference["labels"])
    for key, payload in payloads.items():
        if (
            list(payload["patients"]) != reference_patients
            or list(payload["paths"]) != reference_paths
            or not np.array_equal(
                np.asarray(payload["labels"]),
                reference_labels,
                equal_nan=True,
            )
        ):
            raise ValueError(f"unpaired prediction identity for {key}")
        if summaries[key]["seed"] != key[0] or summaries[key]["arm"] != key[1]:
            raise ValueError(f"summary identity mismatch for {key}")

    observed: dict[str, object] = {"seeds": {}, "comparisons": {}}
    seed_metrics: dict[tuple[int, str], dict[str, object]] = {}
    for seed in seeds:
        observed["seeds"][str(seed)] = {}
        for arm in ARMS:
            value = metrics(
                reference_labels,
                np.asarray(payloads[(seed, arm)]["scores"]),
                findings,
            )
            seed_metrics[(seed, arm)] = value
            observed["seeds"][str(seed)][arm] = value

    for name, (candidate, baseline) in COMPARISONS.items():
        seed_deltas = []
        for seed in seeds:
            left = seed_metrics[(seed, candidate)]
            right = seed_metrics[(seed, baseline)]
            seed_deltas.append(
                {
                    "seed": seed,
                    "macro_auroc": left["macro_auroc"] - right["macro_auroc"],
                    "macro_auprc": left["macro_auprc"] - right["macro_auprc"],
                    "per_finding_auroc": {
                        finding: (
                            left["per_finding"][finding]["auroc"]
                            - right["per_finding"][finding]["auroc"]
                        )
                        for finding in findings
                    },
                }
            )
        observed["comparisons"][name] = {
            "seed_deltas": seed_deltas,
            "mean_macro_auroc": float(
                np.mean([row["macro_auroc"] for row in seed_deltas])
            ),
            "mean_macro_auprc": float(
                np.mean([row["macro_auprc"] for row in seed_deltas])
            ),
            "mean_per_finding_auroc": {
                finding: float(
                    np.mean(
                        [
                            row["per_finding_auroc"][finding]
                            for row in seed_deltas
                        ]
                    )
                )
                for finding in findings
            },
        }

    patients_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(reference_patients):
        patients_to_indices[patient].append(index)
    unique_patients = np.asarray(sorted(patients_to_indices))
    promotion = lock["promotion"]
    replicates = int(promotion["primary_bootstrap"]["replicates"])
    rng = np.random.default_rng(int(promotion["primary_bootstrap"]["seed"]))
    bootstrap: dict[str, list[float]] = {
        name: [] for name in COMPARISONS
    }
    for _ in range(replicates):
        sampled = rng.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        indices = np.concatenate(
            [np.asarray(patients_to_indices[str(patient)]) for patient in sampled]
        )
        labels = reference_labels[indices]
        for name, (candidate, baseline) in COMPARISONS.items():
            seed_values = []
            for seed in seeds:
                left = metrics(
                    labels,
                    np.asarray(payloads[(seed, candidate)]["scores"])[indices],
                    findings,
                )
                right = metrics(
                    labels,
                    np.asarray(payloads[(seed, baseline)]["scores"])[indices],
                    findings,
                )
                seed_values.append(left["macro_auroc"] - right["macro_auroc"])
            bootstrap[name].append(float(np.mean(seed_values)))
    cis = {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "replicates": len(values),
        }
        for name, values in bootstrap.items()
    }

    checks: dict[str, bool] = {}
    for name in COMPARISONS:
        comparison = observed["comparisons"][name]
        checks[f"{name}_mean_auroc"] = (
            comparison["mean_macro_auroc"]
            >= promotion["mean_macro_auroc_delta_min"][name]
        )
        checks[f"{name}_all_seeds_positive"] = all(
            row["macro_auroc"] > 0
            for row in comparison["seed_deltas"]
        )
        checks[f"{name}_mean_auprc"] = (
            comparison["mean_macro_auprc"]
            >= promotion["mean_macro_auprc_delta_min"][name]
        )
        checks[f"{name}_per_finding_floor"] = all(
            value >= promotion["per_finding_mean_auroc_delta_min"]
            for value in comparison["mean_per_finding_auroc"].values()
        )
    primary = str(promotion["primary_bootstrap"]["comparison"])
    checks["primary_bootstrap_lower_bound"] = (
        cis[primary]["lower"]
        > promotion["primary_bootstrap"]["lower_bound_strictly_above"]
    )
    passed = all(checks.values())
    a3_a0 = observed["comparisons"]["A3_minus_A0"]
    a3_a2 = observed["comparisons"]["A3_minus_A2"]
    if passed:
        verdict = "REPLICATION_PASS"
    elif (
        a3_a2["mean_macro_auroc"]
        >= promotion["mean_macro_auroc_delta_min"]["A3_minus_A2"]
        and all(
            row["macro_auroc"] > 0 for row in a3_a2["seed_deltas"]
        )
        and not checks["A3_minus_A0_mean_auroc"]
    ):
        verdict = "A3_BEATS_A2_NOT_A0"
    else:
        verdict = "TERMINAL_NO_GO"
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_fresh_replication_verdict",
        "pass": passed,
        "verdict": verdict,
        "checks": checks,
        "observed": observed,
        "patient_bootstrap_auroc": cis,
        "rows": len(reference_paths),
        "patients": len(unique_patients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
