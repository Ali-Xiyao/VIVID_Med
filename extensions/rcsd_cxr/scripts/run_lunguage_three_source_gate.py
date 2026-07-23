"""Formal, patient-held-out G2 gate for three frozen report label sources."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_lunguage_two_source_diagnostic import (
    apply_temperature,
    choose_temperature,
    estimate_likelihoods,
    load_gold,
    load_source,
    patient_fold,
    posterior_for,
    sha256_file,
    summarize,
)


SOURCE_NAMES = ("chexpert", "negbio", "chexbert")


def evaluate_gate(metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    single = {name: metrics[name] for name in SOURCE_NAMES}
    best_f1_name = max(single, key=lambda name: float(single[name]["macro_f1"]))
    best_nll_name = min(single, key=lambda name: float(single[name]["nll"]))
    best_ece_name = min(single, key=lambda name: float(single[name]["ece"]))
    fused = metrics["fusion"]
    f1_reference = float(single[best_f1_name]["macro_f1"])
    nll_reference = float(single[best_nll_name]["nll"])
    ece_reference = float(single[best_ece_name]["ece"])
    f1_gain = float(fused["macro_f1"]) - f1_reference
    nll_relative_change = float(fused["nll"]) / nll_reference - 1.0
    ece_relative_change = (
        float(fused["ece"]) / ece_reference - 1.0
        if ece_reference
        else float("nan")
    )
    thresholds = {
        "macro_f1_gain_at_least": 0.01,
        "nll_relative_change_at_most": -0.05,
        "ece_at_most_or_relative_change_at_most": [0.05, -0.20],
        "reliability_correctness_auroc_at_least": 0.65,
        "confidence_quartile_accuracy_gap_at_least": 0.10,
        "mapped_findings_at_least": 8,
        "predicted_states_required": 3,
    }
    checks = {
        "macro_f1": f1_gain >= thresholds["macro_f1_gain_at_least"],
        "nll": nll_relative_change <= thresholds["nll_relative_change_at_most"],
        "ece": (
            float(fused["ece"])
            <= thresholds["ece_at_most_or_relative_change_at_most"][0]
            or ece_relative_change
            <= thresholds["ece_at_most_or_relative_change_at_most"][1]
        ),
        "reliability": float(fused["reliability_correctness_auroc"])
        >= thresholds["reliability_correctness_auroc_at_least"],
        "quartile_gap": float(
            fused["top_bottom_confidence_quartile_accuracy_gap"]
        )
        >= thresholds["confidence_quartile_accuracy_gap_at_least"],
        "finding_coverage": len(fused["per_finding"])
        >= thresholds["mapped_findings_at_least"],
        "state_coverage": len(fused["predicted_state_counts"])
        == thresholds["predicted_states_required"],
    }
    return {
        "formal": True,
        "reference_sources": {
            "macro_f1": best_f1_name,
            "nll": best_nll_name,
            "ece": best_ece_name,
        },
        "macro_f1_gain": f1_gain,
        "nll_relative_change": nll_relative_change,
        "ece_relative_change": ece_relative_change,
        "thresholds": thresholds,
        "checks": checks,
        "g2_pass": all(checks.values()),
        "failure_action": (
            None
            if all(checks.values())
            else "drop posterior-fusion claim and freeze the best single source"
        ),
    }


def run_gate(
    gold_path: Path,
    chexpert_path: Path,
    negbio_path: Path,
    chexbert_path: Path,
) -> dict[str, object]:
    gold, audit = load_gold(gold_path)
    studies = {key[0] for key in gold}
    sources = {
        "chexpert": load_source(chexpert_path, studies),
        "negbio": load_source(negbio_path, studies),
        "chexbert": load_source(chexbert_path, studies),
    }
    patient_by_study = audit.pop("patient_by_study")
    samples = sorted(
        (study, finding, truth) for (study, finding), truth in gold.items()
    )
    methods = {
        "chexpert": ("chexpert",),
        "negbio": ("negbio",),
        "chexbert": ("chexbert",),
        "fusion": SOURCE_NAMES,
    }
    held_truth = {name: [] for name in methods}
    held_prob = {name: [] for name in methods}
    held_findings = {name: [] for name in methods}
    fold_temperatures: dict[str, list[float]] = defaultdict(list)
    split_counts = []

    # Each outer fold has a disjoint, fixed next-fold calibration surface.
    # Likelihood estimation, temperature calibration, and evaluation patients
    # are therefore disjoint.
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        train = [
            index
            for index, item in enumerate(samples)
            if patient_fold(patient_by_study[item[0]])
            not in {outer_fold, calibration_fold}
        ]
        calibration = [
            index
            for index, item in enumerate(samples)
            if patient_fold(patient_by_study[item[0]]) == calibration_fold
        ]
        test = [
            index
            for index, item in enumerate(samples)
            if patient_fold(patient_by_study[item[0]]) == outer_fold
        ]
        split_counts.append(
            {
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "train_samples": len(train),
                "calibration_samples": len(calibration),
                "test_samples": len(test),
            }
        )
        likelihoods, _, priors = estimate_likelihoods(samples, sources, train)
        for method, names in methods.items():
            calibration_truth = [samples[index][2] for index in calibration]
            calibration_prob = [
                posterior_for(
                    samples[index][0],
                    samples[index][1],
                    names,
                    sources,
                    likelihoods,
                    priors,
                )
                for index in calibration
            ]
            temperature = choose_temperature(
                calibration_truth, calibration_prob
            )
            fold_temperatures[method].append(temperature)
            for index in test:
                study, finding, truth = samples[index]
                probability = posterior_for(
                    study,
                    finding,
                    names,
                    sources,
                    likelihoods,
                    priors,
                )
                held_truth[method].append(truth)
                held_prob[method].append(
                    apply_temperature(probability, temperature)
                )
                held_findings[method].append(finding)

    metrics = {
        name: summarize(
            held_truth[name], held_prob[name], held_findings[name]
        )
        for name in methods
    }
    gate = evaluate_gate(metrics)
    return {
        "schema_version": 1,
        "artifact": "lunguage_three_source_g2_gate",
        "formal": True,
        "protocol": {
            "outer_folds": 5,
            "patient_disjoint": True,
            "calibration": "fixed next patient-hash fold",
            "likelihood_training": "remaining three patient-hash folds",
            "mapping_frozen_before_chexbert_scoring": True,
        },
        "mapping_audit": audit,
        "gold_samples": len(samples),
        "gold_studies": len(studies),
        "source_coverage": {
            name: len(values) for name, values in sources.items()
        },
        "split_counts": split_counts,
        "fold_temperatures": dict(fold_temperatures),
        "metrics": metrics,
        "gate": gate,
        "hashes": {
            "gold": sha256_file(gold_path),
            "chexpert": sha256_file(chexpert_path),
            "negbio": sha256_file(negbio_path),
            "chexbert": sha256_file(chexbert_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--chexpert", required=True, type=Path)
    parser.add_argument("--negbio", required=True, type=Path)
    parser.add_argument("--chexbert", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_gate(
            args.gold, args.chexpert, args.negbio, args.chexbert
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "artifact": "lunguage_three_source_g2_gate",
            "pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        result["pass"] = True
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
