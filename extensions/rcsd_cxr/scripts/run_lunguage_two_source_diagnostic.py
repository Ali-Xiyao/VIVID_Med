"""Five-fold diagnostic for official CheXpert/NegBio against LUNGUAGE gold."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from rcsd_cxr.gold_mapping import FINDINGS, aggregate_study_gold


SOURCE_VALUE = {"0": 0, "0.0": 0, "1": 1, "1.0": 1, "-1": 2, "-1.0": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gold(path: Path) -> tuple[dict[tuple[str, str], int], dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return aggregate_study_gold(csv.DictReader(handle))


def load_source(path: Path, studies: set[str]) -> dict[tuple[str, str], int]:
    if path.suffix == ".gz":
        handle_context = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle_context = path.open("r", encoding="utf-8", newline="")
    with handle_context as handle:
        reader = csv.DictReader(handle)
        required = {"study_id", *FINDINGS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"source table missing columns: {sorted(missing)}")
        result: dict[tuple[str, str], int] = {}
        for row in reader:
            study_id = str(row["study_id"]).strip()
            if study_id not in studies:
                continue
            for finding in FINDINGS:
                value = str(row[finding] or "").strip()
                if value in SOURCE_VALUE:
                    result[(study_id, finding)] = SOURCE_VALUE[value]
    return result


def patient_fold(patient_id: str, folds: int = 5) -> int:
    token = hashlib.sha256(patient_id.encode("utf-8")).digest()
    return int.from_bytes(token[:4], "big") % folds


def estimate_likelihoods(
    samples: list[tuple[str, str, int]],
    sources: dict[str, dict[tuple[str, str], int]],
    train_indices: list[int],
    kappa: float = 50.0,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    pooled = {name: np.ones((3, 3), dtype=np.float64) for name in sources}
    local = {(name, finding): np.ones((3, 3), dtype=np.float64) for name in sources for finding in FINDINGS}
    priors = {finding: np.ones(3, dtype=np.float64) for finding in FINDINGS}
    for index in train_indices:
        study, finding, truth = samples[index]
        priors[finding][truth] += 1
        for name, values in sources.items():
            observed = values.get((study, finding))
            if observed is not None:
                pooled[name][truth, observed] += 1
                local[(name, finding)][truth, observed] += 1
    pooled_prob = {name: value / value.sum(axis=1, keepdims=True) for name, value in pooled.items()}
    likelihoods: dict[tuple[str, str], np.ndarray] = {}
    for key, counts in local.items():
        n = max(float(counts.sum() - 9.0), 0.0)
        rho = n / (n + kappa)
        local_prob = counts / counts.sum(axis=1, keepdims=True)
        likelihoods[key] = rho * local_prob + (1.0 - rho) * pooled_prob[key[0]]
    prior_prob = {name: value / value.sum() for name, value in priors.items()}
    return likelihoods, pooled_prob, prior_prob


def posterior_for(
    study: str,
    finding: str,
    source_names: tuple[str, ...],
    sources: dict[str, dict[tuple[str, str], int]],
    likelihoods: dict[tuple[str, str], np.ndarray],
    priors: dict[str, np.ndarray],
) -> np.ndarray:
    logits = np.log(priors[finding] + 1e-12)
    observed_count = 0
    for name in source_names:
        observed = sources[name].get((study, finding))
        if observed is None:
            continue
        logits += np.log(likelihoods[(name, finding)][:, observed] + 1e-12)
        observed_count += 1
    if not observed_count:
        return priors[finding].copy()
    logits -= logits.max()
    prob = np.exp(logits)
    return prob / prob.sum()


def apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(prob, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    value = np.exp(logits)
    return value / value.sum()


def choose_temperature(y_true: list[int], probabilities: list[np.ndarray]) -> float:
    candidates = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    return min(
        candidates,
        key=lambda value: log_loss(
            y_true,
            np.stack([apply_temperature(item, value) for item in probabilities]),
            labels=[0, 1, 2],
        ),
    )


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    value = 0.0
    for low in np.linspace(0.0, 1.0, bins, endpoint=False):
        high = low + 1.0 / bins
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            value += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def summarize(y_true: list[int], probabilities: list[np.ndarray], finding_names: list[str]) -> dict[str, object]:
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.stack(probabilities)
    pred = probs.argmax(axis=1)
    per_finding = {}
    for finding in FINDINGS:
        mask = np.asarray([item == finding for item in finding_names])
        if not mask.any():
            continue
        per_finding[finding] = {
            "n": int(mask.sum()),
            "state_counts": dict(Counter(int(v) for v in truth[mask])),
            "macro_f1": float(f1_score(truth[mask], pred[mask], labels=[0, 1, 2], average="macro", zero_division=0)),
        }
    confidence = probs.max(axis=1)
    correct = (pred == truth).astype(np.int64)
    reliability_auc = float("nan")
    if len(set(correct.tolist())) == 2:
        reliability_auc = float(roc_auc_score(correct, confidence))
    order = np.argsort(confidence)
    quartile = max(len(order) // 4, 1)
    quartile_gap = float(correct[order[-quartile:]].mean() - correct[order[:quartile]].mean())
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "nll": float(log_loss(truth, probs, labels=[0, 1, 2])),
        "ece": ece_score(truth, probs),
        "predicted_state_counts": dict(Counter(int(v) for v in pred)),
        "reliability_correctness_auroc": reliability_auc,
        "top_bottom_confidence_quartile_accuracy_gap": quartile_gap,
        "per_finding": per_finding,
    }


def run_diagnostic(gold_path: Path, chexpert_path: Path, negbio_path: Path) -> dict[str, object]:
    gold, audit = load_gold(gold_path)
    studies = {key[0] for key in gold}
    sources = {
        "chexpert": load_source(chexpert_path, studies),
        "negbio": load_source(negbio_path, studies),
    }
    patient_by_study = audit.pop("patient_by_study")
    samples = sorted((study, finding, truth) for (study, finding), truth in gold.items())
    methods = {"chexpert": ("chexpert",), "negbio": ("negbio",), "fusion": ("chexpert", "negbio")}
    held_truth = {name: [] for name in methods}
    held_prob = {name: [] for name in methods}
    held_findings = {name: [] for name in methods}
    fold_temperatures = defaultdict(list)
    for fold in range(5):
        train = [i for i, item in enumerate(samples) if patient_fold(patient_by_study[item[0]]) != fold]
        test = [i for i, item in enumerate(samples) if patient_fold(patient_by_study[item[0]]) == fold]
        likelihoods, _, priors = estimate_likelihoods(samples, sources, train)
        for method, names in methods.items():
            train_truth = [samples[i][2] for i in train]
            train_prob = [posterior_for(samples[i][0], samples[i][1], names, sources, likelihoods, priors) for i in train]
            temperature = choose_temperature(train_truth, train_prob)
            fold_temperatures[method].append(temperature)
            for i in test:
                study, finding, truth = samples[i]
                prob = posterior_for(study, finding, names, sources, likelihoods, priors)
                held_truth[method].append(truth)
                held_prob[method].append(apply_temperature(prob, temperature))
                held_findings[method].append(finding)
    metrics = {
        name: summarize(held_truth[name], held_prob[name], held_findings[name])
        for name in methods
    }
    best_name = max(("chexpert", "negbio"), key=lambda name: metrics[name]["macro_f1"])
    best = metrics[best_name]
    fused = metrics["fusion"]
    gate = {
        "diagnostic_only": True,
        "best_single_source": best_name,
        "macro_f1_gain": fused["macro_f1"] - best["macro_f1"],
        "nll_relative_change": fused["nll"] / best["nll"] - 1.0,
        "ece_relative_change": fused["ece"] / best["ece"] - 1.0 if best["ece"] else float("nan"),
        "candidate_thresholds": {
            "macro_f1_gain_at_least": 0.01,
            "nll_relative_change_at_most": -0.05,
            "ece_at_most_or_relative_change_at_most": [0.05, -0.20],
        },
    }
    gate["two_source_candidate_pass"] = bool(
        gate["macro_f1_gain"] >= 0.01
        and gate["nll_relative_change"] <= -0.05
        and (fused["ece"] <= 0.05 or gate["ece_relative_change"] <= -0.20)
        and len(fused["predicted_state_counts"]) == 3
    )
    return {
        "schema_version": 1,
        "artifact": "lunguage_two_source_diagnostic",
        "source_scope": "official CheXpert and NegBio only; not the final G2 gate",
        "mapping_audit": audit,
        "gold_samples": len(samples),
        "gold_studies": len(studies),
        "source_coverage": {name: len(value) for name, value in sources.items()},
        "fold_temperatures": dict(fold_temperatures),
        "metrics": metrics,
        "gate": gate,
        "hashes": {
            "gold": sha256_file(gold_path),
            "chexpert": sha256_file(chexpert_path),
            "negbio": sha256_file(negbio_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--chexpert", required=True, type=Path)
    parser.add_argument("--negbio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_diagnostic(args.gold, args.chexpert, args.negbio)
    except Exception as error:
        result = {"schema_version": 1, "artifact": "lunguage_two_source_diagnostic", "pass": False, "error_type": type(error).__name__, "error": str(error)}
    else:
        result["pass"] = True
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
