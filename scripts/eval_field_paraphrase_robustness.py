"""
Evaluate field paraphrase robustness for a trained frozen-LM VIVID model.

For the same image and finding states, compare teacher-forcing token NLL under:
- original: training-time clinical field names.
- clinical_paraphrase: medically plausible field synonyms.
- lay_paraphrase: less formal but semantically related field descriptions.

This is an EMNLP-style diagnostic for whether schema supervision is robust to
semantic paraphrases or mainly tied to the exact training-time field tokens.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_schema_key_robustness import (
    create_model,
    create_val_loader,
    load_checkpoint,
    prompt_from_config,
    score_targets,
)


VARIANTS = ("clinical_paraphrase", "lay_paraphrase")

CLINICAL_PARAPHRASES = {
    "No Finding": "No acute cardiopulmonary abnormality",
    "Enlarged Cardiomediastinum": "Widened cardiomediastinal silhouette",
    "Cardiomegaly": "Enlarged cardiac silhouette",
    "Lung Opacity": "Pulmonary opacity",
    "Lung Lesion": "Focal pulmonary lesion",
    "Edema": "Pulmonary edema",
    "Consolidation": "Airspace consolidation",
    "Pneumonia": "Pulmonary infection",
    "Atelectasis": "Subsegmental atelectatic change",
    "Pneumothorax": "Pleural air collection",
    "Pleural Effusion": "Pleural fluid",
    "Pleural Other": "Other pleural abnormality",
    "Fracture": "Osseous fracture",
    "Support Devices": "Medical support devices",
}

LAY_PARAPHRASES = {
    "No Finding": "No visible problem",
    "Enlarged Cardiomediastinum": "Wide central chest shadow",
    "Cardiomegaly": "Large heart",
    "Lung Opacity": "Cloudy lung area",
    "Lung Lesion": "Spot in the lung",
    "Edema": "Fluid in the lungs",
    "Consolidation": "Dense lung patch",
    "Pneumonia": "Lung infection",
    "Atelectasis": "Collapsed lung area",
    "Pneumothorax": "Air outside the lung",
    "Pleural Effusion": "Fluid around the lung",
    "Pleural Other": "Other lining abnormality",
    "Fracture": "Broken bone",
    "Support Devices": "Tubes or lines",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_schema(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False)


def rename_keys_preserving_order(values: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    renamed: Dict[str, Any] = {}
    for key, value in values.items():
        renamed[mapping.get(key, key)] = copy.deepcopy(value)
    return renamed


def paraphrase_schema(target_json: str, variant: str) -> str:
    schema = json.loads(target_json)
    mapping = CLINICAL_PARAPHRASES if variant == "clinical_paraphrase" else LAY_PARAPHRASES
    mutated = copy.deepcopy(schema)

    for section in ("findings", "answerability", "uncertainty", "provenance"):
        values = mutated.get(section)
        if isinstance(values, dict):
            mutated[section] = rename_keys_preserving_order(values, mapping)

    return dump_schema(mutated)


def summarize(rows: List[Dict[str, Any]], sample_count: int, original_stats: Dict[str, float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "original": {
            "n": sample_count,
            "loss_sum": original_stats["loss_sum"],
            "token_count": int(original_stats["token_count"]),
            "nll": original_stats["loss_sum"] / max(int(original_stats["token_count"]), 1),
        }
    }

    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            result[variant] = {"n": 0}
            continue
        margins = np.array([row["margin"] for row in variant_rows], dtype=np.float64)
        variant_nlls = np.array([row["variant_nll"] for row in variant_rows], dtype=np.float64)
        original_nlls = np.array([row["original_nll"] for row in variant_rows], dtype=np.float64)
        token_counts = np.array([row["variant_token_count"] for row in variant_rows], dtype=np.float64)
        original_better = np.array([1.0 if row["original_nll"] < row["variant_nll"] else 0.0 for row in variant_rows])

        result[variant] = {
            "n": len(variant_rows),
            "variant_nll_mean": float(variant_nlls.mean()),
            "original_nll_mean": float(original_nlls.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "original_better_rate": float(original_better.mean()),
            "variant_token_count_mean": float(token_counts.mean()),
            "relative_delta_vs_original": float(margins.mean() / original_nlls.mean()) if original_nlls.mean() else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate field paraphrase robustness for VIVID")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_yaml(args.config)
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = create_val_loader(
        config=config,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = create_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    prompt_text = prompt_from_config(config)

    rows: List[Dict[str, Any]] = []
    original_loss_sum = 0.0
    original_token_count = 0
    sample_count = 0

    for batch in tqdm(loader, desc="Field paraphrase eval"):
        images = batch["images"].to(device)
        target_jsons = batch["target_jsons"]
        sample_ids = batch["sample_ids"]
        original_paths = batch["original_paths"]
        sample_count += len(target_jsons)

        original_nlls, original_tokens, original_loss_sums = score_targets(model, images, target_jsons, prompt_text)
        original_loss_sum += sum(original_loss_sums)
        original_token_count += sum(original_tokens)

        for variant in VARIANTS:
            paraphrased = [paraphrase_schema(text, variant) for text in target_jsons]
            variant_nlls, variant_tokens, _ = score_targets(model, images, paraphrased, prompt_text)
            for i, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "variant": variant,
                        "sample_id": sample_id,
                        "original_path": original_paths[i],
                        "original_nll": float(original_nlls[i]),
                        "variant_nll": float(variant_nlls[i]),
                        "variant_token_count": int(variant_tokens[i]),
                        "margin": float(variant_nlls[i] - original_nlls[i]),
                    }
                )

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "prompt_text": prompt_text,
        "paraphrase_maps": {
            "clinical_paraphrase": CLINICAL_PARAPHRASES,
            "lay_paraphrase": LAY_PARAPHRASES,
        },
        "summary": summarize(
            rows,
            sample_count=sample_count,
            original_stats={"loss_sum": original_loss_sum, "token_count": original_token_count},
        ),
        "rows": rows,
    }
    save_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
