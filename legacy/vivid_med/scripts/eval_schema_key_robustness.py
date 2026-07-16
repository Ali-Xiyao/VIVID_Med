"""
Evaluate schema key/order robustness for a trained frozen-LM VIVID model.

For the same image and finding states, compare teacher-forcing token NLL under:
- original: training-time schema order and clinical field names.
- reversed_order: same clinical keys and states, reversed finding key order.
- shuffled_order: same clinical keys and states, deterministic shuffled key order.
- clinical_key_shift: cyclically shifted clinical key names, preserving values.
- generic_keys: replace finding names with field_00, field_01, ...

This is a diagnostic for fixed JSON/key dependency. It is not a training script.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel


VARIANTS = ("reversed_order", "shuffled_order", "clinical_key_shift", "generic_keys")


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


def create_val_loader(config: Dict[str, Any], max_samples: int, batch_size: int, num_workers: int) -> DataLoader:
    data_cfg = config["data"]
    dataset = CheXpertUMSDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg.get("val_ums_path") or data_cfg["train_ums_path"],
        transform=get_val_transforms(data_cfg["image_size"]),
        is_train=False,
        use_common_labels_only=data_cfg.get("use_common_labels_only", False),
        selected_labels=data_cfg.get("selected_labels"),
        max_samples=max_samples,
        json_include_all_labels=data_cfg.get("json_include_all_labels", False),
        json_missing_state=data_cfg.get("json_missing_state"),
        json_null_state=data_cfg.get("json_null_state"),
        dense_subset_top_k=data_cfg.get("val_dense_top_k"),
        dense_subset_min_answerable=data_cfg.get("val_dense_min_answerable"),
        field_query_training=None,
        target_format=data_cfg.get("target_format", "json"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def create_model(config: Dict[str, Any], device: torch.device) -> VIVIDModel:
    model_cfg = config["model"]
    spd_cfg = model_cfg.get("spd", {}) or {}
    return VIVIDModel(
        vit_model_name=model_cfg["vit_model_name"],
        vit_pretrained=model_cfg.get("vit_pretrained", True),
        vit_output_type=model_cfg.get("vit_output_type", "cls"),
        num_prefix_tokens=model_cfg["num_prefix_tokens"],
        projector_dropout=model_cfg.get("projector_dropout", 0.1),
        projector_mlp_hidden_dim=model_cfg.get("projector_mlp_hidden_dim"),
        llm_model_name=model_cfg["llm_model_name"],
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        max_text_length=model_cfg.get("max_text_length", 512),
        load_llm=True,
        spd_enabled=bool(spd_cfg.get("enabled", False)),
        spd_num_groups=int(spd_cfg.get("num_groups", 3)),
        spd_tokens_per_group=int(spd_cfg.get("tokens_per_group", 2)),
    ).to(device)


def load_checkpoint(model: VIVIDModel, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    model.vit.load_state_dict(state["vit"])
    model.projector.load_state_dict(state["projector"])


def prompt_from_config(config: Dict[str, Any]) -> str:
    prompt_cfg = config.get("prompt", {})
    if "template" in prompt_cfg:
        return prompt_cfg["template"]
    if config["data"].get("target_format", "json") == "text":
        return "Describe the findings in this chest X-ray:\n"
    return "Generate a structured medical report:\n"


def per_sample_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[List[float], List[int], List[float]]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    batch_size = shift_logits.shape[0]

    flat_losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(batch_size, -1)
    valid = shift_labels != -100
    loss_sums = (flat_losses * valid.float()).sum(dim=1)
    token_counts = valid.sum(dim=1).clamp_min(1)
    nlls = loss_sums / token_counts
    return (
        [float(x) for x in nlls.detach().cpu()],
        [int(x) for x in valid.sum(dim=1).detach().cpu()],
        [float(x) for x in loss_sums.detach().cpu()],
    )


@torch.no_grad()
def score_targets(
    model: VIVIDModel,
    images: torch.Tensor,
    target_texts: List[str],
    prompt_text: str,
) -> Tuple[List[float], List[int], List[float]]:
    outputs = model(images=images, prompt_text=prompt_text, target_text=target_texts)
    return per_sample_token_nll(outputs["logits"], outputs["labels"])


def dump_schema(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False)


def reorder_findings(schema: Dict[str, Any], order: List[str]) -> Dict[str, Any]:
    mutated = copy.deepcopy(schema)
    findings = schema.get("findings", {})
    mutated["findings"] = {name: copy.deepcopy(findings[name]) for name in order}
    return mutated


def make_variant(target_json: str, variant: str, sample_index: int, seed: int) -> str:
    schema = json.loads(target_json)
    findings = schema.get("findings", {})
    keys = list(findings.keys())

    if variant == "reversed_order":
        return dump_schema(reorder_findings(schema, list(reversed(keys))))

    if variant == "shuffled_order":
        shuffled = list(keys)
        rng = random.Random(seed + sample_index * 9973)
        rng.shuffle(shuffled)
        return dump_schema(reorder_findings(schema, shuffled))

    if variant == "clinical_key_shift":
        if len(keys) <= 1:
            return dump_schema(schema)
        shifted = keys[1:] + keys[:1]
        mutated = copy.deepcopy(schema)
        mutated["findings"] = {
            new_name: copy.deepcopy(findings[old_name])
            for old_name, new_name in zip(keys, shifted)
        }
        return dump_schema(mutated)

    if variant == "generic_keys":
        mutated = copy.deepcopy(schema)
        mutated["findings"] = {
            f"field_{idx:02d}": copy.deepcopy(findings[name])
            for idx, name in enumerate(keys)
        }
        return dump_schema(mutated)

    raise ValueError(f"Unknown variant: {variant}")


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
    parser = argparse.ArgumentParser(description="Evaluate schema key/order robustness for VIVID")
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
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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

    original_stats = {"loss_sum": 0.0, "token_count": 0}
    rows: List[Dict[str, Any]] = []
    sample_count = 0

    for batch in tqdm(loader, desc="Schema key robustness"):
        images = batch["images"].to(device)
        target_jsons = batch["target_jsons"]
        batch_size = len(target_jsons)
        sample_ids = batch.get("sample_ids") or [None] * batch_size
        original_paths = batch.get("original_paths") or [None] * batch_size

        original_nlls, original_token_counts, original_loss_sums = score_targets(
            model, images, target_jsons, prompt_text
        )
        original_stats["loss_sum"] += float(sum(original_loss_sums))
        original_stats["token_count"] += int(sum(original_token_counts))

        global_indices = list(range(sample_count, sample_count + batch_size))
        sample_count += batch_size

        variant_scores: Dict[str, Tuple[List[float], List[int], List[float]]] = {}
        for variant in VARIANTS:
            variant_targets = [
                make_variant(target_json, variant, sample_index=global_idx, seed=args.seed)
                for target_json, global_idx in zip(target_jsons, global_indices)
            ]
            variant_scores[variant] = score_targets(model, images, variant_targets, prompt_text)

        for local_idx, global_idx in enumerate(global_indices):
            for variant in VARIANTS:
                variant_nlls, variant_token_counts, variant_loss_sums = variant_scores[variant]
                margin = variant_nlls[local_idx] - original_nlls[local_idx]
                rows.append(
                    {
                        "sample_index": global_idx,
                        "sample_id": sample_ids[local_idx],
                        "original_path": original_paths[local_idx],
                        "variant": variant,
                        "original_nll": original_nlls[local_idx],
                        "variant_nll": variant_nlls[local_idx],
                        "margin": margin,
                        "original_token_count": original_token_counts[local_idx],
                        "variant_token_count": variant_token_counts[local_idx],
                        "variant_loss_sum": variant_loss_sums[local_idx],
                    }
                )

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "prompt_text": prompt_text,
        "summary": summarize(rows, sample_count, original_stats),
        "rows": rows,
    }
    save_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
