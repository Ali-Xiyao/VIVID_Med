"""
Counterfactual schema grounding evaluation for VIVID.

For each image and positive UMS target z+, construct counterfactual targets z-:
- state_flip: flip one answerable finding state.
- field_swap: swap states between two answerable findings.
- image_swap: score another sample's schema against the current image.
- null_to_present: corrupt one unanswerable/null finding into present.

The frozen-LM schema scorer should assign lower token NLL to z+ than z-.
This is an EMNLP-style grounding diagnostic, not a training script.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel


STATE_SET = {"present", "absent", "uncertain"}


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


def parse_schema(target_json: str) -> Dict[str, Any]:
    return json.loads(target_json)


def dump_schema(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False)


def flip_state(state: Optional[str], rng: random.Random) -> str:
    if state == "present":
        return "absent"
    if state == "absent":
        return "present"
    if state == "uncertain":
        return rng.choice(["present", "absent"])
    return "present"


def finding_states(schema: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    findings = schema.get("findings", {})
    rows = []
    for name, item in findings.items():
        if isinstance(item, dict):
            rows.append((name, item.get("state")))
    return rows


def make_state_flip(target_json: str, rng: random.Random) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    schema = parse_schema(target_json)
    candidates = [(name, state) for name, state in finding_states(schema) if state in STATE_SET]
    if not candidates:
        return None, None
    name, state = rng.choice(candidates)
    mutated = copy.deepcopy(schema)
    mutated["findings"][name]["state"] = flip_state(state, rng)
    meta = {"field": name, "original_state": state, "new_state": mutated["findings"][name]["state"]}
    return dump_schema(mutated), meta


def make_field_swap(target_json: str, rng: random.Random) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    schema = parse_schema(target_json)
    candidates = [(name, state) for name, state in finding_states(schema) if state in STATE_SET]
    pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:] if a[1] != b[1]]
    if not pairs:
        return None, None
    (name_a, state_a), (name_b, state_b) = rng.choice(pairs)
    mutated = copy.deepcopy(schema)
    mutated["findings"][name_a]["state"] = state_b
    mutated["findings"][name_b]["state"] = state_a
    meta = {
        "field_a": name_a,
        "field_b": name_b,
        "state_a": state_a,
        "state_b": state_b,
    }
    return dump_schema(mutated), meta


def make_null_to_present(target_json: str, rng: random.Random) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    schema = parse_schema(target_json)
    candidates = [(name, state) for name, state in finding_states(schema) if state is None]
    if not candidates:
        return None, None
    name, state = rng.choice(candidates)
    mutated = copy.deepcopy(schema)
    mutated["findings"][name]["state"] = "present"
    meta = {"field": name, "original_state": state, "new_state": "present"}
    return dump_schema(mutated), meta


def summarize_variant(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return {"n": 0}
    margins = np.array([row["margin"] for row in valid], dtype=np.float64)
    positives = np.array([row["positive_nll"] for row in valid], dtype=np.float64)
    negatives = np.array([row["negative_nll"] for row in valid], dtype=np.float64)
    correct = np.array([1.0 if row["correct"] else 0.0 for row in valid], dtype=np.float64)

    result = {
        "n": len(valid),
        "pairwise_accuracy": float(correct.mean()),
        "positive_nll_mean": float(positives.mean()),
        "negative_nll_mean": float(negatives.mean()),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
    }

    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for row in valid:
        field = row.get("field")
        if field:
            by_field.setdefault(field, []).append(row)
    if by_field:
        result["per_field"] = {}
        for field, field_rows in sorted(by_field.items()):
            field_correct = [1.0 if row["correct"] else 0.0 for row in field_rows]
            field_margins = [float(row["margin"]) for row in field_rows]
            result["per_field"][field] = {
                "n": len(field_rows),
                "pairwise_accuracy": float(np.mean(field_correct)),
                "mean_margin": float(np.mean(field_margins)),
            }
    return result


def build_rows(
    variant: str,
    sample_ids: List[Any],
    original_paths: List[str],
    positive_nlls: List[float],
    negative_nlls: List[Optional[float]],
    metas: List[Optional[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows = []
    for i, sample_id in enumerate(sample_ids):
        valid = negative_nlls[i] is not None
        row = {
            "variant": variant,
            "sample_id": sample_id,
            "original_path": original_paths[i],
            "valid": bool(valid),
            "positive_nll": positive_nlls[i],
            "negative_nll": None if negative_nlls[i] is None else float(negative_nlls[i]),
            "margin": None,
            "correct": None,
        }
        if metas[i]:
            row.update(metas[i])
            if "field_a" in metas[i] and "field_b" in metas[i]:
                row["field"] = f"{metas[i]['field_a']}|{metas[i]['field_b']}"
        if valid:
            row["margin"] = float(negative_nlls[i] - positive_nlls[i])
            row["correct"] = bool(positive_nlls[i] < negative_nlls[i])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual schema grounding")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
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

    all_rows: Dict[str, List[Dict[str, Any]]] = {
        "state_flip": [],
        "field_swap": [],
        "image_swap": [],
        "null_to_present": [],
    }
    positive_nll_all: List[float] = []
    positive_tokens_all: List[int] = []
    sample_count = 0

    for batch in tqdm(loader, desc="Counterfactual eval"):
        images = batch["images"].to(device)
        target_jsons = batch["target_jsons"]
        sample_ids = batch["sample_ids"]
        original_paths = batch["original_paths"]
        batch_size = len(target_jsons)
        sample_count += batch_size

        positive_nlls, positive_tokens, _ = score_targets(model, images, target_jsons, prompt_text)
        positive_nll_all.extend(positive_nlls)
        positive_tokens_all.extend(positive_tokens)

        for variant, maker in (
            ("state_flip", make_state_flip),
            ("field_swap", make_field_swap),
            ("null_to_present", make_null_to_present),
        ):
            neg_texts: List[str] = []
            valid_indices: List[int] = []
            metas: List[Optional[Dict[str, Any]]] = [None] * batch_size
            neg_nlls: List[Optional[float]] = [None] * batch_size
            for i, target_json in enumerate(target_jsons):
                neg_text, meta = maker(target_json, rng)
                metas[i] = meta
                if neg_text is not None:
                    neg_texts.append(neg_text)
                    valid_indices.append(i)
            if neg_texts:
                neg_images = images[torch.tensor(valid_indices, device=device)]
                scored, _, _ = score_targets(model, neg_images, neg_texts, prompt_text)
                for idx, nll in zip(valid_indices, scored):
                    neg_nlls[idx] = nll
            all_rows[variant].extend(
                build_rows(variant, sample_ids, original_paths, positive_nlls, neg_nlls, metas)
            )

        if batch_size > 1:
            rolled = list(target_jsons[1:]) + [target_jsons[0]]
            metas = [{"source_sample_id": sample_ids[(i + 1) % batch_size]} for i in range(batch_size)]
            neg_nlls, _, _ = score_targets(model, images, rolled, prompt_text)
            all_rows["image_swap"].extend(
                build_rows(variant="image_swap", sample_ids=sample_ids, original_paths=original_paths,
                           positive_nlls=positive_nlls, negative_nlls=neg_nlls, metas=metas)
            )
        else:
            all_rows["image_swap"].extend(
                build_rows(variant="image_swap", sample_ids=sample_ids, original_paths=original_paths,
                           positive_nlls=positive_nlls, negative_nlls=[None], metas=[None])
            )

    summary = {variant: summarize_variant(rows) for variant, rows in all_rows.items()}
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "prompt_text": prompt_text,
        "positive": {
            "n": len(positive_nll_all),
            "nll_mean": float(np.mean(positive_nll_all)) if positive_nll_all else None,
            "token_count_mean": float(np.mean(positive_tokens_all)) if positive_tokens_all else None,
        },
        "summary": summary,
        "rows": all_rows,
    }
    save_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
