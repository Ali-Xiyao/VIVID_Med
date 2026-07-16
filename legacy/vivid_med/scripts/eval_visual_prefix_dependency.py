"""
Evaluate whether a trained frozen-LM VIVID model uses the visual prefix.

For the same target JSON, compare token NLL under:
- image: normal image prefix
- blank: zero image prefix
- shuffled: batch-shuffled image prefix

If image NLL is not lower than blank/shuffled, teacher forcing may be relying
mostly on text/label priors rather than image-conditioned supervision.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel


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
    model = VIVIDModel(
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
    return model


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


def token_loss_stats(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    valid = flat_labels != -100
    losses = torch.nn.functional.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="none")
    return {
        "loss_sum": float(losses.sum().detach().cpu()),
        "token_count": int(valid.sum().detach().cpu()),
        "nll": float(losses.mean().detach().cpu()) if valid.any() else float("nan"),
    }


@torch.no_grad()
def evaluate_variant(
    model: VIVIDModel,
    images: torch.Tensor,
    target_jsons: List[str],
    prompt_text: str,
    variant: str,
) -> Dict[str, float]:
    if variant == "image":
        variant_images = images
    elif variant == "blank":
        variant_images = torch.zeros_like(images)
    elif variant == "shuffled":
        if images.shape[0] == 1:
            variant_images = torch.zeros_like(images)
        else:
            variant_images = images[torch.roll(torch.arange(images.shape[0], device=images.device), shifts=1)]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    outputs = model(
        images=variant_images,
        prompt_text=prompt_text,
        target_text=target_jsons,
    )
    return token_loss_stats(outputs["logits"], outputs["labels"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visual-prefix dependency for VIVID")
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

    accum = {
        "image": {"loss_sum": 0.0, "token_count": 0},
        "blank": {"loss_sum": 0.0, "token_count": 0},
        "shuffled": {"loss_sum": 0.0, "token_count": 0},
    }

    sample_count = 0
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["images"].to(device)
        target_jsons = batch["target_jsons"]
        sample_count += len(target_jsons)
        for variant in accum:
            stats = evaluate_variant(model, images, target_jsons, prompt_text, variant)
            accum[variant]["loss_sum"] += stats["loss_sum"]
            accum[variant]["token_count"] += stats["token_count"]

    summary = {}
    for variant, values in accum.items():
        token_count = max(int(values["token_count"]), 1)
        summary[variant] = {
            "loss_sum": values["loss_sum"],
            "token_count": int(values["token_count"]),
            "nll": values["loss_sum"] / token_count,
        }

    image_nll = summary["image"]["nll"]
    for variant in ("blank", "shuffled"):
        summary[variant]["delta_vs_image"] = summary[variant]["nll"] - image_nll
        summary[variant]["relative_delta_vs_image"] = (
            summary[variant]["delta_vs_image"] / image_nll if image_nll else None
        )

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "prompt_text": prompt_text,
        "summary": summary,
    }
    save_json(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
