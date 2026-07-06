"""
Evaluate visual-prefix dependency for counterfactual schema scoring.

For each image and positive schema z+, construct counterfactual schemas z-,
then compare pairwise schema scoring under three prefix conditions:
- image: normal image prefix.
- blank: zero image prefix.
- shuffled: batch-shuffled image prefix.

This extends the positive-only prefix dependency diagnostic to the EMNLP P0
counterfactual setting: a grounded scorer should prefer z+ over z- most
reliably under the correct image prefix.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eval_counterfactual_schema_grounding import (
    build_rows,
    create_model,
    create_val_loader,
    load_checkpoint,
    load_yaml,
    make_field_swap,
    make_null_to_present,
    make_state_flip,
    prompt_from_config,
    save_json,
    score_targets,
    summarize_variant,
)


PREFIX_VARIANTS = ("image", "blank", "shuffled")
COUNTERFACTUAL_VARIANTS = ("state_flip", "field_swap", "image_swap", "null_to_present")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prefix_images(images: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "image":
        return images
    if variant == "blank":
        return torch.zeros_like(images)
    if variant == "shuffled":
        if images.shape[0] == 1:
            return torch.zeros_like(images)
        order = torch.roll(torch.arange(images.shape[0], device=images.device), shifts=1)
        return images[order]
    raise ValueError(f"Unknown prefix variant: {variant}")


def build_counterfactual_targets(
    target_jsons: List[str],
    sample_ids: List[Any],
    rng: random.Random,
) -> Dict[str, Tuple[List[Optional[str]], List[Optional[Dict[str, Any]]]]]:
    batch_size = len(target_jsons)
    targets: Dict[str, Tuple[List[Optional[str]], List[Optional[Dict[str, Any]]]]] = {}

    for variant, maker in (
        ("state_flip", make_state_flip),
        ("field_swap", make_field_swap),
        ("null_to_present", make_null_to_present),
    ):
        neg_texts: List[Optional[str]] = [None] * batch_size
        metas: List[Optional[Dict[str, Any]]] = [None] * batch_size
        for i, target_json in enumerate(target_jsons):
            neg_text, meta = maker(target_json, rng)
            neg_texts[i] = neg_text
            metas[i] = meta
        targets[variant] = (neg_texts, metas)

    if batch_size > 1:
        image_swap_texts = list(target_jsons[1:]) + [target_jsons[0]]
        image_swap_metas = [{"source_sample_id": sample_ids[(i + 1) % batch_size]} for i in range(batch_size)]
        targets["image_swap"] = (image_swap_texts, image_swap_metas)
    else:
        targets["image_swap"] = ([None], [None])

    return targets


def score_optional_targets(
    model,
    images: torch.Tensor,
    target_texts: List[Optional[str]],
    prompt_text: str,
) -> List[Optional[float]]:
    valid_indices = [i for i, text in enumerate(target_texts) if text is not None]
    scored: List[Optional[float]] = [None] * len(target_texts)
    if not valid_indices:
        return scored

    valid_images = images[torch.tensor(valid_indices, device=images.device)]
    valid_texts = [target_texts[i] for i in valid_indices]
    nlls, _, _ = score_targets(model, valid_images, valid_texts, prompt_text)
    for idx, nll in zip(valid_indices, nlls):
        scored[idx] = nll
    return scored


def summarize_positive(nlls: List[float]) -> Dict[str, Optional[float]]:
    if not nlls:
        return {"n": 0, "nll_mean": None}
    values = np.array(nlls, dtype=np.float64)
    return {
        "n": int(values.size),
        "nll_mean": float(values.mean()),
        "nll_median": float(np.median(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual prefix dependency for VIVID")
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

    rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        prefix: {variant: [] for variant in COUNTERFACTUAL_VARIANTS}
        for prefix in PREFIX_VARIANTS
    }
    positive_nlls: Dict[str, List[float]] = {prefix: [] for prefix in PREFIX_VARIANTS}
    sample_count = 0

    for batch in tqdm(loader, desc="Counterfactual prefix eval"):
        images = batch["images"].to(device)
        target_jsons = batch["target_jsons"]
        sample_ids = batch["sample_ids"]
        original_paths = batch["original_paths"]
        sample_count += len(target_jsons)

        cf_targets = build_counterfactual_targets(target_jsons, sample_ids, rng)

        for prefix in PREFIX_VARIANTS:
            prefix_images = make_prefix_images(images, prefix)
            pos_nlls, _, _ = score_targets(model, prefix_images, target_jsons, prompt_text)
            positive_nlls[prefix].extend(pos_nlls)

            for cf_variant in COUNTERFACTUAL_VARIANTS:
                neg_texts, metas = cf_targets[cf_variant]
                neg_nlls = score_optional_targets(model, prefix_images, neg_texts, prompt_text)
                built = build_rows(
                    variant=cf_variant,
                    sample_ids=sample_ids,
                    original_paths=original_paths,
                    positive_nlls=pos_nlls,
                    negative_nlls=neg_nlls,
                    metas=metas,
                )
                for row in built:
                    row["prefix_variant"] = prefix
                rows[prefix][cf_variant].extend(built)

    summary = {
        prefix: {
            "positive": summarize_positive(positive_nlls[prefix]),
            "counterfactual": {
                variant: summarize_variant(rows[prefix][variant])
                for variant in COUNTERFACTUAL_VARIANTS
            },
        }
        for prefix in PREFIX_VARIANTS
    }

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "sample_count": sample_count,
        "prompt_text": prompt_text,
        "summary": summary,
        "rows": rows,
    }
    save_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
