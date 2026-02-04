"""
Evaluate VIVID model by generating UMS-JSON and computing metrics.

用法:
    python eval_vivid.py --config ../configs/cxr_chexpert.yaml --checkpoint ../outputs/cxr_chexpert/checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel
from evaluation.metrics import compute_classification_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    parsed = try_parse(text)
    if parsed is not None:
        return parsed

    start_positions = [i for i, ch in enumerate(text) if ch == "{"]
    fallback = None
    for start in start_positions:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    parsed = try_parse(candidate)
                    if parsed is not None:
                        if "findings" in parsed:
                            return parsed
                        if fallback is None:
                            fallback = parsed
                    break
    return fallback


def apply_uncertain_policy(arr: np.ndarray, policy: str) -> np.ndarray:
    arr = arr.copy()
    if policy == "ignore":
        arr[arr == -1] = np.nan
    elif policy == "positive":
        arr[arr == -1] = 1.0
    elif policy == "negative":
        arr[arr == -1] = 0.0
    else:
        raise ValueError(f"Unknown uncertain_policy: {policy}")
    return arr


@torch.no_grad()
def evaluate(
    model: VIVIDModel,
    dataloader: DataLoader,
    label_names: List[str],
    device: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    uncertain_policy: str,
    threshold: float,
):
    model.eval()

    pred_matrix = []
    true_matrix = []
    json_success = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].cpu().numpy()

        texts = model.generate(
            images=images,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        for text, y_true in zip(texts, labels):
            total_samples += 1
            pred_row = np.full((len(label_names),), np.nan, dtype=float)

            parsed = parse_json_from_text(text)
            if parsed is not None:
                json_success += 1
                findings = parsed.get("findings", {}) if isinstance(parsed, dict) else {}
                if isinstance(findings, dict):
                    for i, name in enumerate(label_names):
                        if name in findings and isinstance(findings[name], dict):
                            state = findings[name].get("state")
                            if state == "present":
                                pred_row[i] = 1.0
                            elif state == "absent":
                                pred_row[i] = 0.0
                            elif state == "uncertain":
                                pred_row[i] = -1.0

            pred_matrix.append(pred_row)
            true_matrix.append(y_true)

    y_true = np.stack(true_matrix, axis=0)
    y_pred_raw = np.stack(pred_matrix, axis=0)

    y_true = apply_uncertain_policy(y_true, uncertain_policy)
    y_pred = apply_uncertain_policy(y_pred_raw, uncertain_policy)

    # NaN 预测按 0 处理（仅对 y_true 有效位置才参与评估）
    y_pred = np.nan_to_num(y_pred, nan=0.0)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
        label_names=label_names,
        threshold=threshold,
    )

    json_success_rate = json_success / total_samples if total_samples > 0 else 0.0
    pred_nan_rate = np.isnan(y_pred_raw).mean()

    return {
        "num_samples": total_samples,
        "json_success_rate": json_success_rate,
        "pred_nan_rate": float(pred_nan_rate),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VIVID model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "cxr_chexpert.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output", type=str, default=None, help="Output metrics json path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    requested_device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(requested_device, str) and requested_device.startswith("cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("CUDA not available, falling back to CPU")
    else:
        device = requested_device

    dataset = CheXpertUMSDataset(
        data_root=data_cfg["data_root"],
        ums_jsonl_path=data_cfg.get("val_ums_path") or data_cfg["train_ums_path"],
        transform=get_val_transforms(data_cfg["image_size"]),
        is_train=False,
        use_common_labels_only=data_cfg.get("use_common_labels_only", False),
        max_samples=args.max_samples or data_cfg.get("max_val_samples"),
    )

    batch_size = args.batch_size or config["training"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model_cfg = config["model"]
    model = VIVIDModel(
        vit_model_name=model_cfg["vit_model_name"],
        vit_pretrained=model_cfg["vit_pretrained"],
        vit_output_type=model_cfg.get("vit_output_type", "cls"),
        num_prefix_tokens=model_cfg["num_prefix_tokens"],
        projector_dropout=model_cfg.get("projector_dropout", 0.1),
        llm_model_name=model_cfg["llm_model_name"],
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        load_llm=True,
    ).to(device)

    # 加载 checkpoint（只含 vit/projector）
    state = torch.load(args.checkpoint, map_location=device)
    model.vit.load_state_dict(state["vit"])
    model.projector.load_state_dict(state["projector"])

    prompt_text = config.get("prompt", {}).get(
        "template",
        "Generate a structured medical report in JSON format for this chest X-ray image:\n",
    )

    eval_cfg = config.get("evaluation", {})
    uncertain_policy = eval_cfg.get("uncertain_policy", "ignore")
    threshold = float(eval_cfg.get("threshold", 0.5))

    result = evaluate(
        model=model,
        dataloader=loader,
        label_names=dataset.label_names,
        device=device,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        uncertain_policy=uncertain_policy,
        threshold=threshold,
    )

    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
