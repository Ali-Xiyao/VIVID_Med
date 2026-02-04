"""
Sample and inspect VIVID generated outputs.

用法:
    python sample_vivid_outputs.py --config ../configs/cxr_chexpert.yaml --checkpoint ../outputs/cxr_chexpert/checkpoints/best.pt --num_samples 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import CheXpertUMSDataset, get_val_transforms
from data.chexpert_dataset import collate_fn
from models import VIVIDModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Sample VIVID outputs")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "cxr_chexpert.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default=None, help="Save outputs as jsonl")
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
        max_samples=args.num_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
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

    state = torch.load(args.checkpoint, map_location=device)
    model.vit.load_state_dict(state["vit"])
    model.projector.load_state_dict(state["projector"])

    prompt_text = config.get("prompt", {}).get(
        "template",
        "Generate a structured medical report in JSON format for this chest X-ray image:\n",
    )

    outputs = []
    count = 0
    for batch in loader:
        images = batch["images"].to(device)
        texts = model.generate(
            images=images,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

        for i, text in enumerate(texts):
            record = {
                "sample_id": batch["sample_ids"][i],
                "original_path": batch["original_paths"][i],
                "target_json": batch["target_jsons"][i],
                "generated_text": text,
                "parsed_json": parse_json_from_text(text),
            }
            outputs.append(record)
            count += 1
            print("=" * 80)
            print(f"sample_id: {record['sample_id']}")
            print(f"original_path: {record['original_path']}")
            print("generated_text:")
            print(record["generated_text"])
            print("parsed_json:", "OK" if record["parsed_json"] else "FAILED")
            if count >= args.num_samples:
                break
        if count >= args.num_samples:
            break

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in outputs:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved samples to {out_path}")


if __name__ == "__main__":
    main()
