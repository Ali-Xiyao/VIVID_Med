"""Audit local Qwen3-VL components and run a dummy image-text loss check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_MODEL_PATH = r"H:\Xiyao_Wang\001_models\qwen3-vl-2b-thinking-new"


def load_transformers():
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    return AutoConfig, AutoModelForImageTextToText, AutoProcessor


def choose_dtype(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def module_name_matches(name: str, keywords: list[str]) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def classify_parameter(name: str) -> str:
    lowered = name.lower()
    connector_keywords = ["merger", "merge", "projector", "connector", "mm_projector", "adapter"]
    vision_keywords = ["visual", "vision", "image", "vit"]
    language_keywords = ["language", "llm", "lm_head", "embed_tokens", "model.layers", "model.norm"]

    if module_name_matches(lowered, connector_keywords):
        return "visual_connector"
    if module_name_matches(lowered, vision_keywords):
        return "vision_tower"
    if module_name_matches(lowered, language_keywords):
        return "language_decoder"
    return "other"


def count_parameters(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {
        "vision_tower": {"parameters": 0, "trainable": 0},
        "visual_connector": {"parameters": 0, "trainable": 0},
        "language_decoder": {"parameters": 0, "trainable": 0},
        "other": {"parameters": 0, "trainable": 0},
    }
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        count = int(param.numel())
        groups[group]["parameters"] += count
        if param.requires_grad:
            groups[group]["trainable"] += count
    groups["total"] = {
        "parameters": sum(group["parameters"] for group in groups.values()),
        "trainable": sum(group["trainable"] for group in groups.values()),
    }
    return groups


def list_candidate_modules(model: torch.nn.Module) -> dict[str, list[str]]:
    candidates = {
        "vision_tower": [],
        "visual_connector": [],
        "language_decoder": [],
    }
    for name, module in model.named_modules():
        if not name:
            continue
        lowered = name.lower()
        if module_name_matches(lowered, ["visual", "vision", "image", "vit"]):
            candidates["vision_tower"].append(name)
        if module_name_matches(lowered, ["merger", "projector", "connector", "mm_projector", "adapter"]):
            candidates["visual_connector"].append(name)
        if module_name_matches(lowered, ["language", "llm", "lm_head", "embed_tokens", "model.layers", "model.norm"]):
            candidates["language_decoder"].append(name)
    return {key: value[:60] for key, value in candidates.items()}


def apply_freeze_plan(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        group = classify_parameter(name)
        param.requires_grad = group in {"vision_tower", "visual_connector"}


def make_dummy_inputs(processor: Any, device: torch.device) -> dict[str, torch.Tensor]:
    image = Image.new("RGB", (256, 256), color=(32, 32, 32))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Is there visual evidence of cardiomegaly?"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "No. This is a dummy audit image."}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def run_forward_loss(model: torch.nn.Module, processor: Any, device: torch.device) -> dict[str, Any]:
    inputs = make_dummy_inputs(processor, device)
    if "input_ids" not in inputs:
        raise RuntimeError("processor output did not include input_ids")
    labels = inputs["input_ids"].clone()
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError("model forward did not return loss")
    tensor_shapes = {
        key: list(value.shape)
        for key, value in inputs.items()
        if torch.is_tensor(value)
    }
    return {
        "loss": float(loss.detach().cpu()),
        "input_shapes": tensor_shapes,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen3-VL Component Audit",
        "",
        f"- Model path: `{payload['model_path']}`",
        f"- Config class: `{payload['config_class']}`",
        f"- Model class: `{payload['model_class']}`",
        f"- Processor class: `{payload['processor_class']}`",
        f"- Model type: `{payload['model_type']}`",
        f"- Device: `{payload['device']}`",
        f"- DType: `{payload['dtype']}`",
        "",
        "## Parameter Groups",
        "",
        "| Group | Parameters | Trainable after freeze plan |",
        "| --- | ---: | ---: |",
    ]
    for group, stats in payload["parameter_groups_after_freeze"].items():
        lines.append(f"| {group} | {stats['parameters']} | {stats['trainable']} |")
    lines.extend(
        [
            "",
            "## Freeze Plan",
            "",
            "- Train: `vision_tower`, `visual_connector`.",
            "- Freeze: `language_decoder` and unmatched `other` parameters by default.",
            "",
            "## Dummy Forward Loss",
            "",
            f"- Status: `{payload['forward_loss']['status']}`",
        ]
    )
    if payload["forward_loss"]["status"] == "ok":
        lines.append(f"- Loss: `{payload['forward_loss']['loss']}`")
        lines.append(f"- Input shapes: `{json.dumps(payload['forward_loss']['input_shapes'])}`")
    else:
        lines.append(f"- Error: `{payload['forward_loss']['error']}`")
    lines.extend(
        [
            "",
            "## Candidate Module Names",
            "",
        ]
    )
    for group, names in payload["candidate_modules"].items():
        lines.append(f"### {group}")
        if not names:
            lines.append("- none found")
        else:
            for name in names[:30]:
                lines.append(f"- `{name}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-json", default="outputs/final_tables/qwen3vl_component_audit.json")
    parser.add_argument("--output-md", default="outputs/final_tables/qwen3vl_component_audit.md")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    dtype = choose_dtype(args.dtype)

    AutoConfig, AutoModelForImageTextToText, AutoProcessor = load_transformers()
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    before = count_parameters(model)
    apply_freeze_plan(model)
    after = count_parameters(model)
    candidate_modules = list_candidate_modules(model)

    forward_loss: dict[str, Any]
    if args.skip_forward:
        forward_loss = {"status": "skipped"}
    else:
        try:
            forward_loss = {"status": "ok", **run_forward_loss(model, processor, device)}
        except Exception as exc:  # noqa: BLE001 - audit must preserve failure evidence.
            forward_loss = {"status": "failed", "error": repr(exc)}

    payload = {
        "model_path": str(model_path),
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "model_type": getattr(config, "model_type", None),
        "device": str(device),
        "dtype": str(dtype),
        "parameter_groups_before_freeze": before,
        "parameter_groups_after_freeze": after,
        "candidate_modules": candidate_modules,
        "forward_loss": forward_loss,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload, indent=2))

    if forward_loss["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
