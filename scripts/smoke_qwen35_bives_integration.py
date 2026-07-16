"""Bounded real-Qwen3.5 BiVES integration gate for the server."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.losses import BiVESLoss, BiVESLossConfig
from bives_cxr.model import BiVESModelConfig
from scripts.train_bives_cxr import (
    BiVESExperiment,
    Qwen35BiVESCollator,
    move_to_device,
    require_uniform_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def synthetic_group() -> list[dict[str, object]]:
    sizes = ((448, 448), (640, 320), (320, 640), (512, 384))
    states = ("support", "contradict", "uncertain", "insufficient")
    rows: list[dict[str, object]] = []
    for index, (size, state) in enumerate(zip(sizes, states)):
        image = Image.new("RGB", size, (32 + 40 * index, 64, 160 - 20 * index))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (
                size[0] // 4,
                size[1] // 4,
                3 * size[0] // 4,
                3 * size[1] // 4,
            ),
            outline=(255, 255, 255),
            width=max(2, min(size) // 64),
        )
        rows.append(
            {
                "sample_id": f"synthetic-{state}",
                "patient_id": f"synthetic-patient-{index}",
                "group_id": "synthetic-effusion-right-quartet",
                "canonical_statement_id": "synthetic-effusion-right",
                "statement_text": "A right pleural effusion is present.",
                "state": state,
                "state_index": index,
                "statement_index": 0,
                "image": image,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.steps != 2:
        raise ValueError("the bounded integration contract requires exactly 2 steps")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real Qwen3.5 integration gate requires a CUDA server allocation")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_path = args.output_dir / "stages.jsonl"

    def stage(name: str, **payload: object) -> None:
        row = {"stage": name, **payload}
        with stage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    visual, processor, config = load_qwen35_visual_and_processor(
        args.model_path,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
    )
    merge = int(config["vision_config"]["spatial_merge_size"])
    visual_parameters = sum(parameter.numel() for parameter in visual.parameters())
    visual = visual.to(device).eval()
    stage("selective_visual_loaded", parameters=visual_parameters)

    rows = synthetic_group()
    collator = Qwen35BiVESCollator(
        processor,
        image_size=448,
        include_group_indices=True,
    )
    batch = collator(rows)
    probe_pixels = batch["pixel_values"].to(device=device, dtype=dtype)
    probe_grid = batch["image_grid_thw"].to(device)

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    torch.cuda.reset_peak_memory_stats(device)
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_implementation,
    ).to(device).eval()
    stage("official_full_loaded")
    full_total_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    full_visual_parameters = sum(
        parameter.numel() for parameter in full_model.model.visual.parameters()
    )
    selective_state = visual.state_dict()
    official_state = full_model.model.visual.state_dict()
    parameter_mismatch_keys = [
        key
        for key in selective_state
        if key not in official_state
        or not torch.equal(selective_state[key], official_state[key])
    ]
    stage("visual_parameters_compared", mismatch_count=len(parameter_mismatch_keys))
    with torch.no_grad():
        selective_adapter = Qwen35VisionAdapter(visual, merge)
        official_adapter = Qwen35VisionAdapter(full_model, merge)
        selective_tokens = selective_adapter(
            probe_pixels,
            probe_grid,
        ).tokens
        selective_repeat = selective_adapter(
            probe_pixels,
            probe_grid,
        ).tokens
        official_tokens = official_adapter(
            probe_pixels,
            probe_grid,
        ).tokens
        official_repeat = official_adapter(
            probe_pixels,
            probe_grid,
        ).tokens
    alignment_difference = (selective_tokens - official_tokens).abs().float()
    selective_repeat_difference = (selective_tokens - selective_repeat).abs().float()
    official_repeat_difference = (official_tokens - official_repeat).abs().float()
    alignment_peak_bytes = torch.cuda.max_memory_allocated(device)
    stage(
        "visual_outputs_compared",
        max_abs_error=float(alignment_difference.max()),
        mean_abs_error=float(alignment_difference.mean()),
    )
    del (
        official_tokens,
        official_repeat,
        selective_tokens,
        selective_repeat,
        official_adapter,
        selective_adapter,
        official_state,
        selective_state,
        full_model,
    )
    gc.collect()
    torch.cuda.empty_cache()
    stage("official_full_released")

    torch.cuda.reset_peak_memory_stats(device)
    adapter = Qwen35VisionAdapter(visual, merge)
    experiment = BiVESExperiment(
        adapter,
        num_statements=1,
        statement_dim=64,
        head_config=BiVESModelConfig(
            visual_dim=int(config["vision_config"]["hidden_size"]),
            statement_dim=64,
            fusion_dim=128,
            gate_mode="soft_topk",
            topk=16,
            num_controls=4,
            contextual_layers=1,
            contextual_heads=4,
            contextual_dropout=0.0,
        ),
    ).to(device)
    for parameter in experiment.backbone.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in list(experiment.statement_table.parameters())
            + list(experiment.head.parameters())
            if parameter.requires_grad
        ],
        lr=1e-4,
    )
    loss_fn = BiVESLoss(
        BiVESLossConfig(
            lambda_pair=0.1,
            lambda_u_pol=0.1,
            lambda_min=0.0,
        )
    )
    device_batch = move_to_device(batch, device)
    step_rows: list[dict[str, object]] = []
    for step in range(1, args.steps + 1):
        outputs, grids = experiment(
            device_batch["pixel_values"],
            device_batch["image_grid_thw"],
            device_batch["statement_indices"],
            device_batch["content_valid_mask"],
            run_interventions=True,
            control_seeds=device_batch["control_seeds"],
        )
        losses = loss_fn(
            outputs,
            device_batch["targets"],
            require_uniform_grid(grids),
            device_batch["support_pair_indices"],
            device_batch["contradict_pair_indices"],
            device_batch["uncertain_indices"],
        )
        losses["total"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        original_probs = outputs["original"]["state_probs"]
        step_rows.append(
            {
                "step": step,
                "loss": float(losses["total"].detach().cpu()),
                "control_loss": float(losses["control"].detach().cpu()),
                "gate_cardinality": outputs["evidence_hard_mask"].sum(dim=-1).tolist(),
                "keep_max_abs_diff": float(
                    (outputs["keep"]["state_probs"] - original_probs)
                    .abs()
                    .max()
                    .detach()
                ),
                "control_max_abs_diff": float(
                    max(
                        (branch["state_probs"] - original_probs).abs().max().detach()
                        for branch in outputs["controls"]
                    )
                ),
            }
        )
        stage("training_step_complete", step=step, loss=step_rows[-1]["loss"])

    letterboxed, content_box = collator._letterbox(rows[1]["image"])
    overlay = letterboxed.copy()
    ImageDraw.Draw(overlay).rectangle(content_box, outline=(255, 0, 0), width=4)
    overlay_path = args.output_dir / "content_mask_overlay.png"
    overlay.save(overlay_path)

    language_parameters = sum(
        parameter.numel()
        for name, parameter in experiment.named_parameters()
        if "embed_tokens" in name or "lm_head" in name or "language" in name
    )
    payload = {
        "status": "passed",
        "model_path": str(args.model_path),
        "device": str(device),
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "official_full_parameters": full_total_parameters,
        "official_visual_parameters": full_visual_parameters,
        "selective_visual_parameters": visual_parameters,
        "training_language_parameters": language_parameters,
        "visual_parameter_mismatch_count": len(parameter_mismatch_keys),
        "visual_parameter_mismatch_examples": parameter_mismatch_keys[:5],
        "alignment_max_abs_error": float(alignment_difference.max()),
        "alignment_mean_abs_error": float(alignment_difference.mean()),
        "selective_repeat_max_abs_error": float(selective_repeat_difference.max()),
        "official_repeat_max_abs_error": float(official_repeat_difference.max()),
        "alignment_peak_memory_bytes": int(alignment_peak_bytes),
        "training_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "content_valid_patches": batch["content_valid_mask"].sum(dim=-1).tolist(),
        "grid_thw": batch["image_grid_thw"].tolist(),
        "content_mask_overlay": str(overlay_path),
        "steps": step_rows,
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "torchvision",
                "transformers",
                "safetensors",
                "numpy",
                "pillow",
                "scikit-learn",
            )
        },
    }
    if language_parameters != 0:
        raise RuntimeError("BiVES training experiment retained language-model parameters")
    if payload["alignment_max_abs_error"] > 1e-5:
        raise RuntimeError("vision-only output does not align with the official full model")
    if any(
        any(cardinality != 16 for cardinality in row["gate_cardinality"])
        for row in step_rows
    ):
        raise RuntimeError("integration gate violated fixed-K cardinality")
    if any(row["control_loss"] <= 0 for row in step_rows):
        raise RuntimeError("control closure remained trivial during integration smoke")
    output_path = args.output_dir / "qwen35_bives_integration.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stage("artifact_written", path=str(output_path))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
