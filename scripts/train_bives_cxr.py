"""Train the Qwen3.5-only BiVES-CXR evidence model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.audit import audit_manifests
from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.calibration import fit_decoder_temperatures
from bives_cxr.data import (
    BiVESManifestDataset,
    SameStatementStateBatchSampler,
    build_group_loss_indices,
    limit_rows_to_complete_groups,
    read_manifest,
    statement_text_by_id,
)
from bives_cxr.losses import BiVESLoss, BiVESLossConfig
from bives_cxr.metrics import (
    classification_metrics,
    finalize_intervention_metrics,
    intervention_metric_counts,
    patient_bootstrap_confidence_intervals,
)
from bives_cxr.model import BiVESCXR, BiVESModelConfig
from bives_cxr.interventions import CONTROL_PROTOCOL_VERSION, bives_control_seed
from bives_cxr.statement_cache import load_statement_embedding_matrix
from bives_cxr.provenance import (
    build_source_snapshot,
    build_run_lock,
    canonical_json_sha256,
    file_sha256 as provenance_file_sha256,
    resolved_config_text,
)
from bives_cxr.dataset_lock import validate_dataset_lock


class BiVESExperiment(nn.Module):
    """Qwen3.5 patch encoder plus canonical statement table and BiVES head."""

    def __init__(
        self,
        backbone: Qwen35VisionAdapter,
        num_statements: int,
        statement_dim: int,
        head_config: BiVESModelConfig,
        frozen_statement_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        if frozen_statement_embeddings is None:
            self.statement_table = nn.Embedding(num_statements, statement_dim)
        else:
            if frozen_statement_embeddings.shape != (num_statements, statement_dim):
                raise ValueError(
                    "frozen statement embedding matrix must have shape "
                    f"{(num_statements, statement_dim)}"
                )
            self.statement_table = nn.Embedding.from_pretrained(
                frozen_statement_embeddings.float(),
                freeze=True,
            )
        self.head = BiVESCXR(head_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        statement_indices: torch.Tensor,
        content_valid_mask: torch.Tensor | None = None,
        run_interventions: bool = True,
        control_seeds: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor] | torch.Tensor], list[tuple[int, int]]]:
        first_backbone_parameter = next(self.backbone.parameters(), None)
        if first_backbone_parameter is not None and pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=first_backbone_parameter.dtype)
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            patches = self.backbone(pixel_values, image_grid_thw)
        else:
            self.backbone.eval()
            with torch.no_grad():
                patches = self.backbone(pixel_values, image_grid_thw)
        head_dtype = self.head.visual_projection.weight.dtype
        patch_tokens = patches.tokens.to(dtype=head_dtype)
        statement_embeddings = self.statement_table(statement_indices).to(dtype=head_dtype)
        valid_mask = patches.valid_mask
        if content_valid_mask is not None:
            if content_valid_mask.shape != valid_mask.shape:
                raise ValueError(
                    f"content_valid_mask shape {tuple(content_valid_mask.shape)} "
                    f"does not match visual grid {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask & content_valid_mask.bool()
        outputs = self.head(
            patch_tokens,
            statement_embeddings,
            valid_mask,
            run_interventions=run_interventions,
            control_seeds=control_seeds,
        )
        return outputs, patches.grid_hw


class Qwen35BiVESCollator:
    def __init__(
        self,
        processor: Any,
        image_size: int = 448,
        include_group_indices: bool = True,
        split: str = "train",
        training_seed: int = 17,
        evaluation_control_seed: int = 20260717,
        control_protocol_version: str = CONTROL_PROTOCOL_VERSION,
    ) -> None:
        self.processor = processor
        self.image_size = int(image_size)
        self.include_group_indices = bool(include_group_indices)
        self.split = str(split)
        self.training_seed = int(training_seed)
        self.evaluation_control_seed = int(evaluation_control_seed)
        self.control_protocol_version = str(control_protocol_version)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
        image = image.convert("RGB")
        scale = min(self.image_size / image.width, self.image_size / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        left = (self.image_size - resized.width) // 2
        top = (self.image_size - resized.height) // 2
        canvas.paste(resized, (left, top))
        return canvas, (left, top, left + resized.width, top + resized.height)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        content_boxes = []
        for item in batch:
            image, content_box = self._letterbox(item["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item["statement_text"]},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            images.append(image)
            content_boxes.append(content_box)
        processor_output = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        encoded = {
            "pixel_values": processor_output["pixel_values"],
            "image_grid_thw": processor_output["image_grid_thw"],
        }
        masks: list[torch.Tensor] = []
        for grid, box in zip(encoded["image_grid_thw"], content_boxes):
            height, width = int(grid[1]), int(grid[2])
            left, top, right, bottom = box
            x_centers = (torch.arange(width, dtype=torch.float32) + 0.5) * self.image_size / width
            y_centers = (torch.arange(height, dtype=torch.float32) + 0.5) * self.image_size / height
            mask = (
                (y_centers[:, None] >= top)
                & (y_centers[:, None] < bottom)
                & (x_centers[None, :] >= left)
                & (x_centers[None, :] < right)
            ).reshape(-1)
            masks.append(mask)
        max_patches = max(mask.numel() for mask in masks)
        content_valid = torch.zeros((len(masks), max_patches), dtype=torch.bool)
        for index, mask in enumerate(masks):
            content_valid[index, : mask.numel()] = mask
        encoded["content_valid_mask"] = content_valid
        encoded["statement_indices"] = torch.tensor([item["statement_index"] for item in batch], dtype=torch.long)
        encoded["targets"] = torch.tensor([item["state_index"] for item in batch], dtype=torch.long)
        encoded["sample_ids"] = [str(item["sample_id"]) for item in batch]
        encoded["patient_ids"] = [str(item["patient_id"]) for item in batch]
        encoded["canonical_statement_ids"] = [
            str(item["canonical_statement_id"]) for item in batch
        ]
        encoded["group_ids"] = [str(item["group_id"]) for item in batch]
        encoded["control_protocol_version"] = self.control_protocol_version
        encoded["control_seeds"] = torch.tensor(
            [
                bives_control_seed(
                    split=self.split,
                    sample_id=str(item["sample_id"]),
                    training_seed=self.training_seed,
                    evaluation_control_seed=self.evaluation_control_seed,
                    epoch=self.epoch,
                    protocol_version=self.control_protocol_version,
                )
                for item in batch
            ],
            dtype=torch.int64,
        )
        if self.include_group_indices:
            encoded.update(build_group_loss_indices(batch))
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run the synthetic CPU core smoke without loading Qwen3.5.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch", "cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


DEVICE_FIELDS = {
    "pixel_values",
    "image_grid_thw",
    "content_valid_mask",
    "statement_indices",
    "targets",
    "support_pair_indices",
    "contradict_pair_indices",
    "uncertain_indices",
    "control_seeds",
}


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if key in DEVICE_FIELDS and torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def require_uniform_grid(grid_hw: list[tuple[int, int]]) -> tuple[int, int]:
    unique = set(grid_hw)
    if len(unique) != 1:
        raise ValueError(f"BiVES batches currently require one padded image grid; got {sorted(unique)}")
    return grid_hw[0]


def assert_full_sample_coverage(
    expected_ids: list[str],
    evaluated_ids: list[str],
) -> None:
    if len(evaluated_ids) != len(expected_ids):
        raise RuntimeError(
            f"primary evaluation covered {len(evaluated_ids)}/{len(expected_ids)} rows"
        )
    if len(set(evaluated_ids)) != len(evaluated_ids):
        raise RuntimeError("primary evaluation emitted duplicate sample IDs")
    if set(evaluated_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(evaluated_ids))[:5]
        unexpected = sorted(set(evaluated_ids) - set(expected_ids))[:5]
        raise RuntimeError(
            f"primary evaluation sample-ID mismatch; missing={missing}, unexpected={unexpected}"
        )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    return provenance_file_sha256(path)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    environment_commit = os.environ.get("BIVES_GIT_COMMIT", "").strip()
    if environment_commit:
        return environment_commit
    marker = Path(__file__).resolve().parents[1] / ".bives_source_commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip():
        return marker.read_text(encoding="utf-8").strip()
    return "unknown"


def limit_to_complete_groups(dataset: BiVESManifestDataset, max_records: int | None) -> None:
    dataset.rows = limit_rows_to_complete_groups(dataset.rows, max_records)


def load_frozen_statement_embeddings(
    path: str | Path,
    statement_to_index: dict[str, int],
    expected_text_by_id: dict[str, str],
    expected_cache: dict[str, Any] | None = None,
) -> torch.Tensor:
    return load_statement_embedding_matrix(
        path,
        statement_to_index,
        expected_text_by_id,
        expected_cache=expected_cache,
    )


@torch.no_grad()
def evaluate(
    experiment: BiVESExperiment,
    loader: DataLoader,
    loss_fn: BiVESLoss,
    device: torch.device,
    *,
    assert_full_coverage: bool = True,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 17,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    experiment.eval()
    loss_sum = 0.0
    samples = 0
    mechanism: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    all_probabilities: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_patient_ids: list[str] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs, grids = experiment(
            batch["pixel_values"],
            batch["image_grid_thw"],
            batch["statement_indices"],
            batch["content_valid_mask"],
            run_interventions=True,
            control_seeds=batch["control_seeds"],
        )
        losses = loss_fn(
            outputs,
            batch["targets"],
            require_uniform_grid(grids),
        )
        batch_size = int(batch["targets"].numel())
        loss_sum += float(losses["total"].cpu()) * batch_size
        probabilities = outputs["original"]["state_probs"]
        samples += batch_size
        all_probabilities.append(probabilities.detach().float().cpu())
        all_targets.append(batch["targets"].detach().long().cpu())
        all_patient_ids.extend(batch["patient_ids"])
        for key, value in intervention_metric_counts(outputs, batch["targets"]).items():
            mechanism[key] = mechanism.get(key, 0.0) + value
        for index, sample_id in enumerate(batch["sample_ids"]):
            evidence_indices = torch.where(
                outputs["evidence_hard_mask"][index].detach().cpu()
            )[0].tolist()
            control_indices = [
                torch.where(mask.detach().cpu() > 0.5)[0].tolist()
                for mask in outputs["control_masks"][index]
            ]
            grid_h, grid_w = grids[index]
            control_probabilities = [
                branch["state_probs"][index].detach().float().cpu().tolist()
                for branch in outputs["controls"]
            ]
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": batch["patient_ids"][index],
                    "group_id": batch["group_ids"][index],
                    "canonical_statement_id": batch["canonical_statement_ids"][index],
                    "target": int(batch["targets"][index].item()),
                    "control_seed": int(batch["control_seeds"][index].item()),
                    "control_protocol_version": batch["control_protocol_version"],
                    "grid_h": int(grid_h),
                    "grid_w": int(grid_w),
                    "evidence_topk_indices": evidence_indices,
                    "control_topk_indices": control_indices,
                    "original_probs": probabilities[index].detach().float().cpu().tolist(),
                    "evidence_pos": float(
                        outputs["original"]["evidence_pos"][index].detach().float().cpu()
                    ),
                    "evidence_neg": float(
                        outputs["original"]["evidence_neg"][index].detach().float().cpu()
                    ),
                    "keep_probs": outputs["keep"]["state_probs"][index].detach().float().cpu().tolist(),
                    "drop_probs": outputs["drop"]["state_probs"][index].detach().float().cpu().tolist(),
                    "control_probs": outputs["control"]["state_probs"][index].detach().float().cpu().tolist(),
                    "control_branch_probs": control_probabilities,
                }
            )
    if samples == 0:
        raise ValueError("evaluation loader produced no samples")
    if assert_full_coverage:
        dataset = loader.dataset
        expected_ids = [str(row["sample_id"]) for row in dataset.rows]
        evaluated_ids = [row["sample_id"] for row in rows]
        assert_full_sample_coverage(expected_ids, evaluated_ids)
    probability_matrix = torch.cat(all_probabilities).numpy()
    target_vector = torch.cat(all_targets).numpy()
    experiment.train()
    result: dict[str, Any] = {
        "loss": float(loss_sum / samples) if samples else float("nan"),
        "evaluated_sample_count": samples,
        "unique_sample_count": len({row["sample_id"] for row in rows}),
        **classification_metrics(probability_matrix, target_vector),
    }
    result.update(finalize_intervention_metrics(mechanism))
    if bootstrap_replicates > 0:
        result["patient_bootstrap_95ci"] = patient_bootstrap_confidence_intervals(
            probability_matrix,
            target_vector,
            all_patient_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            prediction_rows=rows,
        )
    return result, rows


@torch.no_grad()
def evaluate_grouped_mechanisms(
    experiment: BiVESExperiment,
    loader: DataLoader,
    device: torch.device,
    pair_margin: float,
) -> dict[str, float]:
    """Evaluate same-statement relations separately from primary row coverage."""

    experiment.eval()
    pair_violations: list[torch.Tensor] = []
    uncertain_polarities: list[torch.Tensor] = []
    groups = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs, _ = experiment(
            batch["pixel_values"],
            batch["image_grid_thw"],
            batch["statement_indices"],
            batch["content_valid_mask"],
            run_interventions=False,
            control_seeds=batch["control_seeds"],
        )
        original = outputs["original"]
        rho = (original["evidence_pos"] - original["evidence_neg"]) / (
            original["total_evidence"] + 1e-8
        )
        support = batch["support_pair_indices"].long()
        contradict = batch["contradict_pair_indices"].long()
        uncertain = batch["uncertain_indices"].long()
        pair_violations.append(
            torch.relu(pair_margin - rho[support] + rho[contradict]).detach().cpu()
        )
        uncertain_polarities.append(rho[uncertain].abs().detach().cpu())
        groups += int(support.numel())
    experiment.train()
    if groups == 0:
        raise ValueError("grouped evaluator produced no complete S/C/U/I groups")
    return {
        "groups": float(groups),
        "pair_margin_violation": float(torch.cat(pair_violations).mean()),
        "uncertain_absolute_polarity": float(torch.cat(uncertain_polarities).mean()),
    }


def set_decoder_temperatures(
    experiment: BiVESExperiment,
    temperatures: dict[str, float],
) -> None:
    for name in ("tau_a", "tau_d", "tau_p"):
        value = float(temperatures[name])
        if not math.isfinite(value) or not 1e-4 <= value <= 1e4:
            raise ValueError(f"{name} must be finite and bounded in [1e-4, 1e4]")
        getattr(experiment.head.decoder, name).fill_(value)


def load_checkpoint_model_state(
    experiment: BiVESExperiment,
    checkpoint: dict[str, Any],
) -> None:
    experiment.statement_table.load_state_dict(checkpoint["statement_table"])
    experiment.head.load_state_dict(checkpoint["bives_head"])
    if "backbone" in checkpoint:
        experiment.backbone.load_state_dict(checkpoint["backbone"])


def synthetic_smoke() -> None:
    from scripts.smoke_bives_cxr import main

    main()


def main() -> None:
    args = parse_args()
    if args.smoke:
        synthetic_smoke()
        return
    if args.config is None:
        raise SystemExit("--config is required unless --smoke is used")

    config = load_config(args.config)
    if str(config["model"]["family"]).lower() != "qwen3.5":
        raise ValueError("active BiVES-CXR configs are Qwen3.5-only")
    if args.debug:
        config["data"]["max_train_samples"] = min(int(config["data"].get("max_train_samples") or 8), 8)
        config["data"]["max_val_samples"] = min(int(config["data"].get("max_val_samples") or 4), 4)
        config["training"]["max_steps"] = min(int(config["training"].get("max_steps", 2)), 2)
        config.setdefault("sampling", {})["groups_per_batch"] = 1
        config["training"]["eval_interval"] = 1
        config["training"]["output_dir"] = str(config["training"]["output_dir"]).rstrip("/\\") + "_debug"
        config.setdefault("evaluation", {})["run_calibration"] = False
        config["evaluation"]["run_test"] = False

    evaluation_config = config.setdefault("evaluation", {})
    if bool(evaluation_config.get("run_test", False)):
        raise ValueError(
            "scripts/train_bives_cxr.py never evaluates the locked test split; "
            "use scripts/evaluate_bives_final.py with an explicit release flag"
        )

    set_seed(int(config.get("seed", 17)))
    device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir} already contains metrics_final.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        resolved_config_text(config), encoding="utf-8"
    )

    manifest_paths = {
        split: config["data"][key]
        for split, key in (
            ("train", "train_manifest"),
            ("val", "val_manifest"),
            ("calibration", "calibration_manifest"),
        )
        if config["data"].get(key)
        and (split != "calibration" or bool(evaluation_config.get("run_calibration", True)))
    }
    audit_config = config.get("audit", {})
    audit_report = audit_manifests(
        manifest_paths,
        data_root=config["data"]["data_root"],
        check_images=bool(audit_config.get("check_images", True)),
        require_complete_statements=bool(audit_config.get("require_complete_statements", True)),
        check_decodable=bool(audit_config.get("check_decodable", True)),
        reject_constant_images=bool(audit_config.get("reject_constant_images", True)),
        require_provenance=bool(audit_config.get("require_provenance", True)),
        verify_image_sha256=bool(audit_config.get("verify_image_sha256", True)),
        require_matching_protocol=bool(
            audit_config.get("require_matching_protocol", True)
        ),
    )
    save_json(output_dir / "manifest_readiness_audit.json", audit_report)
    if audit_report["status"] != "pass":
        raise SystemExit(
            f"manifest readiness gate failed before Qwen3.5 load; "
            f"see {output_dir / 'manifest_readiness_audit.json'}"
        )

    formal_manifest_paths = {
        split: config["data"][key]
        for split, key in (
            ("train", "train_manifest"), ("val", "val_manifest"),
            ("calibration", "calibration_manifest"), ("test", "test_manifest"),
        )
    }
    dataset_lock = validate_dataset_lock(
        config["data"]["dataset_lock"],
        formal_manifest_paths,
        data_root=config["data"]["data_root"],
        audit_options={
            "check_images": bool(audit_config.get("check_images", True)),
            "require_complete_statements": bool(audit_config.get("require_complete_statements", True)),
            "check_decodable": bool(audit_config.get("check_decodable", True)),
            "reject_constant_images": bool(audit_config.get("reject_constant_images", True)),
            "require_provenance": bool(audit_config.get("require_provenance", True)),
            "verify_image_sha256": bool(audit_config.get("verify_image_sha256", True)),
            "require_matching_protocol": bool(audit_config.get("require_matching_protocol", True)),
        },
    )

    train_rows = read_manifest(config["data"]["train_manifest"])
    val_rows = read_manifest(config["data"]["val_manifest"])
    if args.debug:
        train_rows = limit_rows_to_complete_groups(train_rows, config["data"].get("max_train_samples"))
        val_rows = limit_rows_to_complete_groups(val_rows, config["data"].get("max_val_samples"))
    train_dataset = BiVESManifestDataset(config["data"]["train_manifest"], config["data"]["data_root"], rows=train_rows)
    val_dataset = BiVESManifestDataset(
        config["data"]["val_manifest"],
        config["data"]["data_root"],
        statement_to_index=train_dataset.statement_to_index,
        rows=val_rows,
    )
    calibration_dataset = (
        BiVESManifestDataset(
            config["data"]["calibration_manifest"],
            config["data"]["data_root"],
            statement_to_index=train_dataset.statement_to_index,
        )
        if config["data"].get("calibration_manifest")
        and bool(evaluation_config.get("run_calibration", True))
        else None
    )

    expected_statement_texts = statement_text_by_id(
        train_dataset.rows
        + val_dataset.rows
        + (calibration_dataset.rows if calibration_dataset is not None else [])
    )

    statement_embedding_config = config["model"].get(
        "statement_embeddings",
        {"mode": "learned_id"},
    )
    statement_embedding_mode = str(statement_embedding_config.get("mode", "learned_id"))
    frozen_statement_embeddings = None
    if statement_embedding_mode == "frozen_cached":
        frozen_statement_embeddings = load_frozen_statement_embeddings(
            statement_embedding_config["path"],
            train_dataset.statement_to_index,
            expected_statement_texts,
            expected_cache=statement_embedding_config,
        )
        statement_dim = int(frozen_statement_embeddings.shape[1])
    elif statement_embedding_mode == "learned_id":
        statement_dim = int(config["model"].get("statement_dim", 512))
    else:
        raise ValueError(f"unsupported statement embedding mode: {statement_embedding_mode}")

    source_snapshot = build_source_snapshot(require_clean=True)
    run_lock = build_run_lock(
        config,
        git_commit=git_commit(),
        source_snapshot=source_snapshot,
        dataset_lock=dataset_lock,
    )
    run_lock_sha256 = canonical_json_sha256(run_lock)
    save_json(output_dir / "run_lock.json", run_lock)

    visual_model, processor, qwen_config = load_qwen35_visual_and_processor(
        config["model"]["path"],
        dtype=str(config["model"].get("dtype", "bf16")),
        attention_implementation=str(
            config["model"].get("attention_implementation", "eager")
        ),
    )
    freeze_backbone = bool(config["model"].get("freeze_backbone", True))
    for parameter in visual_model.parameters():
        parameter.requires_grad = not freeze_backbone
    visual_adapter = Qwen35VisionAdapter(
        visual_model,
        spatial_merge_size=int(qwen_config["vision_config"]["spatial_merge_size"]),
    ).to(device)
    visual_dim = int(qwen_config["vision_config"]["hidden_size"])
    intervention_config = config["bives"].get("interventions", {})
    if str(intervention_config.get("replacement", "zero")) != "zero":
        raise ValueError("active BiVES mainline currently supports only zero intervention replacement")
    head_config = BiVESModelConfig(
        visual_dim=visual_dim,
        statement_dim=statement_dim,
        fusion_dim=int(config["bives"].get("fusion_dim", 512)),
        evidence_max=float(config["bives"].get("evidence_max", 8.0)),
        gate_mode=str(config["bives"]["mask"].get("type", "soft_topk")),
        topk=int(config["bives"]["mask"].get("topk", 16)),
        gate_temperature=float(config["bives"]["mask"].get("temperature", 0.5)),
        tau_a=float(config["bives"]["decoder"].get("tau_a", 1.0)),
        tau_d=float(config["bives"]["decoder"].get("tau_d", 1.0)),
        tau_p=float(config["bives"]["decoder"].get("tau_p", 1.0)),
        num_controls=int(intervention_config.get("num_controls", 4)),
        control_mode=str(intervention_config.get("control_mode", "random_disjoint")),
        contextual_layers=int(config["bives"].get("contextual_layers", 1)),
        contextual_heads=int(config["bives"].get("contextual_heads", 4)),
        contextual_dropout=float(config["bives"].get("contextual_dropout", 0.0)),
    )
    experiment = BiVESExperiment(
        visual_adapter,
        num_statements=len(train_dataset.statement_to_index),
        statement_dim=statement_dim,
        head_config=head_config,
        frozen_statement_embeddings=frozen_statement_embeddings,
    ).to(device)
    image_size = int(config["data"].get("image_size", 448))
    grouped_collator = Qwen35BiVESCollator(
        processor,
        image_size=image_size,
        include_group_indices=True,
        split="train",
        training_seed=int(config.get("seed", 17)),
        evaluation_control_seed=int(evaluation_config["control_seed"]),
    )
    sampling = config.get("sampling", {})
    if sampling.get("type") != "same_statement_state_group":
        raise ValueError(
            "active BiVES training requires sampling.type=same_statement_state_group"
        )
    groups_per_batch = int(sampling.get("groups_per_batch", 1))
    sampler_kwargs = {
        "groups_per_batch": groups_per_batch,
        "states": tuple(sampling.get("states", ("support", "contradict", "uncertain", "insufficient"))),
        "seed": int(config.get("seed", 17)),
        "require_complete_groups": bool(sampling.get("require_complete_groups", True)),
    }
    train_batch_sampler = SameStatementStateBatchSampler(
        train_dataset,
        shuffle=True,
        **sampler_kwargs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=grouped_collator,
    )
    evaluation_batch_size = int(evaluation_config.get("batch_size", groups_per_batch * 4))

    def make_primary_eval_loader(
        dataset: BiVESManifestDataset | None,
        split: str,
    ) -> DataLoader | None:
        if dataset is None:
            return None
        collator = Qwen35BiVESCollator(
            processor,
            image_size=image_size,
            include_group_indices=False,
            split=split,
            training_seed=int(config.get("seed", 17)),
            evaluation_control_seed=int(evaluation_config["control_seed"]),
        )
        return DataLoader(
            dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=int(config["data"].get("num_workers", 0)),
            collate_fn=collator,
        )

    def make_grouped_eval_loader(
        dataset: BiVESManifestDataset,
        split: str,
    ) -> DataLoader:
        collator = Qwen35BiVESCollator(
            processor,
            image_size=image_size,
            include_group_indices=True,
            split=split,
            training_seed=int(config.get("seed", 17)),
            evaluation_control_seed=int(evaluation_config["control_seed"]),
        )
        return DataLoader(
            dataset,
            batch_sampler=SameStatementStateBatchSampler(
                dataset,
                shuffle=False,
                **sampler_kwargs,
            ),
            num_workers=int(config["data"].get("num_workers", 0)),
            collate_fn=collator,
        )

    val_loader = make_primary_eval_loader(val_dataset, "val")
    assert val_loader is not None
    val_grouped_loader = make_grouped_eval_loader(val_dataset, "val")
    calibration_loader = make_primary_eval_loader(calibration_dataset, "calibration")
    loss_config = BiVESLossConfig(**config.get("loss", {}))
    if head_config.gate_mode == "soft_topk" and loss_config.lambda_min != 0:
        raise ValueError(
            "fixed exact-K is a budgeted evidence set; lambda_min must be 0 "
            "until an adaptive hard-concrete/L0 gate is implemented"
        )
    loss_fn = BiVESLoss(loss_config)
    primary_eval_loss_fn = BiVESLoss(
        replace(loss_config, lambda_pair=0.0, lambda_u_pol=0.0)
    )
    head_parameters = [
        parameter
        for parameter in list(experiment.statement_table.parameters()) + list(experiment.head.parameters())
        if parameter.requires_grad
    ]
    parameter_groups = [
        {
            "params": head_parameters,
            "lr": float(config["training"].get("lr_new_layers", 1e-4)),
        }
    ]
    if not freeze_backbone:
        parameter_groups.append(
            {
                "params": [parameter for parameter in experiment.backbone.parameters() if parameter.requires_grad],
                "lr": float(config["training"].get("lr_backbone", 1e-5)),
            }
        )
    optimizer = AdamW(parameter_groups, weight_decay=float(config["training"].get("weight_decay", 0.05)))

    max_steps = int(config["training"]["max_steps"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(config["training"].get("min_lr", 0.0)),
    )
    eval_interval = int(config["training"].get("eval_interval", 100))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    gradient_accumulation_steps = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    selection_metric = str(evaluation_config.get("selection_metric", "nll"))
    selection_mode = str(evaluation_config.get("selection_mode", "min"))
    if selection_mode not in {"min", "max"}:
        raise ValueError("evaluation.selection_mode must be min or max")
    step = 0
    best_selection_score = float("inf") if selection_mode == "min" else float("-inf")
    events: list[dict[str, Any]] = []
    resume_epoch = 0
    resume_batch_index = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("run_lock_sha256") != run_lock_sha256:
            raise ValueError("resume checkpoint belongs to a different run lock")
        load_checkpoint_model_state(experiment, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        best_selection_score = float(
            checkpoint.get(
                "best_selection_score",
                checkpoint.get("best_val_loss", best_selection_score),
            )
        )
        events = list(checkpoint.get("events", []))
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_batch_index = int(checkpoint.get("batch_index", 0))
        accumulated = int(checkpoint.get("accumulated_micro_steps", 0))
        if accumulated != 0:
            raise ValueError(
                "checkpoint contains partial gradient accumulation without serialized gradients"
            )
        restore_rng_state(checkpoint["rng_state"])
    started = time.time()
    experiment.train()
    epoch = resume_epoch
    cursor_epoch = resume_epoch
    cursor_batch_index = resume_batch_index
    optimizer.zero_grad(set_to_none=True)
    accumulated_micro_steps = 0
    while step < max_steps:
        train_batch_sampler.set_epoch(epoch)
        grouped_collator.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == resume_epoch and batch_index < resume_batch_index:
                continue
            batch = move_to_device(batch, device)
            outputs, grids = experiment(
                batch["pixel_values"],
                batch["image_grid_thw"],
                batch["statement_indices"],
                batch["content_valid_mask"],
                run_interventions=True,
                control_seeds=batch["control_seeds"],
            )
            losses = loss_fn(
                outputs,
                batch["targets"],
                require_uniform_grid(grids),
                batch["support_pair_indices"],
                batch["contradict_pair_indices"],
                batch["uncertain_indices"],
            )
            (losses["total"] / gradient_accumulation_steps).backward()
            accumulated_micro_steps += 1
            if accumulated_micro_steps < gradient_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(experiment.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_micro_steps = 0
            step += 1
            if batch_index + 1 >= len(train_loader):
                cursor_epoch = epoch + 1
                cursor_batch_index = 0
            else:
                cursor_epoch = epoch
                cursor_batch_index = batch_index + 1
            if step % eval_interval == 0 or step == max_steps:
                validation, validation_rows = evaluate(
                    experiment,
                    val_loader,
                    primary_eval_loss_fn,
                    device,
                    assert_full_coverage=True,
                )
                grouped_validation = evaluate_grouped_mechanisms(
                    experiment,
                    val_grouped_loader,
                    device,
                    pair_margin=loss_config.pair_margin,
                )
                event = {
                    "step": step,
                    "train_loss": float(losses["total"].detach().cpu()),
                    "grouped_mechanisms": grouped_validation,
                    **validation,
                }
                events.append(event)
                print(json.dumps(event, ensure_ascii=False))
                save_json(output_dir / f"metrics_step_{step}.json", event)
                with (output_dir / f"val_predictions_step_{step}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for row in validation_rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if selection_metric not in validation:
                    raise KeyError(
                        f"selection metric {selection_metric!r} is absent from validation metrics"
                    )
                selection_score = float(validation[selection_metric])
                improved = (
                    selection_score < best_selection_score
                    if selection_mode == "min"
                    else selection_score > best_selection_score
                )
                if improved:
                    best_selection_score = selection_score
                checkpoint = {
                    "step": step,
                    "best_selection_metric": selection_metric,
                    "best_selection_mode": selection_mode,
                    "best_selection_score": best_selection_score,
                    "best_val_loss": validation["loss"],
                    "events": events,
                    "model_family": "Qwen3.5",
                    "model_path": str(config["model"]["path"]),
                    "statement_to_index": train_dataset.statement_to_index,
                    "statement_text_by_id": expected_statement_texts,
                    "run_lock": run_lock,
                    "run_lock_sha256": run_lock_sha256,
                    "statement_table": experiment.statement_table.state_dict(),
                    "bives_head": experiment.head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": cursor_epoch,
                    "batch_index": cursor_batch_index,
                    "accumulated_micro_steps": accumulated_micro_steps,
                    "sampler_state": {
                        "epoch": cursor_epoch,
                        "next_batch_index": cursor_batch_index,
                    },
                    "rng_state": capture_rng_state(),
                    "config": config,
                }
                if not freeze_backbone:
                    checkpoint["backbone"] = experiment.backbone.state_dict()
                (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
                if improved:
                    torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")
            if step >= max_steps:
                break
        epoch += 1
        resume_batch_index = 0

    checkpoint = {
        "step": step,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "statement_to_index": train_dataset.statement_to_index,
        "statement_text_by_id": expected_statement_texts,
        "run_lock": run_lock,
        "run_lock_sha256": run_lock_sha256,
        "statement_table": experiment.statement_table.state_dict(),
        "bives_head": experiment.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": cursor_epoch,
        "batch_index": cursor_batch_index,
        "accumulated_micro_steps": accumulated_micro_steps,
        "sampler_state": {
            "epoch": cursor_epoch,
            "next_batch_index": cursor_batch_index,
        },
        "rng_state": capture_rng_state(),
        "best_selection_metric": selection_metric,
        "best_selection_mode": selection_mode,
        "best_selection_score": best_selection_score,
        "events": events,
        "config": config,
    }
    if not freeze_backbone:
        checkpoint["backbone"] = experiment.backbone.state_dict()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / "checkpoints" / "final.pt")
    best_path = output_dir / "checkpoints" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("training completed without a validation-selected best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    load_checkpoint_model_state(experiment, best_checkpoint)
    selected_best_step = int(best_checkpoint["step"])
    base_checkpoint_sha256 = file_sha256(best_path)
    uncalibrated_temperatures = {
        name: float(getattr(experiment.head.decoder, name).detach().cpu())
        for name in ("tau_a", "tau_d", "tau_p")
    }

    def write_predictions(name: str, prediction_rows: list[dict[str, Any]]) -> None:
        with (output_dir / f"{name}_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_metrics: dict[str, Any] = {}
    calibration_temperatures: dict[str, float] | None = None
    calibration_pre_rows: list[dict[str, Any]] = []
    if calibration_loader is not None:
        calibration_pre, calibration_pre_rows = evaluate(
            experiment,
            calibration_loader,
            primary_eval_loss_fn,
            device,
            assert_full_coverage=True,
        )
        split_metrics["calibration_pre"] = calibration_pre
        write_predictions("calibration_pre", calibration_pre_rows)
        calibration_temperatures = fit_decoder_temperatures(
            torch.tensor([row["evidence_pos"] for row in calibration_pre_rows]),
            torch.tensor([row["evidence_neg"] for row in calibration_pre_rows]),
            torch.tensor([row["target"] for row in calibration_pre_rows]),
            initial=(
                uncalibrated_temperatures["tau_a"],
                uncalibrated_temperatures["tau_d"],
                uncalibrated_temperatures["tau_p"],
            ),
            max_iter=int(evaluation_config.get("calibration_max_iter", 100)),
        )

    if calibration_temperatures is not None:
        set_decoder_temperatures(experiment, calibration_temperatures)
        calibration_post, calibration_post_rows = evaluate(
            experiment,
            calibration_loader,
            primary_eval_loss_fn,
            device,
            assert_full_coverage=True,
        )
        split_metrics["calibration_post"] = calibration_post
        write_predictions("calibration_post", calibration_post_rows)
        calibrated_checkpoint = {
            **best_checkpoint,
            "selected_best_step": selected_best_step,
            "uncalibrated_temperatures": uncalibrated_temperatures,
            "calibrated_temperatures": calibration_temperatures,
            "bives_head": experiment.head.state_dict(),
        }
        torch.save(
            calibrated_checkpoint,
            output_dir / "checkpoints" / "best_calibrated.pt",
        )
        calibration_artifact = {
                "format_version": 2,
                "calibration_algorithm": "three_temperature_lbfgs_v1",
                "selected_best_step": selected_best_step,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "run_lock_sha256": run_lock_sha256,
                "statement_cache_sha256": run_lock["statement_cache_sha256"],
                "statement_vocabulary_sha256": run_lock[
                    "statement_vocabulary_sha256"
                ],
                "uncalibrated_temperatures": uncalibrated_temperatures,
                "calibrated_temperatures": calibration_temperatures,
                "calibration_manifest": str(config["data"]["calibration_manifest"]),
                "calibration_manifest_sha256": file_sha256(
                    config["data"]["calibration_manifest"]
                ),
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "evaluation_control_seed": int(evaluation_config["control_seed"]),
                "calibration_predictions_file": "calibration_pre_predictions.jsonl",
                "calibration_predictions_sha256": file_sha256(
                    output_dir / "calibration_pre_predictions.jsonl"
                ),
                "calibration_pre_nll": float(calibration_pre["nll"]),
                "calibration_post_nll": float(calibration_post["nll"]),
            }
        calibration_artifact["canonical_artifact_sha256"] = canonical_json_sha256(
            calibration_artifact
        )
        save_json(output_dir / "calibration_artifact.json", calibration_artifact)

    final_metrics = {
        "step": step,
        "elapsed_seconds": time.time() - started,
        "model_family": "Qwen3.5",
        "model_path": str(config["model"]["path"]),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "best_selection_metric": selection_metric,
        "best_selection_mode": selection_mode,
        "best_selection_score": best_selection_score,
        "selected_best_step": selected_best_step,
        "selected_checkpoint": str(best_path),
        "uncalibrated_temperatures": uncalibrated_temperatures,
        "calibrated_temperatures": calibration_temperatures,
        "split_metrics": split_metrics,
        "manifest_sha256": {
            split: file_sha256(path)
            for split, path in manifest_paths.items()
        },
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("transformers", "safetensors", "numpy", "pillow")
        },
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pid": os.getpid(),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "control_protocol_version": CONTROL_PROTOCOL_VERSION,
        "events": events,
    }
    save_json(output_dir / "metrics_final.json", final_metrics)


if __name__ == "__main__":
    main()
