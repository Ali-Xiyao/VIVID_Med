"""Qwen3.5 frozen-vision dense verifier used by the ARISE oracle bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.polarity_runtime import (
    extract_qwen35_patch_batch,
    load_locked_polarity_checkpoint,
    score_statements,
)
from bives_cxr.provenance import file_sha256
from bives_cxr.qwen35_localization_audit import QWEN35_2B_SNAPSHOT_SHA256

from .mil_verifier import MILVerifierConfig, PatchMILVerifier


EXPECTED_DENSE_CHECKPOINT_SHA256 = (
    "7429c5a071388ffd56a75e03e24af5903e0306e733cc598ab724bab0c91bea45"
)
EXPECTED_CACHE_LOCK_SHA256 = (
    "503ead96a0de948b56361b8097bd14cc1ba0942878b56cb3aa0ba2e39d3ec0f2"
)
EXPECTED_POOLED_MODEL_SHA256 = {
    "consolidation": "be783df5c038d9be80933bc9181fde7597f4dd8850ab76b289eb497be9d5bfea",
    "pleural_effusion": "a4f09f5ea93e4314479f8ddb764053037fed6245079647c18031ee67e4f131e7",
}
EXPECTED_MIL_CHECKPOINT_SHA256 = (
    "e5d1de32e126f3dcc58dff7c283f2612d226ef4018a8871128a2332604e655bb"
)
EXPECTED_BOX_MIL_CHECKPOINT_SHA256 = (
    "c4cbd7de800d0bfddb88147c36e4e44bb1fe8109eb2156067414f3378cdd9b5a"
)
EXPECTED_BOX_MIL_TRAIN_CACHE_LOCK_SHA256 = (
    "2885eb65553451264bbdb751b8437f5db80bc518229449432463e67d07c550ad"
)
EXPECTED_BOX_MIL_VAL_CACHE_LOCK_SHA256 = (
    "fdb401ce161bd278d1c3cc29ac6e9542084616d22e0102ff1292cad7e72e8682"
)
EXPECTED_BOX_SUPERVISION_SHA256 = (
    "d282d8596215c7cdadbd100593baf2b0c000d4cac3eece75898220ecba041c75"
)
ORACLE_MODEL_IDS = {
    "trained_dense_sc_b1": "arise_dense_qwen35_2b_b1_step300",
    "frozen_pooled_logistic_margin": "arise_dense_qwen35_2b_pooled_logistic",
    "arise_patch_mil_dense_sc_v1": "arise_dense_qwen35_2b_patch_mil_step300",
    "arise_patch_mil_vindr_box_overlap_v2": "arise_dense_qwen35_2b_patch_mil_vindr_box_step200",
}


def oracle_model_id_for_head(head: str) -> str:
    """Resolve one explicitly frozen scorer head to its audit-row identity."""

    try:
        return ORACLE_MODEL_IDS[head]
    except KeyError as error:
        raise ValueError(f"unregistered ARISE oracle head: {head}") from error


def dense_oracle_progress_identity_matches(
    existing: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Accept one exact pre-backend B1 identity and no other migration."""

    if existing == current:
        return True
    if current.get("mil_checkpoint_sha256") is None:
        pre_mil = dict(current)
        pre_mil.pop("mil_checkpoint_sha256", None)
        if existing == pre_mil:
            return True
    if current.get("backend") == "b1_dense" and current.get("pooled_model_sha256") is None:
        legacy = dict(current)
        legacy.pop("backend", None)
        legacy.pop("pooled_model_sha256", None)
        legacy.pop("mil_checkpoint_sha256", None)
        return existing == legacy
    return False


def pooled_logistic_margin(
    pooled: np.ndarray,
    *,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: np.ndarray,
) -> float:
    """Apply a frozen standardized binary logistic head fail-closed."""

    feature = np.asarray(pooled, dtype=np.float64).reshape(-1)
    mean = np.asarray(scaler_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(scaler_scale, dtype=np.float64).reshape(-1)
    weight = np.asarray(coefficient, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64).reshape(-1)
    if mean.shape != feature.shape or scale.shape != feature.shape:
        raise ValueError("ARISE pooled-verifier scaler shape changed")
    if weight.shape != (1, feature.size) or bias.shape != (1,):
        raise ValueError("ARISE pooled-verifier head shape changed")
    if not all(np.isfinite(array).all() for array in (feature, mean, scale, weight, bias)):
        raise ValueError("ARISE pooled-verifier payload is non-finite")
    if np.any(scale <= 0.0):
        raise ValueError("ARISE pooled-verifier scaler is non-positive")
    scaled = (feature - mean) / scale
    value = float(weight.reshape(-1) @ scaled + bias[0])
    if not np.isfinite(value):
        raise ValueError("ARISE pooled-verifier margin is non-finite")
    return value


def reconstruct_phase_h_explanation_mask(
    target_geometry: dict[str, Any],
    *,
    content_mask: np.ndarray,
    grid_size: int = 4,
) -> np.ndarray:
    """Recover the frozen 4x4 top-cell mask from its Phase-H geometry."""

    content = np.asarray(content_mask, dtype=bool)
    if content.ndim != 2 or not content.any():
        raise ValueError("content_mask must be a non-empty 2D mask")
    bbox = target_geometry.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Phase-H explanation geometry is missing bbox")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if not (0 <= x0 < x1 <= content.shape[1] and 0 <= y0 < y1 <= content.shape[0]):
        raise ValueError("Phase-H explanation bbox is invalid")
    y_edges = np.linspace(0, content.shape[0], grid_size + 1, dtype=np.int64)
    x_edges = np.linspace(0, content.shape[1], grid_size + 1, dtype=np.int64)
    row = min(grid_size - 1, max(0, y0 * grid_size // content.shape[0]))
    column = min(grid_size - 1, max(0, x0 * grid_size // content.shape[1]))
    mask = np.zeros_like(content)
    mask[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]] = True
    mask &= content
    if int(mask.sum()) != int(target_geometry.get("area_pixels", -1)):
        raise ValueError("reconstructed Phase-H explanation area changed")
    coordinates = np.argwhere(mask)
    reconstructed_bbox = [
        int(coordinates[:, 1].min()),
        int(coordinates[:, 0].min()),
        int(coordinates[:, 1].max()) + 1,
        int(coordinates[:, 0].max()) + 1,
    ]
    if reconstructed_bbox != [x0, y0, x1, y1]:
        raise ValueError("reconstructed Phase-H explanation bbox changed")
    return mask


class DenseVerifierScorer:
    """Score support with a trained dense head after full visual re-encoding."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        checkpoint_path: str | Path,
        cache_lock_path: str | Path,
        device: str,
        dtype: str = "bf16",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ARISE dense verifier requires local CUDA")
        self.model_path = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.cache_lock_path = Path(cache_lock_path)
        if file_sha256(self.checkpoint_path) != EXPECTED_DENSE_CHECKPOINT_SHA256:
            raise ValueError("ARISE dense checkpoint SHA256 changed")
        if file_sha256(self.cache_lock_path) != EXPECTED_CACHE_LOCK_SHA256:
            raise ValueError("ARISE dense cache-lock SHA256 changed")
        self.verifier, self.statement_to_index, checkpoint = load_locked_polarity_checkpoint(
            self.checkpoint_path,
            self.device,
        )
        if checkpoint.get("variant") != "B1_dense" or int(checkpoint.get("step", -1)) != 300:
            raise ValueError("ARISE oracle requires the locked B1 dense step-300 checkpoint")
        if checkpoint.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
            raise ValueError("ARISE dense checkpoint/cache binding changed")
        if set(self.statement_to_index) != {"consolidation", "pleural_effusion"}:
            raise ValueError("ARISE dense statement vocabulary changed")
        visual, self.processor, config = load_qwen35_visual_and_processor(
            self.model_path,
            dtype=dtype,
            attention_implementation="eager",
        )
        self.visual = visual.to(self.device).eval()
        self.adapter = Qwen35VisionAdapter(
            self.visual,
            spatial_merge_size=int(config["vision_config"]["spatial_merge_size"]),
        ).to(self.device).eval()
        self.validation = checkpoint["validation"]

    @torch.no_grad()
    def score(
        self,
        images: Sequence[Image.Image],
        *,
        statement_id: str,
        statement_text: str,
    ) -> list[float]:
        if statement_id not in self.statement_to_index:
            raise ValueError(f"unknown dense-verifier statement: {statement_id}")
        extracted = extract_qwen35_patch_batch(
            [image.convert("RGB") for image in images],
            [statement_text] * len(images),
            adapter=self.adapter,
            visual=self.visual,
            processor=self.processor,
            device=self.device,
            image_size=448,
        )
        values = []
        statement_index = self.statement_to_index[statement_id]
        for item in extracted:
            output = score_statements(
                self.verifier,
                item["patch_tokens"],
                item["valid_mask"],
                [statement_index],
            )
            value = float(output["signed_evidence"][0].detach().cpu())
            if not np.isfinite(value):
                raise ValueError("ARISE dense-verifier score is non-finite")
            values.append(value)
        return values

    def identity(self) -> dict[str, Any]:
        return {
            "family": "Qwen3.5",
            "scale": "2B",
            "vision_snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
            "head": "trained_dense_sc_b1",
            "checkpoint_sha256": EXPECTED_DENSE_CHECKPOINT_SHA256,
            "checkpoint_step": 300,
            "cache_lock_sha256": EXPECTED_CACHE_LOCK_SHA256,
            "score": "dense signed_evidence margin",
            "validation_macro": self.validation["macro"],
        }


class PooledLogisticVerifierScorer:
    """Calibrated-margin repair over the same frozen Qwen3.5 visual tokens."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        pooled_model_dir: str | Path,
        cache_lock_path: str | Path,
        device: str,
        dtype: str = "bf16",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ARISE pooled verifier requires local CUDA")
        self.model_path = Path(model_path)
        self.pooled_model_dir = Path(pooled_model_dir)
        self.cache_lock_path = Path(cache_lock_path)
        if file_sha256(self.cache_lock_path) != EXPECTED_CACHE_LOCK_SHA256:
            raise ValueError("ARISE pooled cache-lock SHA256 changed")
        self.models: dict[str, dict[str, np.ndarray]] = {}
        for finding, expected_hash in EXPECTED_POOLED_MODEL_SHA256.items():
            path = self.pooled_model_dir / f"{finding}_model.npz"
            if file_sha256(path) != expected_hash:
                raise ValueError(f"ARISE pooled model SHA256 changed: {finding}")
            with np.load(path, allow_pickle=False) as payload:
                model = {name: np.asarray(payload[name], dtype=np.float64) for name in payload.files}
            required = {"scaler_mean", "scaler_scale", "coefficient", "intercept"}
            if set(model) != required:
                raise ValueError(f"ARISE pooled model fields changed: {finding}")
            dimension = model["scaler_mean"].size
            if (
                model["scaler_mean"].shape != (dimension,)
                or model["scaler_scale"].shape != (dimension,)
                or model["coefficient"].shape != (1, dimension)
                or model["intercept"].shape != (1,)
            ):
                raise ValueError(f"ARISE pooled model shape changed: {finding}")
            self.models[finding] = model
        visual, self.processor, config = load_qwen35_visual_and_processor(
            self.model_path,
            dtype=dtype,
            attention_implementation="eager",
        )
        self.visual = visual.to(self.device).eval()
        self.adapter = Qwen35VisionAdapter(
            self.visual,
            spatial_merge_size=int(config["vision_config"]["spatial_merge_size"]),
        ).to(self.device).eval()

    @torch.no_grad()
    def score(
        self,
        images: Sequence[Image.Image],
        *,
        statement_id: str,
        statement_text: str,
    ) -> list[float]:
        if statement_id not in self.models:
            raise ValueError(f"unknown pooled-verifier statement: {statement_id}")
        extracted = extract_qwen35_patch_batch(
            [image.convert("RGB") for image in images],
            [statement_text] * len(images),
            adapter=self.adapter,
            visual=self.visual,
            processor=self.processor,
            device=self.device,
            image_size=448,
        )
        model = self.models[statement_id]
        values = []
        for item in extracted:
            pooled = (
                item["patch_tokens"][item["valid_mask"]]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            value = pooled_logistic_margin(pooled, **model)
            values.append(value)
        return values

    def identity(self) -> dict[str, Any]:
        return {
            "family": "Qwen3.5",
            "scale": "2B",
            "vision_snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
            "head": "frozen_pooled_logistic_margin",
            "model_sha256": EXPECTED_POOLED_MODEL_SHA256,
            "cache_lock_sha256": EXPECTED_CACHE_LOCK_SHA256,
            "score": "logistic decision_function margin",
        }


class MILVerifierScorer:
    """Score full reencodings with the frozen ARISE patch-MIL verifier."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        checkpoint_path: str | Path,
        cache_lock_path: str | Path | None,
        device: str,
        dtype: str = "bf16",
        checkpoint_identity: str = "dense_sc",
        train_cache_lock_path: str | Path | None = None,
        val_cache_lock_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("ARISE MIL verifier requires local CUDA")
        self.model_path = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.cache_lock_path = Path(cache_lock_path) if cache_lock_path else None
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_identity == "dense_sc":
            if file_sha256(self.checkpoint_path) != EXPECTED_MIL_CHECKPOINT_SHA256:
                raise ValueError("ARISE MIL checkpoint SHA256 changed")
            if self.cache_lock_path is None or file_sha256(self.cache_lock_path) != EXPECTED_CACHE_LOCK_SHA256:
                raise ValueError("ARISE MIL cache-lock SHA256 changed")
            if (
                checkpoint.get("schema_version") != "arise-patch-mil-checkpoint-v1"
                or checkpoint.get("variant") != "arise_patch_mil_dense_sc_v1"
                or int(checkpoint.get("step", -1)) != 300
            ):
                raise ValueError("ARISE MIL checkpoint identity changed")
            if checkpoint.get("cache_lock_sha256") != EXPECTED_CACHE_LOCK_SHA256:
                raise ValueError("ARISE MIL checkpoint/cache binding changed")
            self._identity = {
                "head": "arise_patch_mil_dense_sc_v1",
                "checkpoint_sha256": EXPECTED_MIL_CHECKPOINT_SHA256,
                "checkpoint_step": 300,
                "cache_lock_sha256": EXPECTED_CACHE_LOCK_SHA256,
            }
        elif checkpoint_identity == "vindr_box_v2":
            train_lock = Path(train_cache_lock_path) if train_cache_lock_path else None
            val_lock = Path(val_cache_lock_path) if val_cache_lock_path else None
            if file_sha256(self.checkpoint_path) != EXPECTED_BOX_MIL_CHECKPOINT_SHA256:
                raise ValueError("ARISE box-MIL checkpoint SHA256 changed")
            if train_lock is None or file_sha256(train_lock) != EXPECTED_BOX_MIL_TRAIN_CACHE_LOCK_SHA256:
                raise ValueError("ARISE box-MIL train cache-lock SHA256 changed")
            if val_lock is None or file_sha256(val_lock) != EXPECTED_BOX_MIL_VAL_CACHE_LOCK_SHA256:
                raise ValueError("ARISE box-MIL validation cache-lock SHA256 changed")
            if (
                checkpoint.get("schema_version") != "arise-patch-mil-vindr-box-checkpoint-v2"
                or checkpoint.get("variant") != "arise_patch_mil_vindr_box_overlap_v2"
                or int(checkpoint.get("step", -1)) != 200
                or bool(checkpoint.get("test_opened", True))
            ):
                raise ValueError("ARISE box-MIL checkpoint identity changed")
            if (
                checkpoint.get("initial_checkpoint_sha256") != EXPECTED_MIL_CHECKPOINT_SHA256
                or checkpoint.get("train_cache_lock_sha256") != EXPECTED_BOX_MIL_TRAIN_CACHE_LOCK_SHA256
                or checkpoint.get("val_cache_lock_sha256") != EXPECTED_BOX_MIL_VAL_CACHE_LOCK_SHA256
                or checkpoint.get("box_supervision_sha256") != EXPECTED_BOX_SUPERVISION_SHA256
            ):
                raise ValueError("ARISE box-MIL provenance binding changed")
            self._identity = {
                "head": "arise_patch_mil_vindr_box_overlap_v2",
                "checkpoint_sha256": EXPECTED_BOX_MIL_CHECKPOINT_SHA256,
                "checkpoint_step": 200,
                "initial_checkpoint_sha256": EXPECTED_MIL_CHECKPOINT_SHA256,
                "train_cache_lock_sha256": EXPECTED_BOX_MIL_TRAIN_CACHE_LOCK_SHA256,
                "val_cache_lock_sha256": EXPECTED_BOX_MIL_VAL_CACHE_LOCK_SHA256,
                "box_supervision_sha256": EXPECTED_BOX_SUPERVISION_SHA256,
            }
        else:
            raise ValueError("unknown ARISE MIL checkpoint identity")
        self.statement_to_index = dict(checkpoint["statement_to_index"])
        if set(self.statement_to_index) != {"consolidation", "pleural_effusion"}:
            raise ValueError("ARISE MIL statement vocabulary changed")
        config = MILVerifierConfig(**checkpoint["model_config"])
        self.verifier = PatchMILVerifier(config).to(self.device).eval()
        self.verifier.load_state_dict(checkpoint["model"], strict=True)
        self.validation = checkpoint["validation"]
        visual, self.processor, vision_config = load_qwen35_visual_and_processor(
            self.model_path,
            dtype=dtype,
            attention_implementation="eager",
        )
        self.visual = visual.to(self.device).eval()
        self.adapter = Qwen35VisionAdapter(
            self.visual,
            spatial_merge_size=int(vision_config["vision_config"]["spatial_merge_size"]),
        ).to(self.device).eval()

    @torch.no_grad()
    def score(
        self,
        images: Sequence[Image.Image],
        *,
        statement_id: str,
        statement_text: str,
    ) -> list[float]:
        if statement_id not in self.statement_to_index:
            raise ValueError(f"unknown MIL-verifier statement: {statement_id}")
        extracted = extract_qwen35_patch_batch(
            [image.convert("RGB") for image in images],
            [statement_text] * len(images),
            adapter=self.adapter,
            visual=self.visual,
            processor=self.processor,
            device=self.device,
            image_size=448,
        )
        statement_index = self.statement_to_index[statement_id]
        values = []
        for item in extracted:
            output = self.verifier(
                item["patch_tokens"].unsqueeze(0),
                item["valid_mask"].unsqueeze(0),
                torch.tensor([statement_index], device=self.device),
            )
            value = float(output["margin"][0].detach().cpu())
            if not np.isfinite(value):
                raise ValueError("ARISE MIL-verifier margin is non-finite")
            values.append(value)
        return values

    def identity(self) -> dict[str, Any]:
        return {
            "family": "Qwen3.5",
            "scale": "2B",
            "vision_snapshot_sha256": QWEN35_2B_SNAPSHOT_SHA256,
            "score": "patch MIL smooth-max/mean margin",
            "validation": self.validation,
            **self._identity,
        }
