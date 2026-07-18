"""Deterministic pixel masks and paired metrics for expert-box interventions."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


LOCAL_MEAN_RING_WIDTH = 8
MASKED_GAUSSIAN_SIGMA = 8.0
MASKED_GAUSSIAN_TRUNCATE = 3.0


def union_box_mask(
    width: int,
    height: int,
    boxes: Iterable[dict[str, Any]],
    *,
    dilation_fraction: float = 0.0,
) -> np.ndarray:
    """Rasterize the clipped union of expert boxes with optional small dilation."""

    if width <= 0 or height <= 0 or dilation_fraction < 0:
        raise ValueError("width/height must be positive and dilation_fraction non-negative")
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x0 = float(box["x_min"])
        y0 = float(box["y_min"])
        x1 = float(box["x_max"])
        y1 = float(box["y_max"])
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid expert box: {box}")
        dx = (x1 - x0) * dilation_fraction
        dy = (y1 - y0) * dilation_fraction
        left = max(0, int(np.floor(x0 - dx)))
        top = max(0, int(np.floor(y0 - dy)))
        right = min(width, int(np.ceil(x1 + dx)))
        bottom = min(height, int(np.ceil(y1 + dy)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    if not mask.any():
        raise ValueError("expert box union is empty after clipping")
    return mask


def transform_mask_to_letterbox(
    mask: np.ndarray,
    content_box: tuple[int, int, int, int],
    image_size: int,
) -> np.ndarray:
    """Resize an original-image boolean mask into the shared letterbox canvas."""

    if mask.ndim != 2 or image_size <= 0:
        raise ValueError("mask must be 2D and image_size positive")
    left, top, right, bottom = content_box
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (right - left, bottom - top), Image.Resampling.NEAREST
    )
    canvas = np.zeros((image_size, image_size), dtype=bool)
    canvas[top:bottom, left:right] = np.asarray(resized) > 0
    return canvas


def deterministic_disjoint_control_mask(
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
) -> np.ndarray:
    """Sample an exact-area control from content pixels disjoint from target."""

    if target_mask.shape != content_mask.shape or target_mask.ndim != 2:
        raise ValueError("target/content masks must be shape-matched 2D arrays")
    target = target_mask.astype(bool)
    content = content_mask.astype(bool)
    area = int(target.sum())
    candidates = np.flatnonzero(content.reshape(-1) & ~target.reshape(-1))
    if area <= 0 or len(candidates) < area:
        raise ValueError("not enough disjoint content pixels for equal-area control")
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little")
    selected = np.random.default_rng(seed).choice(candidates, size=area, replace=False)
    control = np.zeros(target.size, dtype=bool)
    control[selected] = True
    return control.reshape(target.shape)


def deterministic_random_mask(
    area: int,
    content_mask: np.ndarray,
    *,
    seed_key: str,
) -> np.ndarray:
    """Sample a deterministic exact-area random localization baseline."""

    content = content_mask.astype(bool)
    candidates = np.flatnonzero(content.reshape(-1))
    if area <= 0 or len(candidates) < area:
        raise ValueError("random mask area must fit inside content")
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little")
    selected = np.random.default_rng(seed).choice(candidates, size=area, replace=False)
    mask = np.zeros(content.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(content.shape)


def patch_gate_to_pixel_mask(
    gate: torch.Tensor,
    grid_hw: tuple[int, int],
    image_size: int,
) -> np.ndarray:
    """Project an exact-K patch gate to a deterministic input-pixel mask."""

    height, width = map(int, grid_hw)
    flat = gate.detach().to(device="cpu").bool().reshape(-1)
    if flat.numel() != height * width:
        raise ValueError("gate length does not match grid_hw")
    grid = flat.reshape(1, 1, height, width).float()
    pixels = F.interpolate(grid, size=(image_size, image_size), mode="nearest")
    return pixels[0, 0].bool().numpy()


def delete_pixels(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    if mask.shape != array.shape[:2]:
        raise ValueError("delete mask must match image geometry")
    array[mask] = 0
    return Image.fromarray(array, mode="RGB")


def retain_pixels(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    if mask.shape != array.shape[:2]:
        raise ValueError("retention mask must match image geometry")
    array[~mask] = 0
    return Image.fromarray(array, mode="RGB")


def replace_with_local_ring_mean(
    image: Image.Image,
    mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    ring_width: int = LOCAL_MEAN_RING_WIDTH,
) -> Image.Image:
    """Replace masked pixels by the fixed exterior-ring RGB mean."""

    array = np.asarray(image.convert("RGB")).copy()
    if mask.shape != array.shape[:2] or content_mask.shape != mask.shape:
        raise ValueError("local-mean masks must match image geometry")
    target = mask.astype(bool)
    content = content_mask.astype(bool)
    if ring_width <= 0 or not target.any() or bool((target & ~content).any()):
        raise ValueError("local-mean mask must be non-empty, contained, and use a positive ring")
    ring = (
        ndimage.binary_dilation(target, iterations=int(ring_width))
        & content
        & ~target
    )
    if not ring.any():
        raise ValueError("local-mean exterior ring is empty")
    mean_rgb = np.rint(array[ring].astype(np.float64).mean(axis=0)).astype(np.uint8)
    array[target] = mean_rgb
    return Image.fromarray(array, mode="RGB")


def replace_with_masked_gaussian_blur(
    image: Image.Image,
    mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    sigma: float = MASKED_GAUSSIAN_SIGMA,
    truncate: float = MASKED_GAUSSIAN_TRUNCATE,
) -> Image.Image:
    """Replace masked pixels by a content-normalized deterministic Gaussian blur."""

    array = np.asarray(image.convert("RGB")).copy()
    if mask.shape != array.shape[:2] or content_mask.shape != mask.shape:
        raise ValueError("Gaussian-blur masks must match image geometry")
    target = mask.astype(bool)
    content = content_mask.astype(bool)
    if sigma <= 0 or truncate <= 0 or not target.any() or bool((target & ~content).any()):
        raise ValueError("Gaussian-blur mask must be non-empty and contained")
    weights = ndimage.gaussian_filter(
        content.astype(np.float64),
        sigma=float(sigma),
        mode="constant",
        cval=0.0,
        truncate=float(truncate),
    )
    if bool((weights[target] <= 0).any()):
        raise ValueError("Gaussian normalization has zero support inside intervention mask")
    blurred = np.empty_like(array)
    for channel in range(3):
        numerator = ndimage.gaussian_filter(
            array[:, :, channel].astype(np.float64) * content,
            sigma=float(sigma),
            mode="constant",
            cval=0.0,
            truncate=float(truncate),
        )
        blurred[:, :, channel] = np.clip(
            np.rint(numerator / np.maximum(weights, np.finfo(np.float64).tiny)),
            0,
            255,
        ).astype(np.uint8)
    array[target] = blurred[target]
    return Image.fromarray(array, mode="RGB")


def paired_intervention_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    records = [{**row, "_unit_id": str(row.get("unit_id", index))} for index, row in enumerate(rows)]
    if not records:
        raise ValueError("intervention rows are empty")
    per_finding: dict[str, dict[str, float | int]] = {}
    for finding in sorted({str(row["canonical_statement_id"]) for row in records}):
        subset = [row for row in records if row["canonical_statement_id"] == finding]
        target = np.asarray([float(row["original_score"]) - float(row["target_drop_score"]) for row in subset])
        control = np.asarray([float(row["original_score"]) - float(row["control_drop_score"]) for row in subset])
        keep = np.asarray([float(row["keep_score"]) for row in subset])
        per_finding[finding] = {
            "records": len(subset),
            "mean_target_deletion_effect": float(target.mean()),
            "mean_control_deletion_effect": float(control.mean()),
            "mean_tcig": float((target - control).mean()),
            "mean_evidence_only_score": float(keep.mean()),
            "target_greater_than_control_fraction": float((target > control).mean()),
        }
        if all("topk_localization_gain" in row for row in subset):
            per_finding[finding]["mean_topk_localization_gain"] = float(
                np.mean([float(row["topk_localization_gain"]) for row in subset])
            )
    units = sorted({row["_unit_id"] for row in records})
    by_unit = {unit: [row for row in records if row["_unit_id"] == unit] for unit in units}
    rng = np.random.default_rng(bootstrap_seed)
    replicate_tcig: dict[str, list[float]] = {finding: [] for finding in per_finding}
    replicate_localization: dict[str, list[float]] = {finding: [] for finding in per_finding}
    replicate_macro: list[float] = []
    for _ in range(int(bootstrap_replicates)):
        sample = [row for unit in rng.choice(units, size=len(units), replace=True) for row in by_unit[str(unit)]]
        values = []
        for finding in per_finding:
            subset = [row for row in sample if row["canonical_statement_id"] == finding]
            if not subset:
                continue
            tcig = np.mean(
                [
                    (float(row["original_score"]) - float(row["target_drop_score"]))
                    - (float(row["original_score"]) - float(row["control_drop_score"]))
                    for row in subset
                ]
            )
            replicate_tcig[finding].append(float(tcig))
            if all("topk_localization_gain" in row for row in subset):
                replicate_localization[finding].append(
                    float(np.mean([float(row["topk_localization_gain"]) for row in subset]))
                )
            values.append(float(tcig))
        if len(values) == len(per_finding):
            replicate_macro.append(float(np.mean(values)))
    ci = {
        finding: {
            "mean_tcig": {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
        }
        for finding, values in replicate_tcig.items()
        if values
    }
    for finding, values in replicate_localization.items():
        if values:
            ci.setdefault(finding, {})["mean_topk_localization_gain"] = {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
    result = {
        "evaluation_axis": "pixel_interventional_evidence_sufficiency",
        "confidence_interval_unit": "image_level_cluster_by_unit_id",
        "patient_level_confidence_interval": False,
        "per_finding": per_finding,
        "macro_mean_tcig": float(np.mean([row["mean_tcig"] for row in per_finding.values()])),
        "image_cluster_bootstrap_95ci": {
            "per_finding": ci,
            "macro_mean_tcig": {
                "lower": float(np.percentile(replicate_macro, 2.5)),
                "upper": float(np.percentile(replicate_macro, 97.5)),
            }
            if replicate_macro
            else None,
        },
        "bootstrap_requested_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }
    return result
