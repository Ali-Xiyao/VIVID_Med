"""Result-blind image-statistics matched controls for ARISE development.

The routines in this module never inspect model scores or evaluation outcomes.
They generate exact-area connected candidates inside the target coordinate
zone, then select the candidate whose pre-intervention image statistics and
geometry most closely match the target.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from typing import Any

import numpy as np

from bives_cxr.rescue_protocol import coordinate_zone, mask_geometry


STAT_MATCHED_CONTROL_VERSION = "arise_stat_matched_connected_perimeter_v2"


def _as_binary_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask")
    if mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return mask.astype(bool, copy=False)


def _pixel_hash(seed_key: str, y: int, x: int) -> int:
    digest = hashlib.sha256(f"{seed_key}:{y}:{x}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _zone_from_centroid(
    x: float,
    y: float,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> tuple[int, int]:
    normalized_x = (x - left) / max(1, right - left)
    normalized_y = (y - top) / max(1, bottom - top)
    horizontal = min(1, max(0, int(normalized_x * 2.0)))
    vertical = min(2, max(0, int(normalized_y * 3.0)))
    return horizontal, vertical


def _grow_candidate(
    valid: np.ndarray,
    *,
    seed_y: int,
    seed_x: int,
    area: int,
    seed_key: str,
) -> np.ndarray | None:
    """Grow one deterministic 4-connected exact-area candidate."""

    height, width = valid.shape
    if not valid[seed_y, seed_x]:
        return None
    selected = np.zeros_like(valid, dtype=bool)
    seen = np.zeros_like(valid, dtype=bool)
    heap: list[tuple[int, int, int, int]] = []
    heapq.heappush(
        heap,
        (0, _pixel_hash(seed_key, seed_y, seed_x), seed_y, seed_x),
    )
    seen[seed_y, seed_x] = True
    count = 0
    while heap and count < area:
        _, _, y, x = heapq.heappop(heap)
        selected[y, x] = True
        count += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if seen[ny, nx] or not valid[ny, nx]:
                continue
            seen[ny, nx] = True
            distance = (ny - seed_y) ** 2 + (nx - seed_x) ** 2
            heapq.heappush(
                heap,
                (distance, _pixel_hash(seed_key, ny, nx), ny, nx),
            )
    if count != area:
        return None
    return selected


def _region_statistics(
    image: np.ndarray,
    gradient: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    values = image[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "gradient_mean": float(gradient[mask].mean()),
    }


def deterministic_stat_matched_connected_control_mask(
    image: np.ndarray,
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
    forbidden_mask: np.ndarray | None = None,
    candidate_limit: int = 48,
    seed_attempt_limit: int = 384,
    max_log_perimeter_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic result-blind statistics-matched control.

    Candidate controls are exact-area, target-disjoint, 4-connected, and have
    their centroid in the same coarse coordinate zone as the target. Selection
    uses only the original image's intensity mean, standard deviation, gradient
    energy, and target/control geometry. Model scores are never accepted.
    """

    pixels = np.asarray(image, dtype=np.float64)
    target = _as_binary_mask(target_mask, name="target_mask")
    content = _as_binary_mask(content_mask, name="content_mask")
    if pixels.ndim != 2 or pixels.shape != target.shape or content.shape != target.shape:
        raise ValueError("image/target/content must be shape-matched 2D arrays")
    if not np.isfinite(pixels[content]).all():
        raise ValueError("image must be finite inside content")
    if candidate_limit < 2 or seed_attempt_limit < candidate_limit:
        raise ValueError("candidate limits must allow at least two candidates")
    if not np.isfinite(max_log_perimeter_ratio) or max_log_perimeter_ratio < 0.0:
        raise ValueError("max_log_perimeter_ratio must be finite and non-negative")
    forbidden = np.zeros_like(target)
    if forbidden_mask is not None:
        forbidden = _as_binary_mask(forbidden_mask, name="forbidden_mask")
        if forbidden.shape != target.shape:
            raise ValueError("forbidden_mask must match target geometry")
    if bool((target & ~content).any()) or not target.any():
        raise ValueError("target must be non-empty and inside content")

    valid = content & ~target & ~forbidden
    area = int(target.sum())
    if int(valid.sum()) < area:
        raise ValueError("insufficient disjoint content for exact-area control")

    target_zone = coordinate_zone(target, content)
    ys, xs = np.nonzero(content)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    valid_y, valid_x = np.nonzero(valid)
    seed_rows = sorted(
        zip(valid_y.tolist(), valid_x.tolist()),
        key=lambda row: (_pixel_hash(f"{seed_key}:seed", row[0], row[1]), row),
    )[:seed_attempt_limit]

    gradient_y, gradient_x = np.gradient(pixels)
    gradient = np.hypot(gradient_x, gradient_y)
    global_std = max(float(pixels[content].std()), np.finfo(np.float64).eps)
    global_gradient = max(float(gradient[content].mean()), np.finfo(np.float64).eps)
    target_stats = _region_statistics(pixels, gradient, target)
    target_geometry = mask_geometry(target, content)
    target_horizontal, target_vertical = _zone_from_centroid(
        float(target_geometry["centroid_x"]),
        float(target_geometry["centroid_y"]),
        left=left,
        right=right,
        top=top,
        bottom=bottom,
    )

    candidates: list[tuple[float, int, np.ndarray, dict[str, Any]]] = []
    seen_hashes: set[str] = set()
    for seed_order, (seed_y, seed_x) in enumerate(seed_rows):
        candidate = _grow_candidate(
            valid,
            seed_y=seed_y,
            seed_x=seed_x,
            area=area,
            seed_key=f"{seed_key}:grow:{seed_y}:{seed_x}",
        )
        if candidate is None:
            continue
        packed_hash = hashlib.sha256(np.packbits(candidate).tobytes()).hexdigest()
        if packed_hash in seen_hashes:
            continue
        geometry = mask_geometry(candidate, content)
        horizontal, vertical = _zone_from_centroid(
            float(geometry["centroid_x"]),
            float(geometry["centroid_y"]),
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        if horizontal != target_horizontal or vertical != target_vertical:
            continue
        seen_hashes.add(packed_hash)
        stats = _region_statistics(pixels, gradient, candidate)
        mean_difference = abs(stats["mean"] - target_stats["mean"]) / global_std
        std_difference = abs(stats["std"] - target_stats["std"]) / global_std
        gradient_difference = (
            abs(stats["gradient_mean"] - target_stats["gradient_mean"])
            / global_gradient
        )
        centroid_distance = math.hypot(
            float(geometry["centroid_x"]) - float(target_geometry["centroid_x"]),
            float(geometry["centroid_y"]) - float(target_geometry["centroid_y"]),
        ) / max(float(np.hypot(*target.shape)), np.finfo(np.float64).eps)
        perimeter_difference = abs(
            math.log(
                float(geometry["perimeter_edges"])
                / float(target_geometry["perimeter_edges"])
            )
        )
        if perimeter_difference > max_log_perimeter_ratio:
            continue
        statistics_objective = mean_difference + std_difference + gradient_difference
        geometry_objective = centroid_distance + 0.10 * perimeter_difference
        objective = statistics_objective + 0.25 * geometry_objective
        certificate = {
            "seed_order": seed_order,
            "seed_y": seed_y,
            "seed_x": seed_x,
            "mask_sha256": packed_hash,
            "statistics": stats,
            "statistics_objective": float(statistics_objective),
            "geometry_objective": float(geometry_objective),
            "objective": float(objective),
            "geometry": geometry,
        }
        candidates.append((float(objective), seed_order, candidate, certificate))
        if len(candidates) >= candidate_limit:
            break
    if len(candidates) < 2:
        raise ValueError("fewer than two valid statistics-matched control candidates")

    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, control, selected = candidates[0]
    if bool((control & (target | forbidden)).any()):
        raise AssertionError("statistics-matched control overlaps a forbidden target")
    control_geometry = mask_geometry(control, content)
    if int(control.sum()) != area or control_geometry["component_count"] != 1:
        raise AssertionError("statistics-matched control violates geometry contract")
    return control, {
        "version": STAT_MATCHED_CONTROL_VERSION,
        "seed_key": str(seed_key),
        "selection_is_result_blind": True,
        "selection_inputs": [
            "original_intensity_mean",
            "original_intensity_std",
            "original_gradient_mean",
            "region_geometry",
        ],
        "model_scores_used": False,
        "candidate_count": len(candidates),
        "seed_attempt_count": len(seed_rows),
        "target_statistics": target_stats,
        "target_geometry": target_geometry,
        "target_zone": target_zone,
        "selected": selected,
        "runner_up_objective": float(candidates[1][0]),
        "disjoint": True,
        "exact_area": True,
        "connected_component_requirement": 1,
        "max_log_perimeter_ratio": float(max_log_perimeter_ratio),
        "true_anatomy_segmentation": False,
    }
