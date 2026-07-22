"""Score-blind fallback controls for VICER intervention-validity V0."""

from __future__ import annotations

import hashlib
import heapq
import math
from typing import Any

import numpy as np

from bives_cxr.rescue_protocol import mask_geometry


FALLBACK_CONTROL_VERSION = "vicer-v0-connected-stat-fallback-v1"


def _rank(seed_key: str, y: int, x: int) -> int:
    digest = hashlib.sha256(f"{seed_key}:{y}:{x}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _grow(valid: np.ndarray, seed: tuple[int, int], area: int, seed_key: str) -> np.ndarray | None:
    height, width = valid.shape
    selected = np.zeros_like(valid, dtype=bool)
    seen = np.zeros_like(valid, dtype=bool)
    heap: list[tuple[int, int, int, int]] = []
    sy, sx = seed
    if not valid[sy, sx]:
        return None
    heapq.heappush(heap, (0, _rank(seed_key, sy, sx), sy, sx))
    seen[sy, sx] = True
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
            distance = (ny - sy) ** 2 + (nx - sx) ** 2
            heapq.heappush(heap, (distance, _rank(seed_key, ny, nx), ny, nx))
    return selected if count == area else None


def deterministic_connected_statistics_control(
    image: np.ndarray,
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
    seed_limit: int = 64,
    candidate_limit: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    pixels = np.asarray(image, dtype=np.float64)
    target = np.asarray(target_mask, dtype=bool)
    content = np.asarray(content_mask, dtype=bool)
    if pixels.shape != target.shape or content.shape != target.shape or target.ndim != 2:
        raise ValueError("VICER fallback image/masks must be shape-matched 2D arrays")
    if not target.any() or bool((target & ~content).any()):
        raise ValueError("VICER fallback target must be non-empty and contained")
    valid = content & ~target
    area = int(target.sum())
    if int(valid.sum()) < area:
        raise ValueError("VICER fallback has insufficient disjoint content")
    gy, gx = np.gradient(pixels)
    gradient = np.hypot(gx, gy)
    global_std = max(float(pixels[content].std()), 1e-12)
    global_gradient = max(float(gradient[content].mean()), 1e-12)

    def stats(mask: np.ndarray) -> tuple[float, float, float]:
        return (
            float(pixels[mask].mean()),
            float(pixels[mask].std()),
            float(gradient[mask].mean()),
        )

    target_stats = stats(target)
    target_geometry = mask_geometry(target, content)
    ys, xs = np.nonzero(valid)
    seeds = sorted(
        zip(ys.tolist(), xs.tolist()),
        key=lambda point: (_rank(f"{seed_key}:seed", *point), point),
    )[:seed_limit]
    candidates = []
    seen = set()
    for order, seed in enumerate(seeds):
        candidate = _grow(valid, seed, area, f"{seed_key}:grow:{seed[0]}:{seed[1]}")
        if candidate is None:
            continue
        digest = hashlib.sha256(np.packbits(candidate).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        candidate_stats = stats(candidate)
        geometry = mask_geometry(candidate, content)
        stat_objective = (
            abs(candidate_stats[0] - target_stats[0]) / global_std
            + abs(candidate_stats[1] - target_stats[1]) / global_std
            + abs(candidate_stats[2] - target_stats[2]) / global_gradient
        )
        perimeter_objective = abs(
            math.log(float(geometry["perimeter_edges"]) / float(target_geometry["perimeter_edges"]))
        )
        objective = stat_objective + 0.10 * perimeter_objective
        candidates.append((objective, order, candidate, candidate_stats, geometry, digest))
        if len(candidates) >= candidate_limit:
            break
    if len(candidates) < 2:
        raise ValueError("VICER fallback found fewer than two exact-area connected controls")
    candidates.sort(key=lambda item: (item[0], item[1]))
    objective, order, control, control_stats, control_geometry, digest = candidates[0]
    if int(control.sum()) != area or bool((control & target).any()):
        raise AssertionError("VICER fallback control violates area/disjointness")
    if int(control_geometry["component_count"]) != 1:
        raise AssertionError("VICER fallback control is not connected")
    return control, {
        "version": FALLBACK_CONTROL_VERSION,
        "seed_key": seed_key,
        "selection_is_result_blind": True,
        "model_scores_used": False,
        "candidate_count": len(candidates),
        "selected_seed_order": order,
        "selected_mask_sha256": digest,
        "objective": float(objective),
        "target_statistics": dict(zip(("mean", "std", "gradient_mean"), target_stats)),
        "control_statistics": dict(zip(("mean", "std", "gradient_mean"), control_stats)),
        "target_geometry": target_geometry,
        "control_geometry": control_geometry,
        "exact_area": True,
        "disjoint": True,
        "connected_component_requirement": 1,
        "same_vertical_band_required": False,
        "true_anatomy_segmentation": False,
    }
