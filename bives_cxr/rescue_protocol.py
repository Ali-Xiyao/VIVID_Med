"""Deterministic data-split and geometry contracts for the BiVES rescue gate."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy import ndimage


RESCUE_SPLIT_VERSION = "bives_vindr_train_image_disjoint_v1"
TOPOLOGY_CONTROL_VERSION = "bives_translated_target_shape_control_v2_complete_translation"
COORDINATE_ZONE_CONTROL_VERSION = "bives_coordinate_zone_connected_control_v1"


def stable_hash_int(value: str) -> int:
    """Return a platform-independent integer hash for deterministic ordering."""

    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")


def deterministic_multilabel_half_split(
    labels_by_unit: Mapping[str, Iterable[str]],
    *,
    seed: int,
    max_swap_rounds: int = 100,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Split image groups in half while balancing every supplied stratum.

    Units, rather than individual finding rows, are assigned. A deterministic
    hash split is improved by pair swaps until no swap reduces the weighted
    squared stratum imbalance. Every stratum must finish within one sample of
    its ideal half; otherwise the lock fails closed.
    """

    normalized = {
        str(unit): tuple(sorted({str(label) for label in labels}))
        for unit, labels in labels_by_unit.items()
    }
    if not normalized or any(not labels for labels in normalized.values()):
        raise ValueError("every rescue split unit must have at least one stratum label")
    units = sorted(normalized, key=lambda unit: (stable_hash_int(f"{seed}:{unit}"), unit))
    design_size = len(units) // 2
    design = set(units[:design_size])
    confirm = set(units[design_size:])
    totals = Counter(label for labels in normalized.values() for label in labels)

    def counts(selected: set[str]) -> Counter[str]:
        return Counter(label for unit in selected for label in normalized[unit])

    def objective(selected_counts: Counter[str]) -> float:
        return float(
            sum(
                ((selected_counts[label] - total / 2.0) ** 2) / max(1, total)
                for label, total in totals.items()
            )
        )

    design_counts = counts(design)
    current = objective(design_counts)
    rounds = 0
    while rounds < max_swap_rounds:
        best: tuple[float, int, str, str, Counter[str]] | None = None
        for left in sorted(design):
            left_labels = normalized[left]
            for right in sorted(confirm):
                candidate_counts = design_counts.copy()
                candidate_counts.subtract(left_labels)
                candidate_counts.update(normalized[right])
                score = objective(candidate_counts)
                improvement = current - score
                if improvement <= 1e-12:
                    continue
                tie = stable_hash_int(f"{seed}:swap:{rounds}:{left}:{right}")
                candidate = (-improvement, tie, left, right, candidate_counts)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
        if best is None:
            break
        _, _, left, right, design_counts = best
        design.remove(left)
        design.add(right)
        confirm.remove(right)
        confirm.add(left)
        current = objective(design_counts)
        rounds += 1

    strata: dict[str, dict[str, int | float]] = {}
    for label, total in sorted(totals.items()):
        design_count = int(design_counts[label])
        confirm_count = int(total - design_count)
        deviation = abs(design_count - total / 2.0)
        if deviation > 1.0:
            raise ValueError(
                f"multilabel split cannot balance {label}: "
                f"design={design_count}, confirm={confirm_count}"
            )
        strata[label] = {
            "total": int(total),
            "protocol_design": design_count,
            "rescue_confirm": confirm_count,
            "absolute_half_deviation": float(deviation),
        }
    assignment = {
        unit: "protocol_design" if unit in design else "rescue_confirm" for unit in units
    }
    return assignment, {
        "version": RESCUE_SPLIT_VERSION,
        "seed": int(seed),
        "units": len(units),
        "protocol_design_units": len(design),
        "rescue_confirm_units": len(confirm),
        "swap_rounds": rounds,
        "objective": current,
        "strata": strata,
    }


def mask_geometry(mask: np.ndarray, content_mask: np.ndarray) -> dict[str, Any]:
    """Summarize exact binary-mask geometry using four-neighbour topology."""

    if mask.shape != content_mask.shape or mask.ndim != 2:
        raise ValueError("mask/content masks must be shape-matched 2D arrays")
    binary = mask.astype(bool)
    content = content_mask.astype(bool)
    if not binary.any() or bool((binary & ~content).any()):
        raise ValueError("geometry mask must be non-empty and contained in content")
    ys, xs = np.nonzero(binary)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    height = bottom - top
    width = right - left

    padded = np.pad(binary, 1, constant_values=False)
    perimeter = int(
        np.sum(padded[1:-1, 1:-1] & ~padded[:-2, 1:-1])
        + np.sum(padded[1:-1, 1:-1] & ~padded[2:, 1:-1])
        + np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, :-2])
        + np.sum(padded[1:-1, 1:-1] & ~padded[1:-1, 2:])
    )

    remaining = binary.copy()
    components = 0
    while remaining.any():
        components += 1
        start_y, start_x = np.argwhere(remaining)[0]
        stack = [(int(start_y), int(start_x))]
        remaining[start_y, start_x] = False
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1] and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((ny, nx))

    content_ys, _ = np.nonzero(content)
    content_top = int(content_ys.min())
    content_bottom = int(content_ys.max()) + 1
    relative_y = (float(ys.mean()) - content_top) / max(1, content_bottom - content_top)
    vertical_band = min(2, max(0, int(relative_y * 3.0)))
    area = int(binary.sum())
    return {
        "area_pixels": area,
        "component_count": components,
        "perimeter_edges": perimeter,
        "compactness": float(4.0 * math.pi * area / max(1, perimeter**2)),
        "bbox": [left, top, right, bottom],
        "bbox_aspect_ratio": float(width / height),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "vertical_band": vertical_band,
    }


def coordinate_zone(mask: np.ndarray, content_mask: np.ndarray) -> dict[str, Any]:
    """Return the frozen coarse coordinate zone for a non-empty mask.

    The zone is a content-coordinate proxy only. It is not a lung or lobar
    segmentation and must not be reported as true anatomy.
    """

    geometry = mask_geometry(mask, content_mask)
    content = content_mask.astype(bool)
    content_ys, content_xs = np.nonzero(content)
    content_left = int(content_xs.min())
    content_right = int(content_xs.max()) + 1
    content_top = int(content_ys.min())
    content_bottom = int(content_ys.max()) + 1
    normalized_x = (float(geometry["centroid_x"]) - content_left) / max(
        1, content_right - content_left
    )
    normalized_y = (float(geometry["centroid_y"]) - content_top) / max(
        1, content_bottom - content_top
    )
    if normalized_x < 0.40:
        horizontal = "left"
    elif normalized_x <= 0.60:
        horizontal = "central"
    else:
        horizontal = "right"
    if normalized_y < 1.0 / 3.0:
        vertical = "upper"
    elif normalized_y < 2.0 / 3.0:
        vertical = "middle"
    else:
        vertical = "lower"
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "normalized_x": float(normalized_x),
        "normalized_y": float(normalized_y),
    }


def _zone_from_centroid(
    centroid_x: float,
    centroid_y: float,
    *,
    content_left: int,
    content_right: int,
    content_top: int,
    content_bottom: int,
) -> dict[str, Any]:
    normalized_x = (centroid_x - content_left) / max(1, content_right - content_left)
    normalized_y = (centroid_y - content_top) / max(1, content_bottom - content_top)
    if normalized_x < 0.40:
        horizontal = "left"
    elif normalized_x <= 0.60:
        horizontal = "central"
    else:
        horizontal = "right"
    if normalized_y < 1.0 / 3.0:
        vertical = "upper"
    elif normalized_y < 2.0 / 3.0:
        vertical = "middle"
    else:
        vertical = "lower"
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "normalized_x": float(normalized_x),
        "normalized_y": float(normalized_y),
    }


def _coordinate_zone_seed_pixels(
    valid_mask: np.ndarray,
    *,
    seed_key: str,
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Create the frozen 17x17 lattice/component-centroid seed set."""

    valid = valid_mask.astype(bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError("connected control has no admissible seed pixel")
    height, width = valid.shape
    ys, xs = np.nonzero(valid)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    nearest = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    points: list[tuple[int, int]] = []
    for lattice_y in range(17):
        y = top + round((bottom - top - 1) * lattice_y / 16.0)
        for lattice_x in range(17):
            x = left + round((right - left - 1) * lattice_x / 16.0)
            points.append((int(nearest[0, y, x]), int(nearest[1, y, x])))

    labels, component_count = ndimage.label(
        valid,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    if component_count:
        centroids = ndimage.center_of_mass(
            valid,
            labels,
            list(range(1, component_count + 1)),
        )
        for centroid_y, centroid_x in centroids:
            y = min(height - 1, max(0, round(float(centroid_y))))
            x = min(width - 1, max(0, round(float(centroid_x))))
            points.append((int(nearest[0, y, x]), int(nearest[1, y, x])))

    deduplicated = sorted(set(points))
    seeds = [
        (
            stable_hash_int(
                f"{COORDINATE_ZONE_CONTROL_VERSION}:{seed_key}:seed:{y}:{x}"
            ),
            y,
            x,
        )
        for y, x in deduplicated
    ]
    seeds.sort()
    return seeds, labels.astype(np.int32, copy=False), np.bincount(labels.ravel())


def _grow_connected_region(
    valid_mask: np.ndarray,
    *,
    seed_y: int,
    seed_x: int,
    area: int,
    marker: int,
    seen_marks: np.ndarray,
    selected_marks: np.ndarray,
    tie_cache: dict[int, bytes],
    tie_prefix: str,
) -> tuple[list[int], int, int, int] | None:
    """Grow one exact-area region using the frozen best-first frontier."""

    height, width = valid_mask.shape

    def tie(flat_index: int) -> bytes:
        cached = tie_cache.get(flat_index)
        if cached is None:
            y, x = divmod(flat_index, width)
            cached = hashlib.sha256(f"{tie_prefix}:{y}:{x}".encode("utf-8")).digest()
            tie_cache[flat_index] = cached
        return cached

    seed_index = seed_y * width + seed_x
    heap: list[tuple[int, bytes, int]] = [(0, tie(seed_index), seed_index)]
    seen_marks[seed_y, seed_x] = marker
    selected_indices: list[int] = []
    centroid_x_sum = 0
    centroid_y_sum = 0
    perimeter = 0
    while heap and len(selected_indices) < area:
        _, _, flat_index = heapq.heappop(heap)
        y, x = divmod(flat_index, width)
        selected_marks[y, x] = marker
        selected_indices.append(flat_index)
        centroid_x_sum += x
        centroid_y_sum += y
        selected_neighbours = 0
        for neighbour_y, neighbour_x in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ):
            if not (0 <= neighbour_y < height and 0 <= neighbour_x < width):
                continue
            if selected_marks[neighbour_y, neighbour_x] == marker:
                selected_neighbours += 1
            if not valid_mask[neighbour_y, neighbour_x]:
                continue
            if seen_marks[neighbour_y, neighbour_x] == marker:
                continue
            seen_marks[neighbour_y, neighbour_x] = marker
            neighbour_index = neighbour_y * width + neighbour_x
            squared_distance = (neighbour_y - seed_y) ** 2 + (neighbour_x - seed_x) ** 2
            heapq.heappush(
                heap,
                (squared_distance, tie(neighbour_index), neighbour_index),
            )
        perimeter += 4 - 2 * selected_neighbours
    if len(selected_indices) != area:
        return None
    return selected_indices, centroid_x_sum, centroid_y_sum, perimeter


def deterministic_coordinate_zone_connected_control_mask(
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the frozen exact-area coordinate-zone connected control.

    The control is a single connected geometry-only nuisance region. It does
    not preserve target topology and is not a true anatomy segmentation.
    """

    if target_mask.shape != content_mask.shape or target_mask.ndim != 2:
        raise ValueError("target/content masks must be shape-matched 2D arrays")
    target = target_mask.astype(bool)
    content = content_mask.astype(bool)
    target_geometry = mask_geometry(target, content)
    target_zone = coordinate_zone(target, content)
    valid = content & ~target
    area = int(target_geometry["area_pixels"])
    if int(valid.sum()) < area:
        raise ValueError("insufficient target-disjoint content for exact-area connected control")

    seeds, component_labels, component_sizes = _coordinate_zone_seed_pixels(
        valid,
        seed_key=seed_key,
    )
    content_ys, content_xs = np.nonzero(content)
    content_left, content_right = int(content_xs.min()), int(content_xs.max()) + 1
    content_top, content_bottom = int(content_ys.min()), int(content_ys.max()) + 1
    seen_marks = np.zeros(target.shape, dtype=np.int32)
    selected_marks = np.zeros(target.shape, dtype=np.int32)
    tie_cache: dict[int, bytes] = {}
    tie_prefix = f"{COORDINATE_ZONE_CONTROL_VERSION}:{seed_key}:frontier"
    candidates_grown = 0
    candidates_zone_eligible = 0
    best: tuple[float, int, int, int, list[int], int] | None = None
    marker = 0
    for seed_order_hash, seed_y, seed_x in seeds:
        component_label = int(component_labels[seed_y, seed_x])
        if component_label <= 0 or int(component_sizes[component_label]) < area:
            continue
        marker += 1
        grown = _grow_connected_region(
            valid,
            seed_y=seed_y,
            seed_x=seed_x,
            area=area,
            marker=marker,
            seen_marks=seen_marks,
            selected_marks=selected_marks,
            tie_cache=tie_cache,
            tie_prefix=tie_prefix,
        )
        if grown is None:
            continue
        candidates_grown += 1
        selected_indices, centroid_x_sum, centroid_y_sum, perimeter = grown
        centroid_x = centroid_x_sum / area
        centroid_y = centroid_y_sum / area
        control_zone = _zone_from_centroid(
            centroid_x,
            centroid_y,
            content_left=content_left,
            content_right=content_right,
            content_top=content_top,
            content_bottom=content_bottom,
        )
        if (
            control_zone["horizontal"] != target_zone["horizontal"]
            or control_zone["vertical"] != target_zone["vertical"]
        ):
            continue
        candidates_zone_eligible += 1
        objective = (
            abs(control_zone["normalized_y"] - target_zone["normalized_y"])
            + abs(control_zone["normalized_x"] - target_zone["normalized_x"])
            + 0.10
            * abs(math.log(perimeter / max(1, int(target_geometry["perimeter_edges"]))))
        )
        candidate = (
            float(objective),
            int(seed_order_hash),
            int(seed_y),
            int(seed_x),
            selected_indices,
            int(perimeter),
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise ValueError("no exact-area connected control has the target coordinate zone")

    objective, seed_order_hash, seed_y, seed_x, selected_indices, grown_perimeter = best
    control = np.zeros_like(target)
    control.flat[np.asarray(selected_indices, dtype=np.int64)] = True
    if bool((control & target).any()):
        raise AssertionError("connected control unexpectedly overlaps target")
    control_geometry = mask_geometry(control, content)
    control_zone = coordinate_zone(control, content)
    if control_geometry["area_pixels"] != area:
        raise AssertionError("connected control does not preserve exact area")
    if control_geometry["component_count"] != 1:
        raise AssertionError("connected control is not one 4-connected component")
    if control_geometry["perimeter_edges"] != grown_perimeter:
        raise AssertionError("incremental connected-control perimeter mismatch")
    if (
        control_zone["horizontal"] != target_zone["horizontal"]
        or control_zone["vertical"] != target_zone["vertical"]
    ):
        raise AssertionError("connected control is outside the target coordinate zone")
    return control, {
        "version": COORDINATE_ZONE_CONTROL_VERSION,
        "seed_key": str(seed_key),
        "seed_count": len(seeds),
        "candidates_grown": candidates_grown,
        "candidates_zone_eligible": candidates_zone_eligible,
        "selected_seed_hash": str(seed_order_hash),
        "selected_seed_y": seed_y,
        "selected_seed_x": seed_x,
        "geometry_objective": float(objective),
        "target": target_geometry,
        "target_zone": target_zone,
        "control": control_geometry,
        "control_zone": control_zone,
        "disjoint": True,
        "connected_component_requirement": 1,
        "true_anatomy_segmentation": False,
    }


def deterministic_translated_control_mask(
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Translate the exact target shape to a disjoint same-band control.

    Every integer translation that keeps the target shape inside the content
    rectangle and in the same vertical third is considered. Exact pixel-mask
    disjointness is determined by full autocorrelation, so bounding boxes may
    overlap when the irregular expert masks themselves do not. Translation
    preserves area, components, perimeter, compactness, and aspect ratio.
    """

    if target_mask.shape != content_mask.shape or target_mask.ndim != 2:
        raise ValueError("target/content masks must be shape-matched 2D arrays")
    target = target_mask.astype(bool)
    content = content_mask.astype(bool)
    target_geometry = mask_geometry(target, content)
    target_left, target_top, target_right, target_bottom = target_geometry["bbox"]
    crop = target[target_top:target_bottom, target_left:target_right]
    content_ys, content_xs = np.nonzero(content)
    content_left, content_right = int(content_xs.min()), int(content_xs.max()) + 1
    content_top, content_bottom = int(content_ys.min()), int(content_ys.max()) + 1

    x_shifts = range(content_left - target_left, content_right - target_right + 1)
    y_shifts = []
    target_band = int(target_geometry["vertical_band"])
    target_centroid_y = float(target_geometry["centroid_y"])
    for shift_y in range(content_top - target_top, content_bottom - target_bottom + 1):
        relative_y = (target_centroid_y + shift_y - content_top) / max(
            1, content_bottom - content_top
        )
        if min(2, max(0, int(relative_y * 3.0))) == target_band:
            y_shifts.append(shift_y)

    fft_shape = (2 * target.shape[0] - 1, 2 * target.shape[1] - 1)
    correlation = np.fft.irfftn(
        np.fft.rfftn(target.astype(np.float64), fft_shape)
        * np.fft.rfftn(target[::-1, ::-1].astype(np.float64), fft_shape),
        fft_shape,
    )
    candidates = [
        (shift_y, shift_x)
        for shift_y in y_shifts
        for shift_x in x_shifts
        if (shift_y != 0 or shift_x != 0)
        and abs(
            float(
                correlation[
                    target.shape[0] - 1 + shift_y,
                    target.shape[1] - 1 + shift_x,
                ]
            )
        )
        < 0.5
    ]
    if not candidates:
        raise ValueError("no disjoint same-band translated target-shape control exists")

    candidate_count = len(candidates)
    selected = stable_hash_int(f"{TOPOLOGY_CONTROL_VERSION}:{seed_key}") % candidate_count
    shift_y, shift_x = candidates[selected]
    control_left = target_left + shift_x
    control_top = target_top + shift_y
    crop_height, crop_width = crop.shape
    control = np.zeros_like(target)
    control[
        control_top : control_top + crop_height,
        control_left : control_left + crop_width,
    ] = crop
    if bool((control & target).any()):
        raise AssertionError("translated control unexpectedly overlaps target")
    control_geometry = mask_geometry(control, content)
    invariant_keys = (
        "area_pixels",
        "component_count",
        "perimeter_edges",
        "compactness",
        "bbox_aspect_ratio",
        "vertical_band",
    )
    for key in invariant_keys:
        if not np.isclose(target_geometry[key], control_geometry[key], rtol=0.0, atol=1e-12):
            raise AssertionError(f"translated control does not preserve {key}")
    return control, {
        "version": TOPOLOGY_CONTROL_VERSION,
        "seed_key": str(seed_key),
        "candidate_count": candidate_count,
        "shift_x": int(shift_x),
        "shift_y": int(shift_y),
        "target": target_geometry,
        "control": control_geometry,
        "disjoint": True,
    }
