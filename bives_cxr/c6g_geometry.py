"""Score-free continuous-location connected controls for MS-CXR C6G."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .c6_ms_cxr_eval import (
    EXPECTED_ROWS,
    GEOMETRY_FORMAT_VERSION as C6F_GEOMETRY_FORMAT_VERSION,
    IMAGE_SIZE,
    _content_geometry,
    validate_ms_cxr_manifest,
)
from .pixel_interventions import transform_mask_to_letterbox, union_box_mask
from .provenance import canonical_json_sha256, file_sha256
from .rescue_protocol import (
    COORDINATE_ZONE_CONTROL_VERSION,
    _coordinate_zone_seed_pixels,
    _grow_connected_region,
    _zone_from_centroid,
    coordinate_zone,
    mask_geometry,
    stable_hash_int,
)


CONTROL_VERSION = "bives_continuous_location_connected_control_v2"
GEOMETRY_FORMAT_VERSION = "bives_c6g_ms_cxr_geometry_v1"
LOCK_FORMAT_VERSION = "bives_c6g_ms_cxr_geometry_lock_v1"
THRESHOLD_FORMAT_VERSION = "bives_c6g_geometry_thresholds_v1"
FRONTIER_ORDERS = (
    "target_distance_then_l2",
    "centroid_l2_then_target_distance",
    "target_distance_then_l1",
)
EXPECTED_SOURCE_HASHES = {
    "frozen_c4_protocol_design_geometry":
        "b94b77bc528033ea44681fce33f868b87c2806d19450a66ff023628fc2e039d9",
    "frozen_c5_confirmation_geometry":
        "58b10019306e63b9ef1d08b94cd01e9540f3591155b3981762bc9657ccf0be41",
}
EXPECTED_C6F_HASHES = {
    "authority": "6599212f1c9b4177379196435a65deffd278440e94f4574f3057d4b107bc207c",
    "config": "5bfd243c7c3a42a113e5e5b50b171d988cecb4a1d6a1426b4c94ad7a8e1ffb5a",
    "execution_log": "acbe78ac76d08b7ef9d4acd62d1c4d861299ab094631afc77a17985552bb82cd",
    "manifest": "ba31d6e9e2cefe55effaef838a2f7cc8bf68d5c07021f22f2614782082f4f711",
    "geometry_rows": "dbde2e628f7e67db05a815c9110b165b427cbe9c3d837741422febbb02ad2f84",
    "geometry_lock": "c8ee9e4cb03bb8a8b068d08165b645591d1905a1675f89da246aa4b566301d91",
    "dataset_lock": "9a3cc26b982536950b01eedfdde73f596f569338ca64ff706a83545a9758f073",
}


class ControlSearchFailure(ValueError):
    """A fail-closed geometry search failure with a complete certificate."""

    def __init__(self, certificate: dict[str, Any]) -> None:
        super().__init__(
            "no qualifying candidate found under "
            "bives_continuous_location_connected_control_v2"
        )
        self.certificate = certificate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def derive_frozen_thresholds(
    c4_rows_path: Path,
    c5_rows_path: Path,
) -> dict[str, Any]:
    """Derive outcome-independent limits from accepted frozen C4/C5 geometry."""

    sources = (
        ("frozen_c4_protocol_design_geometry", c4_rows_path, 375),
        ("frozen_c5_confirmation_geometry", c5_rows_path, 377),
    )
    location_distances: list[float] = []
    perimeter_mismatches: list[float] = []
    source_rows = []
    for name, path, expected_count in sources:
        digest = file_sha256(path)
        if digest != EXPECTED_SOURCE_HASHES[name]:
            raise ValueError(f"{name} rows SHA-256 changed")
        rows = [row for row in _read_jsonl(path) if bool(row.get("feasible"))]
        if len(rows) != expected_count:
            raise ValueError(f"{name} accepted-row count changed")
        for row in rows:
            audit = row.get("control_geometry")
            if not isinstance(audit, dict):
                raise ValueError(f"{name} accepted row is missing geometry audit")
            target_zone = audit["target_zone"]
            control_zone = audit["control_zone"]
            location_distances.append(
                abs(float(control_zone["normalized_x"]) - float(target_zone["normalized_x"]))
                + abs(
                    float(control_zone["normalized_y"])
                    - float(target_zone["normalized_y"])
                )
            )
            perimeter_mismatches.append(
                abs(
                    math.log(
                        float(audit["control"]["perimeter_edges"])
                        / float(audit["target"]["perimeter_edges"])
                    )
                )
            )
        source_rows.append(
            {"name": name, "accepted_rows": len(rows), "rows_sha256": digest}
        )
    return {
        "format_version": THRESHOLD_FORMAT_VERSION,
        "control_version": CONTROL_VERSION,
        "accepted_rows": len(location_distances),
        "max_location_distance": max(location_distances),
        "max_log_perimeter_ratio": max(perimeter_mismatches),
        "log_perimeter_ratio_weight": 0.10,
        "source_rows": source_rows,
    }


def validate_frozen_thresholds(
    payload: dict[str, Any],
    derived: dict[str, Any],
) -> None:
    if payload != derived:
        raise ValueError("C6G threshold artifact disagrees with frozen C4/C5 geometry")


def verify_c6f_immutability(paths: dict[str, Path]) -> dict[str, str]:
    if set(paths) != set(EXPECTED_C6F_HASHES):
        raise ValueError("C6F immutability path set is incomplete")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    if actual != EXPECTED_C6F_HASHES:
        changed = sorted(name for name in actual if actual[name] != EXPECTED_C6F_HASHES[name])
        raise ValueError("frozen C6F artifacts changed: " + ", ".join(changed))
    return actual


def _content_bounds(content: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(content)
    if len(xs) == 0:
        raise ValueError("content mask is empty")
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def _seed_families(
    valid: np.ndarray,
    *,
    seed_key: str,
) -> tuple[list[tuple[int, int, int, str]], np.ndarray, np.ndarray, dict[str, int]]:
    height, width = valid.shape
    ys, xs = np.nonzero(valid)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    nearest = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    points: dict[tuple[int, int], set[str]] = {}

    def add(y: int, x: int, family: str) -> None:
        point = (int(nearest[0, y, x]), int(nearest[1, y, x]))
        points.setdefault(point, set()).add(family)

    for lattice_size, family in ((17, "lattice17"), (33, "lattice33")):
        for lattice_y in range(lattice_size):
            y = top + round(
                (bottom - top - 1) * lattice_y / float(lattice_size - 1)
            )
            for lattice_x in range(lattice_size):
                x = left + round(
                    (right - left - 1) * lattice_x / float(lattice_size - 1)
                )
                add(y, x, family)

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
            add(y, x, "component_centroid")

    distances = ndimage.distance_transform_edt(valid)
    maxima = (
        (distances == ndimage.maximum_filter(distances, size=15, mode="constant"))
        & valid
    )
    ranked_maxima = sorted(
        (
            (-float(distances[y, x]), int(y), int(x))
            for y, x in zip(*np.nonzero(maxima))
        )
    )[:64]
    for _, y, x in ranked_maxima:
        add(y, x, "distance_transform_maximum")

    family_priority = {
        "lattice17": 0,
        "lattice33": 1,
        "component_centroid": 2,
        "distance_transform_maximum": 3,
    }
    seeds = []
    family_counts: Counter[str] = Counter()
    for (y, x), families in points.items():
        family = min(families, key=lambda value: family_priority[value])
        family_counts.update(families)
        seeds.append(
            (
                stable_hash_int(f"{CONTROL_VERSION}:{seed_key}:seed:{y}:{x}"),
                y,
                x,
                family,
            )
        )
    seeds.sort()
    return (
        seeds,
        labels.astype(np.int32, copy=False),
        np.bincount(labels.ravel()),
        dict(sorted(family_counts.items())),
    )


def _target_boundary_seeds(target: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    adjacent = valid & ndimage.binary_dilation(target, structure=structure)
    points = [(int(y), int(x)) for y, x in zip(*np.nonzero(adjacent))]
    if not points:
        return []
    selected = sorted(
        points,
        key=lambda point: hashlib.sha256(str(point).encode("utf-8")).digest(),
    )[:64]
    selected.extend(
        (
            min(points, key=lambda point: (point[0], point[1])),
            max(points, key=lambda point: (point[0], -point[1])),
            min(points, key=lambda point: (point[1], point[0])),
            max(points, key=lambda point: (point[1], -point[0])),
        )
    )
    return sorted(set(selected))


def _grow_priority_region(
    valid: np.ndarray,
    *,
    seed_y: int,
    seed_x: int,
    area: int,
    marker: int,
    seen_marks: np.ndarray,
    selected_marks: np.ndarray,
    target_distance: np.ndarray,
    target_centroid_x: float,
    target_centroid_y: float,
    frontier_order: str,
    tie_prefix: str,
    tie_cache: dict[int, bytes],
) -> tuple[list[int], int, int, int] | None:
    height, width = valid.shape

    def key(y: int, x: int) -> tuple[Any, ...]:
        flat_index = y * width + x
        tie = tie_cache.get(flat_index)
        if tie is None:
            tie = hashlib.sha256(f"{tie_prefix}:{y}:{x}".encode("utf-8")).digest()
            tie_cache[flat_index] = tie
        squared_l2 = (x - target_centroid_x) ** 2 + (y - target_centroid_y) ** 2
        l1 = abs(x - target_centroid_x) + abs(y - target_centroid_y)
        distance = float(target_distance[y, x])
        if frontier_order == "target_distance_then_l2":
            return distance, squared_l2, tie, flat_index
        if frontier_order == "centroid_l2_then_target_distance":
            return squared_l2, distance, tie, flat_index
        if frontier_order == "target_distance_then_l1":
            return distance, l1, tie, flat_index
        raise ValueError(f"unknown frontier order: {frontier_order}")

    heap: list[tuple[Any, ...]] = [key(seed_y, seed_x)]
    seen_marks[seed_y, seed_x] = marker
    indices: list[int] = []
    centroid_x_sum = 0
    centroid_y_sum = 0
    perimeter = 0
    while heap and len(indices) < area:
        *_, flat_index = heapq.heappop(heap)
        y, x = divmod(int(flat_index), width)
        selected_marks[y, x] = marker
        indices.append(int(flat_index))
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
            if not valid[neighbour_y, neighbour_x]:
                continue
            if seen_marks[neighbour_y, neighbour_x] == marker:
                continue
            seen_marks[neighbour_y, neighbour_x] = marker
            heapq.heappush(heap, key(neighbour_y, neighbour_x))
        perimeter += 4 - 2 * selected_neighbours
    if len(indices) != area:
        return None
    return indices, centroid_x_sum, centroid_y_sum, perimeter


def _translated_target_candidate(
    target: np.ndarray,
    content: np.ndarray,
) -> tuple[list[int], int, int, int, str, int] | None:
    target_geometry = mask_geometry(target, content)
    if int(target_geometry["component_count"]) != 1:
        return None
    left, top, right, bottom = target_geometry["bbox"]
    crop = target[top:bottom, left:right]
    content_left, content_right, content_top, content_bottom = _content_bounds(content)
    fft_shape = (2 * target.shape[0] - 1, 2 * target.shape[1] - 1)
    correlation = np.fft.irfftn(
        np.fft.rfftn(target.astype(np.float64), fft_shape)
        * np.fft.rfftn(target[::-1, ::-1].astype(np.float64), fft_shape),
        fft_shape,
    )
    target_zone = coordinate_zone(target, content)
    candidates = 0
    best: tuple[float, int, int] | None = None
    for shift_y in range(content_top - top, content_bottom - bottom + 1):
        for shift_x in range(content_left - left, content_right - right + 1):
            if shift_x == 0 and shift_y == 0:
                continue
            overlap = float(
                correlation[
                    target.shape[0] - 1 + shift_y,
                    target.shape[1] - 1 + shift_x,
                ]
            )
            if abs(overlap) >= 0.5:
                continue
            candidates += 1
            normalized_x = float(target_zone["normalized_x"]) + shift_x / max(
                1, content_right - content_left
            )
            normalized_y = float(target_zone["normalized_y"]) + shift_y / max(
                1, content_bottom - content_top
            )
            distance = abs(normalized_x - float(target_zone["normalized_x"])) + abs(
                normalized_y - float(target_zone["normalized_y"])
            )
            candidate = (float(distance), int(shift_y), int(shift_x))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _, shift_y, shift_x = best
    control = np.zeros_like(target)
    crop_height, crop_width = crop.shape
    control[
        top + shift_y : top + shift_y + crop_height,
        left + shift_x : left + shift_x + crop_width,
    ] = crop
    indices = np.flatnonzero(control).astype(np.int64).tolist()
    ys, xs = np.nonzero(control)
    identity = f"shift_y={shift_y},shift_x={shift_x}"
    return (
        indices,
        int(xs.sum()),
        int(ys.sum()),
        int(target_geometry["perimeter_edges"]),
        identity,
        candidates,
    )


def _candidate_record(
    *,
    family: str,
    identity: str,
    indices: list[int],
    centroid_x_sum: int,
    centroid_y_sum: int,
    perimeter: int,
    area: int,
    target_geometry: dict[str, Any],
    target_zone: dict[str, Any],
    content_bounds: tuple[int, int, int, int],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    content_left, content_right, content_top, content_bottom = content_bounds
    control_zone = _zone_from_centroid(
        centroid_x_sum / area,
        centroid_y_sum / area,
        content_left=content_left,
        content_right=content_right,
        content_top=content_top,
        content_bottom=content_bottom,
    )
    location_distance = abs(
        float(control_zone["normalized_x"]) - float(target_zone["normalized_x"])
    ) + abs(
        float(control_zone["normalized_y"]) - float(target_zone["normalized_y"])
    )
    log_perimeter_ratio = abs(
        math.log(float(perimeter) / float(target_geometry["perimeter_edges"]))
    )
    objective = location_distance + float(
        thresholds["log_perimeter_ratio_weight"]
    ) * log_perimeter_ratio
    reasons = []
    if location_distance > float(thresholds["max_location_distance"]) + 1e-12:
        reasons.append("location_distance_exceeds_frozen_maximum")
    if log_perimeter_ratio > float(thresholds["max_log_perimeter_ratio"]) + 1e-12:
        reasons.append("log_perimeter_ratio_exceeds_frozen_maximum")
    return {
        "family": family,
        "identity": identity,
        "indices": indices,
        "perimeter_edges": int(perimeter),
        "control_zone": control_zone,
        "location_distance": float(location_distance),
        "log_perimeter_ratio": float(log_perimeter_ratio),
        "objective": float(objective),
        "qualifies": not reasons,
        "rejection_reasons": reasons,
    }


def deterministic_continuous_location_connected_control_mask(
    target_mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    seed_key: str,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select the deterministic minimum-objective C6G connected control."""

    if target_mask.shape != content_mask.shape or target_mask.ndim != 2:
        raise ValueError("target/content masks must be shape-matched 2D arrays")
    target = target_mask.astype(bool)
    content = content_mask.astype(bool)
    target_geometry = mask_geometry(target, content)
    target_zone = coordinate_zone(target, content)
    valid = content & ~target
    area = int(target_geometry["area_pixels"])
    if area <= 0:
        raise ValueError("target mask is empty")
    if int(valid.sum()) < area:
        raise ValueError("insufficient target-disjoint content for exact-area connected control")
    bounds = _content_bounds(content)
    candidate_counts: Counter[str] = Counter()
    all_candidates: list[dict[str, Any]] = []

    def consider(
        family: str,
        identity: str,
        grown: tuple[list[int], int, int, int],
    ) -> None:
        indices, x_sum, y_sum, perimeter = grown
        candidate_counts[family] += 1
        all_candidates.append(
            _candidate_record(
                family=family,
                identity=identity,
                indices=indices,
                centroid_x_sum=x_sum,
                centroid_y_sum=y_sum,
                perimeter=perimeter,
                area=area,
                target_geometry=target_geometry,
                target_zone=target_zone,
                content_bounds=bounds,
                thresholds=thresholds,
            )
        )

    seeds, component_labels, component_sizes, seed_family_counts = _seed_families(
        valid,
        seed_key=seed_key,
    )
    seen_marks = np.zeros(target.shape, dtype=np.int32)
    selected_marks = np.zeros(target.shape, dtype=np.int32)
    tie_cache: dict[int, bytes] = {}
    marker = 0
    for seed_hash, seed_y, seed_x, family in seeds:
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
            tie_prefix=f"{CONTROL_VERSION}:{seed_key}:radial_frontier",
        )
        if grown is not None:
            consider(family, f"seed_hash={seed_hash}", grown)

    target_distance = ndimage.distance_transform_edt(~target)
    target_boundary_seeds = _target_boundary_seeds(target, valid)
    for frontier_order in FRONTIER_ORDERS:
        seen_marks.fill(0)
        selected_marks.fill(0)
        tie_cache = {}
        marker = 0
        for seed_y, seed_x in target_boundary_seeds:
            component_label = int(component_labels[seed_y, seed_x])
            if component_label <= 0 or int(component_sizes[component_label]) < area:
                continue
            marker += 1
            grown = _grow_priority_region(
                valid,
                seed_y=seed_y,
                seed_x=seed_x,
                area=area,
                marker=marker,
                seen_marks=seen_marks,
                selected_marks=selected_marks,
                target_distance=target_distance,
                target_centroid_x=float(target_geometry["centroid_x"]),
                target_centroid_y=float(target_geometry["centroid_y"]),
                frontier_order=frontier_order,
                tie_prefix=f"{CONTROL_VERSION}:{seed_key}:{frontier_order}",
                tie_cache=tie_cache,
            )
            if grown is not None:
                consider(
                    "target_boundary_growth",
                    f"order={frontier_order},seed_y={seed_y},seed_x={seed_x}",
                    grown,
                )

    translated = _translated_target_candidate(target, content)
    translated_candidate_count = 0
    if translated is not None:
        indices, x_sum, y_sum, perimeter, identity, translated_candidate_count = translated
        consider(
            "translated_target_shape",
            identity,
            (indices, x_sum, y_sum, perimeter),
        )

    same_zone_candidates = sum(
        candidate["control_zone"]["horizontal"] == target_zone["horizontal"]
        and candidate["control_zone"]["vertical"] == target_zone["vertical"]
        for candidate in all_candidates
    )
    nearest = min(
        all_candidates,
        key=lambda candidate: (
            float(candidate["objective"]),
            str(candidate["family"]),
            str(candidate["identity"]),
        ),
        default=None,
    )
    qualifying = [candidate for candidate in all_candidates if candidate["qualifies"]]
    selected = min(
        qualifying,
        key=lambda candidate: (
            float(candidate["objective"]),
            str(candidate["family"]),
            str(candidate["identity"]),
        ),
        default=None,
    )
    component_sizes = sorted(
        (int(value) for value in component_sizes[1:] if int(value) > 0),
        reverse=True,
    )
    certificate = {
        "version": CONTROL_VERSION,
        "seed_key": str(seed_key),
        "target": target_geometry,
        "target_zone": target_zone,
        "valid_content_area": int(valid.sum()),
        "valid_component_sizes": component_sizes,
        "seed_family_counts": seed_family_counts,
        "radial_seed_count": len(seeds),
        "target_boundary_seed_count": len(target_boundary_seeds),
        "frontier_orders": list(FRONTIER_ORDERS),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "translated_target_feasible_shifts": translated_candidate_count,
        "grown_candidate_count": len(all_candidates),
        "same_zone_candidate_count": int(same_zone_candidates),
        "qualifying_candidate_count": len(qualifying),
        "thresholds": {
            "max_location_distance": float(thresholds["max_location_distance"]),
            "max_log_perimeter_ratio": float(thresholds["max_log_perimeter_ratio"]),
            "log_perimeter_ratio_weight": float(
                thresholds["log_perimeter_ratio_weight"]
            ),
        },
        "nearest_candidate": None,
        "selected_candidate": None,
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "image_decode_authorized": False,
        "scores_accessed": False,
    }
    for name, candidate in (("nearest_candidate", nearest), ("selected_candidate", selected)):
        if candidate is None:
            continue
        certificate[name] = {
            key: value for key, value in candidate.items() if key != "indices"
        }
    if selected is None:
        raise ControlSearchFailure(certificate)

    control = np.zeros_like(target)
    control.flat[np.asarray(selected["indices"], dtype=np.int64)] = True
    if bool((control & target).any()) or bool((control & ~content).any()):
        raise AssertionError("C6G control violates target/content containment")
    control_geometry = mask_geometry(control, content)
    if int(control_geometry["area_pixels"]) != area:
        raise AssertionError("C6G control does not preserve exact area")
    if int(control_geometry["component_count"]) != 1:
        raise AssertionError("C6G control is not one 4-connected component")
    if int(control_geometry["perimeter_edges"]) != int(selected["perimeter_edges"]):
        raise AssertionError("C6G incremental perimeter mismatch")
    audit = {
        **certificate,
        "control": control_geometry,
        "control_zone": coordinate_zone(control, content),
        "disjoint": True,
        "within_content": True,
        "connected_component_requirement": 1,
        "true_anatomy_segmentation": False,
    }
    return control, audit


def _build_geometry_row(payload: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    row, mask_dir_string, thresholds = payload
    width = int(row["native_columns"])
    height = int(row["native_rows"])
    content_box, content = _content_geometry(width, height)
    target = transform_mask_to_letterbox(
        union_box_mask(width, height, row["bounding_boxes"]),
        content_box,
        IMAGE_SIZE,
    ) & content
    base = {
        "format_version": GEOMETRY_FORMAT_VERSION,
        "sample_id": row["sample_id"],
        "canonical_statement_id": row["canonical_statement_id"],
        "target_area_pixels": int(target.sum()),
        "target_area_fraction": float(target.sum() / content.sum()),
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "image_decode_authorized": False,
        "scores_accessed": False,
    }
    try:
        control, audit = deterministic_continuous_location_connected_control_mask(
            target,
            content,
            seed_key=f"{row['sample_id']}:{CONTROL_VERSION}",
            thresholds=thresholds,
        )
    except ControlSearchFailure as error:
        return {
            **base,
            "control_area_pixels": 0,
            "mask_file": None,
            "mask_sha256": None,
            "control_audit": error.certificate,
            "feasible": False,
            "failure": str(error),
        }
    mask_name = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest() + ".npz"
    mask_path = Path(mask_dir_string) / mask_name
    temporary = mask_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        target_mask=target,
        control_mask=control,
        content_mask=content,
    )
    temporary.replace(mask_path)
    return {
        **base,
        "control_area_pixels": int(control.sum()),
        "mask_file": mask_name,
        "mask_sha256": file_sha256(mask_path),
        "control_audit": audit,
        "feasible": True,
        "failure": None,
    }


def build_c6g_geometry(
    rows: list[dict[str, Any]],
    *,
    mask_dir: Path,
    thresholds: dict[str, Any],
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_ms_cxr_manifest(rows)
    if workers <= 0:
        raise ValueError("geometry workers must be positive")
    mask_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(row, str(mask_dir), thresholds) for row in rows]
    if workers == 1:
        geometry_rows = [_build_geometry_row(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            geometry_rows = list(executor.map(_build_geometry_row, payloads))
    by_finding = {
        finding: sorted(
            (
                row
                for row in geometry_rows
                if row["canonical_statement_id"] == finding
            ),
            key=lambda row: (float(row["target_area_fraction"]), str(row["sample_id"])),
        )
        for finding in EXPECTED_ROWS
    }
    for finding, subset in by_finding.items():
        boundaries = np.quantile(
            [float(row["target_area_fraction"]) for row in subset],
            [0.25, 0.5, 0.75],
        )
        for row in subset:
            row["box_area_quartile"] = int(
                np.searchsorted(
                    boundaries,
                    float(row["target_area_fraction"]),
                    side="right",
                )
                + 1
            )
    geometry_rows.sort(key=lambda row: str(row["sample_id"]))
    summary = {
        "format_version": LOCK_FORMAT_VERSION,
        "status": "pass" if all(row["feasible"] for row in geometry_rows) else "fail_geometry",
        "rows": len(geometry_rows),
        "eligible": sum(bool(row["feasible"]) for row in geometry_rows),
        "infeasible": sum(not bool(row["feasible"]) for row in geometry_rows),
        "control_version": CONTROL_VERSION,
        "image_size": IMAGE_SIZE,
        "per_finding": {
            finding: sum(row["canonical_statement_id"] == finding for row in geometry_rows)
            for finding in EXPECTED_ROWS
        },
        "invariant_failures": 0,
        "denominator_exclusions": 0,
        "evaluation_gate_open_geometry": all(row["feasible"] for row in geometry_rows),
        "model_evaluation_authorized": False,
        "gpu_authorized": False,
        "image_decode_authorized": False,
        "scores_accessed": False,
        "thresholds": thresholds,
    }
    if summary["rows"] != 29 or summary["per_finding"] != EXPECTED_ROWS:
        raise ValueError("C6G geometry denominator changed")
    summary["canonical_sha256"] = canonical_json_sha256(summary)
    return geometry_rows, summary


def write_c6g_artifacts(
    *,
    output_dir: Path,
    geometry_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    source_manifest_path: Path,
    authority_path: Path,
    protocol_plan_path: Path,
    threshold_path: Path,
    c6f_hashes: dict[str, str],
    implementation_paths: dict[str, Path],
    source_commit: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "c6g_geometry_rows.jsonl"
    certificate_path = output_dir / "c6g_candidate_certificates.jsonl"
    lock_path = output_dir / "c6g_geometry_lock.json"
    _write_jsonl(rows_path, geometry_rows)
    _write_jsonl(
        certificate_path,
        [
            {
                "sample_id": row["sample_id"],
                "canonical_statement_id": row["canonical_statement_id"],
                "feasible": row["feasible"],
                "failure": row["failure"],
                "certificate": row["control_audit"],
            }
            for row in geometry_rows
        ],
    )
    lock = {
        **summary,
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "authority_sha256": file_sha256(authority_path),
        "protocol_plan_sha256": file_sha256(protocol_plan_path),
        "threshold_artifact_sha256": file_sha256(threshold_path),
        "geometry_rows_sha256": file_sha256(rows_path),
        "candidate_certificates_sha256": file_sha256(certificate_path),
        "c6f_immutable_sha256": c6f_hashes,
        "implementation_sha256": {
            name: file_sha256(path)
            for name, path in sorted(implementation_paths.items())
        },
        "source_commit": str(source_commit),
    }
    lock["canonical_sha256"] = canonical_json_sha256(lock)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "geometry_rows": rows_path,
        "candidate_certificates": certificate_path,
        "geometry_lock": lock_path,
        "lock": lock,
    }
