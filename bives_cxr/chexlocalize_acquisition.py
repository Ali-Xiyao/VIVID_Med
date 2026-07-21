"""Fail-closed selection helpers for CheXlocalize development acquisition."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


ALLOWED_EXACT_PATHS = frozenset(
    {
        "gt_annotations_val.json",
        "gt_segmentations_val.json",
        "gradcam_segmentations_val.json",
    }
)
ALLOWED_PREFIXES = ("gradcam_maps_val/",)
EXPECTED_VALIDATION_FILE_COUNT = 2343
EXPECTED_VALIDATION_TOTAL_BYTES = 3_849_154_259


def is_allowed_validation_path(path: str) -> bool:
    """Return whether a Redivis file path is within the frozen dev-only allowlist."""

    normalized = path.replace("\\", "/").lstrip("/")
    lowered = normalized.lower()
    if "test" in lowered:
        return False
    return normalized in ALLOWED_EXACT_PATHS or normalized.startswith(ALLOWED_PREFIXES)


def select_validation_files(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_count: int = EXPECTED_VALIDATION_FILE_COUNT,
    expected_total_bytes: int = EXPECTED_VALIDATION_TOTAL_BYTES,
) -> list[dict[str, object]]:
    """Select the exact frozen validation release and reject drift or test leakage."""

    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        path = str(row["path"]).replace("\\", "/").lstrip("/")
        if not is_allowed_validation_path(path):
            continue
        if path in seen:
            raise ValueError(f"duplicate CheXlocalize validation path: {path}")
        seen.add(path)
        row["path"] = path
        row["size"] = int(row["size"])
        selected.append(row)

    selected.sort(key=lambda row: str(row["path"]))
    total_bytes = sum(int(row["size"]) for row in selected)
    if len(selected) != expected_count:
        raise ValueError(
            f"CheXlocalize validation file count drift: {len(selected)} != {expected_count}"
        )
    if total_bytes != expected_total_bytes:
        raise ValueError(
            f"CheXlocalize validation size drift: {total_bytes} != {expected_total_bytes}"
        )
    for required in ALLOWED_EXACT_PATHS:
        if required not in seen:
            raise ValueError(f"missing required CheXlocalize validation file: {required}")
    if any("test" in str(row["path"]).lower() for row in selected):
        raise ValueError("CheXlocalize test path leaked into validation acquisition")
    return selected
