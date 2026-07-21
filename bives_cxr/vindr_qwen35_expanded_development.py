"""Result-blind selection for the expanded VinDr Qwen3.5 development matrix."""

from __future__ import annotations

from typing import Any, Iterable


FINDINGS = ("consolidation", "pleural_effusion")
AREA_QUARTILES = (1, 2, 3, 4)
PER_STRATUM = 4


def select_expanded_rows(
    rows: Iterable[dict[str, Any]],
    *,
    excluded_image_ids: set[str],
) -> list[dict[str, Any]]:
    """Select 32 unique non-pilot images under a frozen lexical rule."""

    candidates = [
        dict(row)
        for row in rows
        if row.get("rescue_split") == "protocol_design"
        and int(row.get("binary_label", 0)) == 1
        and row.get("canonical_statement_id") in FINDINGS
        and int(row.get("box_area_quartile", 0)) in AREA_QUARTILES
        and str(row.get("image_id")) not in excluded_image_ids
    ]
    selected: list[dict[str, Any]] = []
    used = set(excluded_image_ids)
    for finding in FINDINGS:
        for quartile in AREA_QUARTILES:
            subset = sorted(
                (
                    row
                    for row in candidates
                    if row["canonical_statement_id"] == finding
                    and int(row["box_area_quartile"]) == quartile
                ),
                key=lambda row: (
                    0 if row.get("reader_consensus") == "3_of_3" else 1,
                    str(row["sample_id"]),
                ),
            )
            chosen = []
            for row in subset:
                image_id = str(row["image_id"])
                if image_id in used:
                    continue
                used.add(image_id)
                chosen.append(row)
                if len(chosen) == PER_STRATUM:
                    break
            if len(chosen) != PER_STRATUM:
                raise ValueError(
                    f"insufficient unique VinDr rows for {finding}|q{quartile}: {len(chosen)}"
                )
            selected.extend(chosen)
    if len(selected) != len(FINDINGS) * len(AREA_QUARTILES) * PER_STRATUM:
        raise AssertionError("expanded VinDr selection count changed")
    if len({str(row["image_id"]) for row in selected}) != len(selected):
        raise AssertionError("expanded VinDr selection contains duplicate images")
    return selected
