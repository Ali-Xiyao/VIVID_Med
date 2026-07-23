"""Conservative LUNGUAGE entity mapping for the twelve RCSD report concepts."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


FINDINGS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Fracture",
    "Support Devices",
)
STATE_TO_INDEX = {"absent": 0, "present": 1, "uncertain": 2}
INDEX_TO_STATE = {value: key for key, value in STATE_TO_INDEX.items()}


def entity_state(row: Mapping[str, object]) -> str | None:
    """Map report assertion/certainty to A/P/U; uncertainty is report-only."""
    certainty = str(row.get("dx_certainty") or "").strip().lower()
    status = str(row.get("dx_status") or "").strip().lower()
    if certainty == "tentative":
        return "uncertain"
    if status == "positive":
        return "present"
    if status == "negative":
        return "absent"
    return None


def normalized_entity(row: Mapping[str, object]) -> str:
    return str(
        row.get("new_normed_ent")
        or row.get("normed_ent")
        or row.get("ent")
        or ""
    ).strip().lower()


def match_findings(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return only high-precision concept matches fixed before source scoring."""
    term = normalized_entity(row)
    category = str(row.get("cat") or "").strip().lower().rstrip(".")
    radiographic = category in {"pf", "cf"}
    matched: list[str] = []

    # Enlarged Cardiomediastinum is intentionally unmapped: generic silhouette
    # entities do not uniquely encode enlargement in LUNGUAGE.
    if radiographic and "cardiomegaly" in term:
        matched.append("Cardiomegaly")
    if radiographic and ("opacity" in term or "infiltrat" in term):
        matched.append("Lung Opacity")
    lesion_terms = {
        "nodule",
        "metastatic nodule",
        "fdg-avid nodule",
        "mass",
        "lesion",
        "metastatic lesion",
    }
    if radiographic and term in lesion_terms:
        matched.append("Lung Lesion")
    edema_terms = (
        "pulmonary edema",
        "pulmonary interstitial edema",
        "perihilar edema",
        "alveolar edema",
        "flash pulmonary edema",
        "reexpansion lung edema",
        "cardiogenic pulmonary edema",
        "subclinical pulmonary edema",
        "congestion/edema",
        "edema",
    )
    if radiographic and term in edema_terms:
        matched.append("Edema")
    if radiographic and ("consolidat" in term):
        matched.append("Consolidation")
    if radiographic and ("pneumonia" in term or "pneumonic" in term):
        matched.append("Pneumonia")
    if radiographic and "atelect" in term:
        matched.append("Atelectasis")
    if radiographic and "pneumothorax" in term:
        matched.append("Pneumothorax")
    effusion_terms = {
        "pleural effusion",
        "pleural effusion fluid",
        "pneumonia effusion",
        "effusion",
    }
    if radiographic and term in effusion_terms:
        matched.append("Pleural Effusion")
    if category == "pf" and "fracture" in term:
        matched.append("Fracture")
    if category == "oth":
        matched.append("Support Devices")
    return tuple(dict.fromkeys(matched))


def aggregate_study_gold(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[tuple[str, str], int], dict[str, object]]:
    """Aggregate entity assertions to study/finding labels without blank=absent."""
    observed: dict[tuple[str, str], list[int]] = defaultdict(list)
    patient_by_study: dict[str, str] = {}
    match_counts = defaultdict(int)
    for row in rows:
        study_id = str(row.get("study_id") or "").strip().removeprefix("s")
        patient_id = str(row.get("subject_id") or "").strip().removeprefix("p")
        if study_id and patient_id:
            patient_by_study[study_id] = patient_id
        state = entity_state(row)
        if not study_id or state is None:
            continue
        for finding in match_findings(row):
            observed[(study_id, finding)].append(STATE_TO_INDEX[state])
            match_counts[finding] += 1

    labels: dict[tuple[str, str], int] = {}
    conflicts = defaultdict(int)
    for key, values in observed.items():
        unique = set(values)
        conflicts[key[1]] += int(len(unique) > 1)
        # Definite present dominates; tentative dominates a definite negative.
        if STATE_TO_INDEX["present"] in unique:
            labels[key] = STATE_TO_INDEX["present"]
        elif STATE_TO_INDEX["uncertain"] in unique:
            labels[key] = STATE_TO_INDEX["uncertain"]
        else:
            labels[key] = STATE_TO_INDEX["absent"]
    audit = {
        "patient_by_study": patient_by_study,
        "entity_match_counts": dict(match_counts),
        "within_study_state_conflicts": dict(conflicts),
    }
    return labels, audit
