"""Frozen hard-UMS and deterministic free-text contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


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
STATES = ("present", "absent", "uncertain")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATES)}

_TEMPLATES = {
    "present": (
        "{name} is present.",
        "{name} is observed.",
        "There is evidence of {name}.",
        "The image shows {name}.",
    ),
    "absent": (
        "{name} is absent.",
        "No {name} is seen.",
        "No evidence of {name}.",
        "The image is negative for {name}.",
    ),
    "uncertain": (
        "{name} is uncertain.",
        "{name} is equivocal.",
        "Possible {name}.",
        "{name} cannot be excluded.",
    ),
}


def parse_ums_target(target: str) -> tuple[list[int], list[bool]]:
    """Parse one hard-UMS target into fixed finding states and masks."""
    payload = json.loads(target)
    if payload.get("modality") != "CXR":
        raise ValueError("hard UMS modality must be CXR")
    findings = payload.get("findings")
    if not isinstance(findings, Mapping) or not findings:
        raise ValueError("hard UMS requires a nonempty findings object")
    unknown = set(findings) - set(FINDINGS)
    if unknown:
        raise ValueError(f"unknown hard-UMS findings: {sorted(unknown)}")
    states = [-100] * len(FINDINGS)
    mask = [False] * len(FINDINGS)
    for name, item in findings.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entry must be an object")
        state = item.get("state")
        if state not in STATE_TO_INDEX:
            raise ValueError(f"{name} has unsupported state {state!r}")
        index = FINDINGS.index(name)
        states[index] = STATE_TO_INDEX[str(state)]
        mask[index] = True
    return states, mask


def _stable_index(row_id: str, finding: str, state: str, count: int) -> int:
    digest = hashlib.sha256(
        f"{row_id}|{finding}|{state}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % count


def render_free_text(target: str, row_id: str) -> str:
    """Render exactly the selected UMS fields with deterministic templates."""
    payload = json.loads(target)
    findings = payload.get("findings")
    if not isinstance(findings, Mapping):
        raise ValueError("hard UMS requires a findings object")
    parse_ums_target(target)
    ordered = sorted(
        findings,
        key=lambda name: hashlib.sha256(
            f"{row_id}|order|{name}".encode("utf-8")
        ).hexdigest(),
    )
    sentences = []
    for name in ordered:
        state = str(findings[name]["state"])
        templates = _TEMPLATES[state]
        template = templates[_stable_index(row_id, name, state, len(templates))]
        sentences.append(template.format(name=name))
    if not sentences:
        raise ValueError("free-text rendering produced no supervised fields")
    return " ".join(sentences)
