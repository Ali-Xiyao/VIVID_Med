"""Deterministic strength-controlled VICER V0 pixel interventions."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from bives_cxr.pixel_interventions import (
    replace_with_local_ring_mean,
    replace_with_masked_gaussian_blur,
)


INTERVENTION_BANK_VERSION = "vicer-v0-intervention-bank-v1"


def _blend_inside(
    source: Image.Image,
    replacement: Image.Image,
    mask: np.ndarray,
    alpha: float,
) -> Image.Image:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("intervention blend alpha must be in (0, 1]")
    original = np.asarray(source.convert("RGB"), dtype=np.float64)
    changed = np.asarray(replacement.convert("RGB"), dtype=np.float64)
    target = np.asarray(mask, dtype=bool)
    if original.shape[:2] != target.shape or changed.shape != original.shape:
        raise ValueError("intervention image/mask geometry changed")
    output = original.copy()
    output[target] = (1.0 - alpha) * original[target] + alpha * changed[target]
    return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")


def apply_v0_intervention(
    image: Image.Image,
    mask: np.ndarray,
    content_mask: np.ndarray,
    *,
    family: str,
    strength: float,
) -> tuple[Image.Image, dict[str, Any]]:
    if family == "masked_gaussian_blur":
        if strength not in {2.0, 4.0, 8.0, 16.0}:
            raise ValueError("unregistered Gaussian sigma")
        output = replace_with_masked_gaussian_blur(
            image,
            mask,
            content_mask,
            sigma=float(strength),
            truncate=3.0,
        )
        parameters = {"sigma": float(strength), "truncate": 3.0}
    elif family == "local_ring_mean":
        if strength not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("unregistered local-mean blend strength")
        replacement = replace_with_local_ring_mean(
            image,
            mask,
            content_mask,
            ring_width=8,
        )
        output = _blend_inside(image, replacement, mask, float(strength))
        parameters = {"alpha": float(strength), "ring_width": 8}
    elif family == "low_frequency_replacement":
        if strength not in {0.25, 0.5, 0.75, 1.0}:
            raise ValueError("unregistered low-frequency blend strength")
        replacement = replace_with_masked_gaussian_blur(
            image,
            mask,
            content_mask,
            sigma=24.0,
            truncate=3.0,
        )
        output = _blend_inside(image, replacement, mask, float(strength))
        parameters = {"alpha": float(strength), "sigma": 24.0, "truncate": 3.0}
    else:
        raise ValueError(f"unknown VICER V0 operator family: {family}")
    return output, {
        "version": INTERVENTION_BANK_VERSION,
        "family": family,
        "strength": float(strength),
        "parameters": parameters,
    }
