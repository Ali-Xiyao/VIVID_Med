"""Deterministic chest-radiograph DICOM loading for active BiVES inputs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut


DICOM_PREPROCESS_VERSION = "bives_cxr_dicom_v1"


@dataclass(frozen=True)
class DicomPreprocessRecord:
    version: str
    photometric_interpretation: str
    voi_applied: bool
    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float
    rows: int
    columns: int
    rgb_sha256: str

    def to_dict(self) -> dict[str, str | bool | float | int]:
        return asdict(self)


def _has_voi_window(dataset: pydicom.dataset.Dataset) -> bool:
    return bool(
        hasattr(dataset, "VOILUTSequence")
        or (hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth"))
    )


def load_cxr_dicom(
    path: str | Path,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> tuple[Image.Image, DicomPreprocessRecord]:
    """Load a single-frame CXR DICOM into deterministic 8-bit RGB pixels.

    Modality LUT/rescale is applied first, followed by an available VOI LUT or
    window. MONOCHROME1 is inverted before robust percentile normalization.
    Constant or non-finite images fail closed instead of becoming synthetic
    ``insufficient`` samples.
    """

    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("DICOM percentiles must satisfy 0 <= lower < upper <= 100")
    source = Path(path)
    dataset = pydicom.dcmread(source)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(
            f"unsupported DICOM PhotometricInterpretation {photometric!r}: {source}"
        )

    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim != 2:
        raise ValueError(f"expected a single-frame 2D DICOM, got shape {pixels.shape}: {source}")
    pixels = np.asarray(apply_modality_lut(pixels, dataset), dtype=np.float64)
    voi_applied = _has_voi_window(dataset)
    if voi_applied:
        pixels = np.asarray(apply_voi_lut(pixels, dataset), dtype=np.float64)
    if not bool(np.isfinite(pixels).all()):
        raise ValueError(f"DICOM contains non-finite pixels after LUT/window: {source}")
    if photometric == "MONOCHROME1":
        pixels = float(pixels.max() + pixels.min()) - pixels

    lower = float(np.percentile(pixels, lower_percentile))
    upper = float(np.percentile(pixels, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"constant or degenerate DICOM after preprocessing: {source}")
    normalized = np.clip((pixels - lower) / (upper - lower), 0.0, 1.0)
    grayscale = np.rint(normalized * 255.0).astype(np.uint8)
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    record = DicomPreprocessRecord(
        version=DICOM_PREPROCESS_VERSION,
        photometric_interpretation=photometric,
        voi_applied=voi_applied,
        lower_percentile=float(lower_percentile),
        upper_percentile=float(upper_percentile),
        lower_value=lower,
        upper_value=upper,
        rows=int(rgb.shape[0]),
        columns=int(rgb.shape[1]),
        rgb_sha256=hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
    )
    return image, record
