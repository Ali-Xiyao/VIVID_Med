from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from bives_cxr.dicom import DICOM_PREPROCESS_VERSION, load_cxr_dicom


def write_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    window: tuple[float, float] | None = None,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    if window is not None:
        dataset.WindowCenter = window[0]
        dataset.WindowWidth = window[1]
    dataset.PixelData = pixels.astype("<u2").tobytes()
    dataset.save_as(path, enforce_file_format=True)


class VinDrDicomTests(unittest.TestCase):
    def test_monochrome2_without_window_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono2.dicom"
            write_dicom(path, np.arange(64, dtype=np.uint16).reshape(8, 8))
            first, first_record = load_cxr_dicom(path)
            second, second_record = load_cxr_dicom(path)
            self.assertEqual(first.mode, "RGB")
            self.assertEqual(first.size, (8, 8))
            self.assertEqual(first.tobytes(), second.tobytes())
            self.assertEqual(first_record.rgb_sha256, second_record.rgb_sha256)
            self.assertEqual(first_record.version, DICOM_PREPROCESS_VERSION)
            self.assertFalse(first_record.voi_applied)

    def test_monochrome1_inverts_brightness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
            mono2 = root / "mono2.dicom"
            mono1 = root / "mono1.dicom"
            write_dicom(mono2, pixels, photometric="MONOCHROME2")
            write_dicom(mono1, pixels, photometric="MONOCHROME1")
            image2, _ = load_cxr_dicom(mono2)
            image1, _ = load_cxr_dicom(mono1)
            values2 = np.asarray(image2)[:, :, 0].astype(np.int16)
            values1 = np.asarray(image1)[:, :, 0].astype(np.int16)
            self.assertLessEqual(int(np.abs(values1 + values2 - 255).max()), 1)

    def test_windowed_dicom_applies_voi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "windowed.dicom"
            pixels = np.arange(256, dtype=np.uint16).reshape(16, 16)
            write_dicom(path, pixels, window=(128.0, 64.0))
            image, record = load_cxr_dicom(path)
            self.assertTrue(record.voi_applied)
            self.assertGreater(len(np.unique(np.asarray(image)[:, :, 0])), 2)

    def test_constant_dicom_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constant.dicom"
            write_dicom(path, np.full((8, 8), 7, dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "constant or degenerate"):
                load_cxr_dicom(path)


if __name__ == "__main__":
    unittest.main()
