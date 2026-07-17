from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "index_mimic_bives_p0_candidates.py"


class P0IntakeTests(unittest.TestCase):
    def _make_pair(self, images: Path, reports: Path, patient: str, study: str, image_id: str) -> None:
        image_dir = images / "files" / "p10" / patient / study
        report_dir = reports / "files" / "p10" / patient
        image_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{image_id}.jpg").write_bytes(b"jpeg-placeholder")
        (report_dir / f"{study}.txt").write_text("private report text", encoding="utf-8")

    def test_index_writes_only_paired_intake_rows_without_report_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, reports = root / "images", root / "reports"
            self._make_pair(images, reports, "p10001", "s50001", "image1")
            missing_report_dir = images / "files" / "p10" / "p10002" / "s50002"
            missing_report_dir.mkdir(parents=True)
            (missing_report_dir / "image2.jpg").write_bytes(b"jpeg-placeholder")
            output, summary = root / "candidates.jsonl", root / "summary.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--images-root",
                    str(images),
                    "--reports-root",
                    str(reports),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"status": "intake_only"', result.stdout)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_status"], "unparsed")
            self.assertNotIn("report_text", rows[0])
            self.assertEqual(rows[0]["patient_id"], "p10001")
            audit = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(audit["missing_report_studies"], 1)
            self.assertEqual(audit["labeling_claim"], "none")

    def test_index_respects_study_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, reports = root / "images", root / "reports"
            self._make_pair(images, reports, "p10001", "s50001", "image1")
            self._make_pair(images, reports, "p10002", "s50002", "image2")
            output = root / "limited.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--images-root",
                    str(images),
                    "--reports-root",
                    str(reports),
                    "--output",
                    str(output),
                    "--max-studies",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
