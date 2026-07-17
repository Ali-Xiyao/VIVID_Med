from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_bives_p0_report_review.py"
VALIDATE = ROOT / "scripts" / "validate_bives_p0_review_packet.py"


class P0ReviewTests(unittest.TestCase):
    def _candidate(self, root: Path, report_text: str) -> Path:
        report = root / "p10001_s50001.txt"
        report.write_text(report_text, encoding="utf-8")
        candidates = root / "candidates.jsonl"
        candidates.write_text(
            json.dumps(
                {
                    "candidate_id": "candidate_1",
                    "source_dataset": "MIMIC-CXR-JPG",
                    "patient_id": "p10001",
                    "study_id": "s50001",
                    "image_id": "image1",
                    "image_path": str(root / "image1.jpg"),
                    "report_path": str(report),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return candidates

    def test_parser_emits_blinded_packet_without_report_text_or_parser_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = self._candidate(root, "Possible small pleural effusion. No pneumothorax.")
            parsed, packet, summary = root / "parsed.jsonl", root / "packet.csv", root / "summary.json"
            subprocess.run(
                [sys.executable, str(PREPARE), "--candidates", str(candidates), "--parsed-output", str(parsed), "--review-packet", str(packet), "--summary", str(summary)],
                check=True,
                capture_output=True,
                text=True,
            )
            parsed_rows = [json.loads(line) for line in parsed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["parser_state_candidate"] for row in parsed_rows}, {"uncertain", "contradict"})
            with packet.open(encoding="utf-8", newline="") as handle:
                review_row = next(csv.DictReader(handle))
            self.assertNotIn("parser_state_candidate", review_row)
            self.assertNotIn("report_text", review_row)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["labeling_claim"], "none")

    def test_validator_fails_closed_then_accepts_two_reviewers_and_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / "packet.csv"
            fields = [
                "candidate_id", "reviewer_1_id", "reviewer_1_state", "reviewer_2_id", "reviewer_2_state", "adjudicator_id", "adjudicated_state",
            ]
            with packet.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"candidate_id": "x"})
            failed = subprocess.run([sys.executable, str(VALIDATE), "--review-packet", str(packet)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            with packet.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "candidate_id": "x", "reviewer_1_id": "r1", "reviewer_1_state": "uncertain", "reviewer_2_id": "r2", "reviewer_2_state": "insufficient", "adjudicator_id": "a1", "adjudicated_state": "uncertain",
                })
            passed = subprocess.run([sys.executable, str(VALIDATE), "--review-packet", str(packet)], check=True, capture_output=True, text=True)
            self.assertIn('"status": "pass"', passed.stdout)
