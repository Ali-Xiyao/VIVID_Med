from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_bives_weak_sc import prepare_weak_sc


class WeakSCBuilderTests(unittest.TestCase):
    def test_builder_filters_balances_and_patient_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for patient_index in range(20):
                image = root / f"image-{patient_index}.jpg"
                image.write_bytes(f"image-{patient_index}".encode("ascii"))
                for finding in ("pleural_effusion", "consolidation"):
                    state = "support" if patient_index % 2 else "contradict"
                    rows.append(
                        {
                            "candidate_id": f"candidate::{patient_index}::{finding}",
                            "canonical_statement_id": finding,
                            "parser_status": "candidate",
                            "parser_state_candidate": state,
                            "review_track": "p0_1_explicit_positive_negative",
                            "patient_id": f"p{patient_index:03d}",
                            "study_id": f"s{patient_index:03d}",
                            "image_id": f"i{patient_index:03d}",
                            "image_path": str(image),
                            "statement_text": f"{finding} is present.",
                            "report_sha256": "a" * 64,
                            "parser_version": "v3",
                            "parser_rules_sha256": "b" * 64,
                            "parser_cue": finding,
                        }
                    )
            rows.append(
                {
                    **rows[0],
                    "candidate_id": "excluded-uncertain",
                    "parser_state_candidate": "uncertain",
                    "review_track": "p0_2_uncertain_insufficient",
                }
            )
            candidates = root / "candidates.jsonl"
            candidates.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "output"
            result = prepare_weak_sc(candidates, output, verify_images=False)
            train = [json.loads(line) for line in (output / "weak_sc_train.jsonl").read_text().splitlines()]
            val = [json.loads(line) for line in (output / "weak_sc_val.jsonl").read_text().splitlines()]
            self.assertFalse({row["patient_id"] for row in train} & {row["patient_id"] for row in val})
            self.assertEqual(result["patient_overlap"], 0)
            for split_rows in (train, val):
                for finding in ("pleural_effusion", "consolidation"):
                    finding_rows = [row for row in split_rows if row["canonical_statement_id"] == finding]
                    self.assertEqual(
                        sum(row["state"] == "support" for row in finding_rows),
                        sum(row["state"] == "contradict" for row in finding_rows),
                    )
                self.assertTrue(all(row["state"] in {"support", "contradict"} for row in split_rows))
                self.assertTrue(all(row["binary_label"] in {0, 1} for row in split_rows))
            self.assertNotIn("excluded-uncertain", {row["source_candidate_id"] for row in train + val})


if __name__ == "__main__":
    unittest.main()
