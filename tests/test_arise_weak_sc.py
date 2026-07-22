import json
import tempfile
import unittest
from pathlib import Path

from arise_cxr.weak_sc import prepare_arise_weak_sc


class AriseWeakSCTests(unittest.TestCase):
    def test_builds_balanced_patient_disjoint_three_finding_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"not-opened-when-hash-disabled")
            candidates = root / "candidates.jsonl"
            rows = []
            for finding in ("consolidation", "pleural_effusion", "pulmonary_edema"):
                for state in ("support", "contradict"):
                    for index in range(10):
                        rows.append(
                            {
                                "candidate_id": f"{finding}-{state}-{index}",
                                "patient_id": f"patient-{finding}-{state}-{index}",
                                "study_id": f"study-{index}",
                                "image_id": f"image-{finding}-{state}-{index}",
                                "image_path": str(image),
                                "canonical_statement_id": finding,
                                "statement_text": f"{finding} is present.",
                                "parser_status": "candidate",
                                "parser_state_candidate": state,
                                "review_track": "p0_1_explicit_positive_negative",
                                "report_sha256": "a" * 64,
                                "parser_version": "unit-v1",
                                "parser_rules_sha256": "b" * 64,
                                "parser_cue": state,
                            }
                        )
            candidates.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "output"
            lock = prepare_arise_weak_sc(
                candidates,
                output,
                findings=("consolidation", "pleural_effusion", "pulmonary_edema"),
                validation_fraction=0.2,
                seed=7,
                verify_images=False,
            )
            self.assertEqual(lock["patient_overlap"], 0)
            self.assertEqual(lock["findings"], [
                "consolidation",
                "pleural_effusion",
                "pulmonary_edema",
            ])
            self.assertEqual(set(lock["train_counts"].values()), {8})
            self.assertEqual(set(lock["val_counts"].values()), {2})
            self.assertFalse(lock["test_opened"])
            self.assertEqual(lock["status"], "feasibility_only_unhashed")

    def test_rejects_single_finding(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidates = Path(temporary) / "candidates.jsonl"
            candidates.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least two"):
                prepare_arise_weak_sc(
                    candidates,
                    Path(temporary) / "output",
                    findings=("consolidation",),
                    verify_images=False,
                )


if __name__ == "__main__":
    unittest.main()
