import json
import unittest

import torch

from rcsd_cxr.d0_d1_contract import render_hard_ums_target
from rcsd_cxr.token_objective import (
    finding_block_spans,
    prepare_token_batch,
    token_cross_entropy,
)


class CharTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        maximum = max(len(text) for text in texts)
        input_ids = []
        attention = []
        offsets = []
        for text in texts:
            ids = [1 + ord(char) % 97 for char in text]
            count = len(ids)
            input_ids.append(ids + [0] * (maximum - count))
            attention.append([1] * count + [0] * (maximum - count))
            offsets.append(
                [(index, index + 1) for index in range(count)]
                + [(0, 0)] * (maximum - count)
            )
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "offset_mapping": torch.tensor(offsets),
        }


class TokenObjectiveTests(unittest.TestCase):
    def test_serialization_matches_frozen_simple_target(self):
        target = render_hard_ums_target(
            {
                "Cardiomegaly": "present",
                "Edema": "absent",
                "Fracture": None,
            }
        )
        self.assertEqual(
            target,
            '{"modality": "CXR", "findings": {"Cardiomegaly": '
            '{"state": "present", "score": null}, "Edema": '
            '{"state": "absent", "score": null}}, "study_view": null}',
        )
        self.assertEqual(
            list(finding_block_spans(target)),
            ["Cardiomegaly", "Edema"],
        )

    def test_d0_and_d1_have_identical_ids_and_labels(self):
        target = render_hard_ums_target(
            {"Cardiomegaly": "present", "Edema": "absent"}
        )
        weights = {"Cardiomegaly": 0.25, "Edema": 1.0}
        tokenizer = CharTokenizer()
        d0 = prepare_token_batch(
            tokenizer,
            prompt="P:",
            targets=[target],
            finding_weights=[weights],
            variant="d0",
        )
        d1 = prepare_token_batch(
            tokenizer,
            prompt="P:",
            targets=[target],
            finding_weights=[weights],
            variant="d1",
        )
        torch.testing.assert_close(d0["input_ids"], d1["input_ids"])
        torch.testing.assert_close(d0["labels"], d1["labels"])
        self.assertLess(float(d1["token_weights"].min()), 1.0)
        self.assertEqual(
            json.loads(target)["findings"]["Cardiomegaly"]["state"],
            "present",
        )

    def test_weighted_loss_normalizes_by_weight_sum(self):
        logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]]
        )
        labels = torch.tensor([[-100, 1, 1]])
        weights = torch.tensor([[0.0, 1.0, 0.25]])
        value = token_cross_entropy(
            logits, labels, token_weights=weights
        )
        shifted = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 2),
            labels[:, 1:].reshape(-1),
            reduction="none",
        )
        expected = (shifted[0] + 0.25 * shifted[1]) / 1.25
        torch.testing.assert_close(value, expected)


if __name__ == "__main__":
    unittest.main()
