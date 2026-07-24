"""CPU-only contract smoke for VIVID-GDS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_gds.contracts import parse_ums_target, render_free_text  # noqa: E402
from vivid_gds.model import UMSSchemaHead  # noqa: E402
from vivid_gds.objective import (  # noqa: E402
    masked_schema_cross_entropy,
    schema_accuracy,
)


def main() -> int:
    target = json.dumps(
        {
            "modality": "CXR",
            "findings": {
                "Cardiomegaly": {"state": "present", "score": None},
                "Edema": {"state": "uncertain", "score": None},
            },
            "study_view": None,
        }
    )
    states, mask = parse_ums_target(target)
    text = render_free_text(target, "row-1")
    head = UMSSchemaHead(dropout=0.0)
    logits = head(torch.randn(2, 768))
    labels = torch.tensor([states, states])
    loss = masked_schema_cross_entropy(logits, labels)
    correct, observed = schema_accuracy(logits, labels)
    checks = {
        "selected_fields": sum(mask) == 2,
        "missing_is_mask": states[0] == -100,
        "deterministic_free_text": text == render_free_text(target, "row-1"),
        "schema_shape": list(logits.shape) == [2, 12, 3],
        "finite_loss": bool(torch.isfinite(loss)),
        "observed_fields": observed == 4,
        "accuracy_bounded": 0 <= correct <= observed,
    }
    result = {"pass": all(checks.values()), "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
