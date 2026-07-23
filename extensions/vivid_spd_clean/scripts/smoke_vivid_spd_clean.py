"""CPU smoke for the exact projector identities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.model import (  # noqa: E402
    HistoricalPrefixProjector,
    HistoricalSPDProjector,
)


def main() -> int:
    torch.manual_seed(0)
    tokens = torch.randn(2, 197, 768)
    prefix = HistoricalPrefixProjector()(tokens)
    spd_model = HistoricalSPDProjector()
    spd = spd_model(tokens)
    result = {
        "prefix_shape": list(prefix.shape),
        "spd_shape": list(spd.shape),
        "spd_orthogonality_finite": bool(
            torch.isfinite(spd_model.orthogonality_loss())
        ),
    }
    expected = {
        "prefix_shape": [2, 201, 1536],
        "spd_shape": [2, 205, 1536],
        "spd_orthogonality_finite": True,
    }
    if result != expected:
        raise AssertionError({"actual": result, "expected": expected})
    print(json.dumps({"pass": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
