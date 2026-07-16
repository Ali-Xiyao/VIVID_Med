"""CPU-only synthetic smoke test for the BiVES-CXR core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr import BiVESCXR, BiVESLoss, BiVESModelConfig


def main() -> None:
    torch.manual_seed(17)
    model = BiVESCXR(
        BiVESModelConfig(
            visual_dim=8,
            statement_dim=6,
            fusion_dim=16,
            gate_mode="sigmoid",
            topk=2,
        )
    )
    patches = torch.randn(4, 4, 8)
    statements = torch.randn(4, 6)
    valid = torch.ones(4, 4, dtype=torch.bool)
    targets = torch.tensor([0, 1, 2, 3])
    outputs = model(patches, statements, valid, run_interventions=True)
    losses = BiVESLoss()(outputs, targets, grid_hw=(2, 2))
    losses["total"].backward()
    probabilities = outputs["original"]["state_probs"]
    payload = {
        "decoder_kind": model.decoder_kind,
        "has_flat_state_head": model.has_flat_state_head,
        "shape": list(probabilities.shape),
        "probability_sums": probabilities.sum(dim=-1).detach().tolist(),
        "loss": float(losses["total"].detach()),
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
    }
    print(json.dumps(payload, indent=2))
    if payload["shape"] != [4, 4] or not payload["finite_gradients"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
