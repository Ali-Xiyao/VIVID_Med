"""Fail-closed import check for the frozen Qwen3.5 fast path."""

from __future__ import annotations

import json

import causal_conv1d
import fla
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_5 import modeling_qwen3_5


def main() -> int:
    fast_path = modeling_qwen3_5.is_fast_path_available
    if callable(fast_path):
        fast_path = fast_path()
    checks = {
        "causal_conv1d_fn_callable": callable(causal_conv1d_fn),
        "causal_conv1d_update_callable": callable(causal_conv1d_update),
        "chunk_gated_delta_rule_callable": callable(
            chunk_gated_delta_rule
        ),
        "fused_recurrent_gated_delta_rule_callable": callable(
            fused_recurrent_gated_delta_rule
        ),
        "qwen35_fast_path_available": bool(fast_path),
    }
    result = {
        "artifact": "qwen35_acceleration_preflight",
        "pass": all(checks.values()),
        "checks": checks,
        "versions": {
            "causal_conv1d": getattr(
                causal_conv1d, "__version__", "unknown"
            ),
            "flash_linear_attention": getattr(
                fla, "__version__", "unknown"
            ),
        },
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
