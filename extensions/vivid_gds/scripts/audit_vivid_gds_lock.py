"""Fail-closed audit for the VIVID-GDS Stage-A lock."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "audit" / "vivid_gds_stage_a_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "artifact": payload.get("artifact") == "vivid_gds_stage_a_lock",
        "method": payload["method"]["name"] == "VIVID-GDS",
        "teacher": payload["method"]["teacher"] == "Qwen3.5-2B",
        "projector": payload["method"]["projector"] == "historical_prefix4",
        "states": payload["method"]["states"]
        == ["present", "absent", "uncertain"],
        "missing_mask": payload["method"]["missing_policy"]
        == "supervision_mask",
        "lambda": payload["method"]["lambda_schema"] == 0.5,
        "ramp": payload["method"]["lambda_ramp_steps"] == 500,
        "budget": payload["budget"]["pilot_steps"] == 3000
        and payload["budget"]["effective_batch_size"] == 32,
        "protected_tests": {"CheXlocalize_test", "VinDr_test"}.issubset(
            payload["forbidden"]
        ),
        "teacher_scaling_forbidden": {
            "Qwen3.5-4B",
            "Qwen3.5-9B",
        }.issubset(payload["forbidden"]),
    }
    result = {
        "schema_version": 1,
        "artifact": "vivid_gds_lock_audit",
        "pass": all(checks.values()),
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
