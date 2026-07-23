"""Fail-closed audit of the strict VIVID/SPD JSON lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "audit"
        / "vivid_spd_clean_lock.json",
    )
    args = parser.parse_args()
    payload = json.loads(args.lock.read_text(encoding="utf-8"))
    checks = {
        "route": payload["route"] == "strict_vivid_spd_clean_extension",
        "teacher": payload["teacher"]["primary_size"] == "2B",
        "hard_ums": payload["target"]["name"] == "deterministic_hard_ums",
        "no_reliability": payload["target"]["reliability_weighting"] is False,
        "prefix4": payload["arms"]["ums_prefix4"]["prefix_tokens"] == 4,
        "spd4x2": (
            payload["arms"]["ums_spd4x2"]["groups"] == 4
            and payload["arms"]["ums_spd4x2"]["tokens_per_group"] == 2
        ),
        "ortho": (
            payload["arms"]["ums_spd4x2"]["orthogonality_weight"] == 0.02
        ),
        "protected": set(payload["protected_surfaces"])
        == {"CheXlocalize_test", "VinDr_test"},
    }
    result = {"pass": all(checks.values()), "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
