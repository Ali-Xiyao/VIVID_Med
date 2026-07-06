"""Build token/row weighting config snippets for next-stage instruction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "tw-role": {
        "loss_weighting": {
            "enabled": True,
            "base_weight": 1.0,
            "answer_type_weights": {
                "counterfactual_choice": 1.5,
                "same_question_different_answer": 1.5,
                "laterality_location": 1.8,
                "finding_verification": 1.1,
                "uncertainty": 1.2,
                "answerability": 1.1,
            },
            "visual_dependency_weights": {"very_high": 1.5, "high": 1.25, "medium": 1.0, "low": 0.9},
            "quality_flag_weights": {"hard_image_shuffle": 1.5, "standardized_ab": 1.3},
        }
    },
    "tw-visual": {
        "loss_weighting": {
            "enabled": True,
            "base_weight": 1.0,
            "answer_type_weights": {
                "same_question_different_answer": 2.0,
                "counterfactual_choice": 1.8,
                "laterality_location": 2.0,
                "finding_verification": 1.2,
                "uncertainty": 1.5,
                "answerability": 1.2,
            },
            "visual_dependency_weights": {"very_high": 2.0, "high": 1.5, "medium": 1.0, "low": 0.7},
            "quality_flag_weights": {"hard_image_shuffle": 2.0, "multi_negative_k4": 1.2, "same_question": 1.3},
        }
    },
    "tw-clinical": {
        "loss_weighting": {
            "enabled": True,
            "base_weight": 1.0,
            "answer_type_weights": {
                "counterfactual_choice": 1.5,
                "same_question_different_answer": 1.6,
                "laterality_location": 1.8,
                "evidence_phrase": 1.3,
                "uncertainty": 1.4,
                "answerability": 1.2,
            },
            "visual_dependency_weights": {"very_high": 1.7, "high": 1.35, "medium": 1.0, "low": 0.85},
            "quality_flag_weights": {"hard_image_shuffle": 1.5, "rare_finding": 1.3},
        }
    },
    "p2-value-only": {
        "loss_masking": {"mode": "json_value_only", "values": ["present", "absent", "uncertain", "null", "true", "false"]}
    },
    "p2-no-punct": {
        "loss_masking": {"mode": "json_no_punct"}
    },
}


def to_simple_yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item) if isinstance(item, str) else item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {json.dumps(item) if isinstance(item, str) else item}")
        return lines
    return [f"{prefix}{value}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-yaml", type=Path)
    parser.add_argument("--wrap-training", action="store_true", help="Wrap snippet under top-level training key.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = dict(PRESETS[args.preset])
    if args.wrap_training:
        payload = {"training": payload}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_yaml:
        args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
        args.output_yaml.write_text("\n".join(to_simple_yaml(payload)) + "\n", encoding="utf-8")
    print(json.dumps({"preset": args.preset, "keys": list(payload.keys())}, indent=2))


if __name__ == "__main__":
    main()
