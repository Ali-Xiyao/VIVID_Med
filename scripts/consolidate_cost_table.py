"""Build training/deployment cost tables from existing VIVID-Med artifacts.

The table is evidence-first: values are filled only when they can be recovered
from configs, logs, or local artifacts. Missing cost fields are recorded
explicitly instead of estimated from memory.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - local environment should have PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "final_tables"
LOG_DIR = ROOT / "outputs" / "logs"


@dataclass(frozen=True)
class CostSpec:
    method: str
    source_run: str | None
    source_config: str | None
    eval_run: str
    eval_config: str | None
    training_llm: str
    frozen_params: str
    trainable_params: str
    deployment_model: str
    deployment_params: str
    deployment_llm: str
    notes: str


METHODS = [
    CostSpec(
        method="Data-matched BCE ViT-B",
        source_run=None,
        source_config=None,
        eval_run="baseline_vit_full14",
        eval_config=None,
        training_llm="no",
        frozen_params="0",
        trainable_params="not_logged",
        deployment_model="ViT-B + classifier head",
        deployment_params="not_logged",
        deployment_llm="no",
        notes="baseline training config was not identified in P0 artifacts",
    ),
    CostSpec(
        method="Frozen-LM UMS / no-SPD",
        source_run="ablation_A_ums_12label",
        source_config="configs/ablation_A_ums_12label.yaml",
        eval_run="lp_A_ums_12label",
        eval_config="configs/lp_A_ums_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="LLM is training-only for source representation run",
    ),
    CostSpec(
        method="no-LM UMS state classifier",
        source_run="ums_classifier_no_llm_12label_full",
        source_config="configs/ums_classifier_no_llm_12label.yaml",
        eval_run="lp_ums_classifier_no_llm_12label_full",
        eval_config="configs/lp_ums_classifier_no_llm_12label.yaml",
        training_llm="no",
        frozen_params="0",
        trainable_params="not_logged; ViT-B + UMS state classifier trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged",
        deployment_llm="no",
        notes="UMS/schema supervision without LM",
    ),
    CostSpec(
        method="Frozen-LM UMS + SPD default",
        source_run="ablation_A_ums_spd_12label",
        source_config="configs/ablation_A_ums_spd_12label.yaml",
        eval_run="lp_A_ums_spd_12label",
        eval_config="configs/lp_A_ums_spd_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector/SPD components trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="historical SPD baseline only",
    ),
    CostSpec(
        method="Frozen-LM UMS + SPD G=2",
        source_run="ablation_spd_g2_12label",
        source_config="configs/ablation_spd_g2_12label.yaml",
        eval_run="lp_spd_g2_12label",
        eval_config="configs/lp_spd_g2_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector/SPD components trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="historical SPD sensitivity only; no new SPD variant",
    ),
    CostSpec(
        method="Frozen-LM free-text target",
        source_run="ablation_A_freetext_12label",
        source_config="configs/ablation_A_freetext_12label.yaml",
        eval_run="lp_A_freetext_12label",
        eval_config="configs/lp_A_freetext_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="free-text supervision baseline",
    ),
    CostSpec(
        method="Random-mask proxy",
        source_run="ablation_random_mask_12label",
        source_config="configs/ablation_random_mask_12label.yaml",
        eval_run="lp_random_mask_12label",
        eval_config="configs/lp_random_mask_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="random mask control",
    ),
    CostSpec(
        method="BiomedCLIP baseline",
        source_run=None,
        source_config=None,
        eval_run="lp_biomedclip_baseline_seed0",
        eval_config="configs/lp_biomedclip_baseline.yaml",
        training_llm="not trained in this repo",
        frozen_params="external pretraining not measured here",
        trainable_params="not_logged; linear probe/head training in this repo",
        deployment_model="BiomedCLIP-derived backbone + linear probe head",
        deployment_params="not_logged",
        deployment_llm="no",
        notes="external pretrained baseline; upstream training cost out of scope",
    ),
    CostSpec(
        method="Frozen-LM UMS + answerability mask",
        source_run="ablation_ums_ansmask_12label",
        source_config="configs/ablation_ums_ansmask_12label.yaml",
        eval_run="lp_ums_ansmask_12label",
        eval_config="configs/lp_ums_ansmask_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="answerability-mask source run",
    ),
    CostSpec(
        method="Frozen-LM UMS + null-as-negative",
        source_run="ablation_ums_null_as_negative_12label",
        source_config="configs/ablation_ums_null_as_negative_12label.yaml",
        eval_run="lp_ums_null_as_negative_12label",
        eval_config="configs/lp_ums_null_as_negative_12label.yaml",
        training_llm="yes; pretrained frozen Qwen2.5-1.5B",
        frozen_params="Qwen2.5-1.5B frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="dense null-negative source run",
    ),
    CostSpec(
        method="Random-LM same-architecture UMS",
        source_run="ablation_ums_random_lm_12label",
        source_config="configs/ablation_ums_random_lm_12label.yaml",
        eval_run="lp_ums_random_lm_12label",
        eval_config="configs/lp_ums_random_lm_12label.yaml",
        training_llm="yes; random frozen Qwen2.5-1.5B architecture",
        frozen_params="Qwen2.5-1.5B random frozen during representation training",
        trainable_params="not_logged; ViT-B + visual projector trainable",
        deployment_model="ViT-B backbone + linear probe head",
        deployment_params="not_logged; no LLM checkpoint required",
        deployment_llm="no",
        notes="same architecture random-LM control",
    ),
]


LOG_HINTS = {
    "ablation_spd_g2_12label": [
        "outputs/logs/ablation_spd_g2_12label.stderr.log",
        "outputs/logs/ablation_spd_g2_12label.stdout.log",
    ],
    "lp_spd_g2_12label": [
        "outputs/logs/lp_spd_g2_12label.stderr.log",
        "outputs/logs/lp_spd_g2_12label.stdout.log",
    ],
    "ums_classifier_no_llm_12label_full": [
        "outputs/logs/ums_classifier_no_llm_12label_full.log"
    ],
    "lp_ums_classifier_no_llm_12label_full": [
        "outputs/logs/lp_ums_classifier_no_llm_12label_full.log"
    ],
    "ablation_ums_ansmask_12label": [
        "outputs/logs/ablation_ums_ansmask_12label_resume_from_best_gpu1.log"
    ],
    "lp_ums_ansmask_12label": ["outputs/logs/lp_ums_ansmask_12label_train.log"],
    "ablation_ums_null_as_negative_12label": [
        "outputs/logs/ablation_ums_null_as_negative_12label_train.log"
    ],
    "lp_ums_null_as_negative_12label": [
        "outputs/logs/lp_ums_null_as_negative_12label_train_gpu0.log"
    ],
    "ablation_ums_random_lm_12label": [
        "outputs/logs/ablation_ums_random_lm_12label_train.log"
    ],
    "lp_ums_random_lm_12label": ["outputs/logs/lp_ums_random_lm_12label_train.log"],
}


TIME_RE = re.compile(r"\[(?P<elapsed>\d+(?::\d{2}){1,2})<")
RATE_RE = re.compile(r"(?P<rate>\d+(?:\.\d+)?\s*(?:it/s|s/it))")
PROGRESS_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_config(path_text: str | None) -> dict[str, Any] | None:
    if path_text is None or yaml is None:
        return None
    path = ROOT / path_text
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else None


def nested(config: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def checkpoint_path(run: str | None) -> Path | None:
    if run is None:
        return None
    run_dir = ROOT / "outputs" / run
    candidates = [
        run_dir / "best.pt",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "final.pt",
        run_dir / "checkpoints" / "final.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    steps = sorted(
        list(run_dir.glob("step_*.pt")) + list((run_dir / "checkpoints").glob("step_*.pt")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return steps[0] if steps else None


def checkpoint_size_mb(run: str | None) -> str:
    path = checkpoint_path(run)
    if path is None:
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.1f}"


def checkpoint_rel(run: str | None) -> str:
    path = checkpoint_path(run)
    return rel(path) if path is not None else ""


def last_training_line(text: str) -> str:
    training_lines = [line for line in text.splitlines() if "Training:" in line]
    if not training_lines:
        return text
    with_elapsed = [line for line in training_lines if TIME_RE.search(line)]
    return with_elapsed[-1] if with_elapsed else training_lines[-1]


def parse_elapsed_hours(text: str) -> float | None:
    matches = TIME_RE.findall(text)
    if not matches:
        return None
    elapsed = matches[-1]
    pieces = [int(piece) for piece in elapsed.split(":")]
    if len(pieces) == 2:
        minutes, seconds = pieces
        hours = 0
    else:
        hours, minutes, seconds = pieces
    return hours + minutes / 60 + seconds / 3600


def parse_last_progress(text: str) -> tuple[int | None, int | None]:
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None, None
    current, total = matches[-1]
    return int(current), int(total)


def parse_last_rate(text: str) -> str:
    matches = RATE_RE.findall(text)
    return matches[-1] if matches else ""


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.count(b"\x00") > max(10, len(raw) // 20):
        return raw.decode("utf-16-le", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_log(run: str | None) -> dict[str, str]:
    if run is None:
        return {}
    log_paths = [ROOT / path for path in LOG_HINTS.get(run, [])]
    existing = [path for path in log_paths if path.exists()]
    if not existing:
        return {}

    # Prefer the largest matching log for the run; stdout files often only hold
    # launcher text while stderr carries tqdm progress.
    path = max(existing, key=lambda item: item.stat().st_size)
    text = read_log_text(path)
    summary_line = last_training_line(text)
    elapsed_hours = parse_elapsed_hours(summary_line)
    current, total = parse_last_progress(summary_line)
    return {
        "log_path": rel(path),
        "observed_wall_clock_hours": fmt(elapsed_hours),
        "observed_progress": f"{current}/{total}" if current is not None else "",
        "observed_step_rate": parse_last_rate(summary_line),
    }


def effective_batch(config: dict[str, Any] | None) -> str:
    batch = nested(config, "training", "batch_size")
    accum = nested(config, "training", "gradient_accumulation_steps")
    if isinstance(batch, int) and isinstance(accum, int):
        return str(batch * accum)
    return ""


def config_summary(config: dict[str, Any] | None) -> dict[str, str]:
    return {
        "max_steps": fmt(nested(config, "training", "max_steps")),
        "batch_size": fmt(nested(config, "training", "batch_size")),
        "grad_accum": fmt(nested(config, "training", "gradient_accumulation_steps")),
        "effective_batch": effective_batch(config),
        "precision": "bf16" if nested(config, "training", "bf16") else "fp16" if nested(config, "training", "fp16") else "",
        "device": fmt(nested(config, "device")),
        "llm_model_name": fmt(nested(config, "model", "llm_model_name")),
        "output_dir": fmt(nested(config, "training", "output_dir")),
    }


def missing_rows(spec: CostSpec, row: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    checks = {
        "trainable_params": row["Trainable params"],
        "peak_gpu_memory": row["Peak GPU memory"],
        "gpu_hours": row["Observed source GPU-hours"],
        "throughput": row["Observed source step rate"],
        "deployment_params": row["Deployment params"],
        "source_wall_clock": row["Observed source wall-clock hours"],
    }
    for field, value in checks.items():
        if not value or "not_logged" in value or value.startswith("unknown"):
            missing.append(
                {
                    "method": spec.method,
                    "field": field,
                    "severity": "cost_provenance",
                    "reason": "not recoverable from current config/checkpoint/log artifacts",
                }
            )
    if spec.source_config and not (ROOT / spec.source_config).exists():
        missing.append(
            {
                "method": spec.method,
                "field": "source_config",
                "severity": "artifact",
                "reason": f"missing {spec.source_config}",
            }
        )
    if spec.source_run and not checkpoint_path(spec.source_run):
        missing.append(
            {
                "method": spec.method,
                "field": "source_checkpoint",
                "severity": "artifact",
                "reason": f"no checkpoint found for outputs/{spec.source_run}",
            }
        )
    return missing


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [row.get(column, "").replace("\n", " ") for column in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for spec in METHODS:
        source_config = read_config(spec.source_config)
        eval_config = read_config(spec.eval_config)
        source_summary = config_summary(source_config)
        eval_summary = config_summary(eval_config)
        source_log = parse_log(spec.source_run)
        eval_log = parse_log(spec.eval_run)

        observed_hours = source_log.get("observed_wall_clock_hours", "")
        observed_gpu_hours = observed_hours
        if observed_gpu_hours:
            observed_gpu_hours = f"{observed_gpu_hours} (observed segment; one configured CUDA device)"
        else:
            observed_gpu_hours = "unknown"

        row = {
            "Method": spec.method,
            "Source training run": spec.source_run or "",
            "Source config": spec.source_config or "",
            "Source checkpoint": checkpoint_rel(spec.source_run),
            "Source checkpoint MB": checkpoint_size_mb(spec.source_run),
            "Training LLM?": spec.training_llm,
            "Frozen params during train": spec.frozen_params,
            "Trainable params": spec.trainable_params,
            "Source max steps": source_summary["max_steps"],
            "Source batch": source_summary["batch_size"],
            "Source grad accum": source_summary["grad_accum"],
            "Source effective batch": source_summary["effective_batch"],
            "Source precision": source_summary["precision"],
            "Source device": source_summary["device"],
            "Observed source wall-clock hours": observed_hours,
            "Observed source progress": source_log.get("observed_progress", ""),
            "Observed source step rate": source_log.get("observed_step_rate", ""),
            "Observed source log": source_log.get("log_path", ""),
            "Peak GPU memory": "not_logged",
            "Observed source GPU-hours": observed_gpu_hours,
            "Evaluation/LP run": spec.eval_run,
            "Evaluation config": spec.eval_config or "",
            "Evaluation checkpoint": checkpoint_rel(spec.eval_run),
            "Evaluation checkpoint MB": checkpoint_size_mb(spec.eval_run),
            "Evaluation max steps": eval_summary["max_steps"],
            "Evaluation effective batch": eval_summary["effective_batch"],
            "Observed eval wall-clock hours": eval_log.get("observed_wall_clock_hours", ""),
            "Observed eval step rate": eval_log.get("observed_step_rate", ""),
            "Observed eval log": eval_log.get("log_path", ""),
            "Deployment model": spec.deployment_model,
            "Deployment params": spec.deployment_params,
            "Deployment LLM?": spec.deployment_llm,
            "Notes": spec.notes,
        }
        rows.append(row)
        missing.extend(missing_rows(spec, row))
    return rows, missing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = build_rows()
    columns = [
        "Method",
        "Source training run",
        "Source config",
        "Source checkpoint",
        "Source checkpoint MB",
        "Training LLM?",
        "Frozen params during train",
        "Trainable params",
        "Source max steps",
        "Source batch",
        "Source grad accum",
        "Source effective batch",
        "Source precision",
        "Source device",
        "Observed source wall-clock hours",
        "Observed source progress",
        "Observed source step rate",
        "Observed source log",
        "Peak GPU memory",
        "Observed source GPU-hours",
        "Evaluation/LP run",
        "Evaluation config",
        "Evaluation checkpoint",
        "Evaluation checkpoint MB",
        "Evaluation max steps",
        "Evaluation effective batch",
        "Observed eval wall-clock hours",
        "Observed eval step rate",
        "Observed eval log",
        "Deployment model",
        "Deployment params",
        "Deployment LLM?",
        "Notes",
    ]
    write_csv(OUTPUT_DIR / "cost_table.csv", rows, columns)
    (OUTPUT_DIR / "cost_table.md").write_text(
        "# Cost Table\n\n" + markdown_table(rows, columns),
        encoding="utf-8",
    )

    missing_columns = ["method", "field", "severity", "reason"]
    write_csv(OUTPUT_DIR / "cost_missing_artifacts.csv", missing, missing_columns)
    (OUTPUT_DIR / "cost_missing_artifacts.md").write_text(
        "# Cost Missing Artifacts\n\n" + markdown_table(missing, missing_columns),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} cost rows to {rel(OUTPUT_DIR)}")
    print(f"Recorded {len(missing)} missing cost/provenance fields")


if __name__ == "__main__":
    main()
