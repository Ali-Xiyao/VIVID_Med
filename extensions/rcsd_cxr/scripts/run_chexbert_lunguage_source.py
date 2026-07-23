"""Run the frozen CheXbert checkpoint on LUNGUAGE findings/impression text.

The script is intentionally source-only: it produces report labels for the
independent G2 gold surface and never reads images or downstream test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer


CONDITIONS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
)
RCSD_FINDINGS = CONDITIONS[:10] + CONDITIONS[11:13]
RAW_TO_VALUE = {0: "", 1: "1", 2: "0", 3: "-1"}


class CheXbertLabeler(nn.Module):
    """Architecture matching the published 14-head CheXbert checkpoint."""

    def __init__(self, config: BertConfig, dropout: float = 0.1) -> None:
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.linear_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, 4) for _ in range(13)]
            + [nn.Linear(config.hidden_size, 2)]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        hidden = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
        hidden = self.dropout(hidden)
        return [head(hidden) for head in self.linear_heads]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_identifier(value: object, prefix: str) -> str:
    return str(value or "").strip().removeprefix(prefix)


def build_reports(path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Freeze one report per study from findings then impression sections."""
    by_study: dict[str, dict[str, object]] = {}
    section_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "study_id", "section", "section_report"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"LUNGUAGE missing report columns: {sorted(missing)}")
        for row in reader:
            section = str(row.get("section") or "").strip().lower()
            if section not in {"find", "impr"}:
                continue
            study_id = normalize_identifier(row.get("study_id"), "s")
            subject_id = normalize_identifier(row.get("subject_id"), "p")
            text = str(row.get("section_report") or "").strip()
            if not study_id or not subject_id or not text:
                continue
            record = by_study.setdefault(
                study_id, {"study_id": study_id, "subject_id": subject_id}
            )
            if record["subject_id"] != subject_id:
                raise ValueError(f"study {study_id} maps to multiple patients")
            section_values[(study_id, section)].add(text)

    conflicts = {
        f"{study}:{section}": len(values)
        for (study, section), values in section_values.items()
        if len(values) != 1
    }
    if conflicts:
        preview = dict(list(conflicts.items())[:5])
        raise ValueError(f"inconsistent section_report values: {preview}")

    reports: list[dict[str, str]] = []
    pattern_counts: Counter[str] = Counter()
    for study_id in sorted(by_study, key=lambda value: int(value)):
        sections = []
        present_sections = []
        for section in ("find", "impr"):
            values = section_values.get((study_id, section), set())
            if values:
                sections.append(next(iter(values)))
                present_sections.append(section)
        if not sections:
            continue
        reports.append(
            {
                "study_id": study_id,
                "subject_id": str(by_study[study_id]["subject_id"]),
                "report": "\n".join(sections),
            }
        )
        pattern_counts["+".join(present_sections)] += 1
    return reports, {
        "reports": len(reports),
        "patients": len({row["subject_id"] for row in reports}),
        "section_patterns": dict(pattern_counts),
        "history_excluded": True,
        "section_order": ["find", "impr"],
        "section_conflicts": 0,
    }


def load_model(
    checkpoint_path: Path, auxiliary_dir: Path, device: torch.device
) -> tuple[CheXbertLabeler, BertTokenizer, dict[str, object]]:
    config_path = auxiliary_dir / "config.json"
    vocab_path = auxiliary_dir / "vocab.txt"
    if not config_path.is_file() or not vocab_path.is_file():
        raise FileNotFoundError(
            "CheXbert auxiliary directory requires config.json and vocab.txt"
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = BertConfig.from_json_file(str(config_path))
    tokenizer = BertTokenizer.from_pretrained(
        str(auxiliary_dir), local_files_only=True
    )
    model = CheXbertLabeler(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.eval().to(device)
    return model, tokenizer, {
        "checkpoint_keys": len(state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }


def run_inference(
    reports: list[dict[str, str]],
    model: CheXbertLabeler,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    output: list[dict[str, str]] = []
    state_counts = {condition: Counter() for condition in RCSD_FINDINGS}
    with torch.inference_mode():
        for start in range(0, len(reports), batch_size):
            batch = reports[start : start + batch_size]
            encoded = tokenizer(
                [row["report"] for row in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            heads = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
            raw = [head.argmax(dim=1).cpu().tolist() for head in heads]
            for offset, report in enumerate(batch):
                row = {
                    "subject_id": report["subject_id"],
                    "study_id": report["study_id"],
                }
                for index, condition in enumerate(CONDITIONS[:13]):
                    if condition == "Pleural Other":
                        continue
                    value = RAW_TO_VALUE[int(raw[index][offset])]
                    row[condition] = value
                    state_counts[condition][value or "missing"] += 1
                output.append(row)
    return output, {
        "rows": len(output),
        "batch_size": batch_size,
        "batches": (len(output) + batch_size - 1) // batch_size,
        "state_counts": {
            condition: dict(counts) for condition, counts in state_counts.items()
        },
    }


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subject_id", "study_id", *RCSD_FINDINGS]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lunguage", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--auxiliary-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda:0"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    reports, report_audit = build_reports(args.lunguage)
    model, tokenizer, model_audit = load_model(
        args.checkpoint, args.auxiliary_dir, device
    )
    rows, inference_audit = run_inference(
        reports, model, tokenizer, device, args.batch_size
    )
    write_source(args.output, rows)
    audit = {
        "schema_version": 1,
        "artifact": "lunguage_chexbert_source",
        "pass": True,
        "text_contract": {
            "included_sections": ["find", "impr"],
            "excluded_sections": ["hist"],
            "separator": "\\n",
            "max_tokens": 512,
        },
        "report_audit": report_audit,
        "model_audit": model_audit,
        "inference_audit": inference_audit,
        "hashes": {
            "lunguage": sha256_file(args.lunguage),
            "checkpoint": sha256_file(args.checkpoint),
            "config": sha256_file(args.auxiliary_dir / "config.json"),
            "vocab": sha256_file(args.auxiliary_dir / "vocab.txt"),
            "output": sha256_file(args.output),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
