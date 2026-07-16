"""Shared implementation for CVCP/CCSH target-plan entry points."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "instruction_data" / "cvcp_ccsh"
FINAL_DIR = ROOT / "outputs" / "final_tables"


def root_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_rows(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[i] for i in indices[:n]]


def clone_row(row: dict[str, Any], run_id: str, index: int, extra_flags: list[str] | None = None) -> dict[str, Any]:
    out = dict(row)
    flags = list(out.get("quality_flags") or [])
    for flag in extra_flags or []:
        if flag not in flags:
            flags.append(flag)
    out["quality_flags"] = flags
    out["source_version"] = run_id
    out["cvcp_run_id"] = run_id
    base_id = out.get("instruction_id") or out.get("sample_id") or f"row_{index}"
    out["instruction_id"] = f"{run_id}_{index:06d}_{base_id}"
    return out


def pool_paths() -> dict[str, Path]:
    base = ROOT / "outputs" / "instruction_data" / "next_stage"
    return {
        "basic": base / "storymix_10k_train.jsonl",
        "basic_val": base / "storymix_qa8_val.jsonl",
        "cf": base / "cf_10k_train.jsonl",
        "cf_val": base / "cf_10k_val.jsonl",
        "sameq": base / "sameq_shuf_3k_train.jsonl",
        "sameq_val": base / "sameq_shuf_val.jsonl",
        "shuf_k2": base / "shuf_k2_train.jsonl",
        "shuf_k2_val": base / "shuf_k2_val.jsonl",
        "shuf_k4": base / "shuf_k4_train.jsonl",
        "shuf_k4_val": base / "shuf_k4_val.jsonl",
        "progressive": base / "cur_v2_progressive_replay_train.jsonl",
    }


def materialize_dataset(run_id: str, source_rows: list[dict[str, Any]], n: int | None, seed: int, flags: list[str]) -> dict[str, Any]:
    selected = sample_rows(source_rows, n, seed)
    rows = [clone_row(row, run_id, idx, flags) for idx, row in enumerate(selected)]
    out = DATA_DIR / f"{run_id}_train.jsonl"
    write_jsonl(out, rows)
    return {
        "run_id": run_id,
        "output": repo_rel(out),
        "rows": len(rows),
        "requested_rows": n or "all",
        "status": "materialized" if rows else "missing_source",
    }


def mix_dataset(run_id: str, sources: list[tuple[str, list[dict[str, Any]], float]], total: int, seed: int, flags: list[str]) -> dict[str, Any]:
    rng = random.Random(seed)
    out_rows: list[dict[str, Any]] = []
    for name, rows, frac in sources:
        take = min(len(rows), max(0, int(round(total * frac))))
        picked = sample_rows(rows, take, rng.randint(0, 10_000_000))
        out_rows.extend(clone_row(row, run_id, len(out_rows), flags + [f"mix_source_{name}"]) for row in picked)
    rng.shuffle(out_rows)
    path = DATA_DIR / f"{run_id}_train.jsonl"
    write_jsonl(path, out_rows)
    return {
        "run_id": run_id,
        "output": repo_rel(path),
        "rows": len(out_rows),
        "requested_rows": total,
        "status": "materialized" if out_rows else "missing_source",
    }


def cmd_generate_cvcp_curriculum(args: argparse.Namespace) -> None:
    paths = pool_paths()
    pools = {name: read_jsonl(path) for name, path in paths.items()}
    rows: list[dict[str, Any]] = []
    rows.append(materialize_dataset("cvcp_v1_sameq_3k", pools["sameq"], 3000, args.seed, ["cvcp_v1", "sameq"]))
    rows.append(materialize_dataset("cvcp_v1_sameq_10k", pools["sameq"], 10000, args.seed + 1, ["cvcp_v1", "sameq"]))
    rows.append(materialize_dataset("cvcp_v1_sameq_full", pools["sameq"], None, args.seed + 2, ["cvcp_v1", "sameq", "available_full"]))
    rows.append(materialize_dataset("cvcp_v2_shuf_k2", pools["shuf_k2"], 5000, args.seed + 3, ["cvcp_v2", "shuf_k2"]))
    rows.append(materialize_dataset("cvcp_v2_shuf_k4", pools["shuf_k4"], 5000, args.seed + 4, ["cvcp_v2", "shuf_k4"]))
    rows.append(materialize_dataset("cvcp_v2_shuf_k8", pools["shuf_k4"], 8000, args.seed + 5, ["cvcp_v2", "shuf_k8_boundary_from_k4_pool"]))
    rows.append(
        mix_dataset(
            "cvcp_v3_prog_3k",
            [("basic", pools["basic"], 0.25), ("cf", pools["cf"], 0.30), ("sameq", pools["sameq"], 0.30), ("shuf", pools["shuf_k4"], 0.15)],
            3000,
            args.seed + 6,
            ["cvcp_v3", "progressive"],
        )
    )
    rows.append(
        mix_dataset(
            "cvcp_v3_prog_10k",
            [("basic", pools["basic"], 0.20), ("cf", pools["cf"], 0.30), ("sameq", pools["sameq"], 0.30), ("shuf", pools["shuf_k4"], 0.20)],
            10000,
            args.seed + 7,
            ["cvcp_v3", "progressive"],
        )
    )
    rows.append(materialize_dataset("cvcp_v3_prog_full", pools["progressive"], None, args.seed + 8, ["cvcp_v3", "available_full"]))
    rows.append(materialize_dataset("cvcp_v4_replay_10k", pools["progressive"], 10000, args.seed + 9, ["cvcp_v4", "replay"]))
    rows.append(materialize_dataset("cvcp_v4_replay_full", pools["progressive"], None, args.seed + 10, ["cvcp_v4", "replay", "available_full"]))
    rows.append(
        mix_dataset(
            "cvcp_v5_cdcs_field",
            [("basic", pools["basic"], 0.15), ("cf", pools["cf"], 0.30), ("sameq", pools["sameq"], 0.35), ("shuf", pools["shuf_k4"], 0.20)],
            10000,
            args.seed + 11,
            ["cvcp_v5", "cdcs_field"],
        )
    )
    rows.append(
        mix_dataset(
            "cvcp_v5_cdcs_hardneg",
            [("cf", pools["cf"], 0.20), ("sameq", pools["sameq"], 0.35), ("shuf", pools["shuf_k4"], 0.45)],
            10000,
            args.seed + 12,
            ["cvcp_v5", "cdcs_hardneg"],
        )
    )
    rows.append(
        mix_dataset(
            "cvcp_v5_cdcs_full",
            [("basic", pools["basic"], 0.10), ("cf", pools["cf"], 0.25), ("sameq", pools["sameq"], 0.35), ("shuf", pools["shuf_k4"], 0.30)],
            args.full_rows,
            args.seed + 13,
            ["cvcp_v5", "cdcs_full"],
        )
    )
    columns = ["run_id", "output", "rows", "requested_rows", "status"]
    write_csv(FINAL_DIR / "cvcp_dataset_manifest.csv", rows, columns)
    write_md_table(FINAL_DIR / "cvcp_dataset_manifest.md", "CVCP Dataset Manifest", rows, columns)
    print(f"wrote {len(rows)} CVCP dataset rows")


def cmd_generate_sameq_cf(args: argparse.Namespace) -> None:
    paths = pool_paths()
    sameq = read_jsonl(paths["sameq"])
    cf = read_jsonl(paths["cf"])
    rows = []
    for pct in args.percent:
        run_id = f"sameq_cf_{pct}"
        rows.append(
            mix_dataset(
                run_id,
                [("sameq", sameq, (100 - pct) / 100.0), ("cf", cf, pct / 100.0)],
                args.rows,
                args.seed + pct,
                ["sameq_cf_compatible", f"cf_percent_{pct}"],
            )
        )
    columns = ["run_id", "output", "rows", "requested_rows", "status"]
    write_csv(FINAL_DIR / "sameq_cf_compatible_manifest.csv", rows, columns)
    write_md_table(FINAL_DIR / "sameq_cf_compatible_manifest.md", "SAMEQ CF-Compatible Manifest", rows, columns)
    print(f"wrote {len(rows)} SAMEQ-CF datasets")


def cmd_generate_shuf_k_cf(args: argparse.Namespace) -> None:
    paths = pool_paths()
    shuf = read_jsonl(paths["shuf_k4"])
    cf = read_jsonl(paths["cf"])
    specs = [("k4_cf_20", 20, "none"), ("k4_cf_20_tw", 20, "tw_visual"), ("k4_cf_30_tw", 30, "tw_visual")]
    rows = []
    for run_id, pct, tw in specs:
        rows.append(
            mix_dataset(
                run_id,
                [("shuf_k4", shuf, (100 - pct) / 100.0), ("cf", cf, pct / 100.0)],
                args.rows,
                args.seed + pct + len(rows),
                ["shuf_k_cf_compatible", f"cf_percent_{pct}", tw],
            )
        )
    columns = ["run_id", "output", "rows", "requested_rows", "status"]
    write_csv(FINAL_DIR / "shuf_k_cf_compatible_manifest.csv", rows, columns)
    write_md_table(FINAL_DIR / "shuf_k_cf_compatible_manifest.md", "SHUF-K CF-Compatible Manifest", rows, columns)
    print(f"wrote {len(rows)} SHUF-K-CF datasets")


def statement_text(row: dict[str, Any], positive: bool) -> str:
    if positive:
        option = row.get("option_" + str(row.get("answer_short", "")).lower())
        if option:
            return str(option)
    if not positive:
        neg = row.get("negative_option")
        if neg:
            option = row.get("option_" + str(neg).lower())
            if option:
                return str(option)
    finding = str(row.get("finding") or "finding").lower()
    state = str(row.get("state") or "")
    if positive:
        if state == "absent":
            return f"There is no {finding}."
        if state == "uncertain":
            return f"The presence of {finding} is uncertain."
        return f"There is {finding}."
    return f"The statement about {finding} is not supported by this image."


def cmd_generate_ccsh_statements(args: argparse.Namespace) -> None:
    source = read_jsonl(root_path(args.input), args.max_rows)
    out_rows = []
    for row in source:
        for label, positive in [("support", True), ("contradict", False)]:
            out = {
                "statement_id": f"{row.get('instruction_id', row.get('sample_id', 'row'))}_{label}",
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "finding": row.get("finding", ""),
                "state": row.get("state", ""),
                "statement": statement_text(row, positive),
                "label": label,
                "binary_label": 1 if positive else 0,
                "source_instruction_id": row.get("instruction_id", ""),
                "source_version": "cvcp_ccsh_statements",
            }
            out_rows.append(out)
    write_jsonl(root_path(args.output), out_rows)
    columns = ["dataset", "rows", "support", "contradict", "output"]
    summary = [
        {
            "dataset": "ccsh_statements",
            "rows": len(out_rows),
            "support": sum(1 for row in out_rows if row["label"] == "support"),
            "contradict": sum(1 for row in out_rows if row["label"] == "contradict"),
            "output": args.output,
        }
    ]
    write_csv(FINAL_DIR / "ccsh_statement_manifest.csv", summary, columns)
    write_md_table(FINAL_DIR / "ccsh_statement_manifest.md", "CCSH Statement Manifest", summary, columns)
    print(f"wrote_rows={len(out_rows)}")


REGION_HINTS = {
    "Pleural Effusion": "costophrenic angle / pleural space",
    "Cardiomegaly": "cardiac silhouette",
    "Pneumothorax": "lung apex / pleural line",
    "Edema": "bilateral perihilar lungs",
    "Atelectasis": "basal or linear opacity region",
    "Consolidation": "focal air-space opacity",
    "Lung Opacity": "lung fields",
    "Fracture": "ribs / osseous structures",
    "Lung Lesion": "focal lung lesion region",
}


def cmd_generate_ceq_targets(args: argparse.Namespace) -> None:
    source = read_jsonl(root_path(args.input), args.max_rows)
    out_rows = []
    for row in source:
        finding = str(row.get("finding") or "global")
        out_rows.append(
            {
                "target_id": f"{row.get('instruction_id', row.get('sample_id', 'row'))}_ceq",
                "sample_id": row.get("sample_id", ""),
                "image_path": row.get("image_path", ""),
                "finding": finding,
                "state": row.get("state", ""),
                "evidence_query": f"Find visual evidence for {finding}.",
                "expected_region": REGION_HINTS.get(finding, "clinically relevant image region"),
                "source_instruction_id": row.get("instruction_id", ""),
                "source_version": "cvcp_ceq_targets",
            }
        )
    write_jsonl(root_path(args.output), out_rows)
    counts = Counter(row["finding"] for row in out_rows)
    rows = [{"finding": key, "rows": value} for key, value in sorted(counts.items())]
    write_csv(FINAL_DIR / "ceq_target_manifest.csv", rows, ["finding", "rows"])
    write_md_table(FINAL_DIR / "ceq_target_manifest.md", "CEQ Target Manifest", rows, ["finding", "rows"])
    print(f"wrote_rows={len(out_rows)}")


def leakage_for_row(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    question = str(row.get("question") or "").lower()
    answer = str(row.get("answer") or row.get("answer_short") or "").lower()
    evidence = str(row.get("evidence_span") or "").strip().lower()
    if evidence and evidence in question:
        flags.append("question_contains_evidence_span")
    if answer and len(answer) > 1 and answer in question:
        flags.append("question_contains_answer")
    if "report says" in question or "according to the report" in question:
        flags.append("question_mentions_report")
    if row.get("option_a") and row.get("option_b"):
        a_len = len(str(row["option_a"]))
        b_len = len(str(row["option_b"]))
        if max(a_len, b_len) / max(1, min(a_len, b_len)) > 1.8:
            flags.append("ab_option_length_imbalance")
    if row.get("hard_negative_expected_answer") == row.get("answer_short"):
        flags.append("hard_negative_same_answer")
    return flags


def cmd_audit_leakage(args: argparse.Namespace) -> None:
    files = [root_path(path) for path in args.input]
    rows = []
    details = []
    for path in files:
        data = read_jsonl(path, args.max_rows)
        flag_counts: Counter[str] = Counter()
        for idx, row in enumerate(data):
            flags = leakage_for_row(row)
            flag_counts.update(flags)
            if flags:
                details.append(
                    {
                        "dataset": repo_rel(path),
                        "row_index": idx,
                        "instruction_id": row.get("instruction_id", ""),
                        "flags": ";".join(flags),
                    }
                )
        rows.append(
            {
                "dataset": repo_rel(path),
                "rows": len(data),
                "flagged_rows": sum(1 for item in details if item["dataset"] == repo_rel(path)),
                "flags": "; ".join(f"{k}={v}" for k, v in sorted(flag_counts.items())),
                "decision": "review" if flag_counts else "pass",
            }
        )
    write_csv(root_path(args.output_csv), rows, ["dataset", "rows", "flagged_rows", "flags", "decision"])
    write_csv(root_path(args.detail_csv), details, ["dataset", "row_index", "instruction_id", "flags"])
    write_md_table(root_path(args.output_md), "Instruction Leakage v3 Audit", rows, ["dataset", "rows", "flagged_rows", "flags", "decision"])
    print(f"audited_files={len(files)}")


def cmd_audit_false_hard_negatives(args: argparse.Namespace) -> None:
    files = [root_path(path) for path in args.input]
    rows = []
    details = []
    for path in files:
        data = read_jsonl(path, args.max_rows)
        counts = Counter()
        for idx, row in enumerate(data):
            flags = []
            if row.get("hard_negative_sample_id") == row.get("sample_id"):
                flags.append("same_sample")
            if row.get("hard_negative_expected_answer") == row.get("answer_short"):
                flags.append("same_answer")
            reasons = row.get("hard_negative_reasons") or []
            if isinstance(reasons, list) and any("opposite" not in str(reason) for reason in reasons):
                flags.append("weak_reason")
            if flags:
                counts.update(flags)
                details.append(
                    {
                        "dataset": repo_rel(path),
                        "row_index": idx,
                        "instruction_id": row.get("instruction_id", ""),
                        "flags": ";".join(flags),
                    }
                )
        rows.append(
            {
                "dataset": repo_rel(path),
                "rows": len(data),
                "flagged_rows": sum(1 for item in details if item["dataset"] == repo_rel(path)),
                "flags": "; ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "decision": "review" if counts else "pass",
            }
        )
    write_csv(root_path(args.output_csv), rows, ["dataset", "rows", "flagged_rows", "flags", "decision"])
    write_csv(root_path(args.detail_csv), details, ["dataset", "row_index", "instruction_id", "flags"])
    write_md_table(root_path(args.output_md), "False Hard Negative Audit", rows, ["dataset", "rows", "flagged_rows", "flags", "decision"])
    print(f"audited_files={len(files)}")


def cmd_train_qwen3vl(args: argparse.Namespace) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "train_qwen3vl_clinical_instruction.py"), "--config", str(root_path(args.config))]
    if args.resume:
        command.extend(["--resume", str(root_path(args.resume))])
    if args.debug:
        command.append("--debug")
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    print(" ".join(command))
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=ROOT))


def cmd_train_module_stack(args: argparse.Namespace) -> None:
    modules = args.module or args.default_modules
    for module in modules:
        output_dir = root_path(args.output_root) / module.lower()
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_case_study_module_ablation.py"),
            "--module",
            module,
            "--train-embeddings",
            str(root_path(args.train_embeddings)),
            "--train-metadata",
            str(root_path(args.train_metadata)),
            "--val-embeddings",
            str(root_path(args.val_embeddings)),
            "--val-metadata",
            str(root_path(args.val_metadata)),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(args.max_steps),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.target_embeddings:
            command.extend(["--target-embeddings", str(root_path(args.target_embeddings))])
        print(" ".join(command))
        if not args.dry_run:
            code = subprocess.call(command, cwd=ROOT)
            if code != 0:
                raise SystemExit(code)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def parse_float(raw: Any) -> float | None:
    try:
        if raw in (None, ""):
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def cmd_summarize(args: argparse.Namespace) -> None:
    mode = args.mode
    rows: list[dict[str, Any]] = []
    if mode in {"locked", "bootstrap"}:
        source = read_csv(FINAL_DIR / "case_study_full_execution_status.csv")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in source:
            grouped[row.get("family", "")].append(row)
        for family, items in sorted(grouped.items()):
            metrics = {}
            for key in ["chexpert_macro_auc", "nih_macro_auc", "hard_shuffle_delta", "cf_acc", "ab_swap_acc"]:
                vals = [value for value in (parse_float(item.get(key)) for item in items) if value is not None]
                mean, std = mean_std(vals)
                metrics[key] = f"{mean:.6f}+/-{std:.6f}" if vals else "NA"
            rows.append({"family": family, "finalist": family, "seeds": len(items), **metrics, "role": "prior_protocol_baseline"})
        columns = ["family", "finalist", "seeds", "chexpert_macro_auc", "nih_macro_auc", "hard_shuffle_delta", "cf_acc", "ab_swap_acc", "role"]
        out_csv = FINAL_DIR / ("bootstrap_locked_comparison.csv" if mode == "bootstrap" else "locked_final_comparison.csv")
        out_md = out_csv.with_suffix(".md")
        write_csv(out_csv, rows, columns)
        write_md_table(out_md, "Locked Final Comparison", rows, columns, "Prior-protocol rows; CVCP module-combo rows must be added after formal runs.")
    elif mode == "external":
        rows = [
            {"dataset": "VinDr-CXR", "status": "partial_image_only", "evidence": "data/dataset/vinbigdata_xhlulu_512png", "notes": "No label/bbox CSV in current image package."},
            {"dataset": "PadChest", "status": "missing", "evidence": "", "notes": "No local PadChest directory found."},
            {"dataset": "NIH", "status": "appendix_available", "evidence": "outputs/final_tables/nih_available_transfer_status.csv", "notes": "Use only as appendix/stress-test under target plan."},
            {"dataset": "MIMIC-CXR", "status": "conditional_available", "evidence": "H:/Xiyao_Wang/000_Public Dataset/mimic-cxr/mimic-cxr", "notes": "Cannot be external if used as instruction source."},
        ]
        columns = ["dataset", "status", "evidence", "notes"]
        write_csv(FINAL_DIR / "external_eval_results.csv", rows, columns)
        write_md_table(FINAL_DIR / "external_eval_results.md", "External Evaluation Readiness", rows, columns)
    elif mode in {"ccsh", "ceq", "module"}:
        source = read_csv(FINAL_DIR / "module_ablation_results.csv")
        for row in source:
            if mode == "ccsh" and row.get("module") != "CCSH":
                continue
            if mode == "ceq" and row.get("module") != "CEQ":
                continue
            rows.append(row)
        columns = list(rows[0].keys()) if rows else ["module", "status"]
        out = FINAL_DIR / ("module_combo_results.csv" if mode == "module" else f"{mode}_consistency_results.csv")
        write_csv(out, rows, columns)
        write_md_table(out.with_suffix(".md"), f"{mode.upper()} Results", rows, columns, "Embedding-level module evidence unless paired with a formal backbone run.")
    elif mode == "ab_swap":
        source = read_csv(FINAL_DIR / "next_stage_ab_swap_counterfactual.csv")
        columns = list(source[0].keys()) if source else ["run_id"]
        write_csv(FINAL_DIR / "cvcp_ab_swap_results.csv", source, columns)
        write_md_table(FINAL_DIR / "cvcp_ab_swap_results.md", "CVCP A/B-Swap Results", source, columns)
    elif mode == "hard_shuffle":
        source = read_csv(FINAL_DIR / "next_stage_visual_dependence.csv")
        columns = list(source[0].keys()) if source else ["run_id"]
        write_csv(FINAL_DIR / "cvcp_hard_shuffle_results.csv", source, columns)
        write_md_table(FINAL_DIR / "cvcp_hard_shuffle_results.md", "CVCP Hard-Shuffle Results", source, columns)
    elif mode == "calibration":
        source = read_csv(FINAL_DIR / "next_stage_calibration_auprc.csv")
        columns = list(source[0].keys()) if source else ["run_id"]
        write_csv(FINAL_DIR / "cvcp_calibration_auprc.csv", source, columns)
        write_md_table(FINAL_DIR / "cvcp_calibration_auprc.md", "CVCP Calibration/AUPRC", source, columns)
    elif mode == "training":
        rows = []
        for source_path, source_name in [
            (FINAL_DIR / "cvcp_dataset_manifest.csv", "cvcp_generated_data"),
            (FINAL_DIR / "case_study_full_execution_status.csv", "case_study_multiseed"),
            (FINAL_DIR / "case_study_extra_execution_status.csv", "case_study_extra"),
            (FINAL_DIR / "next_stage_training_results.csv", "next_stage_training"),
        ]:
            for row in read_csv(source_path):
                rows.append(
                    {
                        "source": source_name,
                        "run_id": row.get("run_id") or row.get("step_id") or row.get("id") or row.get("dataset", ""),
                        "status": row.get("status") or row.get("train_status") or "available",
                        "rows_or_step": row.get("rows") or row.get("train_step") or row.get("global_step") or "",
                        "primary_metric": row.get("chexpert_macro_auc") or row.get("best_val_loss") or row.get("train_best_val_loss") or "",
                        "evidence": row.get("output") or row.get("evidence") or row.get("success_path") or repo_rel(source_path),
                        "notes": "Prior-protocol evidence must be reinterpreted under CVCP/CCSH where applicable.",
                    }
                )
        columns = ["source", "run_id", "status", "rows_or_step", "primary_metric", "evidence", "notes"]
        write_csv(FINAL_DIR / "cvcp_training_results.csv", rows, columns)
        write_md_table(FINAL_DIR / "cvcp_training_results.md", "CVCP Training Results", rows, columns)
    elif mode == "model":
        readiness = read_csv(FINAL_DIR / "cvcp_ccsh_readiness_audit.csv")
        rows = []
        for row in readiness:
            if row.get("area") == "model":
                rows.append(
                    {
                        "model_family": row.get("item", ""),
                        "status": row.get("status", ""),
                        "local_paths": row.get("evidence", ""),
                        "smoke_status": "pending_model_specific_smoke",
                        "decision": "available_for_comparison_queue" if row.get("status") == "available" else "unavailable",
                    }
                )
        columns = ["model_family", "status", "local_paths", "smoke_status", "decision"]
        write_csv(FINAL_DIR / "model_comparison_results.csv", rows, columns)
        write_md_table(FINAL_DIR / "model_comparison_results.md", "Model Comparison Readiness", rows, columns)
    elif mode == "casebook":
        sources = [
            ("SAMEQ/SHUF case study", FINAL_DIR / "case_study_shuf_tw_vs_shuf.md"),
            ("NIH/domain failures", FINAL_DIR / "case_study_nih_transfer.md"),
            ("Curriculum failure", FINAL_DIR / "case_study_curriculum_failure.md"),
            ("Hard-negative quality", FINAL_DIR / "case_study_hard_negative_quality.md"),
            ("Next-stage qualitative cases", FINAL_DIR / "next_stage_qualitative_cases.md"),
        ]
        lines = ["# CVCP/CCSH Casebook", "", "This casebook consolidates existing qualitative case reports for the CVCP/CCSH plan. New CEQ attention maps and external-main cases should be appended after formal runs.", ""]
        rows = []
        for title, path in sources:
            status = "existing" if path.exists() else "missing"
            rows.append({"casebook": title, "status": status, "evidence": repo_rel(path) if path.exists() else ""})
            lines.extend([f"## {title}", ""])
            if path.exists():
                text = path.read_text(encoding="utf-8")
                lines.extend(text.splitlines()[:80])
                lines.append("")
            else:
                lines.extend(["Missing source casebook.", ""])
        (FINAL_DIR / "casebook.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_csv(FINAL_DIR / "casebook_manifest.csv", rows, ["casebook", "status", "evidence"])
    else:
        raise ValueError(f"Unknown summarize mode: {mode}")
    print(f"mode={mode} rows={len(rows)}")


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)


def build_parser(default_command: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=default_command is None)

    p = sub.add_parser("generate_cvcp_curriculum")
    add_common_data_args(p)
    p.add_argument("--full-rows", type=int, default=16000)
    p.set_defaults(func=cmd_generate_cvcp_curriculum)

    p = sub.add_parser("generate_sameq_cf_compatible")
    add_common_data_args(p)
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--percent", type=int, action="append", default=[10, 20, 30])
    p.set_defaults(func=cmd_generate_sameq_cf)

    p = sub.add_parser("generate_shuf_k_cf_compatible")
    add_common_data_args(p)
    p.add_argument("--rows", type=int, default=10000)
    p.set_defaults(func=cmd_generate_shuf_k_cf)

    p = sub.add_parser("generate_ccsh_statements")
    p.add_argument("--input", default="outputs/instruction_data/next_stage/shuf_k4_train.jsonl")
    p.add_argument("--output", default="outputs/instruction_data/cvcp_ccsh/ccsh_statements_train.jsonl")
    p.add_argument("--max-rows", type=int, default=20000)
    p.set_defaults(func=cmd_generate_ccsh_statements)

    p = sub.add_parser("generate_ceq_targets")
    p.add_argument("--input", default="outputs/instruction_data/next_stage/shuf_k4_train.jsonl")
    p.add_argument("--output", default="outputs/instruction_data/cvcp_ccsh/ceq_targets_train.jsonl")
    p.add_argument("--max-rows", type=int, default=20000)
    p.set_defaults(func=cmd_generate_ceq_targets)

    p = sub.add_parser("audit_instruction_leakage_v3")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--output-csv", default="outputs/final_tables/cvcp_instruction_leakage_v3.csv")
    p.add_argument("--detail-csv", default="outputs/final_tables/cvcp_instruction_leakage_v3_detail.csv")
    p.add_argument("--output-md", default="outputs/final_tables/cvcp_instruction_leakage_v3.md")
    p.set_defaults(func=cmd_audit_leakage)

    p = sub.add_parser("audit_false_hard_negatives")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--output-csv", default="outputs/final_tables/cvcp_false_hard_negative_audit.csv")
    p.add_argument("--detail-csv", default="outputs/final_tables/cvcp_false_hard_negative_audit_detail.csv")
    p.add_argument("--output-md", default="outputs/final_tables/cvcp_false_hard_negative_audit.md")
    p.set_defaults(func=cmd_audit_false_hard_negatives)

    p = sub.add_parser("train_qwen3vl")
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train_qwen3vl)

    p = sub.add_parser("train_module_stack")
    p.add_argument("--module", action="append", choices=["CEQ", "AUCH", "HNMB", "DRA", "CCSH", "CDCS"])
    p.add_argument("--default-modules", nargs="+", default=["CCSH"])
    p.add_argument("--train-embeddings", required=True)
    p.add_argument("--train-metadata", required=True)
    p.add_argument("--val-embeddings", required=True)
    p.add_argument("--val-metadata", required=True)
    p.add_argument("--target-embeddings")
    p.add_argument("--output-root", required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train_module_stack)

    p = sub.add_parser("summarize")
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "locked",
            "external",
            "ccsh",
            "ceq",
            "module",
            "ab_swap",
            "hard_shuffle",
            "calibration",
            "bootstrap",
            "training",
            "model",
            "casebook",
        ],
    )
    p.set_defaults(func=cmd_summarize)

    parser.set_defaults(default_command=default_command)
    return parser


def main(default_command: str | None = None) -> None:
    parser = build_parser(default_command)
    argv = sys.argv[1:]
    if default_command:
        argv = [default_command] + argv
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
