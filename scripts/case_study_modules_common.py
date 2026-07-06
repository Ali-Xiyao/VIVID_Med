"""Shared utilities for the case-study/module execution scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs/final_tables"
NEXT_DIAG_DIR = ROOT / "outputs/qwen3vl_next_stage_diagnostics"
P4V2_DIAG_DIR = ROOT / "outputs/qwen3vl_p4v2_diagnostics"


def root_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()


def read_json(path: str | Path) -> dict[str, Any] | None:
    candidate = root_path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    candidate = root_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    candidate = root_path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    candidate = root_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with candidate.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(
    path: str | Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    note: str = "",
) -> None:
    candidate = root_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    if not rows:
        lines.append("No rows.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    candidate.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md_sections(path: str | Path, title: str, sections: list[tuple[str, str]]) -> None:
    candidate = root_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip() if body.strip() else "No details.", ""])
    candidate.write_text("\n".join(lines), encoding="utf-8")


def read_jsonl(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    candidate = root_path(path)
    rows: list[dict[str, Any]] = []
    if not candidate.exists():
        return rows
    with candidate.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(rows) >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    candidate = root_path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.{digits}g}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def load_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(FINAL_DIR / "qwen3vl_p4v2_decision_summary.csv"):
        run_id = row.get("run_id", "")
        out = dict(row)
        out["id"] = normalize_key(run_id)
        out["source_table"] = "qwen3vl_p4v2_decision_summary.csv"
        rows.append(out)
    for row in read_csv_rows(FINAL_DIR / "next_stage_decision_summary.csv"):
        out = dict(row)
        out["source_table"] = "next_stage_decision_summary.csv"
        rows.append(out)
    return rows


def metric_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in (row.get("id"), row.get("run_id")):
            if key:
                lookup[normalize_key(key)] = row
    return lookup


def find_diagnostic_files(run_id: str, suffix: str) -> list[Path]:
    token = normalize_key(run_id)
    candidates: list[Path] = []
    for base in (NEXT_DIAG_DIR, P4V2_DIAG_DIR):
        direct = base / f"{token}_{suffix}"
        if direct.exists():
            candidates.append(direct)
        if base.exists():
            for path in base.rglob(f"*{suffix}"):
                text = normalize_key(path.as_posix())
                if token in text and path not in candidates:
                    candidates.append(path)
    return sorted(candidates, key=lambda p: (len(p.parts), p.as_posix()))


def read_first_diagnostic_csv(run_id: str, suffix: str) -> tuple[Path | None, list[dict[str, str]]]:
    for path in find_diagnostic_files(run_id, suffix):
        rows = read_csv_rows(path)
        if rows:
            return path, rows
    return None, []


def read_first_diagnostic_json(run_id: str, suffix: str) -> tuple[Path | None, dict[str, Any] | None]:
    for path in find_diagnostic_files(run_id, suffix):
        payload = read_json(path)
        if payload is not None:
            return path, payload
    return None, None


def summarize_metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], metric: str) -> str:
    cand = to_float(candidate.get(metric))
    base = to_float(baseline.get(metric))
    if cand is None or base is None:
        return ""
    return fmt(cand - base)


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

