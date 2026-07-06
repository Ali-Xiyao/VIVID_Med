"""Build a requirement ledger for the CVCP/CCSH full experiment plan.

The target plan is intentionally broad and table-heavy. This script extracts
the named scripts, final-table outputs, run rows, datasets, casebooks, and
visualizations into a machine-readable ledger before execution starts.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "vivid_med_cvcp_ccsh_full_next_experiment_plan.md"
DEFAULT_MD = ROOT / "docs" / "cvcp_ccsh_requirement_ledger.md"
DEFAULT_CSV = ROOT / "outputs" / "final_tables" / "cvcp_ccsh_requirement_ledger.csv"

COLUMNS = [
    "requirement_id",
    "type",
    "section",
    "name",
    "status",
    "evidence",
    "notes",
]

SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_./+\-]+\.py")
OUTPUT_RE = re.compile(r"outputs/final_tables/[A-Za-z0-9_./+\-]+(?:\.(?:md|csv|json|png))?")


def clean_cell(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    return value.replace("<br>", "; ").strip()


def is_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    if not stripped:
        return False
    cells = [cell.strip() for cell in stripped.split("|")]
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def split_row(line: str) -> list[str]:
    return [clean_cell(cell) for cell in line.strip().strip("|").split("|")]


def iter_tables(lines: list[str]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    headings: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            headings = headings[: level - 1] + [title]
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            header = split_row(line)
            rows: list[dict[str, str]] = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = split_row(lines[i])
                if len(values) == len(header):
                    rows.append(dict(zip(header, values)))
                i += 1
            tables.append({"section": " > ".join(headings), "header": header, "rows": rows})
            continue
        i += 1
    return tables


def normalize_key(raw: str) -> str:
    key = raw.lower()
    key = key.replace("+", "_plus_")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def dataset_status(name: str) -> tuple[str, str, str]:
    lower = name.lower()
    if "vindr" in lower or "vinbig" in lower:
        image_only = ROOT / "data" / "dataset" / "vinbigdata_xhlulu_512png"
        if image_only.exists():
            return (
                "partial_image_only",
                repo_rel(image_only),
                "VinBigData/VinDr-derived image package exists, but current audit found no label/bbox CSV in that package.",
            )
        return "missing", "", "No local VinDr-CXR/VinBigData directory found."
    if "padchest" in lower:
        path = first_existing([ROOT / "data" / "dataset" / "PadChest", ROOT / "data" / "dataset" / "padchest"])
        if path:
            return "available", repo_rel(path), "Local PadChest-like directory exists; label mapping still needs audit."
        return "missing", "", "No local PadChest directory found in the first dataset audit."
    if "nih" in lower:
        path = ROOT / "data" / "dataset" / "NIH Chest X-rays"
        if path.exists():
            return "available_appendix", repo_rel(path), "Local NIH exists, but the plan says NIH is appendix/stress-test, not main external."
        return "missing", "", "No local NIH directory found."
    if "mimic" in lower:
        candidates = [
            ROOT / "data" / "dataset" / "mimic-cxr",
            Path("H:/Xiyao_Wang/000_Public Dataset/mimic-cxr/mimic-cxr"),
        ]
        path = first_existing(candidates)
        if path:
            return "available_conditional", repo_rel(path), "MIMIC can be source or conditional external depending on train split usage."
        return "missing", "", "No local MIMIC-CXR path found."
    if "chexpert" in lower:
        path = ROOT / "data" / "dataset" / "CheXpert-v1.0-small"
        if path.exists():
            return "available_source", repo_rel(path), "CheXpert small exists locally and is the current source/eval anchor."
    return "open", "", "Availability must be checked by the phase V2 data audit."


def run_evidence(name: str, all_paths: list[str]) -> tuple[str, str, str]:
    key = normalize_key(name)
    if not key:
        return "open", "", "Blank table row."
    if key in {"run", "data", "model", "purpose", "family", "candidate_pool"}:
        return "open", "", "Header-like row ignored in later audits."
    tokens = [token for token in key.split("_") if len(token) >= 2]
    matches = []
    for path in all_paths:
        pkey = normalize_key(path)
        if key and key in pkey:
            matches.append(path)
        elif len(tokens) >= 2 and all(token in pkey for token in tokens[: min(3, len(tokens))]):
            matches.append(path)
    if matches:
        evidence = "; ".join(matches[:3])
        return "candidate_evidence_needs_protocol_audit", evidence, "Name-like artifacts exist; exact protocol still needs row-level audit."
    return "open", "", "No exact candidate artifact found by normalized-name scan."


def add_row(rows: list[dict[str, str]], req_type: str, section: str, name: str, status: str, evidence: str = "", notes: str = "") -> None:
    name = clean_cell(name)
    if not name:
        return
    req_id = f"CVCP-{len(rows) + 1:04d}"
    rows.append(
        {
            "requirement_id": req_id,
            "type": req_type,
            "section": section,
            "name": name,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
    )


def build_rows(plan_path: Path) -> list[dict[str, str]]:
    text = plan_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    all_paths = [repo_rel(path) for path in (ROOT / "scripts").glob("**/*")] + [
        repo_rel(path) for path in (ROOT / "configs").glob("**/*")
    ] + [repo_rel(path) for path in (ROOT / "outputs" / "final_tables").glob("**/*")]

    for script in sorted(set(SCRIPT_RE.findall(text))):
        path = ROOT / script
        add_row(
            rows,
            "script",
            "Codex task checklist",
            script,
            "existing" if path.exists() else "missing",
            script if path.exists() else "",
            "Target-plan named script.",
        )

    for output in sorted(set(OUTPUT_RE.findall(text))):
        path = ROOT / output
        add_row(
            rows,
            "output",
            "Reporting",
            output,
            "existing" if path.exists() else "missing",
            output if path.exists() else "",
            "Target-plan named final-table artifact.",
        )

    seen_table_items: set[tuple[str, str, str]] = set()
    run_headers = {"Run", "Run ID", "Comparison", "Family", "Model"}
    dataset_headers = {"Dataset", "External candidate"}
    casebook_headers = {"Casebook", "Visualization"}

    for table in iter_tables(lines):
        section = table["section"]
        header = set(table["header"])
        for row in table["rows"]:
            if header & run_headers:
                source_key = next((key for key in table["header"] if key in run_headers), "")
                name = row.get(source_key, "")
                if name and not set(name) <= {"-", " "}:
                    status, evidence, notes = run_evidence(name, all_paths)
                    key = ("run_or_metric", section, name)
                    if key not in seen_table_items:
                        seen_table_items.add(key)
                        add_row(rows, "run_or_metric", section, name, status, evidence, notes)
            if header & dataset_headers:
                source_key = next((key for key in table["header"] if key in dataset_headers), "")
                name = row.get(source_key, "")
                status, evidence, notes = dataset_status(name)
                key = ("dataset", section, name)
                if key not in seen_table_items:
                    seen_table_items.add(key)
                    add_row(rows, "dataset", section, name, status, evidence, notes)
            if header & casebook_headers:
                source_key = next((key for key in table["header"] if key in casebook_headers), "")
                name = row.get(source_key, "")
                key = ("qualitative", section, name)
                if key not in seen_table_items:
                    seen_table_items.add(key)
                    add_row(rows, "qualitative", section, name, "open", "", "Casebook or visualization required by target plan.")
            if "Artifact" in header:
                name = row.get("Artifact", "")
                if name.startswith("outputs/") or name.startswith("docs/") or name.startswith("scripts/"):
                    path = ROOT / name
                    key = ("artifact", section, name)
                    if key not in seen_table_items:
                        seen_table_items.add(key)
                        add_row(rows, "artifact", section, name, "existing" if path.exists() else "missing", name if path.exists() else "", "Named artifact row.")

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def md_table(rows: list[dict[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_md(path: Path, rows: list[dict[str, str]], source_plan: Path, csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["type"], row["status"]) for row in rows)
    type_counts = Counter(row["type"] for row in rows)
    lines = [
        "# VIVID-Med CVCP/CCSH Requirement Ledger",
        "",
        f"Source plan: `{repo_rel(source_plan)}`",
        f"Machine-readable CSV: `{repo_rel(csv_path)}`",
        "",
        "This ledger is generated from the target plan and is intentionally conservative. `candidate_evidence_needs_protocol_audit` means a name-like artifact exists, not that the row is complete under the current protocol.",
        "",
        "## Summary",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(type_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Status Counts", "", "| Type | Status | Count |", "| --- | --- | ---: |"])
    for (req_type, status), value in sorted(counts.items()):
        lines.append(f"| {req_type} | {status} | {value} |")
    lines.extend(["", "## Ledger", ""])
    lines.extend(md_table(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.plan)
    write_csv(args.csv, rows)
    write_md(args.md, rows, args.plan, args.csv)
    counts = Counter(row["status"] for row in rows)
    print(f"wrote_rows={len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")
    print(f"md={repo_rel(args.md)}")
    print(f"csv={repo_rel(args.csv)}")


if __name__ == "__main__":
    main()
