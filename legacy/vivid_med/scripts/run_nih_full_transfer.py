"""Build or run NIH full/available transfer commands from existing LP configs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from case_study_modules_common import FINAL_DIR, read_json, root_path, write_csv_rows, write_md_table


COLUMNS = ["run_id", "id", "status", "nih_manifest", "max_samples", "command", "output_dir", "boundary"]


def load_lp_manifest(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path) or {"configs": []}
    return list(payload.get("configs", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lp-manifest", type=Path, default=Path("outputs/next_stage_manifests/lp_config_manifest.json"))
    parser.add_argument("--run-id", action="append", default=["shuf_tw_clinical", "sameq_shuf_3k", "shuf_k4", "shuf_k4_tw_visual"])
    parser.add_argument("--nih-manifest", type=Path, default=Path("data/dataset/processed/nih_external_test_ums.jsonl"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all rows available to the transfer script.")
    parser.add_argument("--run", action="store_true", help="Actually launch commands. Default only writes a manifest.")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "nih_full_transfer_manifest.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "nih_full_transfer_manifest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_lp_manifest(args.lp_manifest)
    wanted = {item.lower().replace("-", "_") for item in args.run_id}
    rows: list[dict[str, Any]] = []
    nih_path = root_path(args.nih_manifest)
    row_count = 0
    if nih_path.exists():
        row_count = sum(1 for line in nih_path.read_text(encoding="utf-8").splitlines() if line.strip())
    for spec in specs:
        sid = str(spec.get("id", ""))
        if sid.lower().replace("-", "_") not in wanted:
            continue
        config = spec.get("config", "")
        output_dir = f"outputs/qwen3vl_next_stage_transfer/{sid}_nih_full_available"
        max_samples = row_count if args.max_samples == 0 else args.max_samples
        cmd = [
            "python",
            "scripts/evaluate_qwen3vl_lp_transfer.py",
            "--lp-config",
            str(config),
            "--val-ums-path",
            str(args.nih_manifest),
            "--data-root",
            "H:/Xiyao_Wang/000_Public Dataset",
            "--output-dir",
            output_dir,
            "--max-samples",
            str(args.max_samples),
            "--verify-images",
        ]
        boundary = "available_manifest_rows" if row_count else "nih_manifest_missing"
        status = "planned"
        if args.run and row_count:
            completed = subprocess.run(cmd, cwd=root_path("."), check=False)
            status = f"exit_{completed.returncode}"
        rows.append(
            {
                "run_id": spec.get("run_id", sid),
                "id": sid,
                "status": status,
                "nih_manifest": str(args.nih_manifest),
                "max_samples": max_samples,
                "command": " ".join(cmd),
                "output_dir": output_dir,
                "boundary": boundary,
            }
        )
    write_csv_rows(args.output_csv, rows, COLUMNS)
    write_md_table(args.output_md, "NIH Full/Available Transfer Manifest", rows, COLUMNS)


if __name__ == "__main__":
    main()
