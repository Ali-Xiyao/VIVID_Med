"""Prepare extra execution manifest for curriculum, embeddings, MMD/UMAP, and modules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"

COLUMNS = ["step_id", "step_type", "module", "command", "success_path", "force_run", "status", "notes"]
MODULES = ["CEQ", "AUCH", "HNMB", "DRA", "CCSH", "CDCS"]


def sq(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''").replace("\\", "/") + "'"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Case Study Extra Execution Manifest", "", "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key in COLUMNS) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(
    step_id: str,
    step_type: str,
    command: str,
    success_path: Path,
    module: str = "",
    notes: str = "",
    force_run: bool = False,
) -> dict[str, Any]:
    status = "planned" if force_run else ("completed_existing" if success_path.exists() else "planned")
    return {
        "step_id": step_id,
        "step_type": step_type,
        "module": module,
        "command": command,
        "success_path": success_path.as_posix(),
        "force_run": "1" if force_run else "0",
        "status": status,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen3vl_instruction/next_stage/shuf_tw_clinical.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/qwen3vl_instruction/next_stage/shuf_tw_clinical/checkpoints/best.pt")
    parser.add_argument("--embedding-root", type=Path, default=ROOT / "outputs/case_study_module_embeddings")
    parser.add_argument("--module-root", type=Path, default=ROOT / "outputs/qwen3vl_case_study_module_ablation")
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "case_study_extra_execution_manifest.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "case_study_extra_execution_manifest.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_root = args.embedding_root
    train_npz = embed_root / "shuf_tw_clinical_instruction_train.npz"
    train_meta = embed_root / "shuf_tw_clinical_instruction_train_metadata.jsonl"
    val_npz = embed_root / "shuf_tw_clinical_instruction_val.npz"
    val_meta = embed_root / "shuf_tw_clinical_instruction_val_metadata.jsonl"
    chexpert_npz = embed_root / "shuf_tw_clinical_chexpert_val.npz"
    chexpert_meta = embed_root / "shuf_tw_clinical_chexpert_val_metadata.jsonl"
    nih_npz = embed_root / "shuf_tw_clinical_nih_available.npz"
    nih_meta = embed_root / "shuf_tw_clinical_nih_available_metadata.jsonl"

    rows: list[dict[str, Any]] = []
    rows.append(
        row(
            "cur_v2_progressive_replay_train",
            "curriculum_train",
            "conda run -n vivid python scripts/train_qwen3vl_curriculum_v2.py --run",
            ROOT / "outputs/qwen3vl_case_study_modules/cur_v2_progressive_replay/metrics_final.json",
            notes="formal curriculum-v2 long training",
        )
    )
    rows.append(
        row(
            "export_instruction_train_embeddings",
            "embedding_export",
            f"conda run -n vivid python scripts/export_qwen3vl_instruction_embeddings.py --config {sq(args.config)} --checkpoint {sq(args.checkpoint)} --instruction-jsonl outputs/instruction_data/glm_validated/d7_hard_shuffle_3k.jsonl --output-npz {sq(train_npz)} --metadata-jsonl {sq(train_meta)} --manifest {sq(embed_root / 'shuf_tw_clinical_instruction_train_manifest.json')} --batch-size 2 --device cuda:0",
            train_npz,
        )
    )
    rows.append(
        row(
            "export_instruction_val_embeddings",
            "embedding_export",
            f"conda run -n vivid python scripts/export_qwen3vl_instruction_embeddings.py --config {sq(args.config)} --checkpoint {sq(args.checkpoint)} --instruction-jsonl outputs/instruction_data/glm_validated/d7_hard_shuffle_val200.jsonl --output-npz {sq(val_npz)} --metadata-jsonl {sq(val_meta)} --manifest {sq(embed_root / 'shuf_tw_clinical_instruction_val_manifest.json')} --batch-size 2 --device cuda:0",
            val_npz,
        )
    )
    rows.append(
        row(
            "export_chexpert_val_ums_embeddings",
            "embedding_export",
            f"conda run -n vivid python scripts/export_qwen3vl_ums_embeddings.py --config {sq(args.config)} --checkpoint {sq(args.checkpoint)} --ums-jsonl data/splits/chexpert_val_fixed.jsonl --data-root {sq('H:/Xiyao_Wang/000_Public Dataset')} --output-npz {sq(chexpert_npz)} --metadata-jsonl {sq(chexpert_meta)} --manifest {sq(embed_root / 'shuf_tw_clinical_chexpert_val_manifest.json')} --batch-size 2 --device cuda:0 --use-common-labels-only",
            chexpert_npz,
        )
    )
    rows.append(
        row(
            "export_nih_available_ums_embeddings",
            "embedding_export",
            f"conda run -n vivid python scripts/export_qwen3vl_ums_embeddings.py --config {sq(args.config)} --checkpoint {sq(args.checkpoint)} --ums-jsonl data/dataset/processed/nih_external_test_ums.jsonl --data-root {sq('H:/Xiyao_Wang/000_Public Dataset')} --output-npz {sq(nih_npz)} --metadata-jsonl {sq(nih_meta)} --manifest {sq(embed_root / 'shuf_tw_clinical_nih_available_manifest.json')} --batch-size 2 --device cuda:0 --use-common-labels-only",
            nih_npz,
            notes="25,596-row NIH available manifest",
        )
    )
    rows.append(
        row(
            "compute_domain_shift_mmd",
            "domain_report",
            f"python scripts/compute_domain_shift_mmd.py --source {sq(chexpert_npz)} --target {sq(nih_npz)} --source-name CheXpert-val --target-name NIH-available --max-rows 4000",
            FINAL_DIR / "domain_shift_mmd.md",
            force_run=True,
        )
    )
    rows.append(
        row(
            "plot_dataset_embedding_projection",
            "domain_report",
            f"python scripts/plot_dataset_embedding_umap.py --embedding {sq(chexpert_npz)} --embedding {sq(nih_npz)} --max-rows 4000",
            FINAL_DIR / "dataset_embedding_projection.csv",
            force_run=True,
        )
    )
    for module in MODULES:
        output_dir = args.module_root / module.lower()
        rows.append(
            row(
                f"train_module_{module.lower()}",
                "module_ablation",
                f"conda run -n vivid python scripts/train_case_study_module_ablation.py --module {module} --train-embeddings {sq(train_npz)} --train-metadata {sq(train_meta)} --val-embeddings {sq(val_npz)} --val-metadata {sq(val_meta)} --target-embeddings {sq(nih_npz)} --output-dir {sq(output_dir)} --max-steps 1000 --batch-size 64 --device cuda:0 --seed 42",
                output_dir / "metrics_final.json",
                module=module,
                notes="formal embedding-level module ablation",
            )
        )
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    (ROOT / "outputs/next_stage_manifests/case_study_extra_execution_manifest.json").write_text(
        json.dumps({"steps": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"steps": len(rows), "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
