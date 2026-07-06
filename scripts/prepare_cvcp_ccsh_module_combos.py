"""Prepare module-combo embedding/training manifest for CVCP/CCSH."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "outputs" / "final_tables"
TRAIN_MANIFEST = FINAL_DIR / "cvcp_ccsh_training_manifest.csv"
OUT_ROOT = Path("F:/Xiyao_Wang/021_260129VIVID_cvcp_ccsh_outputs")

COMBOS = [
    ("base_ccsh", "Base+CCSH", "cvcp_v1_sameq_3k", "", ["CCSH"], "Head-only raw Qwen3VL baseline on SAMEQ rows."),
    ("sameq_ccsh", "SAMEQ+CCSH", "cvcp_v1_sameq_3k", "final", ["CCSH"], "Main same-question consistency module."),
    ("sameq_10k_ccsh", "SAMEQ-10k+CCSH", "cvcp_v1_sameq_10k", "final", ["CCSH"], "Formal SAMEQ-10k bridge row for the v4 paper-ready plan."),
    ("sameq_full_ccsh", "SAMEQ-full+CCSH", "cvcp_v1_sameq_full", "final", ["CCSH"], "Highest-priority SAMEQ-full consistency bridge row for v4."),
    ("shufk_ccsh", "SHUF-K4+CCSH", "cvcp_v2_shuf_k4", "final", ["CCSH"], "Multi-negative consistency module."),
    ("cvcp_prog_ccsh", "CVCP-prog+CCSH", "cvcp_v3_prog_10k", "final", ["CCSH"], "Progressive curriculum consistency module."),
    ("cvcp_replay_ccsh", "CVCP-replay+CCSH", "cvcp_v4_replay_10k", "final", ["CCSH"], "Replay curriculum consistency module."),
    ("sameq_ceq", "SAMEQ+CEQ", "cvcp_v1_sameq_3k", "final", ["CEQ"], "Evidence query alone."),
    ("sameq_ceq_ccsh", "SAMEQ+CEQ+CCSH", "cvcp_v1_sameq_3k", "final", ["CEQ", "CCSH"], "Same-question evidence plus consistency."),
    ("sameq_full_ceq_ccsh", "SAMEQ-full+CEQ+CCSH", "cvcp_v1_sameq_full", "final", ["CEQ", "CCSH"], "Formal SAMEQ-full evidence plus consistency bridge row for v4."),
    ("sameq_cf20_ccsh", "SAMEQ-CF-20+CCSH", "sameq_cf_20", "final", ["CCSH"], "Counterfactual-compatible SAMEQ bridge row."),
    ("sameq_cf20_ceq_ccsh", "SAMEQ-CF-20+CEQ+CCSH", "sameq_cf_20", "final", ["CEQ", "CCSH"], "Counterfactual-compatible evidence plus consistency row."),
    ("shufk_ceq", "SHUF-K4+CEQ", "cvcp_v2_shuf_k4", "final", ["CEQ"], "Multi-negative evidence query."),
    ("shufk_ceq_ccsh", "SHUF-K4+CEQ+CCSH", "cvcp_v2_shuf_k4", "final", ["CEQ", "CCSH"], "Multi-negative evidence plus consistency."),
    ("cvcp_ceq_ccsh", "CVCP+CEQ+CCSH", "cvcp_v4_replay_10k", "final", ["CEQ", "CCSH"], "Curriculum evidence plus consistency."),
    ("hnmb_static", "HNMB-static", "cvcp_v2_shuf_k4", "final", ["HNMB"], "Embedding memory-bank hard negatives."),
    ("sameq_hnmb", "SAMEQ+HNMB", "cvcp_v1_sameq_3k", "final", ["HNMB"], "Same-question memory-bank negatives."),
    ("hnmb_ccsh", "HNMB+CCSH", "cvcp_v2_shuf_k4", "final", ["HNMB", "CCSH"], "Hard-negative memory plus consistency."),
    ("ceq_hnmb_ccsh", "CEQ+HNMB+CCSH", "cvcp_v4_replay_10k", "final", ["CEQ", "HNMB", "CCSH"], "Full evidence/memory/consistency stack."),
    ("sameq_auch", "SAMEQ+AUCH", "cvcp_v1_sameq_3k", "final", ["AUCH"], "Answerability/uncertainty head."),
    ("sameq_auch_ccsh", "SAMEQ+AUCH+CCSH", "cvcp_v1_sameq_3k", "final", ["AUCH", "CCSH"], "Uncertainty plus consistency."),
    ("ceq_auch_ccsh", "CEQ+AUCH+CCSH", "cvcp_v4_replay_10k", "final", ["CEQ", "AUCH", "CCSH"], "Evidence/uncertainty/consistency stack."),
    ("cvcp_cdcs_ccsh", "CVCP-CDCS+CCSH", "cvcp_v5_cdcs_full", "final", ["CDCS", "CCSH"], "Case-driven scheduler plus consistency."),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|").replace("\n", " ") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    runs = {row["id"]: row for row in read_csv(TRAIN_MANIFEST)}
    rows: list[dict[str, Any]] = []
    for combo_id, label, backbone_id, checkpoint_policy, modules, notes in COMBOS:
        run = runs[backbone_id]
        embedding_root = OUT_ROOT / "module_embeddings" / combo_id
        output_root = OUT_ROOT / "module_combos" / combo_id
        checkpoint = ""
        if checkpoint_policy == "final":
            checkpoint = (Path(run["train_output_dir"]) / "checkpoints" / "final.pt").as_posix()
        success_paths = [str(output_root / module.lower() / "metrics_final.json") for module in modules]
        rows.append(
            {
                "combo_id": combo_id,
                "label": label,
                "backbone_run_id": backbone_id,
                "backbone_label": run["run_id"],
                "modules": ";".join(modules),
                "train_config": run["train_config"],
                "train_output_dir": run["train_output_dir"],
                "checkpoint_path": checkpoint,
                "val_instruction_path": run.get("val_instruction_path", ""),
                "train_npz": (embedding_root / "train_embeddings.npz").as_posix(),
                "train_metadata": (embedding_root / "train_metadata.jsonl").as_posix(),
                "train_manifest": (embedding_root / "train_manifest.json").as_posix(),
                "val_npz": (embedding_root / "val_embeddings.npz").as_posix(),
                "val_metadata": (embedding_root / "val_metadata.jsonl").as_posix(),
                "val_manifest": (embedding_root / "val_manifest.json").as_posix(),
                "output_root": output_root.as_posix(),
                "success_paths": ";".join(success_paths),
                "max_export_train": 5000,
                "max_export_val": 1000,
                "max_steps": 1000,
                "batch_size": 64,
                "notes": notes,
            }
        )
    columns = [
        "combo_id",
        "label",
        "backbone_run_id",
        "backbone_label",
        "modules",
        "train_config",
        "train_output_dir",
        "checkpoint_path",
        "val_instruction_path",
        "train_npz",
        "train_metadata",
        "train_manifest",
        "val_npz",
        "val_metadata",
        "val_manifest",
        "output_root",
        "success_paths",
        "max_export_train",
        "max_export_val",
        "max_steps",
        "batch_size",
        "notes",
    ]
    write_csv(FINAL_DIR / "cvcp_ccsh_module_combo_manifest.csv", rows, columns)
    write_md_table(FINAL_DIR / "cvcp_ccsh_module_combo_manifest.md", "CVCP/CCSH Module Combo Manifest", rows, columns)
    print(f"wrote_rows={len(rows)}")


if __name__ == "__main__":
    main()
