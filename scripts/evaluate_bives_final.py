"""Explicit one-shot evaluator for the locked BiVES-CXR test split."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.audit import audit_manifests
from bives_cxr.backbones import Qwen35VisionAdapter, load_qwen35_visual_and_processor
from bives_cxr.data import BiVESManifestDataset, statement_text_by_id
from bives_cxr.interventions import CONTROL_PROTOCOL_VERSION
from bives_cxr.losses import BiVESLoss, BiVESLossConfig
from bives_cxr.model import BiVESModelConfig
from bives_cxr.provenance import validate_calibrated_release_chain
from bives_cxr.statement_cache import (
    load_statement_embedding_matrix,
    validate_ontology_subset,
)
from scripts.train_bives_cxr import (
    BiVESExperiment,
    Qwen35BiVESCollator,
    evaluate,
    file_sha256,
    git_commit,
    load_checkpoint_model_state,
    save_json,
    set_decoder_temperatures,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--statement-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-locked-test",
        "--release-locked-test",
        dest="run_locked_test",
        action="store_true",
        help="Required acknowledgement that this is the single final test release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_locked_test:
        raise SystemExit("--run-locked-test is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"locked-test output directory is not empty: {args.output_dir}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if str(config["model"]["family"]).lower() != "qwen3.5":
        raise ValueError("locked evaluator is Qwen3.5-only")
    calibration = json.loads(args.calibration_artifact.read_text(encoding="utf-8"))
    if not calibration.get("calibrated_temperatures"):
        raise ValueError("calibration artifact has no calibrated temperatures")
    current_commit = git_commit()
    run_lock = validate_calibrated_release_chain(
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        calibration=calibration,
        statement_cache_path=args.statement_cache,
        test_manifest_path=args.test_manifest,
        current_git_commit=current_commit,
    )

    audit_config = config.get("audit", {})
    audit = audit_manifests(
        {"test": args.test_manifest},
        data_root=args.data_root,
        check_images=True,
        require_complete_statements=True,
        check_decodable=True,
        reject_constant_images=bool(audit_config.get("reject_constant_images", True)),
        require_provenance=True,
        verify_image_sha256=True,
        require_matching_protocol=bool(
            audit_config.get("require_matching_protocol", True)
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "locked_test_manifest_audit.json", audit)
    if audit["status"] != "pass":
        raise SystemExit("locked test readiness audit failed")

    statement_to_index = {
        str(key): int(value)
        for key, value in checkpoint["statement_to_index"].items()
    }
    dataset = BiVESManifestDataset(
        args.test_manifest,
        args.data_root,
        statement_to_index=statement_to_index,
    )
    locked_ontology = checkpoint.get("statement_text_by_id")
    if not isinstance(locked_ontology, dict):
        raise ValueError("checkpoint is missing the full locked statement ontology")
    if set(locked_ontology) != set(statement_to_index):
        raise ValueError("checkpoint ontology and statement index do not match")
    test_ontology = statement_text_by_id(dataset.rows)
    validate_ontology_subset(locked_ontology, test_ontology)
    frozen = load_statement_embedding_matrix(
        args.statement_cache,
        statement_to_index,
        locked_ontology,
        expected_cache=config["model"]["statement_embeddings"],
    )
    device = torch.device(
        str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    )
    set_seed(int(config.get("seed", 17)))
    visual, processor, qwen_config = load_qwen35_visual_and_processor(
        config["model"]["path"],
        dtype=str(config["model"].get("dtype", "bf16")),
        attention_implementation=str(
            config["model"].get("attention_implementation", "eager")
        ),
    )
    for parameter in visual.parameters():
        parameter.requires_grad = False
    head_config = BiVESModelConfig(
        visual_dim=int(qwen_config["vision_config"]["hidden_size"]),
        statement_dim=int(frozen.shape[1]),
        fusion_dim=int(config["bives"].get("fusion_dim", 512)),
        evidence_max=float(config["bives"].get("evidence_max", 8.0)),
        gate_mode=str(config["bives"]["mask"].get("type", "soft_topk")),
        topk=int(config["bives"]["mask"].get("topk", 16)),
        gate_temperature=float(config["bives"]["mask"].get("temperature", 0.5)),
        tau_a=float(config["bives"]["decoder"].get("tau_a", 1.0)),
        tau_d=float(config["bives"]["decoder"].get("tau_d", 1.0)),
        tau_p=float(config["bives"]["decoder"].get("tau_p", 1.0)),
        num_controls=int(config["bives"]["interventions"].get("num_controls", 4)),
        control_mode=str(
            config["bives"]["interventions"].get(
                "control_mode", "random_disjoint"
            )
        ),
        contextual_layers=int(config["bives"].get("contextual_layers", 1)),
        contextual_heads=int(config["bives"].get("contextual_heads", 4)),
        contextual_dropout=float(config["bives"].get("contextual_dropout", 0.0)),
    )
    experiment = BiVESExperiment(
        Qwen35VisionAdapter(
            visual,
            spatial_merge_size=int(
                qwen_config["vision_config"]["spatial_merge_size"]
            ),
        ),
        num_statements=len(statement_to_index),
        statement_dim=int(frozen.shape[1]),
        head_config=head_config,
        frozen_statement_embeddings=frozen,
    ).to(device)
    load_checkpoint_model_state(experiment, checkpoint)
    set_decoder_temperatures(
        experiment,
        calibration["calibrated_temperatures"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 8)),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=Qwen35BiVESCollator(
            processor,
            image_size=int(config["data"].get("image_size", 448)),
            include_group_indices=False,
            split="locked_test",
            training_seed=int(config.get("seed", 17)),
            evaluation_control_seed=int(config["evaluation"]["control_seed"]),
        ),
    )
    loss_config = BiVESLossConfig(**config.get("loss", {}))
    metrics, rows = evaluate(
        experiment,
        loader,
        BiVESLoss(replace(loss_config, lambda_pair=0.0, lambda_u_pol=0.0)),
        device,
        assert_full_coverage=True,
        bootstrap_replicates=int(
            config.get("evaluation", {}).get("bootstrap_replicates", 1000)
        ),
        bootstrap_seed=int(config.get("seed", 17)),
    )
    with (args.output_dir / "locked_test_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(
        args.output_dir / "locked_test_metrics.json",
        {
            "release_protocol": "explicit_one_shot_locked_test_v1",
            "control_protocol_version": CONTROL_PROTOCOL_VERSION,
            "evaluation_control_seed": int(run_lock["evaluation_control_seed"]),
            "run_lock_sha256": checkpoint["run_lock_sha256"],
            "metrics": metrics,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "calibration_artifact": str(args.calibration_artifact),
            "calibration_artifact_sha256": file_sha256(args.calibration_artifact),
            "test_manifest": str(args.test_manifest),
            "test_manifest_sha256": file_sha256(args.test_manifest),
            "statement_cache": str(args.statement_cache),
            "statement_cache_sha256": file_sha256(args.statement_cache),
            "git_commit": current_commit,
        },
    )


if __name__ == "__main__":
    main()
