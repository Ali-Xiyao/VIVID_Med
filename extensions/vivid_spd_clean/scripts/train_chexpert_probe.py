"""Frozen-ViT CheXpert expert-development linear probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vivid_spd_clean.io import sha256_file  # noqa: E402


FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)


def patient_bucket(patient_id: str) -> int:
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def parse_label(value: str) -> float:
    text = str(value or "").strip()
    if text in {"0", "0.0"}:
        return 0.0
    if text in {"1", "1.0"}:
        return 1.0
    return float("nan")


class CheXpertRows(Dataset):
    def __init__(self, manifest: Path, image_root: Path) -> None:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        path = self.image_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "image": pixels,
            "labels": torch.tensor(
                [parse_label(row[finding]) for finding in FINDINGS],
                dtype=torch.float32,
            ),
            "patient_id": row["patient_id"],
            "image_path": row["image_path"],
        }


def collate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([row["image"] for row in rows]),
        "labels": torch.stack([row["labels"] for row in rows]),
        "patient_ids": [str(row["patient_id"]) for row in rows],
        "image_paths": [str(row["image_path"]) for row in rows],
    }


def load_backbone(checkpoint: Path) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("vision") is None or payload["vision"].get("backbone") is None:
        raise ValueError("checkpoint lacks vision.backbone")
    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    incompatible = model.load_state_dict(payload["vision"]["backbone"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            {
                "missing": incompatible.missing_keys,
                "unexpected": incompatible.unexpected_keys,
            }
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    dataset: CheXpertRows,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    model.eval().to(device)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    patients: list[str] = []
    paths: list[str] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = model.forward_features(images)
        if tokens.ndim != 3:
            raise ValueError("ViT forward_features must return token sequence")
        features.append(tokens[:, 0].float().cpu())
        labels.append(batch["labels"])
        patients.extend(batch["patient_ids"])
        paths.extend(batch["image_paths"])
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "patient_ids": patients,
        "image_paths": paths,
    }


def masked_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    mask = torch.isfinite(labels)
    targets = torch.nan_to_num(labels, nan=0.0)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return losses[mask].mean()


@torch.inference_mode()
def internal_nll(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> float:
    head.eval()
    return float(
        masked_bce(
            head(features.to(device)),
            labels.to(device),
            pos_weight,
        )
    )


def metrics(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, object]:
    result: dict[str, object] = {"per_finding": {}}
    aucs = []
    auprcs = []
    for index, finding in enumerate(FINDINGS):
        valid = torch.isfinite(labels[:, index])
        truth = labels[valid, index].numpy()
        probability = scores[valid, index].numpy()
        if len(np.unique(truth)) != 2:
            raise ValueError(f"expert labels are degenerate for {finding}")
        auc = float(roc_auc_score(truth, probability))
        auprc = float(average_precision_score(truth, probability))
        result["per_finding"][finding] = {
            "auroc": auc,
            "auprc": auprc,
            "rows": int(valid.sum()),
            "positive": int(truth.sum()),
        }
        aucs.append(auc)
        auprcs.append(auprc)
    result["macro_auroc"] = float(np.mean(aucs))
    result["macro_auprc"] = float(np.mean(auprcs))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--vision-checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--expert-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("linear probe requires CUDA")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    backbone = load_backbone(args.vision_checkpoint)
    train_rows = CheXpertRows(args.train_manifest, args.image_root)
    expert_rows = CheXpertRows(args.expert_manifest, args.image_root)
    train_payload = extract_features(
        backbone,
        train_rows,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    expert_payload = extract_features(
        backbone,
        expert_rows,
        device=device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    del backbone
    torch.cuda.empty_cache()
    buckets = torch.tensor(
        [patient_bucket(value) for value in train_payload["patient_ids"]]
    )
    train_mask = buckets != 0
    validation_mask = buckets == 0
    if not bool(train_mask.any()) or not bool(validation_mask.any()):
        raise ValueError("deterministic probe train/validation split is empty")
    train_features = train_payload["features"][train_mask]
    train_labels = train_payload["labels"][train_mask]
    validation_features = train_payload["features"][validation_mask]
    validation_labels = train_payload["labels"][validation_mask]
    observed = torch.isfinite(train_labels)
    positives = torch.where(observed, train_labels, 0.0).sum(dim=0)
    negatives = observed.sum(dim=0) - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
    head = nn.Linear(train_features.shape[1], len(FINDINGS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    records = []
    best_nll = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(
                head(features.to(device)),
                labels.to(device),
                pos_weight,
            )
            loss.backward()
            optimizer.step()
        value = internal_nll(
            head,
            validation_features,
            validation_labels,
            pos_weight,
            device,
        )
        records.append({"epoch": epoch, "internal_validation_nll": value})
        if value < best_nll:
            best_nll = value
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
    if best_state is None:
        raise AssertionError("no linear-probe checkpoint selected")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        expert_scores = torch.sigmoid(
            head(expert_payload["features"].to(device))
        ).cpu()
    expert_metrics = metrics(expert_payload["labels"], expert_scores)
    torch.save(
        {
            "head": best_state,
            "findings": FINDINGS,
            "best_epoch": best_epoch,
            "internal_validation_nll": best_nll,
            "arm": args.arm,
        },
        args.output_dir / "best_probe.pt",
    )
    prediction_path = args.output_dir / "expert_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["patient_id", "image_path"]
        for finding in FINDINGS:
            fields.extend([f"{finding}_label", f"{finding}_score"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, patient_id in enumerate(expert_payload["patient_ids"]):
            row: dict[str, object] = {
                "patient_id": patient_id,
                "image_path": expert_payload["image_paths"][row_index],
            }
            for finding_index, finding in enumerate(FINDINGS):
                row[f"{finding}_label"] = float(
                    expert_payload["labels"][row_index, finding_index]
                )
                row[f"{finding}_score"] = float(
                    expert_scores[row_index, finding_index]
                )
            writer.writerow(row)
    summary = {
        "schema_version": 1,
        "artifact": "strict_vivid_spd_chexpert_probe",
        "arm": args.arm,
        "pass": True,
        "checkpoint_rule": "minimum internal probe-train patient-split NLL",
        "best_epoch": best_epoch,
        "internal_validation_nll": best_nll,
        "expert_development": expert_metrics,
        "rows": {
            "head_train": int(train_mask.sum()),
            "head_validation": int(validation_mask.sum()),
            "expert_development": len(expert_rows),
        },
        "hashes": {
            "vision_checkpoint": sha256_file(args.vision_checkpoint),
            "train_manifest": sha256_file(args.train_manifest),
            "expert_manifest": sha256_file(args.expert_manifest),
            "predictions": sha256_file(prediction_path),
        },
        "budget": {
            "seed": args.seed,
            "epochs": args.epochs,
            "feature_batch_size": args.feature_batch_size,
            "head_batch_size": args.head_batch_size,
        },
        "elapsed_seconds": time.time() - started,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
