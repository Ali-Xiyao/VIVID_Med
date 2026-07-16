"""Train VSL-CXR CCSH/AUCH deployable readout variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_ece
from models.answerability_uncertainty_head import AnswerabilityUncertaintyHead
from models.clinical_consistency_head import ClinicalConsistencyHead
from scripts.export_qwen3vl_instruction_embeddings import get_visual_module
from scripts.train_qwen3vl_clinical_instruction import load_model_and_processor, load_trainable_checkpoint
from scripts.train_vsl_ceq import VSLCEQHead, pad_patch_tokens, stable_bucket, summarize_checkpoint_meta


STATE_TO_CLASS = {"absent": 0, "uncertain": 1, "present": 2}
LABEL_TO_INDEX = {"contradict": 0, "support": 1, "uncertain": 2}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=to_jsonable), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class CCSHPairDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        max_samples: int | None = None,
        finding_names: list[str] | None = None,
        statement_buckets: int = 512,
        fallback_size: int = 448,
    ) -> None:
        raw_rows = read_jsonl(jsonl_path, max_samples=max_samples)
        self.rows = [
            row for row in raw_rows
            if row.get("image_path") and row.get("statement") and row.get("binary_label") in {0, 1}
        ]
        if not self.rows:
            raise ValueError(f"No usable CCSH rows found in {jsonl_path}")
        self.finding_names = finding_names or sorted({str(row.get("finding") or "global") for row in self.rows})
        self.finding_to_index = {name: idx for idx, name in enumerate(self.finding_names)}
        self.statement_buckets = int(statement_buckets)
        self.fallback_size = int(fallback_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading image {path}: {exc}")
            return Image.new("RGB", (self.fallback_size, self.fallback_size), (0, 0, 0))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        finding = str(row.get("finding") or "global")
        statement = str(row.get("statement") or finding)
        label = str(row.get("label") or "").lower()
        state = str(row.get("state") or "").lower()
        return {
            "image": self._load_image(Path(str(row["image_path"]))),
            "finding": self.finding_to_index.get(finding, 0),
            "statement_bucket": stable_bucket(statement, self.statement_buckets),
            "pair_label": int(row["binary_label"]),
            "label_type": LABEL_TO_INDEX.get(label, 0),
            "state": STATE_TO_CLASS.get(state, 1),
            "sample_id": str(row.get("sample_id") or row.get("statement_id") or index),
        }


class CCSHCollator:
    def __init__(self, processor: Any, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        images = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append(item["image"])
        encoded = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        for key in ("finding", "statement_bucket", "pair_label", "label_type", "state"):
            encoded[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        encoded["sample_id"] = [item["sample_id"] for item in batch]
        return encoded


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def pool_patch_tokens(patch_tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~key_padding_mask).float().unsqueeze(-1)
    return (patch_tokens.float() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class VSLCCSHReadout(nn.Module):
    def __init__(
        self,
        image_dim: int,
        num_findings: int,
        statement_buckets: int,
        variant: str,
        ceq_head: VSLCEQHead | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.ceq_head = ceq_head
        self.statement_embed = nn.Embedding(statement_buckets, image_dim)
        self.finding_embed = nn.Embedding(num_findings, image_dim)
        self.auch = AnswerabilityUncertaintyHead(image_dim) if "auch" in variant else None
        self.auch_proj = nn.Linear(5, image_dim) if self.auch is not None else None
        self.ccsh = ClinicalConsistencyHead(image_dim=image_dim, statement_dim=image_dim, num_classes=2)

    def image_embedding(
        self,
        patch_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        finding: torch.Tensor,
        statement_bucket: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.ceq_head is not None:
            ceq_out = self.ceq_head(patch_tokens, finding, statement_bucket, key_padding_mask)
            image = ceq_out["evidence"]
        else:
            image = pool_patch_tokens(patch_tokens, key_padding_mask)
        if self.auch is not None and self.auch_proj is not None:
            auch_out = self.auch(image)
            aux.update(auch_out)
            probs = torch.softmax(auch_out["state_logits"], dim=-1)
            calibration = torch.cat(
                [
                    torch.sigmoid(auch_out["answerability_logit"]).unsqueeze(-1),
                    torch.sigmoid(auch_out["uncertainty_logit"]).unsqueeze(-1),
                    probs,
                ],
                dim=-1,
            )
            image = image + self.auch_proj(calibration)
        return image, aux

    def forward(
        self,
        patch_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        finding: torch.Tensor,
        statement_bucket: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image, aux = self.image_embedding(patch_tokens.float(), key_padding_mask, finding, statement_bucket)
        statement = self.statement_embed(statement_bucket) + self.finding_embed(finding)
        logits = self.ccsh(image, statement)
        return {"pair_logits": logits, **aux}


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "support": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "ece": float(compute_ece(y_true, y_prob, n_bins=10)),
    }


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], auch_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    pair_loss = F.cross_entropy(outputs["pair_logits"], batch["pair_label"])
    total = pair_loss
    scalars = {"pair_loss": float(pair_loss.detach().cpu())}
    if "state_logits" in outputs and auch_weight > 0:
        state_loss = F.cross_entropy(outputs["state_logits"], batch["state"])
        answerable = (batch["state"] != STATE_TO_CLASS["uncertain"]).float()
        uncertain = (batch["state"] == STATE_TO_CLASS["uncertain"]).float()
        ans_loss = F.binary_cross_entropy_with_logits(outputs["answerability_logit"], answerable)
        unc_loss = F.binary_cross_entropy_with_logits(outputs["uncertainty_logit"], uncertain)
        auch_loss = state_loss + ans_loss + unc_loss
        total = total + auch_weight * auch_loss
        scalars.update(
            {
                "auch_state_loss": float(state_loss.detach().cpu()),
                "auch_answerability_loss": float(ans_loss.detach().cpu()),
                "auch_uncertainty_loss": float(unc_loss.detach().cpu()),
            }
        )
    scalars["loss"] = float(total.detach().cpu())
    return total, scalars


@torch.no_grad()
def evaluate(base_model: torch.nn.Module, readout: VSLCCSHReadout, loader: DataLoader, device: torch.device, auch_weight: float) -> dict[str, Any]:
    base_model.eval()
    readout.eval()
    visual = get_visual_module(base_model)
    losses = []
    labels = []
    probs = []
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = move_tensors_to_device(batch, device)
        visual_outputs = visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
        patch_tokens, key_padding_mask = pad_patch_tokens(visual_outputs, batch["image_grid_thw"], int(batch["pair_label"].shape[0]))
        outputs = readout(patch_tokens, key_padding_mask, batch["finding"], batch["statement_bucket"])
        loss, _ = compute_loss(outputs, batch, auch_weight)
        losses.append(float(loss.detach().cpu()))
        labels.append(batch["pair_label"].cpu().numpy())
        probs.append(torch.softmax(outputs["pair_logits"], dim=-1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels).astype(int)
    y_prob = np.concatenate(probs)
    readout.train()
    return {"val_loss": float(np.mean(losses)), "binary": binary_metrics(y_true, y_prob)}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_run_config(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    write_json(output_dir / "config_snapshot.json", {"source_config": str(config_path), "resolved_config": config})


def create_dataloaders(config: dict[str, Any], processor: Any) -> tuple[DataLoader, DataLoader, CCSHPairDataset]:
    data_cfg = config["data"]
    buckets = int(config.get("ccsh", {}).get("statement_buckets", 512))
    train_dataset = CCSHPairDataset(Path(data_cfg["train_ccsh_path"]), data_cfg.get("max_train_samples"), statement_buckets=buckets)
    val_dataset = CCSHPairDataset(
        Path(data_cfg["val_ccsh_path"]),
        data_cfg.get("max_val_samples"),
        finding_names=train_dataset.finding_names,
        statement_buckets=buckets,
    )
    collator = CCSHCollator(processor, str(data_cfg.get("processor_prompt", "Represent this chest X-ray for clinical consistency scoring.")))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    return train_loader, val_loader, train_dataset


def load_ceq_head(config: dict[str, Any], embed_dim: int, device: torch.device) -> VSLCEQHead | None:
    ccsh_cfg = config.get("ccsh", {})
    path = ccsh_cfg.get("ceq_checkpoint")
    if not path:
        return None
    checkpoint = torch.load(Path(path), map_location="cpu")
    finding_names = checkpoint.get("finding_names") or []
    region_names = checkpoint.get("region_names") or ["generic"]
    variant = str(ccsh_cfg.get("ceq_variant", "region"))
    ceq_head = VSLCEQHead(
        num_findings=len(finding_names),
        embed_dim=embed_dim,
        num_heads=int(ccsh_cfg.get("ceq_num_heads", 8)),
        variant=variant,
        num_regions=len(region_names),
        query_buckets=int(ccsh_cfg.get("statement_buckets", 512)),
    )
    ceq_head.load_state_dict(checkpoint["ceq_head"], strict=False)
    for param in ceq_head.parameters():
        param.requires_grad = False
    ceq_head.to(device)
    ceq_head.eval()
    return ceq_head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def apply_debug_overrides(config: dict[str, Any]) -> None:
    config["data"]["max_train_samples"] = min(int(config["data"].get("max_train_samples") or 8), 8)
    config["data"]["max_val_samples"] = min(int(config["data"].get("max_val_samples") or 4), 4)
    config["training"]["batch_size"] = 1
    config["training"]["eval_batch_size"] = 1
    config["training"]["max_steps"] = min(int(config["training"].get("max_steps", 2)), 2)
    config["training"]["eval_interval"] = 1
    base_output = str(config["training"]["output_dir"]).rstrip("/\\")
    if not base_output.endswith("_debug"):
        config["training"]["output_dir"] = f"{base_output}_debug"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.debug:
        apply_debug_overrides(config)
    set_seed(int(config.get("seed", 42)))
    requested_device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    output_dir = Path(config["training"]["output_dir"])
    if (output_dir / "metrics_final.json").exists():
        raise SystemExit(f"{output_dir / 'metrics_final.json'} already exists; remove it manually to rerun.")
    save_run_config(config, args.config, output_dir)

    base_model, processor = load_model_and_processor(config, device)
    checkpoint_meta: dict[str, Any] = {}
    checkpoint = config.get("model", {}).get("vision_checkpoint")
    if checkpoint:
        checkpoint_meta = summarize_checkpoint_meta(load_trainable_checkpoint(Path(checkpoint), base_model, device))
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    train_loader, val_loader, train_dataset = create_dataloaders(config, processor)
    visual = get_visual_module(base_model)
    sample = move_tensors_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        sample_outputs = visual(sample["pixel_values"], grid_thw=sample["image_grid_thw"])
    embed_dim = int(getattr(sample_outputs, "last_hidden_state", sample_outputs).shape[-1])

    ceq_head = load_ceq_head(config, embed_dim, device)
    variant = str(config.get("ccsh", {}).get("variant", "ccsh")).lower()
    readout = VSLCCSHReadout(
        image_dim=embed_dim,
        num_findings=len(train_dataset.finding_names),
        statement_buckets=int(config.get("ccsh", {}).get("statement_buckets", 512)),
        variant=variant,
        ceq_head=ceq_head,
    ).to(device)
    optimizer = AdamW(readout.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.01)))
    auch_weight = float(config.get("losses", {}).get("auch_weight", 0.0))
    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"].get("eval_interval", 100))
    log_interval = int(config["training"].get("log_interval", 25))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    best_val_loss = float("inf")
    global_step = 0
    started = time.time()
    events: list[dict[str, Any]] = []
    while global_step < max_steps:
        for batch in train_loader:
            batch = move_tensors_to_device(batch, device)
            with torch.no_grad():
                visual_outputs = visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
                patch_tokens, key_padding_mask = pad_patch_tokens(visual_outputs, batch["image_grid_thw"], int(batch["pair_label"].shape[0]))
            outputs = readout(patch_tokens, key_padding_mask, batch["finding"], batch["statement_bucket"])
            loss, scalars = compute_loss(outputs, batch, auch_weight)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(readout.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % log_interval == 0:
                print(json.dumps({"global_step": global_step, **scalars}, ensure_ascii=False))
            if global_step % eval_interval == 0:
                result = evaluate(base_model, readout, val_loader, device, auch_weight)
                event = {"global_step": global_step, **scalars, **result}
                events.append(event)
                print(json.dumps(event, ensure_ascii=False, default=to_jsonable))
                write_json(output_dir / f"metrics_step_{global_step}.json", event)
                if result["val_loss"] < best_val_loss:
                    best_val_loss = float(result["val_loss"])
                    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                    torch.save({"readout": readout.state_dict(), "finding_names": train_dataset.finding_names, "variant": variant}, output_dir / "checkpoints" / "best.pt")
                readout.train()
            if global_step >= max_steps:
                break
    final = evaluate(base_model, readout, val_loader, device, auch_weight)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"readout": readout.state_dict(), "finding_names": train_dataset.finding_names, "variant": variant}, output_dir / "checkpoints" / "final.pt")
    metrics = {
        "global_step": global_step,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else final["val_loss"],
        "final_val_loss": final["val_loss"],
        "elapsed_seconds": time.time() - started,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset),
        "feature_dim": embed_dim,
        "variant": variant,
        "checkpoint_meta": checkpoint_meta,
        "final": final,
        "events": events,
    }
    write_json(output_dir / "metrics_final.json", metrics)
    (output_dir / "training_log.txt").write_text("\n".join(json.dumps(event, ensure_ascii=False, default=to_jsonable) for event in events) + "\n", encoding="utf-8")
    summary_rows = [
        {"metric": "variant", "value": variant},
        {"metric": "global_step", "value": global_step},
        {"metric": "best_val_loss", "value": metrics["best_val_loss"]},
        {"metric": "final_val_loss", "value": final["val_loss"]},
        {"metric": "binary_auc", "value": final["binary"].get("auc")},
        {"metric": "binary_auprc", "value": final["binary"].get("auprc")},
        {"metric": "binary_f1", "value": final["binary"].get("f1")},
        {"metric": "binary_ece", "value": final["binary"].get("ece")},
    ]
    write_csv(output_dir / "summary.csv", summary_rows, ["metric", "value"])
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=to_jsonable))


if __name__ == "__main__":
    main()
