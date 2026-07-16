"""Train VSL-CXR CEQ variants on Qwen3-VL visual patch tokens."""

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
from models.clinical_evidence_query import ClinicalEvidenceClassifier, ClinicalEvidenceQuery
from scripts.export_qwen3vl_instruction_embeddings import get_visual_module
from scripts.train_qwen3vl_clinical_instruction import load_model_and_processor, load_trainable_checkpoint


STATE_TO_CLASS = {"absent": 0, "uncertain": 1, "present": 2}
CLASS_TO_STATE = {value: key for key, value in STATE_TO_CLASS.items()}
DEFAULT_REGIONS = {
    "cardiac": "cardiomediastinal",
    "cardiomediastinal": "cardiomediastinal",
    "perihilar": "lung_fields",
    "lung": "lung_fields",
    "pleural": "pleural",
    "ribs": "osseous",
    "osseous": "osseous",
    "support": "support_devices",
    "clinically relevant": "generic",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            writer.writerow({key: row.get(key, "") for key in columns})


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def summarize_checkpoint_meta(checkpoint: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("global_step", "best_val_loss", "model_path"):
        value = checkpoint.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
    parameter_groups = checkpoint.get("parameter_groups")
    if isinstance(parameter_groups, dict):
        summary["parameter_groups"] = parameter_groups
    state = checkpoint.get("trainable_state_dict")
    if isinstance(state, dict):
        summary["trainable_state_keys"] = len(state)
    return summary


def normalize_region(value: Any) -> str:
    text = str(value or "").strip().lower()
    for key, label in DEFAULT_REGIONS.items():
        if key in text:
            return label
    return "generic"


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


class CEQTargetDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        max_samples: int | None = None,
        fallback_size: int = 448,
        finding_names: list[str] | None = None,
        region_names: list[str] | None = None,
        query_buckets: int = 256,
    ) -> None:
        self.jsonl_path = jsonl_path
        raw_rows = read_jsonl(jsonl_path, max_samples=max_samples)
        self.rows = [row for row in raw_rows if str(row.get("state") or "").lower() in STATE_TO_CLASS and row.get("image_path")]
        if not self.rows:
            raise ValueError(f"No usable CEQ rows found in {jsonl_path}")
        self.finding_names = finding_names or sorted({str(row.get("finding") or "global") for row in self.rows})
        self.finding_to_index = {name: idx for idx, name in enumerate(self.finding_names)}
        self.region_names = region_names or sorted({normalize_region(row.get("expected_region")) for row in self.rows})
        self.region_to_index = {name: idx for idx, name in enumerate(self.region_names)}
        self.query_buckets = int(query_buckets)
        self.fallback_size = int(fallback_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: Path) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - preserve long-run continuity and keep evidence in logs.
            print(f"Error loading image {path}: {exc}")
            return Image.new("RGB", (self.fallback_size, self.fallback_size), (0, 0, 0))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        finding = str(row.get("finding") or "global")
        region = normalize_region(row.get("expected_region"))
        query = str(row.get("evidence_query") or finding)
        return {
            "image": self._load_image(Path(str(row["image_path"]))),
            "finding": self.finding_to_index.get(finding, 0),
            "state": STATE_TO_CLASS[str(row.get("state") or "").lower()],
            "region": self.region_to_index.get(region, 0),
            "query_bucket": stable_bucket(query, self.query_buckets),
            "sample_id": str(row.get("sample_id") or row.get("target_id") or index),
            "finding_name": finding,
            "region_name": region,
        }


class CEQCollator:
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
        encoded["finding"] = torch.tensor([item["finding"] for item in batch], dtype=torch.long)
        encoded["state"] = torch.tensor([item["state"] for item in batch], dtype=torch.long)
        encoded["region"] = torch.tensor([item["region"] for item in batch], dtype=torch.long)
        encoded["query_bucket"] = torch.tensor([item["query_bucket"] for item in batch], dtype=torch.long)
        encoded["sample_id"] = [item["sample_id"] for item in batch]
        return encoded


def move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def pad_patch_tokens(visual_outputs: Any, grid_thw: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = getattr(visual_outputs, "last_hidden_state", visual_outputs)
    counts = grid_thw.prod(dim=1).long().tolist()
    if sum(counts) == int(hidden.shape[0]):
        chunks = torch.split(hidden, counts, dim=0)
    else:
        chunks = torch.chunk(hidden, batch_size, dim=0)
        counts = [int(chunk.shape[0]) for chunk in chunks]
    max_len = max(counts)
    padded = hidden.new_zeros((batch_size, max_len, hidden.shape[-1]))
    mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=hidden.device)
    for idx, chunk in enumerate(chunks):
        length = int(chunk.shape[0])
        padded[idx, :length] = chunk
        mask[idx, :length] = False
    return padded, mask


class VSLCEQHead(nn.Module):
    def __init__(
        self,
        num_findings: int,
        embed_dim: int,
        num_heads: int,
        variant: str,
        num_regions: int,
        query_buckets: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.ceq = ClinicalEvidenceQuery(num_findings=num_findings, embed_dim=embed_dim, num_heads=num_heads)
        self.state_head = ClinicalEvidenceClassifier(embed_dim=embed_dim, num_states=len(STATE_TO_CLASS))
        self.region_head = nn.Linear(embed_dim, num_regions) if variant == "region" else None
        self.query_condition = nn.Embedding(query_buckets, embed_dim) if variant == "statement" else None
        self.statement_attention = (
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            if variant == "statement"
            else None
        )
        self.statement_norm = nn.LayerNorm(embed_dim) if variant == "statement" else None

    def forward(
        self,
        patch_tokens: torch.Tensor,
        finding: torch.Tensor,
        query_bucket: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        patch_tokens = patch_tokens.float()
        if self.variant == "statement":
            base = self.ceq.clinical_queries[finding]
            query = base + self.query_condition(query_bucket)
            attended, attention = self.statement_attention(
                query.unsqueeze(1),
                patch_tokens,
                patch_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            evidence = self.ceq.output(self.statement_norm(attended.squeeze(1) + query))
        else:
            ceq_out = self.ceq(patch_tokens, key_padding_mask=key_padding_mask)
            all_evidence = ceq_out["evidence"]
            attention = ceq_out["attention"][torch.arange(patch_tokens.shape[0], device=patch_tokens.device), finding]
            evidence = all_evidence[torch.arange(patch_tokens.shape[0], device=patch_tokens.device), finding]
        logits = self.state_head(evidence)
        output = {"state_logits": logits, "attention": attention, "evidence": evidence}
        if self.region_head is not None:
            output["region_logits"] = self.region_head(evidence)
        return output

    def diversity_loss(self) -> torch.Tensor:
        queries = F.normalize(self.ceq.clinical_queries.float(), dim=-1)
        sim = queries @ queries.T
        off_diag = sim - torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        return off_diag.pow(2).sum() / max(sim.numel() - sim.shape[0], 1)


def attention_entropy(attention: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    values = attention.float().clamp_min(1e-8)
    if key_padding_mask is not None:
        valid = (~key_padding_mask).float()
        values = values * valid
        denom = valid.sum(dim=-1).clamp_min(2.0).log()
    else:
        denom = torch.full((values.shape[0],), math.log(max(values.shape[-1], 2)), device=values.device)
    entropy = -(values * values.log()).sum(dim=-1) / denom.clamp_min(1e-6)
    return entropy.mean()


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    if y_true.size == 0:
        return {}
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "support": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ece": float(compute_ece(y_true, y_prob, n_bins=10)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    metrics["auprc"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    return metrics


def compute_loss(
    ceq_head: VSLCEQHead,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    key_padding_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    state_loss = F.cross_entropy(outputs["state_logits"], batch["state"])
    total = state_loss
    scalars = {"state_loss": float(state_loss.detach().cpu())}
    if float(loss_cfg.get("diversity_weight", 0.0)) > 0:
        diversity = ceq_head.diversity_loss()
        total = total + float(loss_cfg["diversity_weight"]) * diversity
        scalars["diversity_loss"] = float(diversity.detach().cpu())
    if float(loss_cfg.get("sparsity_weight", 0.0)) > 0:
        sparse = attention_entropy(outputs["attention"], key_padding_mask)
        total = total + float(loss_cfg["sparsity_weight"]) * sparse
        scalars["sparsity_loss"] = float(sparse.detach().cpu())
    if "region_logits" in outputs and float(loss_cfg.get("region_weight", 0.0)) > 0:
        region = F.cross_entropy(outputs["region_logits"], batch["region"])
        total = total + float(loss_cfg["region_weight"]) * region
        scalars["region_loss"] = float(region.detach().cpu())
    scalars["loss"] = float(total.detach().cpu())
    return total, scalars


@torch.no_grad()
def evaluate(
    base_model: torch.nn.Module,
    ceq_head: VSLCEQHead,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: dict[str, Any],
) -> dict[str, Any]:
    base_model.eval()
    ceq_head.eval()
    visual = get_visual_module(base_model)
    losses = []
    correct = 0
    total = 0
    binary_true = []
    binary_prob = []
    region_correct = 0
    region_total = 0
    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = move_tensors_to_device(batch, device)
        visual_outputs = visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
        patch_tokens, key_padding_mask = pad_patch_tokens(visual_outputs, batch["image_grid_thw"], int(batch["state"].shape[0]))
        outputs = ceq_head(patch_tokens, batch["finding"], batch["query_bucket"], key_padding_mask)
        loss, _ = compute_loss(ceq_head, outputs, batch, key_padding_mask, loss_cfg)
        losses.append(float(loss.detach().cpu()))
        pred = outputs["state_logits"].argmax(dim=-1)
        correct += int((pred == batch["state"]).sum().item())
        total += int(batch["state"].numel())
        binary_mask = (batch["state"] == STATE_TO_CLASS["absent"]) | (batch["state"] == STATE_TO_CLASS["present"])
        if bool(binary_mask.any()):
            binary_true.append((batch["state"][binary_mask] == STATE_TO_CLASS["present"]).float().cpu().numpy())
            binary_prob.append(torch.softmax(outputs["state_logits"][binary_mask], dim=-1)[:, STATE_TO_CLASS["present"]].cpu().numpy())
        if "region_logits" in outputs:
            region_pred = outputs["region_logits"].argmax(dim=-1)
            region_correct += int((region_pred == batch["region"]).sum().item())
            region_total += int(batch["region"].numel())
    y_true = np.concatenate(binary_true).astype(int) if binary_true else np.asarray([], dtype=int)
    y_prob = np.concatenate(binary_prob) if binary_prob else np.asarray([], dtype=np.float32)
    result = {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "state_accuracy": float(correct / max(total, 1)),
        "binary": binary_metrics(y_true, y_prob),
    }
    if region_total:
        result["region_accuracy"] = float(region_correct / max(region_total, 1))
    ceq_head.train()
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_run_config(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    write_json(output_dir / "config_snapshot.json", {"source_config": str(config_path), "resolved_config": config})


def create_dataloaders(config: dict[str, Any], processor: Any) -> tuple[DataLoader, DataLoader, CEQTargetDataset]:
    data_cfg = config["data"]
    query_buckets = int(config.get("ceq", {}).get("query_buckets", 256))
    train_dataset = CEQTargetDataset(
        Path(data_cfg["train_ceq_path"]),
        max_samples=data_cfg.get("max_train_samples"),
        query_buckets=query_buckets,
    )
    val_dataset = CEQTargetDataset(
        Path(data_cfg["val_ceq_path"]),
        max_samples=data_cfg.get("max_val_samples"),
        finding_names=train_dataset.finding_names,
        region_names=train_dataset.region_names,
        query_buckets=query_buckets,
    )
    prompt = str(data_cfg.get("processor_prompt", "Find visual evidence for this chest X-ray."))
    collator = CEQCollator(processor, prompt)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("eval_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collator,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset


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
    checkpoint = config.get("model", {}).get("vision_checkpoint")
    checkpoint_meta: dict[str, Any] = {}
    if checkpoint:
        checkpoint_meta = summarize_checkpoint_meta(load_trainable_checkpoint(Path(checkpoint), base_model, device))
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    train_loader, val_loader, train_dataset = create_dataloaders(config, processor)
    visual = get_visual_module(base_model)
    sample_batch = move_tensors_to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        sample_outputs = visual(sample_batch["pixel_values"], grid_thw=sample_batch["image_grid_thw"])
    embed_dim = int(getattr(sample_outputs, "last_hidden_state", sample_outputs).shape[-1])

    ceq_cfg = config.get("ceq", {})
    variant = str(ceq_cfg.get("variant", "basic")).lower()
    ceq_head = VSLCEQHead(
        num_findings=len(train_dataset.finding_names),
        embed_dim=embed_dim,
        num_heads=int(ceq_cfg.get("num_heads", 8)),
        variant=variant,
        num_regions=len(train_dataset.region_names),
        query_buckets=int(ceq_cfg.get("query_buckets", 256)),
    ).to(device)
    optimizer = AdamW(
        ceq_head.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    loss_cfg = config.get("losses", {})
    max_steps = int(config["training"]["max_steps"])
    eval_interval = int(config["training"].get("eval_interval", 100))
    log_interval = int(config["training"].get("log_interval", 25))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    best_val_loss = float("inf")
    global_step = 0
    started = time.time()
    events: list[dict[str, Any]] = []
    ceq_head.train()
    while global_step < max_steps:
        for batch in train_loader:
            batch = move_tensors_to_device(batch, device)
            with torch.no_grad():
                visual_outputs = visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])
                patch_tokens, key_padding_mask = pad_patch_tokens(visual_outputs, batch["image_grid_thw"], int(batch["state"].shape[0]))
            outputs = ceq_head(patch_tokens, batch["finding"], batch["query_bucket"], key_padding_mask)
            loss, scalars = compute_loss(ceq_head, outputs, batch, key_padding_mask, loss_cfg)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(ceq_head.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % log_interval == 0:
                print(json.dumps({"global_step": global_step, **scalars}, ensure_ascii=False))
            if global_step % eval_interval == 0:
                result = evaluate(base_model, ceq_head, val_loader, device, loss_cfg)
                event = {"global_step": global_step, **scalars, **result}
                events.append(event)
                print(json.dumps(event, ensure_ascii=False, default=to_jsonable))
                write_json(output_dir / f"metrics_step_{global_step}.json", event)
                if result["val_loss"] < best_val_loss:
                    best_val_loss = float(result["val_loss"])
                    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                    torch.save({"ceq_head": ceq_head.state_dict(), "finding_names": train_dataset.finding_names, "region_names": train_dataset.region_names}, output_dir / "checkpoints" / "best.pt")
                ceq_head.train()
            if global_step >= max_steps:
                break
    final = evaluate(base_model, ceq_head, val_loader, device, loss_cfg)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"ceq_head": ceq_head.state_dict(), "finding_names": train_dataset.finding_names, "region_names": train_dataset.region_names}, output_dir / "checkpoints" / "final.pt")
    metrics = {
        "global_step": global_step,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else final["val_loss"],
        "final_val_loss": final["val_loss"],
        "elapsed_seconds": time.time() - started,
        "train_records": len(train_loader.dataset),
        "val_records": len(val_loader.dataset),
        "feature_dim": embed_dim,
        "variant": variant,
        "finding_names": train_dataset.finding_names,
        "region_names": train_dataset.region_names,
        "checkpoint_meta": checkpoint_meta,
        "final": final,
        "events": events,
    }
    write_json(output_dir / "metrics_final.json", metrics)
    lines = [json.dumps(event, ensure_ascii=False, default=to_jsonable) for event in events]
    (output_dir / "training_log.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    columns = ["metric", "value"]
    summary_rows = [
        {"metric": "variant", "value": variant},
        {"metric": "global_step", "value": global_step},
        {"metric": "best_val_loss", "value": metrics["best_val_loss"]},
        {"metric": "final_val_loss", "value": final["val_loss"]},
        {"metric": "state_accuracy", "value": final.get("state_accuracy")},
        {"metric": "binary_auc", "value": (final.get("binary") or {}).get("auc")},
        {"metric": "binary_auprc", "value": (final.get("binary") or {}).get("auprc")},
        {"metric": "binary_f1", "value": (final.get("binary") or {}).get("f1")},
        {"metric": "region_accuracy", "value": final.get("region_accuracy", "")},
    ]
    write_csv(output_dir / "summary.csv", summary_rows, columns)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=to_jsonable))


if __name__ == "__main__":
    main()
