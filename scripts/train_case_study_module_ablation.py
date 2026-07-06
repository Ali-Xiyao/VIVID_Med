"""Train formal CEQ/AUCH/HNMB/DRA/CCSH/CDCS ablations on exported embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from evaluation.metrics import compute_ece
from models.answerability_uncertainty_head import AnswerabilityUncertaintyHead
from models.case_driven_curriculum_scheduler import CaseDrivenCurriculumScheduler
from models.clinical_consistency_head import ClinicalConsistencyHead
from models.clinical_evidence_query import ClinicalEvidenceClassifier, ClinicalEvidenceQuery
from models.domain_robust_adapter import DomainRobustAdapter, coral_loss
from models.hard_negative_memory_bank import HardNegativeMemoryBank

STATE_TO_CLASS = {"absent": 0, "uncertain": 1, "present": 2}
CLASS_TO_BINARY = {0: 0, 2: 1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_embeddings(path: Path) -> np.ndarray:
    payload = np.load(path)
    key = "embeddings" if "embeddings" in payload else payload.files[0]
    return np.asarray(payload[key], dtype=np.float32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


class EmbeddingStateDataset(Dataset):
    def __init__(
        self,
        embedding_path: Path,
        metadata_path: Path,
        label_names: list[str] | None = None,
        case_rows: list[dict[str, Any]] | None = None,
        use_cdcs_weights: bool = False,
    ) -> None:
        embeddings = read_embeddings(embedding_path)
        metadata = read_jsonl(metadata_path)
        if len(embeddings) != len(metadata):
            raise ValueError(f"Embedding/metadata length mismatch: {len(embeddings)} vs {len(metadata)}")
        self.label_names = label_names or sorted({str(row.get("finding") or "global") for row in metadata})
        self.label_to_index = {name: idx for idx, name in enumerate(self.label_names)}
        scheduler = CaseDrivenCurriculumScheduler()
        failure_stats = scheduler.summarize_failures(case_rows or [])
        rows = []
        for idx, row in enumerate(metadata):
            state = str(row.get("state") or "").lower()
            finding = str(row.get("finding") or "global")
            if state not in STATE_TO_CLASS or finding not in self.label_to_index:
                continue
            weight = 1.0
            if use_cdcs_weights:
                weight = scheduler.sample_weight(row, failure_stats)
            rows.append(
                {
                    "embedding": embeddings[idx],
                    "state": STATE_TO_CLASS[state],
                    "finding": self.label_to_index[finding],
                    "finding_name": finding,
                    "sample_id": row.get("sample_id") or row.get("instruction_id") or str(idx),
                    "answerability": 1.0,
                    "uncertain": 1.0 if state == "uncertain" else 0.0,
                    "weight": float(weight),
                }
            )
        if not rows:
            raise ValueError(f"No usable state rows found in {metadata_path}")
        self.rows = rows
        self.embedding_dim = int(np.asarray(rows[0]["embedding"]).shape[-1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "embedding": torch.tensor(row["embedding"], dtype=torch.float32),
            "state": torch.tensor(row["state"], dtype=torch.long),
            "finding": torch.tensor(row["finding"], dtype=torch.long),
            "answerability": torch.tensor(row["answerability"], dtype=torch.float32),
            "uncertain": torch.tensor(row["uncertain"], dtype=torch.float32),
            "weight": torch.tensor(row["weight"], dtype=torch.float32),
            "sample_id": row["sample_id"],
        }


class ModuleAblationModel(nn.Module):
    def __init__(self, module_name: str, embedding_dim: int, num_findings: int, hidden_dim: int | None = None):
        super().__init__()
        self.module_name = module_name.upper()
        hidden_dim = hidden_dim or embedding_dim
        self.finding_embed = nn.Embedding(num_findings, embedding_dim)
        if self.module_name == "CEQ":
            self.ceq = ClinicalEvidenceQuery(num_findings=num_findings, embed_dim=embedding_dim, num_heads=8)
            self.state_head = ClinicalEvidenceClassifier(embed_dim=embedding_dim, num_states=3)
        elif self.module_name == "AUCH":
            self.auch = AnswerabilityUncertaintyHead(embed_dim=embedding_dim, hidden_dim=hidden_dim, num_states=3)
        elif self.module_name == "DRA":
            self.adapter = DomainRobustAdapter(embed_dim=embedding_dim, hidden_dim=hidden_dim, domain_count=1)
            self.state_head = nn.Linear(embedding_dim, 3)
        elif self.module_name == "CCSH":
            self.ccsh = ClinicalConsistencyHead(image_dim=embedding_dim, statement_dim=embedding_dim, hidden_dim=hidden_dim, num_classes=3)
        else:
            self.state_head = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, embedding: torch.Tensor, finding: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.module_name == "CEQ":
            patch_tokens = embedding.unsqueeze(1)
            evidence = self.ceq(patch_tokens)["evidence"]
            all_logits = self.state_head(evidence)
            logits = all_logits[torch.arange(embedding.shape[0], device=embedding.device), finding]
            return {"state_logits": logits}
        if self.module_name == "AUCH":
            outputs = self.auch(embedding)
            return {"state_logits": outputs["state_logits"], **outputs}
        if self.module_name == "DRA":
            adapted = self.adapter(embedding, domain_id=0)
            return {"state_logits": self.state_head(adapted), "adapted": adapted}
        if self.module_name == "CCSH":
            statement = self.finding_embed(finding)
            return {"state_logits": self.ccsh(embedding, statement)}
        return {"state_logits": self.state_head(embedding)}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["embedding", "state", "finding", "answerability", "uncertain", "weight"]
    output = {key: torch.stack([item[key] for item in batch]) for key in keys}
    output["sample_id"] = [item["sample_id"] for item in batch]
    return output


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def weighted_state_loss(logits: torch.Tensor, state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, state, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def batch_hnmb_margin(
    model: ModuleAblationModel,
    bank: HardNegativeMemoryBank,
    batch: dict[str, Any],
    device: torch.device,
    margin: float,
) -> torch.Tensor:
    losses = []
    embeddings = batch["embedding"]
    states = batch["state"]
    findings = batch["finding"]
    for idx in range(embeddings.shape[0]):
        state = int(states[idx].detach().cpu())
        if state not in CLASS_TO_BINARY:
            continue
        mined = bank.mine(
            embeddings[idx].detach().cpu(),
            sample_id=batch["sample_id"][idx],
            top_k=1,
            finding=str(int(findings[idx].detach().cpu())),
            state=str(state),
        )
        if not mined:
            continue
        negative = torch.tensor(mined[0]["vector"], dtype=torch.float32, device=device).unsqueeze(0)
        finding = findings[idx].view(1)
        pos_logits = model(embeddings[idx].view(1, -1), finding)["state_logits"]
        neg_logits = model(negative, finding)["state_logits"]
        pos_score = pos_logits[:, 2] - pos_logits[:, 0] if state == 2 else pos_logits[:, 0] - pos_logits[:, 2]
        neg_score = neg_logits[:, 2] - neg_logits[:, 0] if state == 2 else neg_logits[:, 0] - neg_logits[:, 2]
        losses.append(torch.relu(margin - pos_score + neg_score).mean())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def build_memory_bank(dataset: EmbeddingStateDataset) -> HardNegativeMemoryBank:
    bank = HardNegativeMemoryBank()
    for row in dataset.rows:
        bank.add(
            str(row["sample_id"]),
            torch.tensor(row["embedding"], dtype=torch.float32),
            finding=str(row["finding"]),
            state=str(row["state"]),
            vector=np.asarray(row["embedding"], dtype=np.float32).tolist(),
        )
    return bank


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "support": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "brier": float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else None,
        "ece": float(compute_ece(y_true, y_prob, n_bins=10)) if len(y_true) else None,
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        metrics["auprc"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        metrics["auc"] = None
        metrics["auprc"] = None
    return metrics


@torch.no_grad()
def evaluate(model: ModuleAblationModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses = []
    states = []
    probs = []
    correct = []
    total = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["embedding"], batch["finding"])
        logits = outputs["state_logits"]
        loss = weighted_state_loss(logits, batch["state"], batch["weight"])
        losses.append(float(loss.detach().cpu()))
        pred = logits.argmax(dim=-1)
        correct.append((pred == batch["state"]).float().sum().item())
        total += int(batch["state"].numel())
        binary_mask = (batch["state"] == 0) | (batch["state"] == 2)
        if binary_mask.any():
            binary_state = (batch["state"][binary_mask] == 2).float()
            binary_prob = torch.softmax(logits[binary_mask], dim=-1)[:, 2]
            states.append(binary_state.cpu().numpy())
            probs.append(binary_prob.cpu().numpy())
    y_true = np.concatenate(states) if states else np.asarray([], dtype=np.float32)
    y_prob = np.concatenate(probs) if probs else np.asarray([], dtype=np.float32)
    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        "state_accuracy": float(sum(correct) / max(total, 1)),
        "binary": binary_metrics(y_true, y_prob),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=["CEQ", "AUCH", "HNMB", "DRA", "CCSH", "CDCS"])
    parser.add_argument("--train-embeddings", required=True, type=Path)
    parser.add_argument("--train-metadata", required=True, type=Path)
    parser.add_argument("--val-embeddings", required=True, type=Path)
    parser.add_argument("--val-metadata", required=True, type=Path)
    parser.add_argument("--target-embeddings", type=Path, help="Optional target-domain embeddings for DRA alignment.")
    parser.add_argument("--case-csv", type=Path, default=Path("outputs/final_tables/case_study_summary.csv"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dra-coral-weight", type=float, default=0.05)
    parser.add_argument("--hnmb-margin-weight", type=float, default=0.1)
    parser.add_argument("--hnmb-margin", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    case_rows = read_csv_rows(args.case_csv)
    train_dataset = EmbeddingStateDataset(
        args.train_embeddings,
        args.train_metadata,
        case_rows=case_rows,
        use_cdcs_weights=args.module.upper() == "CDCS",
    )
    val_dataset = EmbeddingStateDataset(
        args.val_embeddings,
        args.val_metadata,
        label_names=train_dataset.label_names,
        case_rows=case_rows,
        use_cdcs_weights=False,
    )
    sampler = None
    if args.module.upper() == "CDCS":
        weights = torch.tensor([float(row["weight"]) for row in train_dataset.rows], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    target_matrix = None
    if args.target_embeddings and args.target_embeddings.exists():
        target_matrix = torch.tensor(read_embeddings(args.target_embeddings), dtype=torch.float32, device=device)

    model = ModuleAblationModel(args.module, train_dataset.embedding_dim, len(train_dataset.label_names), args.hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bank = build_memory_bank(train_dataset) if args.module.upper() == "HNMB" else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "config_snapshot.json",
        {
            "module": args.module,
            "train_embeddings": str(args.train_embeddings),
            "train_metadata": str(args.train_metadata),
            "val_embeddings": str(args.val_embeddings),
            "val_metadata": str(args.val_metadata),
            "target_embeddings": str(args.target_embeddings) if args.target_embeddings else "",
            "label_names": train_dataset.label_names,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    )
    started = time.time()
    best_loss = float("inf")
    global_step = 0
    events: list[dict[str, Any]] = []
    model.train()
    while global_step < args.max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = model(batch["embedding"], batch["finding"])
            loss = weighted_state_loss(outputs["state_logits"], batch["state"], batch["weight"])
            if args.module.upper() == "HNMB" and bank is not None:
                loss = loss + args.hnmb_margin_weight * batch_hnmb_margin(model, bank, batch, device, args.hnmb_margin)
            if args.module.upper() == "DRA" and target_matrix is not None and "adapted" in outputs:
                take = min(outputs["adapted"].shape[0], target_matrix.shape[0])
                idx = torch.randint(0, target_matrix.shape[0], (take,), device=device)
                target = model.adapter(target_matrix[idx], domain_id=0)
                loss = loss + args.dra_coral_weight * coral_loss(outputs["adapted"][:take], target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.eval_interval == 0:
                result = evaluate(model, val_loader, device)
                event = {"global_step": global_step, "train_loss": float(loss.detach().cpu()), **result}
                events.append(event)
                print(json.dumps(event, default=to_jsonable))
                if result["val_loss"] is not None and result["val_loss"] < best_loss:
                    best_loss = float(result["val_loss"])
                    torch.save({"model": model.state_dict(), "label_names": train_dataset.label_names, "embedding_dim": train_dataset.embedding_dim}, args.output_dir / "best.pt")
                model.train()
            if global_step >= args.max_steps:
                break
    final = evaluate(model, val_loader, device)
    torch.save({"model": model.state_dict(), "label_names": train_dataset.label_names, "embedding_dim": train_dataset.embedding_dim}, args.output_dir / "final.pt")
    metrics = {
        "module": args.module,
        "global_step": global_step,
        "best_val_loss": best_loss if np.isfinite(best_loss) else final["val_loss"],
        "final": final,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "embedding_dim": train_dataset.embedding_dim,
        "label_names": train_dataset.label_names,
        "elapsed_seconds": time.time() - started,
        "events": events,
    }
    write_json(args.output_dir / "metrics_final.json", metrics)
    (args.output_dir / "training_log.txt").write_text("\n".join(json.dumps(event, ensure_ascii=False, default=to_jsonable) for event in events) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=to_jsonable))


if __name__ == "__main__":
    main()
