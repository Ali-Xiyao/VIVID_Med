"""Build frozen field prototypes from the preregistered Qwen3.5-2B teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from rcsd_cxr.gold_mapping import FINDINGS
from rcsd_cxr.models.field_anchored import FIELD_NAMES


STATES = ("absent", "present", "uncertain")
ANATOMY = {
    "Enlarged Cardiomediastinum": "cardiomediastinal silhouette",
    "Cardiomegaly": "cardiac silhouette",
    "Lung Opacity": "lung parenchyma",
    "Lung Lesion": "lung parenchyma",
    "Edema": "bilateral lung parenchyma",
    "Consolidation": "lung parenchyma",
    "Pneumonia": "lung parenchyma",
    "Atelectasis": "lung parenchyma",
    "Pneumothorax": "pleural space",
    "Pleural Effusion": "pleural space",
    "Fracture": "thoracic bones",
    "Support Devices": "thorax and support device course",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def templates() -> dict[str, object]:
    return {
        "observation": [
            f"Radiographic observation concept: {finding}."
            for finding in FINDINGS
        ],
        "assertion": [
            [
                f"Radiology report assertion: {state} {finding}."
                for state in STATES
            ]
            for finding in FINDINGS
        ],
        "anatomy": [
            f"Relevant chest radiograph anatomy: {ANATOMY[finding]}."
            for finding in FINDINGS
        ],
        "global": [
            [
                (
                    "Structured chest radiograph statement: "
                    f"{state} {finding} in {ANATOMY[finding]}."
                )
                for state in STATES
            ]
            for finding in FINDINGS
        ],
    }


def flatten_templates(
    values: dict[str, object]
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    flat: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for field in FIELD_NAMES:
        start = len(flat)
        value = values[field]
        if field in {"assertion", "global"}:
            for row in value:
                flat.extend(row)
        else:
            flat.extend(value)
        spans[field] = (start, len(flat))
    return flat, spans


@torch.inference_mode()
def encode(
    model: torch.nn.Module,
    tokenizer: object,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        last_index = encoded["attention_mask"].sum(dim=1) - 1
        pooled = hidden[
            torch.arange(hidden.shape[0], device=device), last_index
        ].float()
        vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(vectors, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen prototype build requires CUDA")
    device = torch.device("cuda:0")
    values = templates()
    texts, spans = flatten_templates(values)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    vectors = encode(model, tokenizer, texts, device, args.batch_size)
    hidden_size = vectors.shape[-1]
    observation = vectors[slice(*spans["observation"])]
    assertion = vectors[slice(*spans["assertion"])].reshape(
        len(FINDINGS), len(STATES), hidden_size
    )
    anatomy = vectors[slice(*spans["anatomy"])]
    global_vectors = vectors[slice(*spans["global"])].reshape(
        len(FINDINGS), len(STATES), hidden_size
    )
    payload = {
        "schema_version": 1,
        "model_path": str(args.model),
        "findings": FINDINGS,
        "states": STATES,
        "fields": FIELD_NAMES,
        "observation": observation,
        "assertion": assertion,
        "anatomy": anatomy,
        "global": global_vectors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    audit = {
        "schema_version": 1,
        "artifact": "qwen35_field_prototypes",
        "pass": True,
        "model": str(args.model),
        "teacher_size": "2B",
        "hidden_size": hidden_size,
        "template_count": len(texts),
        "field_shapes": {
            "observation": list(observation.shape),
            "assertion": list(assertion.shape),
            "anatomy": list(anatomy.shape),
            "global": list(global_vectors.shape),
        },
        "templates": values,
        "hashes": {
            "model_config": sha256_file(args.model / "config.json"),
            "prototype_file": sha256_file(args.output),
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
