"""One-sample real-weight forward/backward smoke for the D0/D1 objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rcsd_cxr.d0_d1_contract import render_hard_ums_target
from rcsd_cxr.models.token_distillation import D0D1TokenModel
from rcsd_cxr.token_objective import prepare_token_batch, token_cross_entropy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", required=True, type=Path)
    parser.add_argument("--backbone-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = D0D1TokenModel(
        teacher_path=args.teacher_path,
        backbone_weights=args.backbone_weights,
    ).to(device)
    target = render_hard_ums_target(
        {
            "Cardiomegaly": "present",
            "Edema": "absent",
            "Pleural Effusion": "uncertain",
        }
    )
    tokens = prepare_token_batch(
        model.tokenizer,
        prompt="Generate a structured medical report:\n",
        targets=[target],
        finding_weights=[
            {
                "Cardiomegaly": 1.0,
                "Edema": 0.420619835714305,
                "Pleural Effusion": 0.0,
            }
        ],
        variant="d1",
    )
    tensors = {key: value.to(device) for key, value in tokens.items()}
    images = torch.zeros(1, 3, 224, 224, device=device)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            images,
            tensors["input_ids"],
            tensors["attention_mask"],
            tensors["labels"],
        )
        visual_tokens = int(output["visual_tokens"])
        weights = torch.cat(
            [
                torch.ones(1, visual_tokens, device=device),
                tensors["token_weights"],
            ],
            dim=1,
        )
        token_loss = token_cross_entropy(
            output["logits"],
            output["labels"],
            token_weights=weights,
        )
        loss = token_loss + 0.02 * output["orthogonality"]
    loss.backward()
    groups = model.trainable_parameter_groups()
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "backbone_gradient": any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and int(torch.count_nonzero(parameter.grad)) > 0
            for parameter in groups["backbone"]
        ),
        "projector_gradient": any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and int(torch.count_nonzero(parameter.grad)) > 0
            for parameter in groups["projector"]
        ),
        "teacher_frozen": all(
            parameter.grad is None for parameter in model.teacher.parameters()
        ),
        "visual_tokens_205": visual_tokens == 205,
    }
    result = {
        "schema_version": 1,
        "artifact": "d0_d1_real_weight_smoke",
        "pass": all(checks.values()),
        "checks": checks,
        "token_loss": float(token_loss),
        "orthogonality": float(output["orthogonality"]),
        "teacher_hidden_size": model.teacher_hidden_size,
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "hashes": {
            "teacher_weight_authority": sha256_file(
                (
                    args.teacher_path / "model.safetensors.index.json"
                    if (
                        args.teacher_path / "model.safetensors.index.json"
                    ).is_file()
                    else args.teacher_path / "model.safetensors"
                )
            ),
            "backbone_weights": sha256_file(args.backbone_weights),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
