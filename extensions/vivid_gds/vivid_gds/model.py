"""Frozen-Qwen prefix4 generation with an optional UMS schema bridge."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


ARMS = ("A0_direct", "A1_freetext", "A2_ums", "A3_gds")
GENERATIVE_ARMS = {"A1_freetext", "A2_ums", "A3_gds"}
SCHEMA_ARMS = {"A0_direct", "A3_gds"}


class HistoricalPrefixProjector(nn.Module):
    """The audited historical four-prefix projector."""

    def __init__(
        self,
        vision_dim: int = 768,
        output_dim: int = 1536,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prefix_tokens = nn.Parameter(
            torch.randn(1, 4, output_dim) * 0.02
        )
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_query_tokens(self) -> int:
        return 4

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        projected = self.layer_norm(self.mlp(vision_tokens))
        prefix = self.prefix_tokens.expand(vision_tokens.shape[0], -1, -1)
        return torch.cat([prefix, projected], dim=1)


class UMSSchemaHead(nn.Module):
    def __init__(
        self,
        vision_dim: int = 768,
        hidden_dim: int = 512,
        num_findings: int = 12,
        num_states: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_findings = num_findings
        self.num_states = num_states
        self.layers = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_findings * num_states),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.layers(cls).reshape(
            cls.shape[0], self.num_findings, self.num_states
        )


class VividGDSModel(nn.Module):
    """Stage-A arms with one shared ViT identity."""

    def __init__(
        self,
        *,
        arm: str,
        backbone_weights: Path,
        teacher_path: Path | None = None,
        teacher_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown VIVID-GDS arm: {arm}")
        if arm in GENERATIVE_ARMS and teacher_path is None:
            raise ValueError(f"{arm} requires teacher_path")
        self.arm = arm
        self.backbone = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
        state = load_file(str(backbone_weights))
        incompatible = self.backbone.load_state_dict(state, strict=False)
        allowed = {"head.weight", "head.bias"}
        if set(incompatible.unexpected_keys) - allowed:
            raise ValueError(
                f"unexpected backbone keys: {incompatible.unexpected_keys}"
            )
        if incompatible.missing_keys:
            raise ValueError(f"missing backbone keys: {incompatible.missing_keys}")

        self.schema_head = (
            UMSSchemaHead(vision_dim=int(self.backbone.embed_dim))
            if arm in SCHEMA_ARMS
            else None
        )
        self.tokenizer = None
        self.teacher = None
        self.projector = None
        if arm in GENERATIVE_ARMS:
            assert teacher_path is not None
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(teacher_path),
                local_files_only=True,
                trust_remote_code=True,
                padding_side="right",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.teacher = AutoModelForCausalLM.from_pretrained(
                str(teacher_path),
                local_files_only=True,
                trust_remote_code=True,
                dtype=teacher_dtype,
                attn_implementation="sdpa",
            )
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False
            self.teacher.config.use_cache = False
            if gradient_checkpointing:
                self.teacher.gradient_checkpointing_enable()
            teacher_config = getattr(
                self.teacher.config,
                "text_config",
                self.teacher.config,
            )
            self.projector = HistoricalPrefixProjector(
                vision_dim=int(self.backbone.embed_dim),
                output_dim=int(teacher_config.hidden_size),
            )

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {"backbone": list(self.backbone.parameters())}
        if self.projector is not None:
            groups["projector"] = list(self.projector.parameters())
        if self.schema_head is not None:
            groups["schema_head"] = list(self.schema_head.parameters())
        return groups

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self.backbone.forward_features(images)
        result: dict[str, torch.Tensor] = {"vit_cls": tokens[:, 0]}
        if self.schema_head is not None:
            result["schema_logits"] = self.schema_head(tokens[:, 0])
        if self.teacher is not None and self.projector is not None:
            if input_ids is None or attention_mask is None or labels is None:
                raise ValueError("generative arm requires token tensors")
            visual = self.projector(tokens).to(
                next(self.teacher.parameters()).dtype
            )
            text = self.teacher.get_input_embeddings()(input_ids)
            inputs = torch.cat([visual, text], dim=1)
            visual_attention = torch.ones(
                visual.shape[:2],
                dtype=attention_mask.dtype,
                device=images.device,
            )
            full_attention = torch.cat(
                [visual_attention, attention_mask], dim=1
            )
            visual_labels = torch.full(
                visual.shape[:2],
                -100,
                dtype=labels.dtype,
                device=images.device,
            )
            full_labels = torch.cat([visual_labels, labels], dim=1)
            output = self.teacher(
                inputs_embeds=inputs,
                attention_mask=full_attention,
                use_cache=False,
                return_dict=True,
            )
            result["logits"] = output.logits
            result["labels"] = full_labels
            result["visual_tokens"] = torch.tensor(
                visual.shape[1],
                device=images.device,
                dtype=torch.long,
            )
        return result

    def vision_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        state = {"backbone": self.backbone.state_dict()}
        if self.projector is not None:
            state["projector"] = self.projector.state_dict()
        if self.schema_head is not None:
            state["schema_head"] = self.schema_head.state_dict()
        return state
