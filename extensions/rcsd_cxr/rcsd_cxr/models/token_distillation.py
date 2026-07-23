"""Clean reconstruction of the VIVID hard-UMS SPD token objective."""

from __future__ import annotations

import itertools
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ExactSPDProjector(nn.Module):
    """Historical four-by-two SPD, including projected ViT tokens."""

    def __init__(
        self,
        vision_dim: int = 768,
        output_dim: int = 1536,
        *,
        num_groups: int = 4,
        tokens_per_group: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_groups != 4 or tokens_per_group != 2:
            raise ValueError("D0/D1 SPD is frozen at four groups by two tokens")
        self.num_groups = num_groups
        self.tokens_per_group = tokens_per_group
        self.group_queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, tokens_per_group, vision_dim) * 0.02
                )
                for _ in range(num_groups)
            ]
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    vision_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_groups)
            ]
        )
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, output_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        self._attention_maps: list[torch.Tensor] = []
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_query_tokens(self) -> int:
        return self.num_groups * self.tokens_per_group

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        if vision_tokens.ndim != 3:
            raise ValueError("vision tokens must have shape [batch, tokens, dim]")
        batch = vision_tokens.shape[0]
        groups: list[torch.Tensor] = []
        self._attention_maps = []
        for index, attention in enumerate(self.cross_attentions):
            query = self.group_queries[index].expand(batch, -1, -1)
            value, weights = attention(
                query,
                vision_tokens,
                vision_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            groups.append(value)
            self._attention_maps.append(weights)
        query_tokens = self.layer_norm(self.mlp(torch.cat(groups, dim=1)))
        projected_vision = self.layer_norm(self.mlp(vision_tokens))
        return torch.cat([query_tokens, projected_vision], dim=1)

    def orthogonality_loss(self) -> torch.Tensor:
        if len(self._attention_maps) < 2:
            return self.group_queries[0].sum() * 0.0
        means = [weights.mean(dim=1) for weights in self._attention_maps]
        values = [
            F.cosine_similarity(left, right, dim=-1).abs().mean()
            for left, right in itertools.combinations(means, 2)
        ]
        return torch.stack(values).mean()


class D0D1TokenModel(nn.Module):
    """Trainable ViT/SPD feeding a frozen local Qwen3.5 text teacher."""

    def __init__(
        self,
        *,
        teacher_path: Path,
        backbone_weights: Path,
        teacher_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
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
        teacher_hidden_size = int(teacher_config.hidden_size)
        self.projector = ExactSPDProjector(
            vision_dim=int(self.backbone.embed_dim),
            output_dim=teacher_hidden_size,
        )
        self.teacher_hidden_size = teacher_hidden_size

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "backbone": list(self.backbone.parameters()),
            "projector": list(self.projector.parameters()),
        }

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        vision_tokens = self.backbone.forward_features(images)
        projected = self.projector(vision_tokens)
        return projected.to(next(self.teacher.parameters()).dtype)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_image(images)
        text = self.teacher.get_input_embeddings()(input_ids)
        inputs = torch.cat([visual, text], dim=1)
        visual_attention = torch.ones(
            visual.shape[:2], dtype=attention_mask.dtype, device=images.device
        )
        full_attention = torch.cat([visual_attention, attention_mask], dim=1)
        visual_labels = torch.full(
            visual.shape[:2], -100, dtype=labels.dtype, device=images.device
        )
        full_labels = torch.cat([visual_labels, labels], dim=1)
        output = self.teacher(
            inputs_embeds=inputs,
            attention_mask=full_attention,
            use_cache=False,
            return_dict=True,
        )
        return {
            "logits": output.logits,
            "labels": full_labels,
            "visual_tokens": torch.tensor(
                visual.shape[1], device=images.device, dtype=torch.long
            ),
            "orthogonality": self.projector.orthogonality_loss(),
        }

    def vision_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "backbone": self.backbone.state_dict(),
            "projector": self.projector.state_dict(),
        }
