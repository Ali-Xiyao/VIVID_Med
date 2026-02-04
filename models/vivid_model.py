"""
VIVID Model
整合 ViT + Projector + Frozen LLM 的完整模型

核心架构：
Image → ViT(train) → Projector(train) → Frozen LLM(forward) → JSON token logits

关键点：
- LLM 参数冻结 (requires_grad=False)
- 但 forward 必须保持可导 (不能用 torch.no_grad())
- 梯度只更新 ViT 和 Projector
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union

from .vit import ViTEncoder, get_vit_config
from .projector import VisionProjector


class VIVIDModel(nn.Module):
    """
    VIVID: Vision-Language Structured Alignment Model

    用冻结 LLM 作为"结构化语义解码器/监督空间"，
    训练 ViT 学到可迁移、可验证的医学视觉表征。
    """

    def __init__(
        self,
        # ViT 配置
        vit_model_name: str = "vit_base_patch16_224",
        vit_pretrained: bool = True,
        vit_output_type: str = "all",  # "cls", "mean", "all"
        # Projector 配置
        num_prefix_tokens: int = 16,
        projector_dropout: float = 0.1,
        # LLM 配置
        llm_model_name: str = "Qwen/Qwen3-1.7B",
        llm_device: Optional[str] = None,
        load_llm: bool = True,
        # 其他
        use_flash_attention: bool = True,
    ):
        """
        Args:
            vit_model_name: ViT 模型名称
            vit_pretrained: 是否使用预训练 ViT
            vit_output_type: ViT 输出类型
            num_prefix_tokens: 可学习前缀 token 数量
            projector_dropout: Projector dropout
            llm_model_name: HuggingFace LLM 模型名称
            llm_device: LLM 设备（默认与模型相同）
            load_llm: 是否加载 LLM（调试时可设为 False）
            use_flash_attention: 是否使用 Flash Attention
        """
        super().__init__()

        self.vit_model_name = vit_model_name
        self.llm_model_name = llm_model_name
        self.vit_output_type = vit_output_type
        self.llm_device = llm_device

        # 1. 创建 ViT 编码器（可训练）
        self.vit = ViTEncoder(
            model_name=vit_model_name,
            pretrained=vit_pretrained,
            output_type=vit_output_type,
        )
        vit_embed_dim = self.vit.get_embed_dim()

        # 2. 加载 LLM（冻结）
        self.llm = None
        self.tokenizer = None
        self.llm_embed_dim = None

        if load_llm:
            self._load_llm(llm_model_name, use_flash_attention, llm_device=self.llm_device)
        else:
            # 使用默认值（Qwen3-1.7B）
            self.llm_embed_dim = 1536

        # 3. 创建 Projector（可训练）
        self.projector = VisionProjector(
            vit_embed_dim=vit_embed_dim,
            llm_embed_dim=self.llm_embed_dim,
            num_prefix_tokens=num_prefix_tokens,
            dropout=projector_dropout,
        )

        # 4. Answerability Head（可选，用于预测字段可答性）
        self.answerability_head = None

    def _load_llm(
        self,
        model_name: str,
        use_flash_attention: bool = True,
        llm_device: Optional[str] = None,
    ):
        """
        加载并冻结 LLM

        关键：冻结参数但保持 forward 可导
        支持从 HuggingFace 或 ModelScope 下载
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 下载相关环境变量（提升稳定性，避免 Xet）
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")

        print(f"Loading LLM: {model_name}")

        # 尝试从 ModelScope 下载（国内镜像）
        use_modelscope = False
        try:
            from modelscope import snapshot_download
            use_modelscope = True
            print("ModelScope available, will try ModelScope first")
        except ImportError:
            print("ModelScope not available, using HuggingFace")

        # ModelScope 模型名称映射
        modelscope_mapping = {
            "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        }

        local_model_path = None
        if use_modelscope and model_name in modelscope_mapping:
            try:
                print(f"Downloading from ModelScope: {modelscope_mapping[model_name]}")
                local_model_path = snapshot_download(modelscope_mapping[model_name])
                print(f"Model downloaded to: {local_model_path}")
            except Exception as e:
                print(f"ModelScope download failed: {e}, falling back to HuggingFace")
                local_model_path = None

        # 确定模型路径
        model_path = local_model_path if local_model_path else model_name

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",  # 对于 decoder-only 模型
        )

        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 选择 dtype（CPU 用 float32，CUDA 用 bfloat16）
        target_device = llm_device or ("cuda" if torch.cuda.is_available() else "cpu")
        llm_dtype = torch.bfloat16 if str(target_device).startswith("cuda") else torch.float32

        # 加载模型
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

        common_load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=llm_dtype,
        )

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                **common_load_kwargs,
            )
        except Exception as e:
            print(f"Flash attention not available, falling back to eager: {e}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                **common_load_kwargs,
            )

        # 冻结 LLM 参数
        # 关键：只设置 requires_grad=False，不要用 torch.no_grad()
        for param in self.llm.parameters():
            param.requires_grad = False

        # 获取 LLM embedding 维度
        self.llm_embed_dim = self.llm.config.hidden_size

        print(f"LLM loaded. Hidden size: {self.llm_embed_dim}")
        print(f"LLM parameters frozen: {sum(p.numel() for p in self.llm.parameters()):,}")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数（ViT + Projector）"""
        params = []
        params.extend(self.vit.parameters())
        params.extend(self.projector.parameters())
        if self.answerability_head is not None:
            params.extend(self.answerability_head.parameters())
        return params

    def get_num_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def get_num_frozen_parameters(self) -> int:
        """获取冻结参数数量"""
        if self.llm is None:
            return 0
        return sum(p.numel() for p in self.llm.parameters())

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像为视觉 tokens

        Args:
            images: (B, C, H, W)

        Returns:
            visual_embeds: (B, num_visual_tokens, llm_embed_dim)
        """
        # ViT 编码
        vit_features = self.vit(images)  # (B, N, vit_embed_dim) or (B, vit_embed_dim)

        # Projector 投影
        visual_embeds = self.projector(vit_features)  # (B, num_prefix + N, llm_embed_dim)

        # 转换为 LLM 的数据类型（通常是 bfloat16）
        if self.llm is not None:
            llm_dtype = next(self.llm.parameters()).dtype
            visual_embeds = visual_embeds.to(llm_dtype)

        return visual_embeds

    def prepare_inputs_for_generation(
        self,
        images: torch.Tensor,
        prompt_text: Union[str, List[str]],
        target_text: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        准备生成所需的输入

        Args:
            images: (B, C, H, W)
            prompt_text: 提示文本
            target_text: 目标文本（训练时使用），可为单个字符串或与 batch 对齐的列表

        Returns:
            包含 input_embeds, attention_mask, labels 的字典
        """
        batch_size = images.shape[0]
        device = images.device
        max_length = 512

        if isinstance(prompt_text, list):
            if len(prompt_text) != batch_size:
                raise ValueError(
                    f"prompt_text list length ({len(prompt_text)}) "
                    f"does not match batch size ({batch_size})"
                )
            if not all(isinstance(p, str) for p in prompt_text):
                raise ValueError("prompt_text list must contain only strings")
            prompt_texts = prompt_text
        else:
            if not isinstance(prompt_text, str):
                raise ValueError("prompt_text must be a string or list of strings")
            prompt_texts = [prompt_text] * batch_size

        # 1. 编码图像
        visual_embeds = self.encode_image(images)  # (B, num_visual_tokens, llm_embed_dim)
        num_visual_tokens = visual_embeds.shape[1]

        # 2. 编码文本
        if target_text is not None:
            # 训练模式：prompt + target
            if isinstance(target_text, list):
                if len(target_text) != batch_size:
                    raise ValueError(
                        f"target_text list length ({len(target_text)}) "
                        f"does not match batch size ({batch_size})"
                    )
                if not all(isinstance(t, str) for t in target_text):
                    raise ValueError("target_text list must contain only strings")
                full_texts = [p + t for p, t in zip(prompt_texts, target_text)]
            else:
                if not isinstance(target_text, str):
                    raise ValueError("target_text must be a string or list of strings")
                full_texts = [p + target_text for p in prompt_texts]
        else:
            # 推理模式：只有 prompt
            full_texts = prompt_texts

        # Tokenize
        text_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # 获取文本 embeddings
        text_embeds = self.llm.get_input_embeddings()(text_inputs.input_ids)

        # 3. 拼接 [visual_embeds, text_embeds]
        input_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 4. 创建 attention mask
        visual_attention = torch.ones(
            batch_size, num_visual_tokens, dtype=torch.long, device=device
        )
        attention_mask = torch.cat([visual_attention, text_inputs.attention_mask], dim=1)

        # 5. 创建 labels（训练时）
        labels = None
        if target_text is not None:
            # 对于 visual tokens 和 prompt，labels 设为 -100（不计算 loss）
            prompt_tokens = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)

            # Labels: -100 for visual + prompt, actual tokens for target
            labels = text_inputs.input_ids.clone()
            for i, length in enumerate(prompt_lengths):
                labels[i, : int(length.item())] = -100  # 不计算 prompt 的 loss

            # 在前面添加 visual tokens 的 -100
            visual_labels = torch.full(
                (batch_size, num_visual_tokens), -100, dtype=torch.long, device=device
            )
            labels = torch.cat([visual_labels, labels], dim=1)

            # 忽略 padding tokens
            labels = labels.masked_fill(attention_mask == 0, -100)

        return {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_text: Optional[Union[str, List[str]]] = None,
        target_text: Optional[Union[str, List[str]]] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        两种使用方式：
        1. 直接传入 input_ids（已经 tokenize 的文本）
        2. 传入 prompt_text 和 target_text（自动处理）

        Args:
            images: (B, C, H, W) 输入图像
            input_ids: (B, L) 文本 token ids
            attention_mask: (B, L) attention mask
            labels: (B, L) 训练标签
            prompt_text: 提示文本
            target_text: 目标文本（可为单个字符串或与 batch 对齐的列表）
            return_dict: 是否返回字典

        Returns:
            包含 loss, logits 等的字典
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Set load_llm=True or call _load_llm()")

        # 如果提供了 prompt_text，使用自动处理
        if prompt_text is not None:
            prepared = self.prepare_inputs_for_generation(
                images, prompt_text, target_text
            )
            inputs_embeds = prepared["inputs_embeds"]
            attention_mask = prepared["attention_mask"]
            labels = prepared["labels"]
        else:
            # 手动处理
            # 1. 编码图像
            visual_embeds = self.encode_image(images)
            batch_size = images.shape[0]
            num_visual_tokens = visual_embeds.shape[1]
            device = images.device

            # 2. 获取文本 embeddings
            text_embeds = self.llm.get_input_embeddings()(input_ids)

            # 3. 拼接
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            # 4. 更新 attention mask
            visual_attention = torch.ones(
                batch_size, num_visual_tokens, dtype=torch.long, device=device
            )
            attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

            # 5. 更新 labels
            if labels is not None:
                visual_labels = torch.full(
                    (batch_size, num_visual_tokens), -100, dtype=torch.long, device=device
                )
                labels = torch.cat([visual_labels, labels], dim=1)

        # LLM forward（保持可导！）
        # 注意：不要用 torch.no_grad()，否则梯度无法回传
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        if return_dict:
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            }
        else:
            return outputs

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        生成文本（推理时使用）

        Args:
            images: (B, C, H, W)
            prompt_text: 提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否采样

        Returns:
            生成的文本列表
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded")

        batch_size = images.shape[0]
        device = images.device

        # 编码图像
        visual_embeds = self.encode_image(images)
        num_visual_tokens = visual_embeds.shape[1]

        # 编码 prompt
        prompt_inputs = self.tokenizer(
            [prompt_text] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_inputs.input_ids)

        # 拼接
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)

        # Attention mask
        visual_attention = torch.ones(
            batch_size, num_visual_tokens, dtype=torch.long, device=device
        )
        attention_mask = torch.cat([visual_attention, prompt_inputs.attention_mask], dim=1)

        # 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        # 解码
        # 注意：outputs 包含了 prompt tokens，需要跳过
        generated_texts = []
        for i, output in enumerate(outputs):
            # 跳过 visual tokens + prompt 部分
            input_len = num_visual_tokens + prompt_inputs.input_ids.shape[1]
            generated = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            generated_texts.append(generated)

        return generated_texts
