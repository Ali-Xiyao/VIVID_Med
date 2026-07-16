"""
VIVID-Med 快速测试脚本

用于验证：
1. 数据加载是否正常
2. 模型是否能正常创建
3. 前向传播是否正常
4. 梯度是否能正常回传到 ViT 和 Projector

用法:
    python test_pipeline.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 50)
    print("Testing Data Loading...")
    print("=" * 50)

    from data import CheXpertUMSDataset, get_train_transforms
    from data.chexpert_dataset import collate_fn
    from torch.utils.data import DataLoader

    # 数据路径
    data_root = Path(__file__).parent.parent / "data" / "dataset"
    ums_path = data_root / "processed" / "chexpert_ums.jsonl"

    if not ums_path.exists():
        print(f"ERROR: UMS file not found: {ums_path}")
        print("Please run prepare_chexpert.py first")
        return None

    # 创建数据集
    dataset = CheXpertUMSDataset(
        data_root=str(data_root),
        ums_jsonl_path=str(ums_path),
        transform=get_train_transforms(224),
        is_train=True,
        max_samples=10,  # 只加载 10 个样本用于测试
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试单个样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Target JSON (truncated): {sample['target_json'][:100]}...")

    # 测试 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")

    print("\n✓ Data loading test passed!")
    return dataloader


def test_model_creation(load_llm: bool = False):
    """测试模型创建"""
    print("\n" + "=" * 50)
    print("Testing Model Creation...")
    print("=" * 50)

    from models import VIVIDModel

    # 创建模型（不加载 LLM 以加快测试）
    model = VIVIDModel(
        vit_model_name="vit_base_patch16_224",
        vit_pretrained=True,
        vit_output_type="cls",
        num_prefix_tokens=16,
        llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        load_llm=load_llm,
    )

    print(f"ViT embed dim: {model.vit.get_embed_dim()}")
    print(f"Projector output dim: {model.projector.llm_embed_dim}")
    print(f"Num prefix tokens: {model.projector.num_prefix_tokens}")
    print(f"Trainable parameters: {model.get_num_trainable_parameters():,}")

    if load_llm:
        print(f"Frozen parameters: {model.get_num_frozen_parameters():,}")

    print("\n✓ Model creation test passed!")
    return model


def test_forward_pass(model, dataloader, device="cpu"):
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("Testing Forward Pass...")
    print("=" * 50)

    model = model.to(device)
    model.eval()

    batch = next(iter(dataloader))
    images = batch["images"].to(device)

    print(f"Input images shape: {images.shape}")

    # 测试图像编码
    with torch.no_grad():
        visual_embeds = model.encode_image(images)
        print(f"Visual embeddings shape: {visual_embeds.shape}")

    print("\n✓ Forward pass test passed!")
    return visual_embeds


def test_gradient_flow(model, dataloader, device="cpu"):
    """测试梯度流动"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow...")
    print("=" * 50)

    if model.llm is None:
        print("Skipping gradient test (LLM not loaded)")
        return

    model = model.to(device)
    model.train()

    batch = next(iter(dataloader))
    images = batch["images"].to(device)
    target_jsons = batch["target_jsons"]

    # 前向传播
    outputs = model(
        images=images,
        prompt_text="Generate a medical report:\n",
        target_text=target_jsons,
    )

    loss = outputs["loss"]
    print(f"Loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    vit_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.vit.parameters() if p.requires_grad)
    proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.projector.parameters())
    llm_has_grad = any(p.grad is not None for p in model.llm.parameters())

    print(f"ViT has gradients: {vit_has_grad}")
    print(f"Projector has gradients: {proj_has_grad}")
    print(f"LLM has gradients (should be False): {llm_has_grad}")

    if vit_has_grad and proj_has_grad and not llm_has_grad:
        print("\n✓ Gradient flow test passed!")
        print("  - Gradients flow to ViT and Projector")
        print("  - LLM parameters remain frozen")
    else:
        print("\n✗ Gradient flow test FAILED!")
        if not vit_has_grad:
            print("  - ERROR: No gradients in ViT")
        if not proj_has_grad:
            print("  - ERROR: No gradients in Projector")
        if llm_has_grad:
            print("  - ERROR: LLM should not have gradients")


def test_verifier():
    """测试 Verifier"""
    print("\n" + "=" * 50)
    print("Testing Verifier...")
    print("=" * 50)

    from evaluation import UMSVerifier

    verifier = UMSVerifier()

    # 测试有效的 UMS JSON
    valid_ums = {
        "modality": "CXR",
        "anatomy": ["chest"],
        "findings": {
            "Pneumonia": {"state": "present", "score": 0.9},
            "Cardiomegaly": {"state": "absent", "score": 0.1},
        },
        "laterality": None,
        "study_view": "AP",
        "geometry": {"bbox": None, "mask": None, "keypoints": None},
        "measurements": {},
        "answerability": {
            "Pneumonia": True,
            "Cardiomegaly": True,
            "laterality": False,
        },
        "uncertainty": {},
        "provenance": {},
        "verifier": {"pass": True, "failure_type": None, "confidence": 1.0, "role": "positive"},
    }

    result = verifier.verify(valid_ums)
    print(f"Valid UMS - Passed: {result.passed}, Role: {result.role.value}")

    # 测试无效的 UMS JSON（错误的 modality）
    invalid_ums = valid_ums.copy()
    invalid_ums["modality"] = "INVALID"

    result = verifier.verify(invalid_ums)
    print(f"Invalid modality - Passed: {result.passed}, Failure: {result.failure_type.value}")

    # 测试无效的 laterality
    invalid_ums2 = valid_ums.copy()
    invalid_ums2["laterality"] = "middle"  # 无效值

    result = verifier.verify(invalid_ums2)
    print(f"Invalid laterality - Passed: {result.passed}, Failure: {result.failure_type.value}")

    print("\n✓ Verifier test passed!")


def main():
    print("=" * 50)
    print("VIVID-Med Pipeline Test")
    print("=" * 50)

    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 测试数据加载
    dataloader = test_data_loading()
    if dataloader is None:
        return

    # 2. 测试模型创建（不加载 LLM）
    model = test_model_creation(load_llm=False)

    # 3. 测试前向传播（不需要 LLM）
    test_forward_pass(model, dataloader, device)

    # 4. 测试 Verifier
    test_verifier()

    # 5. 完整测试（加载 LLM）
    print("\n" + "=" * 50)
    print("Full Test with LLM (optional)")
    print("=" * 50)

    run_full_test = input("\nRun full test with LLM? This will download ~3GB model. (y/n): ")

    if run_full_test.lower() == 'y':
        print("\nLoading LLM...")
        model_full = test_model_creation(load_llm=True)
        test_gradient_flow(model_full, dataloader, device)
    else:
        print("Skipping full test.")

    print("\n" + "=" * 50)
    print("All basic tests completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run full training: python train_cxr.py --debug")
    print("2. Check outputs in ./outputs/cxr_chexpert/")


if __name__ == "__main__":
    main()
