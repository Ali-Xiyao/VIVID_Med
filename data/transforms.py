"""
图像预处理与数据增强
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """
    训练时的数据增强
    - CXR 使用标准的 ImageNet 风格预处理
    - 添加适度的增强（不要过度，医学图像敏感）
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),  # CXR 可以水平翻转
        transforms.RandomAffine(
            degrees=10,      # 轻微旋转
            translate=(0.05, 0.05),  # 轻微平移
            scale=(0.95, 1.05),      # 轻微缩放
        ),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 均值
            std=[0.229, 0.224, 0.225]    # ImageNet 标准差
        ),
    ])


def get_val_transforms(image_size: int = 224):
    """
    验证/测试时的预处理（无增强）
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_grayscale_transforms(image_size: int = 224, is_train: bool = True):
    """
    灰度图像的预处理（CXR 原始为灰度）
    将单通道复制为三通道以适配 ViT
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
