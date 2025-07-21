import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from PIL import Image
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
import random
import os
from pathlib import Path

class NaViTDataset(Dataset):
    """NaViT专用数据集类
    
    支持多分辨率图像的数据集，可以保持图像的原始宽高比
    """
    
    def __init__(
        self,
        data_dir: str,                           # 数据目录路径
        split: str = 'train',                    # 数据集分割：'train', 'val', 'test'
        min_size: int = 128,                     # 最小图像尺寸
        max_size: int = 512,                     # 最大图像尺寸
        preserve_aspect_ratio: bool = True,      # 是否保持宽高比
        augment: bool = True,                    # 是否使用数据增强
        normalize: bool = True,                  # 是否进行归一化
        class_to_idx: Optional[dict] = None      # 类别到索引的映射
    ):
        super().__init__()
        
        # 保存配置参数
        self.data_dir = Path(data_dir)
        self.split = split
        self.min_size = min_size
        self.max_size = max_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.augment = augment and (split == 'train')  # 只在训练时使用数据增强
        self.normalize = normalize
        
        # 加载数据文件列表和标签
        self.samples, self.class_to_idx = self._load_samples(class_to_idx)
        self.classes = list(self.class_to_idx.keys())
        self.num_classes = len(self.classes)
        
        # 定义基础变换
        self.base_transform = self._get_base_transform()
        
        # 定义数据增强变换（仅训练时使用）
        if self.augment:
            self.augment_transform = self._get_augment_transform()
        else:
            self.augment_transform = None
            
        # 定义归一化变换
        if self.normalize:
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准均值
                std=[0.229, 0.224, 0.225]    # ImageNet标准标准差
            )
        else:
            self.normalize_transform = None
    
    def _load_samples(self, class_to_idx: Optional[dict] = None) -> Tuple[List[Tuple[str, int]], dict]:
        """加载样本文件路径和标签
        
        Args:
            class_to_idx: 预定义的类别到索引映射
            
        Returns:
            samples: 样本列表，每个元素为(文件路径, 标签索引)
            class_to_idx: 类别到索引的映射字典
        """
        samples = []
        
        # 构建数据集路径
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"数据集路径不存在: {split_dir}")
        
        # 如果没有提供类别映射，自动构建
        if class_to_idx is None:
            # 获取所有类别文件夹
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            class_names = sorted([d.name for d in class_dirs])
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 遍历每个类别文件夹
        for class_name, class_idx in class_to_idx.items():
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                print(f"警告: 类别文件夹不存在: {class_dir}")
                continue
            
            # 遍历类别文件夹中的所有图像文件
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    samples.append((str(img_path), class_idx))
        
        print(f"加载了 {len(samples)} 个样本，{len(class_to_idx)} 个类别")
        return samples, class_to_idx
    
    def _get_base_transform(self) -> transforms.Compose:
        """获取基础变换
        
        Returns:
            基础变换的组合
        """
        transform_list = []
        
        # 转换为PIL图像（如果需要）
        transform_list.append(transforms.ToPILImage() if not isinstance(self.samples[0], str) else lambda x: x)
        
        return transforms.Compose(transform_list)
    
    def _get_augment_transform(self) -> transforms.Compose:
        """获取数据增强变换
        
        Returns:
            数据增强变换的组合
        """
        return transforms.Compose([
            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机旋转
            transforms.RandomRotation(degrees=10),
            # 随机颜色抖动
            transforms.ColorJitter(
                brightness=0.2,  # 亮度变化范围
                contrast=0.2,    # 对比度变化范围
                saturation=0.2,  # 饱和度变化范围
                hue=0.1          # 色调变化范围
            ),
            # 随机高斯模糊
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            # 随机灰度转换
            transforms.RandomGrayscale(p=0.1),
        ])
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """调整图像尺寸
        
        根据配置调整图像尺寸，可以选择是否保持宽高比
        
        Args:
            image: 输入PIL图像
            
        Returns:
            调整尺寸后的PIL图像
        """
        original_width, original_height = image.size
        
        if self.preserve_aspect_ratio:
            # 保持宽高比的调整
            # 计算缩放比例，确保最长边不超过max_size，最短边不小于min_size
            scale = min(
                self.max_size / max(original_width, original_height),
                max(self.min_size / min(original_width, original_height), 1.0)
            )
            
            # 计算新的尺寸
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 确保尺寸在合理范围内
            new_width = max(self.min_size, min(new_width, self.max_size))
            new_height = max(self.min_size, min(new_height, self.max_size))
            
        else:
            # 不保持宽高比，随机选择尺寸
            new_width = random.randint(self.min_size, self.max_size)
            new_height = random.randint(self.min_size, self.max_size)
        
        # 调整图像尺寸
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_image
    
    def __len__(self) -> int:
        """返回数据集大小
        
        Returns:
            数据集中样本的数量
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 图像张量，形状为 (C, H, W)
            label: 标签索引
        """
        # 获取文件路径和标签
        img_path, label = self.samples[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 调整图像尺寸
            image = self._resize_image(image)
            
            # 应用数据增强（仅训练时）
            if self.augment_transform is not None:
                image = self.augment_transform(image)
            
            # 转换为张量
            image = transforms.ToTensor()(image)
            
            # 应用归一化
            if self.normalize_transform is not None:
                image = self.normalize_transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回一个默认图像和标签
            default_image = torch.zeros(3, self.min_size, self.min_size)
            return default_image, 0

class CIFAR10NaViTDataset(Dataset):
    """基于CIFAR-10的NaViT数据集
    
    将CIFAR-10数据集适配为支持多分辨率的NaViT格式
    """
    
    def __init__(
        self,
        root: str = './data',                    # 数据根目录
        train: bool = True,                      # 是否为训练集
        download: bool = True,                   # 是否下载数据
        min_size: int = 32,                      # 最小图像尺寸
        max_size: int = 256,                     # 最大图像尺寸
        preserve_aspect_ratio: bool = True,      # 是否保持宽高比
        augment: bool = True,                    # 是否使用数据增强
        multi_scale_prob: float = 0.5            # 多尺度训练的概率
    ):
        super().__init__()
        
        # 保存配置参数
        self.min_size = min_size
        self.max_size = max_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.augment = augment and train  # 只在训练时使用数据增强
        self.multi_scale_prob = multi_scale_prob
        
        # 加载CIFAR-10数据集
        self.cifar10 = CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # 我们将自定义变换
        )
        
        # CIFAR-10的类别名称
        self.classes = self.cifar10.classes
        self.num_classes = len(self.classes)
        
        # 定义数据增强变换（仅训练时使用）
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                ], p=0.3),
            ])
        else:
            self.augment_transform = None
        
        # 定义归一化变换
        self.normalize_transform = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10标准均值
            std=[0.2023, 0.1994, 0.2010]    # CIFAR-10标准标准差
        )
    
    def _get_random_size(self) -> Tuple[int, int]:
        """获取随机尺寸
        
        Returns:
            (width, height): 随机生成的图像尺寸
        """
        if self.preserve_aspect_ratio:
            # 保持正方形宽高比
            size = random.randint(self.min_size, self.max_size)
            return size, size
        else:
            # 随机宽高比
            width = random.randint(self.min_size, self.max_size)
            height = random.randint(self.min_size, self.max_size)
            return width, height
    
    def __len__(self) -> int:
        """返回数据集大小
        
        Returns:
            数据集中样本的数量
        """
        return len(self.cifar10)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 图像张量，形状为 (C, H, W)
            label: 标签索引
        """
        # 从CIFAR-10获取原始数据
        image, label = self.cifar10[idx]
        
        # 确保图像是PIL格式
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        
        # 决定是否使用多尺度
        if random.random() < self.multi_scale_prob:
            # 使用随机尺寸
            width, height = self._get_random_size()
            image = image.resize((width, height), Image.LANCZOS)
        else:
            # 使用固定尺寸（原始CIFAR-10尺寸的放大版本）
            target_size = random.choice([64, 96, 128, 160, 192, 224])
            image = image.resize((target_size, target_size), Image.LANCZOS)
        
        # 应用数据增强（仅训练时）
        if self.augment_transform is not None:
            image = self.augment_transform(image)
        
        # 转换为张量
        image = transforms.ToTensor()(image)
        
        # 应用归一化
        image = self.normalize_transform(image)
        
        return image, label

class NaViTCollator:
    """NaViT专用的数据整理器
    
    用于将不同尺寸的图像打包成批次
    """
    
    def __init__(self, max_seq_len: int = 2048, pack_sequences: bool = True):
        """
        Args:
            max_seq_len: 最大序列长度（以tokens计算）
            pack_sequences: 是否启用序列打包
        """
        self.max_seq_len = max_seq_len
        self.pack_sequences = pack_sequences
    
    def _calculate_num_patches(self, image: torch.Tensor, patch_size: int = 16) -> int:
        """计算图像的patch数量
        
        Args:
            image: 图像张量，形状为 (C, H, W)
            patch_size: patch大小
            
        Returns:
            patch数量
        """
        _, height, width = image.shape
        h_patches = height // patch_size
        w_patches = width // patch_size
        return h_patches * w_patches
    
    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[List[torch.Tensor]], torch.Tensor]]:
        """整理批次数据
        
        Args:
            batch: 批次数据，每个元素为 (image, label)
            
        Returns:
            如果不使用序列打包:
                images: 图像张量，形状为 (batch_size, C, H, W)
                labels: 标签张量，形状为 (batch_size,)
            如果使用序列打包:
                packed_images: 打包的图像列表
                labels: 标签张量
        """
        images, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if not self.pack_sequences:
            # 不使用序列打包，需要将所有图像调整为相同尺寸
            # 找到批次中的最大尺寸
            max_height = max(img.shape[1] for img in images)
            max_width = max(img.shape[2] for img in images)
            
            # 将所有图像填充到相同尺寸
            padded_images = []
            for img in images:
                c, h, w = img.shape
                # 创建填充后的图像
                padded_img = torch.zeros(c, max_height, max_width)
                padded_img[:, :h, :w] = img
                padded_images.append(padded_img)
            
            images_tensor = torch.stack(padded_images)
            return images_tensor, labels
        
        else:
            # 使用序列打包
            packed_groups = []
            current_group = []
            current_tokens = 0
            
            for img in images:
                # 计算当前图像的token数量
                img_tokens = self._calculate_num_patches(img)
                
                # 检查是否可以添加到当前组
                if current_tokens + img_tokens <= self.max_seq_len:
                    current_group.append(img)
                    current_tokens += img_tokens
                else:
                    # 当前组已满，开始新组
                    if current_group:  # 确保当前组不为空
                        packed_groups.append(current_group)
                    current_group = [img]
                    current_tokens = img_tokens
            
            # 添加最后一组
            if current_group:
                packed_groups.append(current_group)
            
            return packed_groups, labels

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cifar10: bool = False,
    pack_sequences: bool = True,
    max_seq_len: int = 2048
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        num_workers: 数据加载进程数
        pin_memory: 是否使用固定内存
        use_cifar10: 是否使用CIFAR-10数据集
        pack_sequences: 是否启用序列打包
        max_seq_len: 最大序列长度
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 创建数据整理器
    collator = NaViTCollator(max_seq_len=max_seq_len, pack_sequences=pack_sequences)
    
    if use_cifar10:
        # 使用CIFAR-10数据集
        print("使用CIFAR-10数据集")
        
        # 创建训练集
        train_dataset = CIFAR10NaViTDataset(
            root=data_dir,
            train=True,
            download=True,
            augment=True,
            multi_scale_prob=0.7  # 70%概率使用多尺度
        )
        
        # 创建测试集（CIFAR-10没有单独的验证集）
        test_dataset = CIFAR10NaViTDataset(
            root=data_dir,
            train=False,
            download=True,
            augment=False,
            multi_scale_prob=0.0  # 测试时不使用多尺度
        )
        
        # 从训练集中分割出验证集
        train_size = int(0.9 * len(train_dataset))  # 90%用于训练
        val_size = len(train_dataset) - train_size   # 10%用于验证
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
    else:
        # 使用自定义数据集
        print(f"使用自定义数据集: {data_dir}")
        
        # 创建训练集
        train_dataset = NaViTDataset(
            data_dir=data_dir,
            split='train',
            augment=True,
            preserve_aspect_ratio=True
        )
        
        # 创建验证集
        val_dataset = NaViTDataset(
            data_dir=data_dir,
            split='val',
            augment=False,
            preserve_aspect_ratio=True
        )
        
        # 创建测试集
        test_dataset = NaViTDataset(
            data_dir=data_dir,
            split='test',
            augment=False,
            preserve_aspect_ratio=True
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True          # 丢弃最后一个不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # 验证时不打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # 测试时不打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")
    print(f"  批次大小: {batch_size}")
    print(f"  序列打包: {pack_sequences}")
    
    return train_loader, val_loader, test_loader

# 测试函数
if __name__ == "__main__":
    # 测试CIFAR-10数据集
    print("测试CIFAR-10数据集:")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='./data',
        batch_size=4,
        use_cifar10=True,
        pack_sequences=True,
        max_seq_len=1024
    )
    
    # 测试训练数据加载器
    print("\n测试训练数据加载器:")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 2:  # 只测试前2个批次
            break
            
        print(f"批次 {i+1}:")
        if isinstance(images, list):
            # 序列打包格式
            print(f"  打包组数: {len(images)}")
            total_images = sum(len(group) for group in images)
            print(f"  总图像数: {total_images}")
            for j, group in enumerate(images):
                print(f"  组 {j+1}: {len(group)} 张图像")
                for k, img in enumerate(group):
                    print(f"    图像 {k+1}: {img.shape}")
        else:
            # 标准格式
            print(f"  图像形状: {images.shape}")
        
        print(f"  标签形状: {labels.shape}")
        print(f"  标签值: {labels.tolist()}")
        print()
    
    # 测试验证数据加载器
    print("测试验证数据加载器:")
    for i, (images, labels) in enumerate(val_loader):
        if i >= 1:  # 只测试1个批次
            break
            
        print(f"批次 {i+1}:")
        if isinstance(images, list):
            print(f"  打包组数: {len(images)}")
            total_images = sum(len(group) for group in images)
            print(f"  总图像数: {total_images}")
        else:
            print(f"  图像形状: {images.shape}")
        
        print(f"  标签形状: {labels.shape}")
        break