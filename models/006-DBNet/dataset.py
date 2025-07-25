import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
import json
import random
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TextDetectionDataset(Dataset):
    """
    文本检测数据集类
    
    支持多种数据格式：
    - ICDAR格式：每个图像对应一个txt标注文件
    - COCO格式：JSON标注文件
    - 自定义格式：包含图像路径和多边形标注的列表
    """
    
    def __init__(self, 
                 data_root: str,
                 annotation_file: str,
                 image_size: Tuple[int, int] = (640, 640),
                 is_training: bool = True,
                 augment: bool = True):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录路径
            annotation_file: 标注文件路径
            image_size: 图像尺寸 (height, width)
            is_training: 是否为训练模式
            augment: 是否使用数据增强
        """
        self.data_root = data_root  # 数据根目录
        self.annotation_file = annotation_file  # 标注文件路径
        self.image_size = image_size  # 目标图像尺寸
        self.is_training = is_training  # 训练/验证模式标志
        self.augment = augment  # 数据增强标志
        
        # 加载数据列表
        self.data_list = self._load_annotations()
        
        # 设置数据增强策略
        self.transform = self._get_transforms()
        
        print(f"加载数据集完成: {len(self.data_list)} 个样本")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        加载标注数据
        
        Returns:
            数据列表，每个元素包含图像路径和标注信息
        """
        data_list = []
        
        # 检查标注文件格式
        if self.annotation_file.endswith('.json'):
            # JSON格式标注文件
            data_list = self._load_json_annotations()
        elif self.annotation_file.endswith('.txt'):
            # 文本格式标注文件
            data_list = self._load_txt_annotations()
        else:
            raise ValueError(f"不支持的标注文件格式: {self.annotation_file}")
        
        return data_list
    
    def _load_json_annotations(self) -> List[Dict[str, Any]]:
        """
        加载JSON格式的标注文件
        
        Returns:
            数据列表
        """
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        data_list = []
        for item in annotations:
            # 构建图像完整路径
            image_path = os.path.join(self.data_root, item['image_path'])
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在 {image_path}")
                continue
            
            # 解析多边形标注
            polygons = []
            texts = []
            
            for ann in item['annotations']:
                # 多边形坐标 (x1,y1,x2,y2,...)
                polygon = np.array(ann['polygon']).reshape(-1, 2)
                polygons.append(polygon)
                
                # 文本内容（可选）
                text = ann.get('text', '')
                texts.append(text)
            
            data_list.append({
                'image_path': image_path,
                'polygons': polygons,
                'texts': texts
            })
        
        return data_list
    
    def _load_txt_annotations(self) -> List[Dict[str, Any]]:
        """
        加载文本格式的标注文件
        
        文件格式：每行包含 "图像路径\t标注文件路径"
        
        Returns:
            数据列表
        """
        data_list = []
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析图像路径和标注路径
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"警告: 标注格式错误 {line}")
                continue
            
            image_path, gt_path = parts
            
            # 构建完整路径
            image_path = os.path.join(self.data_root, image_path)
            gt_path = os.path.join(self.data_root, gt_path)
            
            # 检查文件是否存在
            if not os.path.exists(image_path) or not os.path.exists(gt_path):
                print(f"警告: 文件不存在 {image_path} 或 {gt_path}")
                continue
            
            # 解析标注文件
            polygons, texts = self._parse_gt_file(gt_path)
            
            data_list.append({
                'image_path': image_path,
                'polygons': polygons,
                'texts': texts
            })
        
        return data_list
    
    def _parse_gt_file(self, gt_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        解析ICDAR格式的标注文件
        
        Args:
            gt_path: 标注文件路径
            
        Returns:
            多边形列表和文本列表
        """
        polygons = []
        texts = []
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析标注行：x1,y1,x2,y2,x3,y3,x4,y4,text
            parts = line.split(',')
            
            if len(parts) < 8:
                print(f"警告: 标注格式错误 {line}")
                continue
            
            # 提取坐标点
            try:
                coords = [float(x) for x in parts[:8]]
                polygon = np.array(coords).reshape(-1, 2)
                polygons.append(polygon)
                
                # 提取文本内容
                text = ','.join(parts[8:]) if len(parts) > 8 else ''
                texts.append(text)
                
            except ValueError:
                print(f"警告: 坐标解析错误 {line}")
                continue
        
        return polygons, texts
    
    def _get_transforms(self) -> A.Compose:
        """
        获取数据增强变换
        
        Returns:
            Albumentations变换组合
        """
        if self.is_training and self.augment:
            # 训练时的数据增强
            transform = A.Compose([
                # 几何变换
                A.RandomRotate90(p=0.2),  # 随机90度旋转
                A.Rotate(limit=10, p=0.3),  # 小角度随机旋转
                A.HorizontalFlip(p=0.5),  # 水平翻转
                
                # 尺寸变换
                A.RandomScale(scale_limit=0.2, p=0.3),  # 随机缩放
                A.Resize(height=self.image_size[0], width=self.image_size[1]),  # 调整到目标尺寸
                
                # 颜色变换
                A.ColorJitter(
                    brightness=0.2,  # 亮度变化
                    contrast=0.2,    # 对比度变化
                    saturation=0.2,  # 饱和度变化
                    hue=0.1,         # 色调变化
                    p=0.5
                ),
                
                # 噪声和模糊
                A.GaussNoise(var_limit=(10, 50), p=0.2),  # 高斯噪声
                A.GaussianBlur(blur_limit=3, p=0.1),      # 高斯模糊
                
                # 归一化和转换为张量
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet均值
                    std=[0.229, 0.224, 0.225]    # ImageNet标准差
                ),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            # 验证时只进行基本变换
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        return transform
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集样本数量
        """
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像和标签的字典
        """
        # 获取数据项
        data_item = self.data_list[idx]
        
        # 加载图像
        image = self._load_image(data_item['image_path'])
        
        # 获取多边形标注
        polygons = data_item['polygons']
        
        # 应用数据增强
        if len(polygons) > 0:
            # 将多边形转换为关键点格式
            keypoints = []
            for polygon in polygons:
                for point in polygon:
                    keypoints.append(point)
            
            # 应用变换
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            
            # 重新组织关键点为多边形
            transformed_keypoints = transformed['keypoints']
            transformed_polygons = []
            
            start_idx = 0
            for polygon in polygons:
                end_idx = start_idx + len(polygon)
                transformed_polygon = np.array(transformed_keypoints[start_idx:end_idx])
                transformed_polygons.append(transformed_polygon)
                start_idx = end_idx
            
            polygons = transformed_polygons
        else:
            # 没有标注时只变换图像
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # 生成训练标签
        gt_text, gt_mask = self._generate_targets(polygons, self.image_size)
        
        return {
            'image': image,           # 输入图像张量 (3, H, W)
            'gt_text': gt_text,       # 文本区域标签 (1, H, W)
            'gt_mask': gt_mask,       # 有效区域掩码 (1, H, W)
            'polygons': polygons      # 原始多边形标注（用于可视化）
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像数组 (H, W, 3)
        """
        try:
            # 使用OpenCV加载图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            
            # 转换颜色空间：BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            print(f"图像加载错误 {image_path}: {e}")
            # 返回默认图像
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _generate_targets(self, polygons: List[np.ndarray], 
                         image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练目标标签
        
        Args:
            polygons: 多边形标注列表
            image_size: 图像尺寸 (height, width)
            
        Returns:
            文本区域标签和有效区域掩码
        """
        height, width = image_size
        
        # 初始化标签图
        gt_text = np.zeros((height, width), dtype=np.float32)
        gt_mask = np.ones((height, width), dtype=np.float32)  # 默认全部有效
        
        # 遍历所有多边形
        for polygon in polygons:
            if len(polygon) < 3:
                continue  # 跳过无效多边形
            
            # 确保坐标在图像范围内
            polygon = np.clip(polygon, 0, [width-1, height-1])
            
            # 填充文本区域
            self._fill_polygon(gt_text, polygon, 1.0)
        
        # 转换为张量
        gt_text = torch.from_numpy(gt_text).unsqueeze(0)  # (1, H, W)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)  # (1, H, W)
        
        return gt_text, gt_mask
    
    def _fill_polygon(self, image: np.ndarray, polygon: np.ndarray, value: float):
        """
        在图像中填充多边形区域
        
        Args:
            image: 目标图像数组
            polygon: 多边形顶点坐标
            value: 填充值
        """
        # 转换为整数坐标
        polygon = polygon.astype(np.int32)
        
        # 使用OpenCV填充多边形
        cv2.fillPoly(image, [polygon], value)


class SyntheticTextDataset(Dataset):
    """
    合成文本数据集
    
    用于生成简单的合成文本数据，便于快速测试和原型开发
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 image_size: Tuple[int, int] = (640, 640),
                 max_texts: int = 10):
        """
        初始化合成数据集
        
        Args:
            num_samples: 生成样本数量
            image_size: 图像尺寸
            max_texts: 每张图像最大文本数量
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_texts = max_texts
        
        # 预定义文本内容
        self.text_samples = [
            'Hello', 'World', 'Text', 'Detection', 'DBNet',
            'PyTorch', 'Deep', 'Learning', 'Computer', 'Vision',
            'Artificial', 'Intelligence', 'Machine', 'Learning',
            'Neural', 'Network', 'Convolutional', 'Feature'
        ]
        
        # 数据变换
        self.transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        生成合成文本样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含图像和标签的字典
        """
        # 创建空白图像
        image = np.ones((*self.image_size, 3), dtype=np.uint8) * 255
        
        # 随机生成文本数量
        num_texts = random.randint(1, self.max_texts)
        
        polygons = []
        
        # 在图像上绘制文本
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        for _ in range(num_texts):
            # 随机选择文本
            text = random.choice(self.text_samples)
            
            # 随机位置和大小
            font_size = random.randint(20, 60)
            x = random.randint(0, self.image_size[1] - len(text) * font_size // 2)
            y = random.randint(0, self.image_size[0] - font_size)
            
            # 绘制文本
            draw.text((x, y), text, fill=(0, 0, 0))
            
            # 创建边界框（简化为矩形）
            text_width = len(text) * font_size // 2
            text_height = font_size
            
            polygon = np.array([
                [x, y],
                [x + text_width, y],
                [x + text_width, y + text_height],
                [x, y + text_height]
            ])
            
            polygons.append(polygon)
        
        # 转换回numpy数组
        image = np.array(pil_image)
        
        # 应用变换
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # 生成标签
        gt_text, gt_mask = self._generate_targets(polygons)
        
        return {
            'image': image,
            'gt_text': gt_text,
            'gt_mask': gt_mask,
            'polygons': polygons
        }
    
    def _generate_targets(self, polygons: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练标签
        
        Args:
            polygons: 多边形列表
            
        Returns:
            文本标签和掩码
        """
        height, width = self.image_size
        
        # 初始化标签
        gt_text = np.zeros((height, width), dtype=np.float32)
        gt_mask = np.ones((height, width), dtype=np.float32)
        
        # 填充多边形
        for polygon in polygons:
            polygon = np.clip(polygon, 0, [width-1, height-1])
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(gt_text, [polygon], 1.0)
        
        # 转换为张量
        gt_text = torch.from_numpy(gt_text).unsqueeze(0)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)
        
        return gt_text, gt_mask


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # 加速GPU传输
        drop_last=True if shuffle else False  # 训练时丢弃最后不完整批次
    )


def visualize_sample(sample: Dict[str, torch.Tensor], save_path: str = None):
    """
    可视化数据样本
    
    Args:
        sample: 数据样本字典
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    
    # 提取数据
    image = sample['image']  # (3, H, W)
    gt_text = sample['gt_text']  # (1, H, W)
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    # 转换为numpy格式
    image = image.permute(1, 2, 0).numpy()
    gt_text = gt_text.squeeze(0).numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 文本标签
    axes[1].imshow(gt_text, cmap='gray')
    axes[1].set_title('文本区域标签')
    axes[1].axis('off')
    
    # 叠加显示
    overlay = image.copy()
    overlay[:, :, 0] = np.where(gt_text > 0.5, 1.0, overlay[:, :, 0])
    axes[2].imshow(overlay)
    axes[2].set_title('叠加显示')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # 测试代码：验证数据集功能
    
    print("测试合成文本数据集...")
    
    # 创建合成数据集
    synthetic_dataset = SyntheticTextDataset(
        num_samples=100,
        image_size=(640, 640),
        max_texts=5
    )
    
    # 创建数据加载器
    dataloader = create_dataloader(
        dataset=synthetic_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # 测试时使用单进程
    )
    
    # 测试数据加载
    print(f"数据集大小: {len(synthetic_dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    # 获取一个批次的数据
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n批次 {batch_idx + 1}:")
        print(f"图像形状: {batch['image'].shape}")
        print(f"文本标签形状: {batch['gt_text'].shape}")
        print(f"掩码形状: {batch['gt_mask'].shape}")
        
        # 可视化第一个样本
        if batch_idx == 0:
            sample = {
                'image': batch['image'][0],
                'gt_text': batch['gt_text'][0]
            }
            
            print("生成可视化图像...")
            visualize_sample(sample, 'sample_visualization.png')
        
        # 只测试前几个批次
        if batch_idx >= 2:
            break
    
    print("\n数据集测试完成！")
    
    # 测试真实数据集（如果有标注文件）
    # 注意：需要准备实际的数据和标注文件
    """
    print("\n测试真实文本检测数据集...")
    
    try:
        real_dataset = TextDetectionDataset(
            data_root='./data',
            annotation_file='./data/train.txt',
            image_size=(640, 640),
            is_training=True,
            augment=True
        )
        
        real_dataloader = create_dataloader(
            dataset=real_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        print(f"真实数据集大小: {len(real_dataset)}")
        
        # 测试一个批次
        for batch in real_dataloader:
            print(f"真实数据批次形状: {batch['image'].shape}")
            break
            
    except Exception as e:
        print(f"真实数据集测试跳过: {e}")
    """