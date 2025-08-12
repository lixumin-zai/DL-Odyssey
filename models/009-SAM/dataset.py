import torch  # 导入 PyTorch 深度学习框架
from torch.utils.data import Dataset, DataLoader  # 从 PyTorch 中导入 Dataset 和 DataLoader 类，用于创建和管理数据集
from PIL import Image  # 导入 Pillow 库中的 Image 模块，用于图像文件的读取和处理
import numpy as np  # 导入 NumPy 库，用于高效的数值计算，特别是数组操作
import os  # 导入 os 模块，用于与操作系统交互，例如文件路径操作

class SAMDataset(Dataset):  # 定义一个名为 SAMDataset 的类，它继承自 torch.utils.data.Dataset，这是创建自定义数据集的标准方法
    """
    一个用于加载 SAM 模型所需数据的自定义数据集。

    Args:
        image_dir (str): 包含图像的目录路径。
        mask_dir (str): 包含掩码的目录路径。
        transform (callable, optional): 应用于图像和掩码的转换。默认为 None。
    """
    def __init__(self, image_dir, mask_dir, transform=None):  # 类的初始化方法
        self.image_dir = image_dir  # 存储图像文件所在的目录路径
        self.mask_dir = mask_dir    # 存储掩码文件所在的目录路径
        self.transform = transform  # 存储数据增强或预处理的转换函数
        self.images = os.listdir(image_dir)  # 使用 os.listdir 获取图像目录中所有文件的名称，并存储在 self.images 列表中

    def __len__(self):
        """返回数据集中样本的总数。这是 Dataset 类的必要方法。"""
        return len(self.images)  # 返回图像列表的长度，即数据集中图像的总数

    def __getitem__(self, idx):
        """
        根据给定的索引 `idx`，获取并返回一个数据样本。这是 Dataset 类的必要方法。

        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: 包含图像、掩码和图像路径的元组。
        """
        # 构建指定索引的图像文件的完整路径
        img_path = os.path.join(self.image_dir, self.images[idx])
        # 构建对应掩码文件的完整路径。这里假设掩码文件名与图像文件名相同，但扩展名不同（例如 .jpg -> .png）
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        
        # 使用 Pillow 的 Image.open 打开图像文件，并使用 .convert("RGB") 确保图像是三通道的 RGB 格式
        # 然后使用 np.array 将图像转换为 NumPy 数组，方便后续处理
        image = np.array(Image.open(img_path).convert("RGB"))
        # 使用 Pillow 的 Image.open 打开掩码文件，并使用 .convert("L") 将其转换为单通道的灰度图
        # 将其数据类型设置为 np.float32，因为模型通常需要 float 类型的输入
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # 在分割任务中，通常用 0 表示背景，用 1 表示目标物体。这里将掩码中的 255.0（通常是白色）像素值替换为 1.0
        mask[mask == 255.0] = 1.0

        if self.transform:  # 检查是否提供了数据转换/增强函数
            # 如果提供了 transform，则将其应用于图像和掩码
            # 这通常用于数据增强，如旋转、缩放、裁剪等，以提高模型的泛化能力
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]  # 获取增强后的图像
            mask = augmentations["mask"]  # 获取增强后的掩码

        return image, mask, img_path  # 返回处理好的图像、掩码和原始图像路径，供 DataLoader 使用