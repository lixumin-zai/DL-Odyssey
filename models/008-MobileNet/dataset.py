import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 定义一个伪图像数据集 (用于演示)
class FakeImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, img_size=(224, 224)):
        """
        初始化伪图像数据集
        :param num_samples: 样本数量
        :param num_classes: 类别数量
        :param img_size: 图像尺寸
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        # 定义图像预处理流程
        self.transform = transforms.Compose([
            transforms.Resize(img_size), # 调整图像大小
            transforms.ToTensor(), # 将图像转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
        ])

    def __len__(self):
        """ 返回数据集中的样本数量 """
        return self.num_samples

    def __getitem__(self, idx):
        """
        获取一个样本
        :param idx: 样本索引
        :return: (图像, 标签)
        """
        # 生成一个随机的RGB图像
        random_image = torch.randn(3, *self.img_size)
        # 将Tensor转换为PIL图像，以便进行transform
        pil_image = transforms.ToPILImage()(random_image)
        # 应用预处理
        img = self.transform(pil_image)
        # 生成一个随机标签
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label

# 创建数据加载器
def create_dataloaders(batch_size=32, num_workers=4):
    """
    创建训练和验证数据加载器
    :param batch_size: 批处理大小
    :param num_workers: 工作线程数
    :return: (训练数据加载器, 验证数据加载器)
    """
    # 创建训练数据集实例
    train_dataset = FakeImageDataset(num_samples=1000, num_classes=10)
    # 创建验证数据集实例
    val_dataset = FakeImageDataset(num_samples=200, num_classes=10)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # 打乱数据
        num_workers=num_workers, # 使用多线程加载数据
        pin_memory=True # 将数据加载到CUDA固定内存中，以加快传输速度
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, # 不打乱数据
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == '__main__':
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(batch_size=4)
    # 从训练加载器中获取一个批次的数据
    images, labels = next(iter(train_loader))
    # 打印图像和标签的形状
    print("图像批次的形状:", images.shape) # 应该为 (4, 3, 224, 224)
    print("标签批次的形状:", labels.shape) # 应该为 (4,)
    print("标签:", labels)