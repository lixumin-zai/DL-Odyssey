import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir # 存储图像的目录路径
        self.mask_dir = mask_dir # 存储掩码的目录路径
        self.transform = transform # 数据预处理和增强的操作
        self.images = os.listdir(image_dir) # 获取图像目录下的所有文件名

    def __len__(self):
        return len(self.images) # 返回数据集中图像的总数

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) # 构建单个图像的完整路径
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png")) # 构建对应掩码的完整路径，假设掩码是png格式
        image = np.array(Image.open(img_path).convert("RGB")) # 打开图像并转换为RGB格式的numpy数组
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # 打开掩码并转换为灰度图格式的numpy数组
        mask[mask == 255.0] = 1.0 # 将掩码中的255.0（通常是白色）转换为1.0，代表前景

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) # 对图像和掩码应用预处理/增强
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask