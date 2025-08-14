import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image

# 导入我们自己定义的模型和数据集
from model import UNet
from dataset import CustomDataset

class UNetLightning(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNetLightning, self).__init__() # 调用父类的构造函数
        self.n_channels = n_channels # 初始化输入通道数
        self.n_classes = n_classes # 初始化输出类别数

        self.model = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=True) # 实例化U-Net模型
        self.loss_fn = torch.nn.BCEWithLogitsLoss() # 定义损失函数为带有Logits的二元交叉熵损失，它结合了Sigmoid和BCELoss，更稳定

    def forward(self, x):
        return self.model(x) # 定义前向传播过程

    def training_step(self, batch, batch_idx):
        x, y = batch # 从批次中解包输入图像和真实掩码
        y_hat = self(x) # 执行前向传播，得到预测结果
        loss = self.loss_fn(y_hat, y) # 计算预测值和真实值之间的损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # 记录训练损失，用于监控
        return loss # 返回损失值，PyTorch Lightning会自动处理反向传播

    def validation_step(self, batch, batch_idx):
        x, y = batch # 从批次中解包输入图像和真实掩码
        y_hat = self(x) # 执行前向传播，得到预测结果
        loss = self.loss_fn(y_hat, y) # 计算验证损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # 记录验证损失
        return loss # 返回验证损失

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # 配置优化器，这里使用Adam，并设置学习率
        return optimizer # 返回优化器

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 2, num_workers: int = 2):
        super().__init__() # 调用父类的构造函数
        self.data_dir = data_dir # 数据集所在的目录
        self.batch_size = batch_size # 设置每个批次的样本数
        self.num_workers = num_workers # 设置用于数据加载的子进程数
        self.transform = transforms.Compose([ # 定义图像预处理的流程
            transforms.ToTensor(), # 将PIL图像或numpy.ndarray转换为FloatTensor
            transforms.Resize((256, 256)), # 将图像大小调整为256x256
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 使用ImageNet的均值和标准差进行归一化
        ])

    def setup(self, stage=None):
        # 创建虚拟数据，用于演示
        image_dir = os.path.join(self.data_dir, 'images') # 定义图像存储路径
        mask_dir = os.path.join(self.data_dir, 'masks') # 定义掩码存储路径
        os.makedirs(image_dir, exist_ok=True) # 创建图像目录
        os.makedirs(mask_dir, exist_ok=True) # 创建掩码目录

        # 生成一些随机的图像和掩码文件
        for i in range(10):
            img = Image.new('RGB', (512, 512), color = 'red') # 创建一个红色的512x512图像
            img.save(os.path.join(image_dir, f'dummy_{i}.jpg')) # 保存为jpg文件
            mask = Image.new('L', (512, 512), color = 255) # 创建一个白色的512x512灰度掩码
            mask.save(os.path.join(mask_dir, f'dummy_{i}.png')) # 保存为png文件

        # 实例化完整的数据集
        full_dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, transform=self.transform) # 加载数据集
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset)) # 计算训练集大小（80%）
        val_size = len(full_dataset) - train_size # 计算验证集大小（20%）
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size]) # 随机划分数据集

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True) # 创建训练数据加载器，并打乱顺序

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) # 创建验证数据加载器，不打乱顺序

if __name__ == '__main__':
    # 初始化数据模块
    data_module = UNetDataModule(data_dir='./data', batch_size=2) # 实例化数据模块

    # 初始化模型
    model = UNetLightning(n_channels=3, n_classes=1) # 实例化U-Net的Lightning模块

    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=5, # 设置最大训练轮数
        gpus=1 if torch.cuda.is_available() else 0, # 如果有可用的GPU，则使用1个GPU
        progress_bar_refresh_rate=20 # 设置进度条的刷新频率
    )

    # 开始训练
    trainer.fit(model, data_module) # 调用fit方法开始训练和验证