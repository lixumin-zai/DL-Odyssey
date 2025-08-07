import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model import MobileNetV2
from dataset import create_dataloaders

# 定义一个基于 PyTorch Lightning 的模型模块
class MobileNetV2Lightning(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        """
        初始化 Lightning 模块
        :param num_classes: 分类数
        :param learning_rate: 学习率
        """
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        # 实例化 MobileNetV2 模型
        self.model = MobileNetV2(num_classes=num_classes)
        # 定义损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 模型输出
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        定义单个训练步的操作
        :param batch: 一个批次的数据
        :param batch_idx: 批次索引
        :return: 训练损失
        """
        # 从批次中解包图像和标签
        images, labels = batch
        # 获取模型预测
        predictions = self(images)
        # 计算损失
        loss = self.loss_fn(predictions, labels)
        # 记录训练损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        定义单个验证步的操作
        :param batch: 一个批次的数据
        :param batch_idx: 批次索引
        """
        # 从批次中解包图像和标签
        images, labels = batch
        # 获取模型预测
        predictions = self(images)
        # 计算损失
        loss = self.loss_fn(predictions, labels)
        # 计算准确率
        acc = (predictions.argmax(dim=1) == labels).float().mean()
        # 记录验证损失和准确率
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        配置优化器
        :return: 优化器
        """
        # 使用 Adam 优化器
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def main():
    """ 主训练函数 """
    # --- 配置 ---
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_CLASSES = 10
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 5 # 为了快速演示，只训练5个周期

    # --- 数据加载 ---
    train_loader, val_loader = create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # --- 模型初始化 ---
    model = MobileNetV2Lightning(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

    # --- 训练器初始化 ---
    # 检查是否有可用的GPU
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, # 最大训练周期
        accelerator=accelerator, # 使用的硬件 (gpu或cpu)
        devices=1, # 使用的设备数量
        log_every_n_steps=10, # 每10步记录一次日志
    )

    # --- 开始训练 ---
    print(f"开始在 {accelerator.upper()} 上训练...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("训练完成！")

if __name__ == '__main__':
    main()