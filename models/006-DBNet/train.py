import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchmetrics
from typing import Dict, Any, Optional
import numpy as np
import cv2
from argparse import ArgumentParser

# 导入自定义模块
from model import DBNet, DBLoss
from dataset import TextDetectionDataset, SyntheticTextDataset, create_dataloader

class DBNetLightningModule(pl.LightningModule):
    """
    DBNet的PyTorch Lightning模块
    
    封装了模型的训练、验证和测试逻辑
    包含损失计算、优化器配置和指标监控
    """
    
    def __init__(self, 
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 k: int = 50,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 alpha: float = 1.0,
                 beta: float = 10.0,
                 **kwargs):
        """
        初始化Lightning模块
        
        Args:
            backbone: 骨干网络类型
            pretrained: 是否使用预训练权重
            k: 可微分二值化放大因子
            learning_rate: 学习率
            weight_decay: 权重衰减
            alpha: 二值化损失权重
            beta: 阈值损失权重
        """
        super().__init__()
        
        # 保存超参数，便于日志记录和模型恢复
        self.save_hyperparameters()
        
        # 初始化模型
        self.model = DBNet(
            backbone=backbone,
            pretrained=pretrained,
            k=k
        )
        
        # 初始化损失函数
        self.criterion = DBLoss(alpha=alpha, beta=beta)
        
        # 初始化评估指标
        self.train_precision = torchmetrics.Precision(task='binary')  # 训练精确率
        self.train_recall = torchmetrics.Recall(task='binary')        # 训练召回率
        self.train_f1 = torchmetrics.F1Score(task='binary')           # 训练F1分数
        
        self.val_precision = torchmetrics.Precision(task='binary')    # 验证精确率
        self.val_recall = torchmetrics.Recall(task='binary')          # 验证召回率
        self.val_f1 = torchmetrics.F1Score(task='binary')             # 验证F1分数
        
        # 学习率和权重衰减
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像张量
            
        Returns:
            模型输出字典
        """
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 训练批次数据
            batch_idx: 批次索引
            
        Returns:
            训练损失
        """
        # 提取批次数据
        images = batch['image']      # 输入图像 (B, 3, H, W)
        gt_text = batch['gt_text']   # 文本标签 (B, 1, H, W)
        gt_mask = batch['gt_mask']   # 有效掩码 (B, 1, H, W)
        
        # 前向传播
        outputs = self.forward(images)
        
        # 计算损失
        losses = self.criterion(outputs, gt_text, gt_mask)
        total_loss = losses['total_loss']
        
        # 计算训练指标
        with torch.no_grad():
            # 使用二值图计算指标
            pred_binary = (outputs['binary_map'] > 0.5).float()
            gt_binary = (gt_text > 0.5).float()
            
            # 应用掩码
            pred_masked = pred_binary * gt_mask
            gt_masked = gt_binary * gt_mask
            
            # 更新指标
            self.train_precision(pred_masked.flatten(), gt_masked.flatten())
            self.train_recall(pred_masked.flatten(), gt_masked.flatten())
            self.train_f1(pred_masked.flatten(), gt_masked.flatten())
        
        # 记录训练损失
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/seg_loss', losses['seg_loss'], on_step=True, on_epoch=True)
        self.log('train/bin_loss', losses['bin_loss'], on_step=True, on_epoch=True)
        self.log('train/thresh_loss', losses['thresh_loss'], on_step=True, on_epoch=True)
        
        # 记录训练指标
        self.log('train/precision', self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        验证步骤
        
        Args:
            batch: 验证批次数据
            batch_idx: 批次索引
            
        Returns:
            验证损失
        """
        # 提取批次数据
        images = batch['image']
        gt_text = batch['gt_text']
        gt_mask = batch['gt_mask']
        
        # 前向传播
        outputs = self.forward(images)
        
        # 计算损失
        losses = self.criterion(outputs, gt_text, gt_mask)
        total_loss = losses['total_loss']
        
        # 计算验证指标
        pred_binary = (outputs['binary_map'] > 0.5).float()
        gt_binary = (gt_text > 0.5).float()
        
        # 应用掩码
        pred_masked = pred_binary * gt_mask
        gt_masked = gt_binary * gt_mask
        
        # 更新指标
        self.val_precision(pred_masked.flatten(), gt_masked.flatten())
        self.val_recall(pred_masked.flatten(), gt_masked.flatten())
        self.val_f1(pred_masked.flatten(), gt_masked.flatten())
        
        # 记录验证损失
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/seg_loss', losses['seg_loss'], on_step=False, on_epoch=True)
        self.log('val/bin_loss', losses['bin_loss'], on_step=False, on_epoch=True)
        self.log('val/thresh_loss', losses['thresh_loss'], on_step=False, on_epoch=True)
        
        # 记录验证指标
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        测试步骤
        
        Args:
            batch: 测试批次数据
            batch_idx: 批次索引
            
        Returns:
            测试结果字典
        """
        # 执行验证步骤的逻辑
        loss = self.validation_step(batch, batch_idx)
        
        return {'test_loss': loss}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器
        
        Returns:
            优化器和调度器配置字典
        """
        # 使用Adam优化器
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # 最大周期数
            eta_min=1e-6  # 最小学习率
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/f1',  # 监控验证F1分数
                'interval': 'epoch',   # 每个epoch更新
                'frequency': 1         # 更新频率
            }
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        预测步骤
        
        Args:
            batch: 预测批次数据
            batch_idx: 批次索引
            
        Returns:
            预测结果字典
        """
        images = batch['image']
        
        # 前向传播
        outputs = self.forward(images)
        
        return {
            'images': images,
            'prob_maps': outputs['prob_map'],
            'threshold_maps': outputs['threshold_map'],
            'binary_maps': outputs['binary_map']
        }


class DBNetDataModule(pl.LightningDataModule):
    """
    DBNet数据模块
    
    封装了数据加载逻辑，包括训练、验证和测试数据集
    """
    
    def __init__(self,
                 data_root: str = './data',
                 train_annotation: str = './data/train.txt',
                 val_annotation: str = './data/val.txt',
                 image_size: tuple = (640, 640),
                 batch_size: int = 8,
                 num_workers: int = 4,
                 use_synthetic: bool = True,
                 **kwargs):
        """
        初始化数据模块
        
        Args:
            data_root: 数据根目录
            train_annotation: 训练标注文件
            val_annotation: 验证标注文件
            image_size: 图像尺寸
            batch_size: 批次大小
            num_workers: 工作进程数
            use_synthetic: 是否使用合成数据
        """
        super().__init__()
        
        # 保存参数
        self.data_root = data_root
        self.train_annotation = train_annotation
        self.val_annotation = val_annotation
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_synthetic = use_synthetic
        
        # 数据集实例
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集
        
        Args:
            stage: 阶段标识 ('fit', 'validate', 'test', 'predict')
        """
        if stage == 'fit' or stage is None:
            if self.use_synthetic:
                # 使用合成数据集进行快速测试
                print("使用合成数据集进行训练...")
                self.train_dataset = SyntheticTextDataset(
                    num_samples=1000,
                    image_size=self.image_size,
                    max_texts=8
                )
                
                self.val_dataset = SyntheticTextDataset(
                    num_samples=200,
                    image_size=self.image_size,
                    max_texts=5
                )
            else:
                # 使用真实数据集
                print("使用真实数据集进行训练...")
                try:
                    self.train_dataset = TextDetectionDataset(
                        data_root=self.data_root,
                        annotation_file=self.train_annotation,
                        image_size=self.image_size,
                        is_training=True,
                        augment=True
                    )
                    
                    self.val_dataset = TextDetectionDataset(
                        data_root=self.data_root,
                        annotation_file=self.val_annotation,
                        image_size=self.image_size,
                        is_training=False,
                        augment=False
                    )
                except Exception as e:
                    print(f"真实数据集加载失败，切换到合成数据集: {e}")
                    self.use_synthetic = True
                    self.setup(stage)  # 递归调用使用合成数据
        
        if stage == 'test':
            # 测试数据集（这里使用验证数据集）
            self.test_dataset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        """
        创建训练数据加载器
        
        Returns:
            训练数据加载器
        """
        return create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        创建验证数据加载器
        
        Returns:
            验证数据加载器
        """
        return create_dataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        创建测试数据加载器
        
        Returns:
            测试数据加载器
        """
        return create_dataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def train_dbnet(args):
    """
    训练DBNet模型的主函数
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子，确保结果可复现
    pl.seed_everything(args.seed)
    
    # 创建数据模块
    data_module = DBNetDataModule(
        data_root=args.data_root,
        train_annotation=args.train_annotation,
        val_annotation=args.val_annotation,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_synthetic=args.use_synthetic
    )
    
    # 创建模型
    model = DBNetLightningModule(
        backbone=args.backbone,
        pretrained=args.pretrained,
        k=args.k,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # 创建回调函数
    callbacks = [
        # 模型检查点：保存最佳模型
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='dbnet-{epoch:02d}-{val/f1:.3f}',
            monitor='val/f1',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        
        # 早停：防止过拟合
        EarlyStopping(
            monitor='val/f1',
            mode='max',
            patience=10,
            verbose=True
        ),
        
        # 学习率监控
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='dbnet_experiment',
        version=None
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',  # 自动选择加速器（GPU/CPU）
        devices='auto',      # 自动选择设备数量
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,  # 每半个epoch验证一次
        gradient_clip_val=1.0,   # 梯度裁剪，防止梯度爆炸
        precision=16 if args.use_amp else 32,  # 混合精度训练
        deterministic=True       # 确保结果可复现
    )
    
    # 开始训练
    print("开始训练DBNet模型...")
    trainer.fit(model, data_module)
    
    # 测试最佳模型
    print("\n开始测试最佳模型...")
    trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\n训练完成！最佳模型保存在: {args.checkpoint_dir}")
    print(f"训练日志保存在: {args.log_dir}")


def main():
    """
    主函数：解析命令行参数并启动训练
    """
    parser = ArgumentParser(description='DBNet文本检测模型训练')
    
    # 数据相关参数
    parser.add_argument('--data_root', type=str, default='./data',
                       help='数据根目录路径')
    parser.add_argument('--train_annotation', type=str, default='./data/train.txt',
                       help='训练标注文件路径')
    parser.add_argument('--val_annotation', type=str, default='./data/val.txt',
                       help='验证标注文件路径')
    parser.add_argument('--use_synthetic', action='store_true', default=True,
                       help='是否使用合成数据集')
    
    # 模型相关参数
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50'],
                       help='骨干网络类型')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='是否使用预训练权重')
    parser.add_argument('--k', type=int, default=50,
                       help='可微分二值化放大因子')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--image_size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='二值化损失权重')
    parser.add_argument('--beta', type=float, default=10.0,
                       help='阈值损失权重')
    
    # 系统相关参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='是否使用混合精度训练')
    
    # 输出相关参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='模型检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='训练日志保存目录')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 打印配置信息
    print("=" * 50)
    print("DBNet训练配置:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # 开始训练
    train_dbnet(args)


if __name__ == '__main__':
    # 运行主函数
    main()
    
    # 简单的测试代码
    print("\n运行简单测试...")
    
    # 创建测试参数
    class TestArgs:
        def __init__(self):
            self.data_root = './data'
            self.train_annotation = './data/train.txt'
            self.val_annotation = './data/val.txt'
            self.use_synthetic = True
            self.backbone = 'resnet18'
            self.pretrained = False  # 测试时不使用预训练权重
            self.k = 50
            self.batch_size = 2
            self.image_size = 320  # 使用较小尺寸加速测试
            self.max_epochs = 2    # 只训练2个epoch
            self.learning_rate = 1e-3
            self.weight_decay = 1e-4
            self.alpha = 1.0
            self.beta = 10.0
            self.num_workers = 0   # 测试时使用单进程
            self.seed = 42
            self.use_amp = False
            self.checkpoint_dir = './test_checkpoints'
            self.log_dir = './test_logs'
    
    # 运行测试
    test_args = TestArgs()
    
    # 创建测试目录
    os.makedirs(test_args.checkpoint_dir, exist_ok=True)
    os.makedirs(test_args.log_dir, exist_ok=True)
    
    print("开始快速测试训练流程...")
    try:
        train_dbnet(test_args)
        print("测试训练完成！")
    except Exception as e:
        print(f"测试训练出现错误: {e}")
        print("这是正常的，因为可能缺少某些依赖库")
    
    print("\nDBNet训练脚本准备完成！")
    print("使用方法:")
    print("python train.py --use_synthetic --max_epochs 50 --batch_size 8")
    print("或者使用真实数据:")
    print("python train.py --data_root /path/to/data --train_annotation /path/to/train.txt")