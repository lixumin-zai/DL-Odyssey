import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import argparse
import os
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
from pathlib import Path

# 导入我们自定义的模块
from model import NaViT, navit_tiny, navit_small, navit_base, navit_large
from dataset import create_dataloaders

class NaViTLightningModule(pl.LightningModule):
    """NaViT的PyTorch Lightning模块
    
    封装了模型的训练、验证和测试逻辑
    """
    
    def __init__(
        self,
        model_name: str = 'navit_base',          # 模型名称
        num_classes: int = 10,                   # 分类类别数
        learning_rate: float = 1e-4,             # 学习率
        weight_decay: float = 0.05,              # 权重衰减
        warmup_epochs: int = 10,                 # 预热轮数
        max_epochs: int = 100,                   # 最大训练轮数
        optimizer_name: str = 'adamw',           # 优化器名称
        scheduler_name: str = 'cosine',          # 学习率调度器名称
        label_smoothing: float = 0.1,            # 标签平滑
        **model_kwargs                           # 模型的其他参数
    ):
        super().__init__()
        
        # 保存超参数，这样可以在日志中查看
        self.save_hyperparameters()
        
        # 根据模型名称创建模型
        self.model = self._create_model(model_name, num_classes, **model_kwargs)
        
        # 定义损失函数，使用标签平滑的交叉熵
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # 定义评估指标
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
        # Top-5准确率（如果类别数大于5）
        if num_classes > 5:
            self.train_top5_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
            self.val_top5_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
            self.test_top5_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        else:
            self.train_top5_accuracy = None
            self.val_top5_accuracy = None
            self.test_top5_accuracy = None
        
        # 混淆矩阵（仅用于验证和测试）
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        
        # 用于存储验证和测试结果
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _create_model(self, model_name: str, num_classes: int, **kwargs) -> NaViT:
        """根据模型名称创建模型实例
        
        Args:
            model_name: 模型名称
            num_classes: 分类类别数
            **kwargs: 模型的其他参数
            
        Returns:
            创建的模型实例
        """
        # 预定义的模型配置
        model_configs = {
            'navit_tiny': navit_tiny,
            'navit_small': navit_small,
            'navit_base': navit_base,
            'navit_large': navit_large
        }
        
        if model_name in model_configs:
            # 使用预定义配置
            model = model_configs[model_name](num_classes=num_classes, **kwargs)
        else:
            # 使用自定义配置
            model = NaViT(num_classes=num_classes, **kwargs)
        
        return model
    
    def forward(self, x: Union[torch.Tensor, List[List[torch.Tensor]]]) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        return self.model(x)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            损失值
        """
        images, labels = batch
        
        # 前向传播
        logits = self(images)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        
        # 计算Top-5准确率（如果适用）
        if self.train_top5_accuracy is not None:
            self.train_top5_accuracy(logits, labels)
        
        # 记录指标
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.train_top5_accuracy is not None:
            self.log('train/top5_accuracy', self.train_top5_accuracy, on_step=False, on_epoch=True)
        
        # 记录学习率
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            包含损失和预测结果的字典
        """
        images, labels = batch
        
        # 前向传播
        logits = self(images)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        self.val_accuracy(preds, labels)
        self.val_confusion_matrix(preds, labels)
        
        if self.val_top5_accuracy is not None:
            self.val_top5_accuracy(logits, labels)
        
        # 存储结果用于epoch结束时的处理
        output = {
            'val_loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        }
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # 记录指标
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/accuracy', self.val_accuracy, prog_bar=True)
        
        if self.val_top5_accuracy is not None:
            self.log('val/top5_accuracy', self.val_top5_accuracy)
        
        # 清空存储的输出
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """测试步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            包含损失和预测结果的字典
        """
        images, labels = batch
        
        # 前向传播
        logits = self(images)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 计算预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 更新指标
        self.test_accuracy(preds, labels)
        self.test_confusion_matrix(preds, labels)
        
        if self.test_top5_accuracy is not None:
            self.test_top5_accuracy(logits, labels)
        
        # 存储结果
        output = {
            'test_loss': loss,
            'preds': preds,
            'labels': labels,
            'logits': logits
        }
        self.test_step_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self) -> None:
        """测试epoch结束时的处理"""
        # 计算平均损失
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        
        # 记录指标
        self.log('test/loss', avg_loss)
        self.log('test/accuracy', self.test_accuracy)
        
        if self.test_top5_accuracy is not None:
            self.log('test/top5_accuracy', self.test_top5_accuracy)
        
        # 打印混淆矩阵
        confusion_matrix = self.test_confusion_matrix.compute()
        print(f"\n测试集混淆矩阵:\n{confusion_matrix}")
        
        # 清空存储的输出
        self.test_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器
        
        Returns:
            包含优化器和调度器配置的字典
        """
        # 创建优化器
        if self.hparams.optimizer_name.lower() == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),  # AdamW的默认beta值
                eps=1e-8             # 数值稳定性参数
            )
        elif self.hparams.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.hparams.optimizer_name}")
        
        # 创建学习率调度器
        if self.hparams.scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,  # 余弦退火的周期
                eta_min=1e-6                    # 最小学习率
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',  # 每个epoch更新一次
                'frequency': 1
            }
        elif self.hparams.scheduler_name.lower() == 'onecycle':
            # 需要知道总的训练步数
            # 这里假设我们有trainer对象，实际使用时需要传入steps_per_epoch
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                epochs=self.hparams.max_epochs,
                steps_per_epoch=100,  # 这个值需要根据实际数据集大小调整
                pct_start=0.1,        # 预热阶段占总训练的10%
                anneal_strategy='cos'  # 使用余弦退火
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',   # 每个step更新一次
                'frequency': 1
            }
        else:
            # 不使用学习率调度器
            return {'optimizer': optimizer}
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_config
        }
    
    def on_train_epoch_start(self) -> None:
        """训练epoch开始时的处理"""
        # 打印当前epoch信息
        print(f"\n开始训练 Epoch {self.current_epoch + 1}/{self.trainer.max_epochs}")
        print(f"当前学习率: {self.optimizers().param_groups[0]['lr']:.2e}")
    
    def on_validation_epoch_start(self) -> None:
        """验证epoch开始时的处理"""
        print(f"开始验证 Epoch {self.current_epoch + 1}")

def create_callbacks(save_dir: str, monitor_metric: str = 'val/accuracy') -> List[pl.Callback]:
    """创建训练回调函数
    
    Args:
        save_dir: 模型保存目录
        monitor_metric: 监控的指标名称
        
    Returns:
        回调函数列表
    """
    callbacks = []
    
    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename='navit-{epoch:02d}-{val_accuracy:.4f}',
        monitor=monitor_metric,
        mode='max',           # 监控指标越大越好
        save_top_k=3,         # 保存最好的3个模型
        save_last=True,       # 保存最后一个模型
        verbose=True          # 打印保存信息
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode='max',           # 监控指标越大越好
        patience=15,          # 15个epoch没有改善就停止
        verbose=True,         # 打印早停信息
        strict=False          # 允许监控指标暂时不可用
    )
    callbacks.append(early_stopping_callback)
    
    # 学习率监控回调
    lr_monitor_callback = LearningRateMonitor(
        logging_interval='step'  # 每个step记录学习率
    )
    callbacks.append(lr_monitor_callback)
    
    return callbacks

def train_navit(
    data_dir: str,
    model_name: str = 'navit_base',
    num_classes: int = 10,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    use_cifar10: bool = True,
    pack_sequences: bool = True,
    max_seq_len: int = 2048,
    save_dir: str = './outputs',
    gpus: int = 1,
    precision: str = '16-mixed',
    resume_from_checkpoint: Optional[str] = None
) -> NaViTLightningModule:
    """训练NaViT模型
    
    Args:
        data_dir: 数据目录
        model_name: 模型名称
        num_classes: 分类类别数
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        use_cifar10: 是否使用CIFAR-10数据集
        pack_sequences: 是否启用序列打包
        max_seq_len: 最大序列长度
        save_dir: 保存目录
        gpus: GPU数量
        precision: 训练精度
        resume_from_checkpoint: 从检查点恢复训练
        
    Returns:
        训练好的模型
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_cifar10=use_cifar10,
        pack_sequences=pack_sequences,
        max_seq_len=max_seq_len
    )
    
    # 创建模型
    print(f"创建模型: {model_name}")
    model = NaViTLightningModule(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        # 传递模型特定参数
        token_dropout_prob=0.1 if pack_sequences else 0.0  # 序列打包时使用token dropout
    )
    
    # 创建回调函数
    callbacks = create_callbacks(save_dir)
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='navit_logs',
        version=None  # 自动生成版本号
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,     # 每50步记录一次日志
        val_check_interval=1.0,   # 每个epoch验证一次
        gradient_clip_val=1.0,    # 梯度裁剪
        deterministic=False,      # 为了性能，不使用确定性训练
        enable_progress_bar=True, # 显示进度条
        enable_model_summary=True # 显示模型摘要
    )
    
    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  模型名称: {model_name}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 开始训练
    print("\n开始训练...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint
    )
    
    # 测试模型
    print("\n开始测试...")
    trainer.test(model, dataloaders=test_loader)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练NaViT模型')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录路径')
    parser.add_argument('--use_cifar10', action='store_true', default=True,
                        help='是否使用CIFAR-10数据集')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='分类类别数')
    
    # 模型相关参数
    parser.add_argument('--model_name', type=str, default='navit_base',
                        choices=['navit_tiny', 'navit_small', 'navit_base', 'navit_large'],
                        help='模型名称')
    parser.add_argument('--pack_sequences', action='store_true', default=True,
                        help='是否启用序列打包')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='最大序列长度')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='最大训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减')
    
    # 系统相关参数
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['16-mixed', '32', 'bf16-mixed'],
                        help='训练精度')
    parser.add_argument('--save_dir', type=str, default='./outputs',
                        help='保存目录')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # 开始训练
    model = train_navit(
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_cifar10=args.use_cifar10,
        pack_sequences=args.pack_sequences,
        max_seq_len=args.max_seq_len,
        save_dir=args.save_dir,
        gpus=args.gpus,
        precision=args.precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    print("\n训练完成！")

if __name__ == '__main__':
    main()