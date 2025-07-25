import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from typing import Dict, Any, Optional
import os
import argparse
from pathlib import Path

# 导入我们自定义的模块
from model import SigLIPModel, SigLIPLoss, create_siglip_model
from dataset import SigLIPDataModule, create_sample_annotations

class SigLIPLightningModule(pl.LightningModule):
    """SigLIP的PyTorch Lightning模块，封装训练、验证和测试逻辑"""
    
    def __init__(self,
                 model_size: str = 'base',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 100000,
                 temperature_init: float = 1.0,
                 **model_kwargs):
        """
        初始化Lightning模块
        
        Args:
            model_size: 模型大小 ('tiny', 'small', 'base', 'large')
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            max_steps: 最大训练步数
            temperature_init: 温度参数初始值
            **model_kwargs: 传递给模型的额外参数
        """
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 创建模型
        self.model = create_siglip_model(model_size)
        
        # 如果提供了温度初始值，更新模型的温度参数
        if temperature_init != 1.0:
            self.model.temperature.data = torch.tensor(temperature_init)
        
        # 创建损失函数
        self.criterion = SigLIPLoss()
        
        # 创建评估指标
        self.train_accuracy = torchmetrics.Accuracy(task='binary')  # 二分类准确率
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        
        # 用于计算图像-文本检索指标
        self.val_image_features = []  # 存储验证集的图像特征
        self.val_text_features = []   # 存储验证集的文本特征
        
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None):
        """前向传播"""
        return self.model(images, input_ids, attention_mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        # 从batch中提取数据
        images = batch['images']  # shape: (batch_size, 3, 224, 224)
        input_ids = batch['input_ids']  # shape: (batch_size, max_length)
        attention_mask = batch['attention_mask']  # shape: (batch_size, max_length)
        
        # 前向传播
        image_features, text_features, similarity_matrix = self.forward(
            images, input_ids, attention_mask
        )
        
        # 计算损失
        loss = self.criterion(similarity_matrix)
        
        # 计算准确率（将相似度矩阵转换为二分类问题）
        batch_size = similarity_matrix.shape[0]
        # 创建标签：对角线为1（正样本），其他为0（负样本）
        labels = torch.eye(batch_size, device=similarity_matrix.device)
        # 将相似度通过sigmoid转换为概率
        probs = torch.sigmoid(similarity_matrix)
        # 计算准确率
        self.train_accuracy(probs.flatten(), labels.flatten().int())
        
        # 记录指标
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True)
        self.log('train/temperature', self.model.temperature, on_step=True)
        
        # 记录学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/learning_rate', current_lr, on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        # 从batch中提取数据
        images = batch['images']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # 前向传播
        image_features, text_features, similarity_matrix = self.forward(
            images, input_ids, attention_mask
        )
        
        # 计算损失
        loss = self.criterion(similarity_matrix)
        
        # 计算准确率
        batch_size = similarity_matrix.shape[0]
        labels = torch.eye(batch_size, device=similarity_matrix.device)
        probs = torch.sigmoid(similarity_matrix)
        self.val_accuracy(probs.flatten(), labels.flatten().int())
        
        # 记录指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        
        # 存储特征用于计算检索指标
        self.val_image_features.append(image_features.detach())
        self.val_text_features.append(text_features.detach())
        
        return loss
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的处理"""
        # 如果有存储的特征，计算检索指标
        if self.val_image_features and self.val_text_features:
            # 合并所有batch的特征
            all_image_features = torch.cat(self.val_image_features, dim=0)
            all_text_features = torch.cat(self.val_text_features, dim=0)
            
            # 计算图像到文本的检索指标
            i2t_recall_at_1, i2t_recall_at_5 = self._compute_retrieval_metrics(
                all_image_features, all_text_features
            )
            
            # 计算文本到图像的检索指标
            t2i_recall_at_1, t2i_recall_at_5 = self._compute_retrieval_metrics(
                all_text_features, all_image_features
            )
            
            # 记录检索指标
            self.log('val/i2t_recall@1', i2t_recall_at_1)
            self.log('val/i2t_recall@5', i2t_recall_at_5)
            self.log('val/t2i_recall@1', t2i_recall_at_1)
            self.log('val/t2i_recall@5', t2i_recall_at_5)
            
            # 计算平均检索性能
            avg_recall = (i2t_recall_at_1 + i2t_recall_at_5 + t2i_recall_at_1 + t2i_recall_at_5) / 4
            self.log('val/avg_recall', avg_recall)
        
        # 清空存储的特征
        self.val_image_features.clear()
        self.val_text_features.clear()
    
    def _compute_retrieval_metrics(self, query_features: torch.Tensor, 
                                 gallery_features: torch.Tensor) -> tuple:
        """计算检索指标（Recall@1和Recall@5）"""
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(query_features, gallery_features.T)
        
        # 获取排序后的索引
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)
        
        # 计算Recall@1和Recall@5
        num_queries = query_features.shape[0]
        recall_at_1 = 0
        recall_at_5 = 0
        
        for i in range(num_queries):
            # 正确答案是索引i（假设query和gallery是对应的）
            correct_idx = i
            top_5_indices = sorted_indices[i, :5]
            
            # 检查正确答案是否在top-1中
            if correct_idx == top_5_indices[0]:
                recall_at_1 += 1
            
            # 检查正确答案是否在top-5中
            if correct_idx in top_5_indices:
                recall_at_5 += 1
        
        recall_at_1 = recall_at_1 / num_queries
        recall_at_5 = recall_at_5 / num_queries
        
        return recall_at_1, recall_at_5
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 创建优化器
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 创建学习率调度器
        # 1. 预热阶段：线性增长到目标学习率
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,  # 从1%的学习率开始
            end_factor=1.0,     # 增长到100%的学习率
            total_iters=self.hparams.warmup_steps
        )
        
        # 2. 主训练阶段：余弦退火
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps - self.hparams.warmup_steps,
            eta_min=self.hparams.learning_rate * 0.01  # 最小学习率为初始学习率的1%
        )
        
        # 3. 组合调度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 每个step更新学习率
                'frequency': 1
            }
        }

def create_callbacks(save_dir: str) -> list:
    """创建训练回调函数"""
    callbacks = []
    
    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, 'checkpoints'),
        filename='siglip-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',  # 监控验证损失
        mode='min',          # 损失越小越好
        save_top_k=3,        # 保存最好的3个模型
        save_last=True,      # 保存最后一个模型
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=10,         # 10个epoch没有改善就停止
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 学习率监控回调
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks

def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练SigLIP模型')
    parser.add_argument('--data_dir', type=str, default='./sample_data', help='数据目录')
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'], help='模型大小')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='预热步数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--precision', type=str, default='16-mixed', help='训练精度')
    parser.add_argument('--save_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--create_sample_data', action='store_true', help='创建示例数据')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 如果需要，创建示例数据
    if args.create_sample_data:
        print("创建示例数据...")
        os.makedirs(args.data_dir, exist_ok=True)
        train_file, val_file = create_sample_annotations(args.data_dir, num_samples=1000)
    else:
        # 使用现有的标注文件
        train_file = os.path.join(args.data_dir, 'train_annotations.json')
        val_file = os.path.join(args.data_dir, 'val_annotations.json')
    
    # 创建数据模块
    print("设置数据模块...")
    data_module = SigLIPDataModule(
        data_dir=args.data_dir,
        train_annotations=train_file,
        val_annotations=val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.setup()
    
    # 计算最大训练步数
    max_steps = len(data_module.train_dataloader()) * args.max_epochs
    
    # 创建模型
    print(f"创建{args.model_size}大小的SigLIP模型...")
    model = SigLIPLightningModule(
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 创建回调函数
    callbacks = create_callbacks(args.save_dir)
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name='siglip_logs',
        version=None
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,  # 每半个epoch验证一次
        gradient_clip_val=1.0,   # 梯度裁剪
        accumulate_grad_batches=1,  # 梯度累积
        deterministic=False,     # 为了性能，不使用确定性训练
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 开始训练
    print("开始训练...")
    trainer.fit(model, data_module)
    
    # 训练完成后进行测试（如果有测试数据）
    if data_module.test_dataloader() is not None:
        print("开始测试...")
        trainer.test(model, data_module)
    
    print(f"训练完成！模型和日志保存在: {args.save_dir}")
    print(f"最佳模型路径: {trainer.checkpoint_callback.best_model_path}")

if __name__ == '__main__':
    # 设置随机种子以确保可重现性
    pl.seed_everything(42)
    
    # 运行主函数
    main()