import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import argparse
from typing import Any, Dict, Optional

# 导入我们自定义的模块
from model import CLIP, create_clip_model, contrastive_loss
from dataset import CLIPDataModule, create_sample_dataset

class CLIPLightningModule(pl.LightningModule):
    """CLIP模型的PyTorch Lightning封装"""
    
    def __init__(self, 
                 model_size: str = "base",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 100000,
                 temperature_init: float = 0.07):
        """
        初始化CLIP Lightning模块
        
        Args:
            model_size: 模型大小（"base" 或 "large"）
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            max_steps: 最大训练步数
            temperature_init: 温度参数初始值
        """
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 创建CLIP模型
        self.model = create_clip_model(model_size)
        
        # 定义评估指标
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=None)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=None)
        
        # 用于记录最佳验证准确率
        self.best_val_accuracy = 0.0
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor):
        """前向传播"""
        return self.model(images, texts)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        images, texts, captions = batch  # 解包批次数据
        
        # 前向传播：获取图像-文本相似度矩阵
        logits_per_image, logits_per_text = self.model(images, texts)
        
        # 计算对比损失
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 计算准确率（图像到文本的分类准确率）
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=self.device)  # 创建标签（对角线为正确匹配）
        
        # 计算图像到文本的准确率
        image_to_text_acc = self.train_accuracy(logits_per_image, labels)
        
        # 记录训练指标
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', image_to_text_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/temperature', self.model.logit_scale.exp(), on_step=True, on_epoch=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        images, texts, captions = batch  # 解包批次数据
        
        # 前向传播
        logits_per_image, logits_per_text = self.model(images, texts)
        
        # 计算损失
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 计算准确率
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算双向准确率
        image_to_text_acc = self.val_accuracy(logits_per_image, labels)
        text_to_image_acc = self.val_accuracy(logits_per_text, labels)
        
        # 记录验证指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/image_to_text_acc', image_to_text_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/text_to_image_acc', text_to_image_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        # 计算平均准确率
        avg_accuracy = (image_to_text_acc + text_to_image_acc) / 2
        self.log('val/avg_accuracy', avg_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """验证轮次结束时的回调"""
        # 获取当前验证准确率
        current_val_acc = self.trainer.callback_metrics.get('val/avg_accuracy', 0.0)
        
        # 更新最佳验证准确率
        if current_val_acc > self.best_val_accuracy:
            self.best_val_accuracy = current_val_acc
            self.log('val/best_accuracy', self.best_val_accuracy, sync_dist=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 分别设置不同部分的学习率
        # 通常文本编码器使用较小的学习率，视觉编码器使用较大的学习率
        param_groups = [
            {
                'params': self.model.text_encoder.parameters(),
                'lr': self.hparams.learning_rate * 0.1,  # 文本编码器使用较小学习率
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': self.model.vision_encoder.parameters(),
                'lr': self.hparams.learning_rate,  # 视觉编码器使用标准学习率
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [self.model.vision_projection.weight, 
                          self.model.text_projection.weight,
                          self.model.logit_scale],
                'lr': self.hparams.learning_rate,  # 投影层使用标准学习率
                'weight_decay': self.hparams.weight_decay
            }
        ]
        
        # 创建AdamW优化器
        optimizer = AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
        
        # 创建学习率调度器
        # 1. 线性预热阶段
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01,  # 从1%的学习率开始
            end_factor=1.0,     # 预热到100%的学习率
            total_iters=self.hparams.warmup_steps
        )
        
        # 2. 余弦退火阶段
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps - self.hparams.warmup_steps,  # 余弦退火的步数
            eta_min=self.hparams.learning_rate * 0.01  # 最小学习率为初始学习率的1%
        )
        
        # 3. 组合调度器：先预热，再余弦退火
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 每步更新学习率
                'frequency': 1,
                'name': 'learning_rate'
            }
        }
    
    def zero_shot_evaluation(self, dataloader, class_names: list, num_samples: int = 100):
        """零样本评估功能"""
        self.eval()  # 设置为评估模式
        
        correct = 0
        total = 0
        
        # 为每个类别创建文本提示
        text_prompts = [f"a photo of a {name}" for name in class_names]
        
        with torch.no_grad():
            # 编码所有类别的文本描述
            text_inputs = torch.stack([
                torch.tensor(self.tokenizer.tokenize(prompt)) 
                for prompt in text_prompts
            ]).to(self.device)
            
            text_features = self.model.encode_text(text_inputs)  # [num_classes, embed_dim]
            
            # 遍历测试样本
            for batch_idx, (images, _, _) in enumerate(dataloader):
                if batch_idx * dataloader.batch_size >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # 编码图像
                image_features = self.model.encode_image(images)  # [batch_size, embed_dim]
                
                # 计算相似度
                similarities = torch.matmul(image_features, text_features.T)  # [batch_size, num_classes]
                
                # 预测类别
                predictions = torch.argmax(similarities, dim=-1)
                
                # 这里假设真实标签就是batch中的索引（简化版本）
                # 实际应用中需要从数据集中获取真实标签
                batch_size = images.shape[0]
                true_labels = torch.arange(batch_size) % len(class_names)
                true_labels = true_labels.to(self.device)
                
                # 计算准确率
                correct += (predictions == true_labels).sum().item()
                total += batch_size
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

def train_clip_model(args):
    """训练CLIP模型的主函数"""
    
    # 设置随机种子以确保可重复性
    pl.seed_everything(args.seed)
    
    # 创建数据模块
    if not os.path.exists(args.data_dir):
        print(f"数据目录 {args.data_dir} 不存在，创建示例数据集...")
        data_module = create_sample_dataset(args.data_dir, num_samples=args.num_samples)
    else:
        data_module = CLIPDataModule(
            train_data_path=os.path.join(args.data_dir, "train.json"),
            val_data_path=os.path.join(args.data_dir, "val.json"),
            image_dir=os.path.join(args.data_dir, "images"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length
        )
        data_module.setup()
    
    # 创建Lightning模块
    model = CLIPLightningModule(
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        temperature_init=args.temperature_init
    )
    
    # 创建回调函数
    callbacks = [
        # 模型检查点：保存最佳模型
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="clip-{epoch:02d}-{val/avg_accuracy:.4f}",
            monitor="val/avg_accuracy",
            mode="max",
            save_top_k=3,  # 保存最好的3个模型
            save_last=True,  # 保存最后一个模型
            verbose=True
        ),
        
        # 早停：如果验证准确率不再提升则停止训练
        EarlyStopping(
            monitor="val/avg_accuracy",
            mode="max",
            patience=args.patience,
            verbose=True,
            min_delta=0.001  # 最小改进阈值
        ),
        
        # 学习率监控
        LearningRateMonitor(logging_interval="step")
    ]
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="clip_logs",
        version=f"run_{args.model_size}"
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="auto",  # 自动选择加速器（GPU/CPU）
        devices="auto",     # 自动选择设备数量
        precision=args.precision,  # 混合精度训练
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,  # 梯度裁剪
        accumulate_grad_batches=args.accumulate_grad_batches,  # 梯度累积
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 开始训练
    print("开始训练CLIP模型...")
    print(f"模型大小: {args.model_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"最大轮次: {args.max_epochs}")
    print(f"训练样本数: {len(data_module.train_dataset)}")
    print(f"验证样本数: {len(data_module.val_dataset)}")
    
    trainer.fit(model, data_module)
    
    # 训练完成后的评估
    print("\n训练完成！")
    print(f"最佳验证准确率: {model.best_val_accuracy:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    return model, trainer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练CLIP模型")
    
    # 数据相关参数
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录路径")
    parser.add_argument("--num_samples", type=int, default=1000, help="示例数据集的样本数量")
    parser.add_argument("--max_length", type=int, default=77, help="文本最大长度")
    
    # 模型相关参数
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="模型大小")
    parser.add_argument("--temperature_init", type=float, default=0.07, help="温度参数初始值")
    
    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_epochs", type=int, default=100, help="最大训练轮次")
    parser.add_argument("--max_steps", type=int, default=100000, help="最大训练步数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="预热步数")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    
    # 训练器相关参数
    parser.add_argument("--precision", type=str, default="16-mixed", help="训练精度")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="梯度裁剪值")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="梯度累积批次数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器工作进程数")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="日志记录间隔")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="验证检查间隔")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    model, trainer = train_clip_model(args)
    
    print("\n训练脚本执行完成！")

if __name__ == "__main__":
    main()