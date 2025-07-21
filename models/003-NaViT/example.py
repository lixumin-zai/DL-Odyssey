#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NaViT模型使用示例

这个脚本展示了如何使用我们实现的NaViT模型进行训练和推理
包括标准输入和序列打包两种使用方式
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

# 导入我们的模块
from model import NaViT, navit_tiny, navit_small, navit_base
from dataset import CIFAR10NaViTDataset, NaViTCollator, create_dataloaders
from train import NaViTLightningModule

def demo_model_creation():
    """演示模型创建的不同方式"""
    print("=" * 60)
    print("演示1: 模型创建")
    print("=" * 60)
    
    # 方式1: 使用预定义配置
    print("\n1. 使用预定义配置创建模型:")
    models = {
        'NaViT-Tiny': navit_tiny(num_classes=10),
        'NaViT-Small': navit_small(num_classes=10),
        'NaViT-Base': navit_base(num_classes=10)
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {total_params:,} 参数")
    
    # 方式2: 自定义配置
    print("\n2. 自定义配置创建模型:")
    custom_model = NaViT(
        image_size=224,           # 图像尺寸
        patch_size=16,            # patch大小
        num_classes=1000,         # ImageNet类别数
        embed_dim=512,            # embedding维度
        depth=8,                  # Transformer层数
        num_heads=8,              # 注意力头数
        mlp_ratio=4.0,            # MLP比例
        dropout=0.1,              # dropout概率
        token_dropout_prob=0.1    # token dropout概率
    )
    
    custom_params = sum(p.numel() for p in custom_model.parameters())
    print(f"  自定义模型: {custom_params:,} 参数")
    
    return models['NaViT-Base']  # 返回基础模型用于后续演示

def demo_standard_inference(model: NaViT):
    """演示标准输入的推理过程"""
    print("\n" + "=" * 60)
    print("演示2: 标准输入推理")
    print("=" * 60)
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建标准批次输入（所有图像相同尺寸）
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    
    print(f"\n输入: 批次大小={batch_size}, 图像尺寸={height}x{width}")
    
    # 生成随机输入数据
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # 推理计时
    start_time = time.time()
    
    with torch.no_grad():  # 推理时不需要计算梯度
        output = model(input_tensor)
    
    inference_time = time.time() - start_time
    
    # 打印结果
    print(f"输出形状: {output.shape}")
    print(f"推理时间: {inference_time:.4f} 秒")
    
    # 计算预测概率
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    print(f"预测类别: {predicted_classes.tolist()}")
    print(f"最高概率: {probabilities.max(dim=1)[0].tolist()}")

def demo_packed_inference(model: NaViT):
    """演示序列打包输入的推理过程"""
    print("\n" + "=" * 60)
    print("演示3: 序列打包推理")
    print("=" * 60)
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建不同尺寸的图像
    images = [
        torch.randn(3, 224, 224),  # 标准尺寸
        torch.randn(3, 160, 160),  # 较小尺寸
        torch.randn(3, 256, 128),  # 宽图像
        torch.randn(3, 128, 256),  # 高图像
        torch.randn(3, 192, 192)   # 中等尺寸
    ]
    
    print("\n输入图像尺寸:")
    for i, img in enumerate(images):
        print(f"  图像 {i+1}: {img.shape[1]}x{img.shape[2]}")
    
    # 方式1: 手动打包
    print("\n方式1: 手动打包")
    packed_input = [
        [images[0], images[1]],     # 第一组: 2张图像
        [images[2]],                # 第二组: 1张图像
        [images[3], images[4]]      # 第三组: 2张图像
    ]
    
    start_time = time.time()
    
    with torch.no_grad():
        output = model(packed_input)
    
    inference_time = time.time() - start_time
    
    print(f"输出形状: {output.shape}")
    print(f"推理时间: {inference_time:.4f} 秒")
    
    # 方式2: 自动打包
    print("\n方式2: 自动打包（模拟）")
    # 注意: 实际的自动打包需要在数据加载器中实现
    # 这里只是演示概念
    
    # 计算每个图像的token数量
    patch_size = 16
    token_counts = []
    for img in images:
        h_patches = img.shape[1] // patch_size
        w_patches = img.shape[2] // patch_size
        tokens = h_patches * w_patches
        token_counts.append(tokens)
        print(f"  图像 {len(token_counts)}: {img.shape[1]}x{img.shape[2]} -> {tokens} tokens")
    
    total_tokens = sum(token_counts)
    print(f"总token数: {total_tokens}")

def demo_training_setup():
    """演示训练设置"""
    print("\n" + "=" * 60)
    print("演示4: 训练设置")
    print("=" * 60)
    
    # 创建数据集（使用CIFAR-10作为示例）
    print("\n1. 创建数据集:")
    
    try:
        # 创建训练数据集
        train_dataset = CIFAR10NaViTDataset(
            root='./data',
            train=True,
            download=True,  # 如果没有数据会自动下载
            augment=True,
            multi_scale_prob=0.7
        )
        
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  类别数: {train_dataset.num_classes}")
        print(f"  类别名称: {train_dataset.classes}")
        
        # 获取一个样本
        sample_image, sample_label = train_dataset[0]
        print(f"  样本图像形状: {sample_image.shape}")
        print(f"  样本标签: {sample_label} ({train_dataset.classes[sample_label]})")
        
    except Exception as e:
        print(f"  数据集创建失败: {e}")
        print("  请确保有网络连接以下载CIFAR-10数据集")
        return
    
    # 创建数据整理器
    print("\n2. 创建数据整理器:")
    collator = NaViTCollator(max_seq_len=1024, pack_sequences=True)
    
    # 创建小批次数据进行测试
    batch_data = [train_dataset[i] for i in range(4)]
    
    try:
        packed_batch = collator(batch_data)
        if isinstance(packed_batch[0], list):
            print(f"  序列打包成功: {len(packed_batch[0])} 组")
            for i, group in enumerate(packed_batch[0]):
                print(f"    组 {i+1}: {len(group)} 张图像")
        else:
            print(f"  标准批次: {packed_batch[0].shape}")
    except Exception as e:
        print(f"  数据整理失败: {e}")

def demo_model_comparison():
    """演示不同模型配置的性能比较"""
    print("\n" + "=" * 60)
    print("演示5: 模型性能比较")
    print("=" * 60)
    
    # 测试输入
    test_input = torch.randn(1, 3, 224, 224)
    
    models = {
        'NaViT-Tiny': navit_tiny(num_classes=10),
        'NaViT-Small': navit_small(num_classes=10),
        'NaViT-Base': navit_base(num_classes=10)
    }
    
    print("\n模型性能对比:")
    print(f"{'模型名称':<15} {'参数数量':<12} {'推理时间(ms)':<15} {'内存使用(MB)':<15}")
    print("-" * 65)
    
    for name, model in models.items():
        model.eval()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_input)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 估算内存使用（简化计算）
        param_memory = total_params * 4 / (1024 * 1024)  # 假设float32，转换为MB
        
        print(f"{name:<15} {total_params:<12,} {inference_time:<15.2f} {param_memory:<15.1f}")

def demo_visualization():
    """演示结果可视化"""
    print("\n" + "=" * 60)
    print("演示6: 结果可视化")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # 创建一个简单的训练曲线可视化
        epochs = list(range(1, 21))
        train_loss = [2.3 - 0.1 * i + 0.05 * np.sin(i) for i in epochs]
        val_loss = [2.4 - 0.08 * i + 0.1 * np.sin(i * 1.2) for i in epochs]
        train_acc = [10 + 4 * i + 2 * np.sin(i * 0.5) for i in epochs]
        val_acc = [8 + 3.8 * i + 3 * np.sin(i * 0.7) for i in epochs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(epochs, train_loss, label='训练损失', color='blue')
        ax1.plot(epochs, val_loss, label='验证损失', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, train_acc, label='训练准确率', color='blue')
        ax2.plot(epochs, val_acc, label='验证准确率', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('navit_training_curves.png', dpi=150, bbox_inches='tight')
        print("\n训练曲线已保存为 'navit_training_curves.png'")
        
        # 显示图像（如果在支持的环境中）
        try:
            plt.show()
        except:
            print("无法显示图像，但已保存到文件")
            
    except ImportError:
        print("\n需要安装matplotlib来进行可视化")
        print("运行: pip install matplotlib")

def main():
    """主函数 - 运行所有演示"""
    print("NaViT模型使用示例")
    print("这个脚本演示了NaViT模型的各种使用方法")
    
    try:
        # 演示1: 模型创建
        model = demo_model_creation()
        
        # 演示2: 标准推理
        demo_standard_inference(model)
        
        # 演示3: 序列打包推理
        demo_packed_inference(model)
        
        # 演示4: 训练设置
        demo_training_setup()
        
        # 演示5: 模型比较
        demo_model_comparison()
        
        # 演示6: 可视化
        demo_visualization()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
        print("\n快速开始训练:")
        print("python train.py --model_name navit_base --batch_size 32 --max_epochs 50")
        
        print("\n更多选项:")
        print("python train.py --help")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        print("请检查依赖是否正确安装")
        print("运行: pip install -r requirements.txt")

if __name__ == '__main__':
    main()