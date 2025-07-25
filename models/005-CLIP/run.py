#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP模型运行脚本

这个脚本提供了一个统一的入口点来运行CLIP模型的各种功能：
- 训练模型
- 推理和演示
- 模型评估
- 数据预处理

使用方法:
    python run.py train --config base --epochs 10
    python run.py demo --model_path ./checkpoints/best.ckpt
    python run.py eval --model_path ./checkpoints/best.ckpt --data_path ./data/test.json
"""

import argparse
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import get_config, get_training_config, get_data_config
from utils import set_seed, print_system_info, create_logger, ensure_dir
from train import train_clip_model
from example import CLIPInference

def setup_environment(args):
    """设置运行环境"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.log_dir)
    
    # 创建日志记录器
    log_file = os.path.join(args.log_dir, f"{args.command}_{args.config}.log")
    logger = create_logger("CLIP", log_file)
    
    # 打印系统信息
    if args.verbose:
        print_system_info()
    
    return logger

def train_command(args):
    """训练命令"""
    print("开始训练CLIP模型...")
    
    # 获取配置
    model_config = get_config(args.config)
    training_config = get_training_config(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed
    )
    data_config = get_data_config(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        image_dir=args.image_dir,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size
    )
    
    # 更新配置
    if args.epochs:
        model_config = model_config.update(max_epochs=args.epochs)
    if args.learning_rate:
        model_config = model_config.update(learning_rate=args.learning_rate)
    if args.batch_size:
        model_config = model_config.update(batch_size=args.batch_size)
    
    # 开始训练
    train_clip_model(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        resume_from_checkpoint=args.resume_from
    )
    
    print("训练完成！")

def demo_command(args):
    """演示命令"""
    print("运行CLIP模型演示...")
    
    # 导入演示函数
    from example import (
        demo_zero_shot_classification,
        demo_image_text_matching,
        demo_custom_prompts,
        demo_batch_processing
    )
    
    # 根据演示类型运行相应的演示
    if args.demo_type == "classification":
        demo_zero_shot_classification()
    elif args.demo_type == "matching":
        demo_image_text_matching()
    elif args.demo_type == "prompts":
        demo_custom_prompts()
    elif args.demo_type == "batch":
        demo_batch_processing()
    elif args.demo_type == "all":
        demo_zero_shot_classification()
        demo_image_text_matching()
        demo_custom_prompts()
        demo_batch_processing()
    
    print("演示完成！")

def inference_command(args):
    """推理命令"""
    print("运行CLIP模型推理...")
    
    # 初始化推理器
    clip_inference = CLIPInference(
        model_path=args.model_path,
        model_size=args.config,
        device=args.device
    )
    
    if args.task == "classify":
        # 零样本分类
        if not args.image_path:
            print("错误: 分类任务需要指定 --image_path")
            return
        
        if not args.class_names:
            # 使用默认类别
            class_names = ["cat", "dog", "bird", "car", "tree", "house", "person", "flower"]
        else:
            class_names = args.class_names.split(",")
        
        probabilities, predicted_idx = clip_inference.zero_shot_classify(
            args.image_path, class_names
        )
        
        print(f"\n分类结果:")
        print(f"图像: {args.image_path}")
        print(f"预测类别: {class_names[predicted_idx]}")
        print(f"置信度: {probabilities[predicted_idx]:.4f}")
        
        if args.show_all_probs:
            print(f"\n所有类别概率:")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                marker = " ← 预测" if i == predicted_idx else ""
                print(f"  {class_name}: {prob:.4f}{marker}")
    
    elif args.task == "similarity":
        # 图像-文本相似度
        if not args.image_path or not args.texts:
            print("错误: 相似度任务需要指定 --image_path 和 --texts")
            return
        
        texts = args.texts.split(";")
        similarities = clip_inference.image_text_similarity(args.image_path, texts)
        
        print(f"\n相似度结果:")
        print(f"图像: {args.image_path}")
        for text, similarity in zip(texts, similarities):
            print(f"  \"{text}\": {similarity:.4f}")
        
        # 找到最佳匹配
        best_text, best_similarity, best_idx = clip_inference.find_best_text_match(
            args.image_path, texts
        )
        print(f"\n最佳匹配: \"{best_text}\" (相似度: {best_similarity:.4f})")
    
    print("推理完成！")

def eval_command(args):
    """评估命令"""
    print("评估CLIP模型...")
    
    # 这里可以添加模型评估逻辑
    # 例如在测试集上计算准确率、召回率等指标
    
    print("评估功能待实现...")
    print("您可以在这里添加自定义的评估逻辑")

def preprocess_command(args):
    """数据预处理命令"""
    print("预处理数据...")
    
    # 这里可以添加数据预处理逻辑
    # 例如图像预处理、文本清理等
    
    print("数据预处理功能待实现...")
    print("您可以在这里添加自定义的数据预处理逻辑")

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="CLIP模型运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  训练模型:
    python run.py train --config base --epochs 10 --batch_size 128
  
  运行演示:
    python run.py demo --demo_type all
  
  零样本分类:
    python run.py inference --task classify --image_path ./image.jpg --class_names "cat,dog,bird"
  
  图像-文本相似度:
    python run.py inference --task similarity --image_path ./image.jpg --texts "a cat;a dog;a bird"
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 通用参数
    parser.add_argument("--config", type=str, default="base", 
                       choices=["base", "large", "huge"], help="模型配置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--batch_size", type=int, help="批次大小")
    train_parser.add_argument("--learning_rate", type=float, help="学习率")
    train_parser.add_argument("--train_data", type=str, default="./data/train.json", help="训练数据路径")
    train_parser.add_argument("--val_data", type=str, default="./data/val.json", help="验证数据路径")
    train_parser.add_argument("--image_dir", type=str, default="./data/images", help="图像目录")
    train_parser.add_argument("--resume_from", type=str, help="从检查点恢复训练")
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    demo_parser.add_argument("--demo_type", type=str, default="all",
                            choices=["all", "classification", "matching", "prompts", "batch"],
                            help="演示类型")
    demo_parser.add_argument("--model_path", type=str, help="模型路径")
    
    # 推理命令
    inference_parser = subparsers.add_parser("inference", help="模型推理")
    inference_parser.add_argument("--task", type=str, required=True,
                                 choices=["classify", "similarity"], help="推理任务类型")
    inference_parser.add_argument("--model_path", type=str, help="模型路径")
    inference_parser.add_argument("--image_path", type=str, help="图像路径")
    inference_parser.add_argument("--class_names", type=str, help="分类类别（逗号分隔）")
    inference_parser.add_argument("--texts", type=str, help="文本列表（分号分隔）")
    inference_parser.add_argument("--show_all_probs", action="store_true", help="显示所有类别概率")
    
    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--data_path", type=str, help="测试数据路径")
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="数据预处理")
    preprocess_parser.add_argument("--input_dir", type=str, required=True, help="输入目录")
    preprocess_parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 检查是否指定了命令
    if not args.command:
        parser.print_help()
        return
    
    # 设置环境
    logger = setup_environment(args)
    
    try:
        # 根据命令执行相应的功能
        if args.command == "train":
            train_command(args)
        elif args.command == "demo":
            demo_command(args)
        elif args.command == "inference":
            inference_command(args)
        elif args.command == "eval":
            eval_command(args)
        elif args.command == "preprocess":
            preprocess_command(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()