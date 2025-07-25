import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

def set_seed(seed: int = 42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)           # 设置Python随机种子
    np.random.seed(seed)        # 设置NumPy随机种子
    torch.manual_seed(seed)     # 设置PyTorch随机种子
    
    # 如果使用CUDA，也设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
        # 设置CUDA确定性行为（可能会影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已设置为: {seed}")

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """计算模型参数数量"""
    if trainable_only:
        # 只计算可训练参数
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # 计算所有参数
        return sum(p.numel() for p in model.parameters())

def format_number(num: int) -> str:
    """格式化数字，添加千位分隔符"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def print_model_info(model: nn.Module, model_name: str = "Model"):
    """打印模型信息"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n{model_name} 信息:")
    print(f"  总参数数量: {format_number(total_params)} ({total_params:,})")
    print(f"  可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"  不可训练参数: {format_number(total_params - trainable_params)} ({total_params - trainable_params:,})")
    
    # 计算模型大小（假设float32）
    model_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"  估计模型大小: {model_size_mb:.1f} MB")

def save_config(config: Dict[str, Any], save_path: str):
    """保存配置到JSON文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换不可序列化的对象
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {save_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"配置已从 {config_path} 加载")
    return config

def create_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """创建日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_device(device: str = "auto") -> torch.device:
    """获取计算设备"""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"使用GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("使用CPU")
    
    return torch.device(device)

def move_to_device(data: Any, device: torch.device) -> Any:
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """计算Top-K准确率"""
    batch_size = targets.size(0)
    
    # 获取top-k预测
    _, pred_indices = predictions.topk(top_k, dim=1, largest=True, sorted=True)
    
    # 检查是否匹配
    correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
    
    # 计算准确率
    accuracy = correct.float().sum() / batch_size
    
    return accuracy.item()

def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算两个张量之间的余弦相似度矩阵"""
    # 归一化
    x_norm = nn.functional.normalize(x, dim=-1)
    y_norm = nn.functional.normalize(y, dim=-1)
    
    # 计算余弦相似度
    similarity = torch.matmul(x_norm, y_norm.T)
    
    return similarity

def create_attention_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """创建因果注意力掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    # 将0位置设为负无穷，1位置设为0
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    return mask

def interpolate_pos_embed(pos_embed: torch.Tensor, new_size: Tuple[int, int]) -> torch.Tensor:
    """插值位置编码以适应不同的图像尺寸"""
    # pos_embed shape: [1, num_patches + 1, embed_dim]
    # 第一个token是class token，需要单独处理
    
    cls_token = pos_embed[:, 0:1, :]  # class token
    pos_embed = pos_embed[:, 1:, :]   # patch tokens
    
    # 获取原始网格大小
    num_patches = pos_embed.shape[1]
    old_size = int(num_patches ** 0.5)
    
    if old_size != new_size[0] or old_size != new_size[1]:
        # 重塑为2D网格
        pos_embed = pos_embed.reshape(1, old_size, old_size, -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, embed_dim, old_size, old_size]
        
        # 插值到新尺寸
        pos_embed = nn.functional.interpolate(
            pos_embed, size=new_size, mode='bicubic', align_corners=False
        )
        
        # 重塑回原始格式
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, new_size[0], new_size[1], embed_dim]
        pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])  # [1, new_num_patches, embed_dim]
    
    # 重新组合class token和patch tokens
    pos_embed = torch.cat([cls_token, pos_embed], dim=1)
    
    return pos_embed

def visualize_attention(attention_weights: torch.Tensor, 
                       tokens: List[str] = None,
                       save_path: str = None,
                       figsize: Tuple[int, int] = (10, 8)):
    """可视化注意力权重"""
    # attention_weights shape: [num_heads, seq_len, seq_len] 或 [seq_len, seq_len]
    
    if attention_weights.dim() == 3:
        # 多头注意力，取平均
        attention_weights = attention_weights.mean(dim=0)
    
    # 转换为numpy
    attention_weights = attention_weights.detach().cpu().numpy()
    
    # 创建图形
    plt.figure(figsize=figsize)
    plt.imshow(attention_weights, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    
    # 设置标签
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
    
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_learning_rate_schedule(optimizer: torch.optim.Optimizer,
                                 warmup_steps: int,
                                 total_steps: int,
                                 min_lr_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """创建带预热的余弦学习率调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            # 线性预热
            return step / warmup_steps
        else:
            # 余弦退火
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   save_path: str,
                   **kwargs):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, save_path)
    print(f"检查点已保存到: {save_path}")

def load_checkpoint(checkpoint_path: str,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态（如果提供）
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"检查点已从 {checkpoint_path} 加载")
    print(f"  轮次: {checkpoint['epoch']}")
    print(f"  损失: {checkpoint['loss']:.4f}")
    
    return checkpoint

def calculate_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """估算模型的FLOPs（浮点运算次数）"""
    # 这是一个简化的FLOPs计算，实际计算会更复杂
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Linear):
            # 线性层: input_size * output_size
            total_flops += module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # 卷积层: kernel_size * input_channels * output_channels * output_height * output_width
            kernel_flops = module.kernel_size[0] * module.kernel_size[1]
            output_elements = output.numel() // output.shape[0]  # 除以batch size
            total_flops += kernel_flops * module.in_channels * output_elements
        elif isinstance(module, nn.MultiheadAttention):
            # 注意力层的近似计算
            seq_len = input[0].shape[1]
            embed_dim = module.embed_dim
            total_flops += 4 * seq_len * embed_dim * embed_dim  # Q, K, V, O projections
            total_flops += 2 * seq_len * seq_len * embed_dim    # Attention computation
    
    # 注册钩子
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(flop_count_hook))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        model(dummy_input)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return total_flops

class Timer:
    """简单的计时器类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """获取经过的时间（秒）"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

def ensure_dir(path: str):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_git_commit_hash() -> Optional[str]:
    """获取当前Git提交哈希"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def print_system_info():
    """打印系统信息"""
    print("系统信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Git信息
    git_hash = get_git_commit_hash()
    if git_hash:
        print(f"  Git提交: {git_hash[:8]}")
    
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # 测试一些工具函数
    print("测试工具函数...")
    
    # 测试数字格式化
    print(f"格式化数字测试:")
    print(f"  1234 -> {format_number(1234)}")
    print(f"  1234567 -> {format_number(1234567)}")
    print(f"  1234567890 -> {format_number(1234567890)}")
    
    # 测试时间格式化
    print(f"\n时间格式化测试:")
    print(f"  30.5秒 -> {format_time(30.5)}")
    print(f"  125.3秒 -> {format_time(125.3)}")
    print(f"  3725.8秒 -> {format_time(3725.8)}")
    
    # 测试计时器
    print(f"\n计时器测试:")
    with Timer() as timer:
        time.sleep(0.1)
    print(f"  经过时间: {format_time(timer.elapsed())}")
    
    # 打印系统信息
    print(f"\n")
    print_system_info()