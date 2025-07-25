from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """CLIP模型配置类"""
    
    # 模型架构参数
    embed_dim: int = 512              # 嵌入维度
    image_resolution: int = 224       # 图像分辨率
    vision_layers: int = 12           # Vision Transformer层数
    vision_width: int = 768           # Vision Transformer宽度
    vision_patch_size: int = 16       # 图像patch大小
    context_length: int = 77          # 文本上下文长度
    vocab_size: int = 49408           # 词汇表大小
    transformer_width: int = 512      # 文本Transformer宽度
    transformer_heads: int = 8        # 文本Transformer注意力头数
    transformer_layers: int = 12      # 文本Transformer层数
    
    # 训练参数
    batch_size: int = 256             # 批次大小
    learning_rate: float = 5e-4       # 学习率
    weight_decay: float = 0.2         # 权重衰减
    warmup_steps: int = 2000          # 预热步数
    max_epochs: int = 32              # 最大训练轮数
    temperature: float = 0.07         # 对比学习温度参数
    
    # 优化器参数
    optimizer: str = "adamw"          # 优化器类型
    beta1: float = 0.9                # Adam beta1
    beta2: float = 0.98               # Adam beta2
    eps: float = 1e-6                 # Adam epsilon
    
    # 学习率调度器参数
    scheduler: str = "cosine"         # 学习率调度器类型
    min_lr: float = 0.0               # 最小学习率
    
    # 数据加载参数
    num_workers: int = 4              # 数据加载器工作进程数
    pin_memory: bool = True           # 是否固定内存
    persistent_workers: bool = True   # 是否保持工作进程
    
    # 正则化参数
    dropout: float = 0.0              # Dropout概率
    attention_dropout: float = 0.0    # 注意力Dropout概率
    
    # 设备和精度
    device: str = "auto"              # 计算设备
    mixed_precision: bool = True      # 是否使用混合精度
    
    # 检查点和日志
    save_top_k: int = 3               # 保存最好的k个模型
    monitor: str = "val_loss"         # 监控指标
    mode: str = "min"                 # 监控模式
    
    def __post_init__(self):
        """初始化后的验证和设置"""
        # 自动设置设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 验证参数合理性
        assert self.embed_dim > 0, "嵌入维度必须大于0"
        assert self.vision_layers > 0, "Vision Transformer层数必须大于0"
        assert self.transformer_layers > 0, "文本Transformer层数必须大于0"
        assert 0 < self.learning_rate < 1, "学习率必须在(0,1)范围内"
        assert 0 <= self.weight_decay < 1, "权重衰减必须在[0,1)范围内"
        assert self.temperature > 0, "温度参数必须大于0"
        assert self.batch_size > 0, "批次大小必须大于0"
        
        # 确保某些参数是2的幂（对于效率优化）
        assert self.vision_width % self.vision_heads == 0 if hasattr(self, 'vision_heads') else True
        assert self.transformer_width % self.transformer_heads == 0
    
    @property
    def vision_heads(self) -> int:
        """Vision Transformer注意力头数（通常等于宽度/64）"""
        return self.vision_width // 64
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'embed_dim': self.embed_dim,
            'image_resolution': self.image_resolution,
            'vision_layers': self.vision_layers,
            'vision_width': self.vision_width,
            'vision_patch_size': self.vision_patch_size,
            'context_length': self.context_length,
            'vocab_size': self.vocab_size,
            'transformer_width': self.transformer_width,
            'transformer_heads': self.transformer_heads,
            'transformer_layers': self.transformer_layers,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'max_epochs': self.max_epochs,
            'temperature': self.temperature,
            'optimizer': self.optimizer,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'scheduler': self.scheduler,
            'min_lr': self.min_lr,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'save_top_k': self.save_top_k,
            'monitor': self.monitor,
            'mode': self.mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'ModelConfig':
        """更新配置参数"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)

# 预定义的模型配置
CLIP_BASE_CONFIG = ModelConfig(
    embed_dim=512,
    vision_layers=12,
    vision_width=768,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12
)

CLIP_LARGE_CONFIG = ModelConfig(
    embed_dim=768,
    vision_layers=24,
    vision_width=1024,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12
)

CLIP_HUGE_CONFIG = ModelConfig(
    embed_dim=1024,
    vision_layers=32,
    vision_width=1280,
    transformer_width=1024,
    transformer_heads=16,
    transformer_layers=24
)

# 配置字典，方便根据名称获取配置
CONFIG_REGISTRY = {
    "base": CLIP_BASE_CONFIG,
    "large": CLIP_LARGE_CONFIG,
    "huge": CLIP_HUGE_CONFIG
}

def get_config(config_name: str) -> ModelConfig:
    """根据配置名称获取配置对象"""
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[config_name]

def create_custom_config(**kwargs) -> ModelConfig:
    """创建自定义配置"""
    base_config = CLIP_BASE_CONFIG
    return base_config.update(**kwargs)

# 训练配置类
@dataclass
class TrainingConfig:
    """训练相关配置"""
    
    # 数据路径
    data_path: str = "./data"                    # 数据集路径
    output_dir: str = "./outputs"                # 输出目录
    checkpoint_dir: str = "./checkpoints"        # 检查点目录
    log_dir: str = "./logs"                      # 日志目录
    
    # 训练设置
    resume_from_checkpoint: Optional[str] = None  # 从检查点恢复训练
    seed: int = 42                               # 随机种子
    deterministic: bool = True                   # 是否使用确定性算法
    
    # 验证设置
    val_check_interval: float = 1.0              # 验证检查间隔
    limit_val_batches: float = 1.0               # 限制验证批次数
    
    # 日志设置
    log_every_n_steps: int = 50                  # 每n步记录一次日志
    enable_progress_bar: bool = True             # 是否启用进度条
    
    # 早停设置
    early_stopping_patience: int = 5             # 早停耐心值
    early_stopping_min_delta: float = 0.001     # 早停最小改善
    
    # 梯度相关
    gradient_clip_val: float = 1.0               # 梯度裁剪值
    accumulate_grad_batches: int = 1             # 梯度累积批次
    
    # 分布式训练
    strategy: str = "auto"                       # 训练策略
    devices: str = "auto"                        # 设备数量
    
    def __post_init__(self):
        """初始化后的验证"""
        import os
        
        # 创建必要的目录
        for directory in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 验证参数
        assert self.seed >= 0, "随机种子必须非负"
        assert 0 < self.val_check_interval <= 1, "验证检查间隔必须在(0,1]范围内"
        assert self.early_stopping_patience > 0, "早停耐心值必须大于0"
        assert self.gradient_clip_val > 0, "梯度裁剪值必须大于0"
        assert self.accumulate_grad_batches > 0, "梯度累积批次必须大于0"

# 默认训练配置
DEFAULT_TRAINING_CONFIG = TrainingConfig()

def get_training_config(**kwargs) -> TrainingConfig:
    """获取训练配置"""
    config_dict = DEFAULT_TRAINING_CONFIG.__dict__.copy()
    config_dict.update(kwargs)
    return TrainingConfig(**config_dict)

# 数据配置类
@dataclass
class DataConfig:
    """数据相关配置"""
    
    # 数据集设置
    train_data_path: str = "./data/train.json"   # 训练数据路径
    val_data_path: str = "./data/val.json"       # 验证数据路径
    image_dir: str = "./data/images"             # 图像目录
    
    # 数据预处理
    image_size: int = 224                        # 图像大小
    center_crop: bool = True                     # 是否中心裁剪
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # 标准化均值
    normalize_std: tuple = (0.229, 0.224, 0.225)   # 标准化标准差
    
    # 数据增强
    random_resized_crop: bool = True             # 随机调整大小裁剪
    horizontal_flip: bool = True                 # 水平翻转
    color_jitter: bool = True                    # 颜色抖动
    random_grayscale: float = 0.1                # 随机灰度化概率
    
    # 文本处理
    max_text_length: int = 77                    # 最大文本长度
    truncate_text: bool = True                   # 是否截断文本
    
    # 数据加载
    train_batch_size: int = 256                  # 训练批次大小
    val_batch_size: int = 256                    # 验证批次大小
    num_workers: int = 4                         # 工作进程数
    pin_memory: bool = True                      # 固定内存
    drop_last: bool = True                       # 丢弃最后不完整批次
    
    def __post_init__(self):
        """初始化后的验证"""
        assert self.image_size > 0, "图像大小必须大于0"
        assert self.max_text_length > 0, "最大文本长度必须大于0"
        assert self.train_batch_size > 0, "训练批次大小必须大于0"
        assert self.val_batch_size > 0, "验证批次大小必须大于0"
        assert 0 <= self.random_grayscale <= 1, "随机灰度化概率必须在[0,1]范围内"

# 默认数据配置
DEFAULT_DATA_CONFIG = DataConfig()

def get_data_config(**kwargs) -> DataConfig:
    """获取数据配置"""
    config_dict = DEFAULT_DATA_CONFIG.__dict__.copy()
    config_dict.update(kwargs)
    return DataConfig(**config_dict)