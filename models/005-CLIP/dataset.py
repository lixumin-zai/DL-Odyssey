import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import re
import random

class SimpleTokenizer:
    """简单的文本分词器，用于将文本转换为token序列"""
    
    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_length = max_length  # 最大序列长度
        
        # 特殊token的定义
        self.bos_token = "<|startoftext|>"  # 序列开始token
        self.eos_token = "<|endoftext|>"    # 序列结束token
        self.pad_token = "<|pad|>"          # 填充token
        self.unk_token = "<|unk|>"          # 未知token
        
        # 特殊token的ID
        self.bos_token_id = 49406
        self.eos_token_id = 49407
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        # 构建基础词汇表（这里使用简化版本，实际应该使用BPE编码）
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词汇表映射"""
        # 创建词汇到ID的映射（简化版本）
        self.vocab_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        
        # 添加常用词汇（实际应该从预训练的BPE词汇表加载）
        common_words = [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "photo", "image", "picture", "of", "showing", "depicts", "contains", "features",
            "cat", "dog", "bird", "car", "tree", "house", "person", "man", "woman", "child",
            "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink",
            "big", "small", "large", "tiny", "beautiful", "cute", "happy", "sad", "old", "new",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "this", "that", "these", "those", "here", "there", "where", "when", "what", "who", "how", "why"
        ]
        
        # 为常用词汇分配ID
        current_id = 4  # 从4开始，前面是特殊token
        for word in common_words:
            if word not in self.vocab_to_id:
                self.vocab_to_id[word] = current_id
                current_id += 1
        
        # 填充剩余的词汇表空间（用随机字符串模拟）
        while current_id < self.vocab_size - 100:  # 保留一些空间
            fake_token = f"token_{current_id}"
            self.vocab_to_id[fake_token] = current_id
            current_id += 1
        
        # 创建ID到词汇的反向映射
        self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}
    
    def tokenize(self, text: str) -> List[int]:
        """将文本转换为token ID列表"""
        # 文本预处理：转小写，移除多余空格
        text = text.lower().strip()
        
        # 简单的分词（按空格和标点符号分割）
        # 实际的CLIP使用更复杂的BPE分词
        words = re.findall(r'\b\w+\b', text)
        
        # 转换为token ID
        token_ids = [self.bos_token_id]  # 添加开始token
        
        for word in words:
            # 如果词汇在词汇表中，使用对应ID；否则使用未知token ID
            token_id = self.vocab_to_id.get(word, self.unk_token_id)
            token_ids.append(token_id)
            
            # 检查是否超过最大长度（需要为结束token留空间）
            if len(token_ids) >= self.max_length - 1:
                break
        
        # 添加结束token
        token_ids.append(self.eos_token_id)
        
        # 填充到固定长度
        while len(token_ids) < self.max_length:
            token_ids.append(self.pad_token_id)
        
        # 截断到最大长度
        token_ids = token_ids[:self.max_length]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token ID列表转换回文本"""
        words = []
        for token_id in token_ids:
            # 跳过特殊token
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            
            # 获取对应的词汇
            word = self.id_to_vocab.get(token_id, self.unk_token)
            if word != self.unk_token:
                words.append(word)
        
        return " ".join(words)

class ImageTextDataset(Dataset):
    """图像-文本配对数据集"""
    
    def __init__(self, 
                 data_path: str,
                 image_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 tokenizer: Optional[SimpleTokenizer] = None,
                 max_length: int = 77):
        """
        初始化数据集
        
        Args:
            data_path: 包含图像-文本对信息的JSON文件路径
            image_dir: 图像文件所在目录
            transform: 图像预处理变换
            tokenizer: 文本分词器
            max_length: 文本最大长度
        """
        self.image_dir = image_dir
        self.max_length = max_length
        
        # 初始化分词器
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_length=max_length)
        else:
            self.tokenizer = tokenizer
        
        # 设置图像预处理
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 加载数据
        self.data = self._load_data(data_path)
        
    def _get_default_transform(self) -> transforms.Compose:
        """获取默认的图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小到224x224
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(           # 标准化（使用ImageNet的均值和标准差）
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据文件"""
        if os.path.exists(data_path):
            # 如果数据文件存在，从文件加载
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 如果数据文件不存在，创建示例数据
            print(f"数据文件 {data_path} 不存在，创建示例数据...")
            data = self._create_sample_data()
            
            # 保存示例数据到文件
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        return data
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """创建示例数据（用于演示）"""
        # 创建一些示例的图像-文本对
        sample_data = []
        
        # 示例描述模板
        templates = [
            "a photo of a {object}",
            "an image showing a {object}",
            "a picture of a {object}",
            "a {color} {object}",
            "a {size} {object}",
            "a beautiful {object}",
            "a {object} in the {location}",
        ]
        
        # 示例对象、颜色、大小、位置
        objects = ["cat", "dog", "car", "tree", "house", "person", "bird", "flower"]
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown"]
        sizes = ["big", "small", "large", "tiny"]
        locations = ["park", "garden", "street", "room", "field"]
        
        # 生成示例数据
        for i in range(100):  # 创建100个示例
            # 随机选择模板和属性
            template = random.choice(templates)
            obj = random.choice(objects)
            color = random.choice(colors)
            size = random.choice(sizes)
            location = random.choice(locations)
            
            # 生成描述文本
            if "{color}" in template:
                caption = template.format(object=obj, color=color)
            elif "{size}" in template:
                caption = template.format(object=obj, size=size)
            elif "{location}" in template:
                caption = template.format(object=obj, location=location)
            else:
                caption = template.format(object=obj)
            
            # 创建数据项
            sample_data.append({
                "image_path": f"sample_image_{i:03d}.jpg",  # 示例图像路径
                "caption": caption,
                "image_id": i
            })
        
        return sample_data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """获取单个数据项"""
        # 获取数据项
        item = self.data[idx]
        image_path = item["image_path"]
        caption = item["caption"]
        
        # 加载和预处理图像
        full_image_path = os.path.join(self.image_dir, image_path)
        
        try:
            # 尝试加载真实图像
            image = Image.open(full_image_path).convert("RGB")
        except (FileNotFoundError, IOError):
            # 如果图像文件不存在，创建随机图像（用于演示）
            image = Image.new("RGB", (224, 224), color=(random.randint(0, 255), 
                                                       random.randint(0, 255), 
                                                       random.randint(0, 255)))
        
        # 应用图像变换
        image = self.transform(image)
        
        # 对文本进行分词
        text_tokens = self.tokenizer.tokenize(caption)
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        
        return image, text_tensor, caption

class CLIPDataModule:
    """CLIP数据模块，用于管理训练和验证数据"""
    
    def __init__(self, 
                 train_data_path: str,
                 val_data_path: str,
                 image_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_length: int = 77):
        """
        初始化数据模块
        
        Args:
            train_data_path: 训练数据文件路径
            val_data_path: 验证数据文件路径
            image_dir: 图像目录路径
            batch_size: 批次大小
            num_workers: 数据加载器的工作进程数
            max_length: 文本最大长度
        """
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
        # 初始化分词器
        self.tokenizer = SimpleTokenizer(max_length=max_length)
        
        # 定义训练和验证的图像变换
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self) -> transforms.Compose:
        """获取训练时的图像变换（包含数据增强）"""
        return transforms.Compose([
            transforms.Resize((224, 224)),          # 调整大小
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            transforms.ColorJitter(                 # 随机颜色变换
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),                   # 转换为张量
            transforms.Normalize(                    # 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_val_transform(self) -> transforms.Compose:
        """获取验证时的图像变换（不包含数据增强）"""
        return transforms.Compose([
            transforms.Resize((224, 224)),  # 调整大小
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(           # 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def setup(self):
        """设置数据集"""
        # 创建训练数据集
        self.train_dataset = ImageTextDataset(
            data_path=self.train_data_path,
            image_dir=self.image_dir,
            transform=self.train_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # 创建验证数据集
        self.val_dataset = ImageTextDataset(
            data_path=self.val_data_path,
            image_dir=self.image_dir,
            transform=self.val_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,                    # 训练时打乱数据
            num_workers=self.num_workers,
            pin_memory=True,                 # 固定内存，加速GPU传输
            drop_last=True                   # 丢弃最后一个不完整的批次
        )
    
    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,                   # 验证时不打乱数据
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """返回测试数据加载器（这里使用验证数据）"""
        return self.val_dataloader()

def create_sample_dataset(data_dir: str, num_samples: int = 1000):
    """创建示例数据集文件"""
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建图像目录
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # 初始化分词器
    tokenizer = SimpleTokenizer()
    
    # 创建数据模块
    data_module = CLIPDataModule(
        train_data_path=os.path.join(data_dir, "train.json"),
        val_data_path=os.path.join(data_dir, "val.json"),
        image_dir=image_dir,
        batch_size=16
    )
    
    # 设置数据集
    data_module.setup()
    
    print(f"创建示例数据集完成！")
    print(f"训练样本数: {len(data_module.train_dataset)}")
    print(f"验证样本数: {len(data_module.val_dataset)}")
    
    return data_module

if __name__ == "__main__":
    # 测试代码
    print("测试CLIP数据集...")
    
    # 创建示例数据集
    data_dir = "./sample_data"
    data_module = create_sample_dataset(data_dir)
    
    # 测试数据加载器
    train_loader = data_module.train_dataloader()
    
    print("\n测试数据加载...")
    for batch_idx, (images, texts, captions) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  文本形状: {texts.shape}")
        print(f"  示例标题: {captions[0]}")
        
        # 测试分词器的解码功能
        decoded_text = data_module.tokenizer.decode(texts[0].tolist())
        print(f"  解码文本: {decoded_text}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("\n数据集测试完成！")