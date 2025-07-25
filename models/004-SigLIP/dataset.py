import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path

class SigLIPDataset(Dataset):
    """SigLIP训练数据集类，处理图像-文本对数据"""
    
    def __init__(self, 
                 data_dir: str,
                 annotations_file: str,
                 image_size: int = 224,
                 max_text_length: int = 77,
                 vocab_size: int = 32000,
                 split: str = 'train'):
        """
        初始化数据集
        
        Args:
            data_dir: 图像文件夹路径
            annotations_file: 标注文件路径（JSON格式）
            image_size: 图像尺寸
            max_text_length: 文本最大长度
            vocab_size: 词汇表大小
            split: 数据集分割（train/val/test）
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        self.split = split
        
        # 加载标注数据
        self.annotations = self._load_annotations(annotations_file)
        
        # 创建简单的词汇表（实际应用中应该使用预训练的tokenizer）
        self.vocab = self._create_vocab()
        
        # 定义图像变换
        self.image_transform = self._get_image_transforms()
        
        print(f"加载了 {len(self.annotations)} 个{split}样本")
        
    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        """加载标注文件"""
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果标注文件包含不同split的数据，过滤出当前split
            if isinstance(data, dict) and self.split in data:
                return data[self.split]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"标注文件格式不正确: {annotations_file}")
                
        except FileNotFoundError:
            # 如果标注文件不存在，创建示例数据
            print(f"标注文件 {annotations_file} 不存在，创建示例数据...")
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> List[Dict]:
        """创建示例数据用于测试"""
        dummy_data = []
        
        # 创建一些示例图像-文本对
        sample_texts = [
            "a cat sitting on a chair",
            "a dog running in the park",
            "a bird flying in the sky",
            "a car driving on the road",
            "a person walking on the street",
            "a flower blooming in the garden",
            "a tree standing in the forest",
            "a house with a red roof",
            "a mountain covered with snow",
            "a beach with blue water"
        ]
        
        for i in range(100):  # 创建100个示例
            dummy_data.append({
                'image_path': f'dummy_image_{i:03d}.jpg',
                'caption': random.choice(sample_texts),
                'image_id': i
            })
            
        return dummy_data
    
    def _create_vocab(self) -> Dict[str, int]:
        """创建简单的词汇表（实际应用中应该使用预训练的tokenizer）"""
        # 特殊token
        special_tokens = {
            '[PAD]': 0,    # 填充token
            '[UNK]': 1,    # 未知token
            '[CLS]': 2,    # 分类token
            '[SEP]': 3,    # 分隔token
        }
        
        # 收集所有文本中的词汇
        all_words = set()
        for annotation in self.annotations:
            words = annotation['caption'].lower().split()
            all_words.update(words)
        
        # 创建词汇表
        vocab = special_tokens.copy()
        for i, word in enumerate(sorted(all_words)):
            if len(vocab) < self.vocab_size:
                vocab[word] = len(vocab)
            else:
                break
                
        return vocab
    
    def _get_image_transforms(self) -> transforms.Compose:
        """定义图像预处理变换"""
        if self.split == 'train':
            # 训练时使用数据增强
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # 调整图像大小
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomRotation(degrees=10),   # 随机旋转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
                transforms.ToTensor(),  # 转换为tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
        else:
            # 验证/测试时不使用数据增强
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # 调整图像大小
                transforms.ToTensor(),  # 转换为tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
            ])
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """将文本转换为token序列"""
        # 简单的分词（实际应用中应该使用更复杂的tokenizer）
        words = text.lower().split()
        
        # 添加特殊token
        tokens = ['[CLS]'] + words + ['[SEP]']
        
        # 转换为ID
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['[UNK]'])  # 未知token
        
        # 截断或填充到固定长度
        if len(token_ids) > self.max_text_length:
            token_ids = token_ids[:self.max_text_length]
        else:
            token_ids.extend([self.vocab['[PAD]']] * (self.max_text_length - len(token_ids)))
        
        # 创建attention mask（1表示有效token，0表示padding）
        attention_mask = [1 if token_id != self.vocab['[PAD]'] else 0 for token_id in token_ids]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载并预处理图像"""
        full_path = self.data_dir / image_path
        
        try:
            # 尝试加载真实图像
            image = Image.open(full_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # 如果图像不存在，创建随机图像用于测试
            image = Image.new('RGB', (self.image_size, self.image_size), 
                            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        
        # 应用图像变换
        image_tensor = self.image_transform(image)
        
        return image_tensor
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        annotation = self.annotations[idx]
        
        # 加载图像
        image = self._load_image(annotation['image_path'])
        
        # 处理文本
        input_ids, attention_mask = self._tokenize_text(annotation['caption'])
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_id': annotation.get('image_id', idx),
            'caption': annotation['caption']  # 原始文本，用于调试
        }

class SigLIPDataModule:
    """数据模块，管理训练、验证和测试数据加载器"""
    
    def __init__(self,
                 data_dir: str,
                 train_annotations: str,
                 val_annotations: str,
                 test_annotations: Optional[str] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: int = 224,
                 max_text_length: int = 77,
                 vocab_size: int = 32000):
        """
        初始化数据模块
        
        Args:
            data_dir: 图像文件夹路径
            train_annotations: 训练集标注文件
            val_annotations: 验证集标注文件
            test_annotations: 测试集标注文件（可选）
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            image_size: 图像尺寸
            max_text_length: 文本最大长度
            vocab_size: 词汇表大小
        """
        self.data_dir = data_dir
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.test_annotations = test_annotations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vocab_size = vocab_size
        
        # 初始化数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """设置数据集"""
        # 创建训练数据集
        self.train_dataset = SigLIPDataset(
            data_dir=self.data_dir,
            annotations_file=self.train_annotations,
            image_size=self.image_size,
            max_text_length=self.max_text_length,
            vocab_size=self.vocab_size,
            split='train'
        )
        
        # 创建验证数据集
        self.val_dataset = SigLIPDataset(
            data_dir=self.data_dir,
            annotations_file=self.val_annotations,
            image_size=self.image_size,
            max_text_length=self.max_text_length,
            vocab_size=self.vocab_size,
            split='val'
        )
        
        # 创建测试数据集（如果提供）
        if self.test_annotations:
            self.test_dataset = SigLIPDataset(
                data_dir=self.data_dir,
                annotations_file=self.test_annotations,
                image_size=self.image_size,
                max_text_length=self.max_text_length,
                vocab_size=self.vocab_size,
                split='test'
            )
    
    def train_dataloader(self) -> DataLoader:
        """返回训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练时打乱数据
            num_workers=self.num_workers,
            pin_memory=True,  # 加速GPU传输
            drop_last=True   # 丢弃最后一个不完整的batch
        )
    
    def val_dataloader(self) -> DataLoader:
        """返回验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 验证时不打乱数据
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """返回测试数据加载器"""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 测试时不打乱数据
            num_workers=self.num_workers,
            pin_memory=True
        )

def create_sample_annotations(output_dir: str, num_samples: int = 1000):
    """创建示例标注文件用于测试"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例文本描述
    sample_captions = [
        "a cat sitting on a windowsill",
        "a dog playing in the garden",
        "a bird perched on a tree branch",
        "a car parked on the street",
        "a person walking in the park",
        "a flower blooming in spring",
        "a mountain peak covered in snow",
        "a beach with crystal clear water",
        "a sunset over the ocean",
        "a city skyline at night",
        "a child playing with toys",
        "a book lying on a table",
        "a cup of coffee on a desk",
        "a bicycle leaning against a wall",
        "a train arriving at the station"
    ]
    
    # 创建训练集标注
    train_annotations = []
    for i in range(int(num_samples * 0.8)):  # 80%用于训练
        train_annotations.append({
            'image_path': f'train_image_{i:05d}.jpg',
            'caption': random.choice(sample_captions),
            'image_id': i
        })
    
    # 创建验证集标注
    val_annotations = []
    for i in range(int(num_samples * 0.2)):  # 20%用于验证
        val_annotations.append({
            'image_path': f'val_image_{i:05d}.jpg',
            'caption': random.choice(sample_captions),
            'image_id': i + len(train_annotations)
        })
    
    # 保存标注文件
    train_file = os.path.join(output_dir, 'train_annotations.json')
    val_file = os.path.join(output_dir, 'val_annotations.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"创建了 {len(train_annotations)} 个训练样本和 {len(val_annotations)} 个验证样本")
    print(f"标注文件保存在: {output_dir}")
    
    return train_file, val_file

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """自定义的batch整理函数"""
    # 将batch中的所有样本整理成tensor
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    image_ids = torch.tensor([item['image_id'] for item in batch])
    captions = [item['caption'] for item in batch]  # 保持为字符串列表
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image_ids': image_ids,
        'captions': captions
    }

if __name__ == "__main__":
    # 测试数据集和数据加载器
    print("测试SigLIP数据集...")
    
    # 创建示例数据目录
    data_dir = "./sample_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建示例标注文件
    train_file, val_file = create_sample_annotations(data_dir, num_samples=100)
    
    # 创建数据模块
    data_module = SigLIPDataModule(
        data_dir=data_dir,
        train_annotations=train_file,
        val_annotations=val_file,
        batch_size=4,
        num_workers=0,  # 测试时使用0个工作进程
        image_size=224
    )
    
    # 设置数据集
    data_module.setup()
    
    # 测试训练数据加载器
    train_loader = data_module.train_dataloader()
    print(f"训练数据加载器批次数: {len(train_loader)}")
    
    # 获取一个batch的数据
    for batch in train_loader:
        print(f"图像batch shape: {batch['images'].shape}")
        print(f"文本ID batch shape: {batch['input_ids'].shape}")
        print(f"注意力mask batch shape: {batch['attention_mask'].shape}")
        print(f"图像ID batch shape: {batch['image_ids'].shape}")
        print(f"文本描述: {batch['captions'][:2]}")
        break
    
    # 测试验证数据加载器
    val_loader = data_module.val_dataloader()
    print(f"验证数据加载器批次数: {len(val_loader)}")
    
    print("数据集测试完成！")