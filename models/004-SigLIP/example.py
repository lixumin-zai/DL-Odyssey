import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os
from pathlib import Path

# 导入我们的模型
from model import SigLIPModel, create_siglip_model
from train import SigLIPLightningModule

class SigLIPInference:
    """SigLIP模型推理类，用于零样本图像分类和图像-文本检索"""
    
    def __init__(self, model_path: str = None, model_size: str = 'base', device: str = 'auto'):
        """
        初始化推理类
        
        Args:
            model_path: 预训练模型路径（.ckpt文件）
            model_size: 模型大小，如果没有提供model_path则创建新模型
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            print(f"从检查点加载模型: {model_path}")
            # 从Lightning检查点加载模型
            self.lightning_model = SigLIPLightningModule.load_from_checkpoint(model_path)
            self.model = self.lightning_model.model
        else:
            print(f"创建新的{model_size}模型（随机权重）")
            self.model = create_siglip_model(model_size)
            self.lightning_model = None
        
        # 将模型移到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 定义图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),          # 转换为tensor
            transforms.Normalize(           # 标准化
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 创建简单的词汇表（实际应用中应该使用预训练的tokenizer）
        self.vocab = self._create_simple_vocab()
        
    def _create_simple_vocab(self) -> Dict[str, int]:
        """创建简单的词汇表用于演示"""
        # 基础词汇表
        vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            'a': 4, 'an': 5, 'the': 6, 'this': 7, 'that': 8,
            'cat': 9, 'dog': 10, 'bird': 11, 'car': 12, 'person': 13,
            'man': 14, 'woman': 15, 'child': 16, 'baby': 17,
            'house': 18, 'building': 19, 'tree': 20, 'flower': 21,
            'red': 22, 'blue': 23, 'green': 24, 'yellow': 25, 'black': 26, 'white': 27,
            'big': 28, 'small': 29, 'large': 30, 'tiny': 31,
            'sitting': 32, 'standing': 33, 'running': 34, 'walking': 35, 'flying': 36,
            'on': 37, 'in': 38, 'at': 39, 'with': 40, 'of': 41,
            'photo': 42, 'image': 43, 'picture': 44,
            'beautiful': 45, 'cute': 46, 'lovely': 47
        }
        
        # 添加更多常用词汇
        additional_words = [
            'animal', 'food', 'water', 'sky', 'ground', 'grass', 'road', 'street',
            'happy', 'sad', 'angry', 'surprised', 'calm', 'peaceful',
            'indoor', 'outdoor', 'nature', 'city', 'countryside',
            'morning', 'afternoon', 'evening', 'night', 'day',
            'sunny', 'cloudy', 'rainy', 'snowy', 'windy'
        ]
        
        for word in additional_words:
            if len(vocab) < 1000:  # 限制词汇表大小
                vocab[word] = len(vocab)
        
        return vocab
    
    def _tokenize_text(self, text: str, max_length: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
        """将文本转换为token序列"""
        # 简单分词
        words = text.lower().split()
        tokens = ['[CLS]'] + words + ['[SEP]']
        
        # 转换为ID
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['[UNK]'])
        
        # 截断或填充
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['[PAD]']] * (max_length - len(token_ids)))
        
        # 创建attention mask
        attention_mask = [1 if token_id != self.vocab['[PAD]'] else 0 for token_id in token_ids]
        
        return (
            torch.tensor(token_ids, dtype=torch.long, device=self.device),
            torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        )
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理单张图像"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        image_tensor = self.image_transform(image)
        
        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def encode_image(self, image_path: str) -> torch.Tensor:
        """编码单张图像"""
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_path)
            image_features = self.model.encode_image(image_tensor)
        return image_features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码单个文本"""
        with torch.no_grad():
            input_ids, attention_mask = self._tokenize_text(text)
            input_ids = input_ids.unsqueeze(0)  # 添加batch维度
            attention_mask = attention_mask.unsqueeze(0)
            text_features = self.model.encode_text(input_ids, attention_mask)
        return text_features
    
    def compute_similarity(self, image_path: str, text: str) -> float:
        """计算图像和文本之间的相似度"""
        # 编码图像和文本
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)
        
        # 计算相似度（余弦相似度）
        similarity = torch.cosine_similarity(image_features, text_features, dim=1)
        
        return similarity.item()
    
    def zero_shot_classification(self, image_path: str, candidate_labels: List[str]) -> Dict[str, float]:
        """零样本图像分类"""
        print(f"对图像进行零样本分类: {image_path}")
        print(f"候选标签: {candidate_labels}")
        
        # 编码图像
        image_features = self.encode_image(image_path)
        
        # 编码所有候选标签
        similarities = {}
        for label in candidate_labels:
            # 为标签添加模板
            text_prompt = f"a photo of {label}"
            text_features = self.encode_text(text_prompt)
            
            # 计算相似度
            similarity = torch.cosine_similarity(image_features, text_features, dim=1)
            similarities[label] = similarity.item()
        
        # 按相似度排序
        sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_similarities
    
    def image_text_retrieval(self, image_paths: List[str], text_queries: List[str]) -> Dict:
        """图像-文本检索"""
        print(f"执行图像-文本检索...")
        print(f"图像数量: {len(image_paths)}")
        print(f"查询数量: {len(text_queries)}")
        
        # 编码所有图像
        image_features_list = []
        for img_path in image_paths:
            img_features = self.encode_image(img_path)
            image_features_list.append(img_features)
        
        # 合并图像特征
        all_image_features = torch.cat(image_features_list, dim=0)
        
        # 编码所有文本查询
        text_features_list = []
        for query in text_queries:
            text_features = self.encode_text(query)
            text_features_list.append(text_features)
        
        # 合并文本特征
        all_text_features = torch.cat(text_features_list, dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(all_text_features, all_image_features.T)
        
        # 转换为numpy数组便于处理
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        return {
            'similarity_matrix': similarity_matrix,
            'image_paths': image_paths,
            'text_queries': text_queries
        }
    
    def visualize_similarities(self, image_paths: List[str], text_queries: List[str], 
                             save_path: str = None):
        """可视化图像-文本相似度矩阵"""
        # 执行检索
        results = self.image_text_retrieval(image_paths, text_queries)
        similarity_matrix = results['similarity_matrix']
        
        # 创建热力图
        plt.figure(figsize=(12, 8))
        
        # 创建标签（只显示文件名）
        image_labels = [Path(p).stem for p in image_paths]
        
        # 绘制热力图
        sns.heatmap(
            similarity_matrix,
            xticklabels=image_labels,
            yticklabels=text_queries,
            annot=True,
            fmt='.3f',
            cmap='viridis',
            center=0
        )
        
        plt.title('图像-文本相似度矩阵')
        plt.xlabel('图像')
        plt.ylabel('文本查询')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相似度矩阵保存到: {save_path}")
        
        plt.show()

def create_demo_images():
    """创建演示用的示例图像"""
    demo_dir = Path('./demo_images')
    demo_dir.mkdir(exist_ok=True)
    
    # 创建一些彩色示例图像
    colors_and_names = [
        ((255, 0, 0), 'red_square.jpg'),
        ((0, 255, 0), 'green_square.jpg'),
        ((0, 0, 255), 'blue_square.jpg'),
        ((255, 255, 0), 'yellow_square.jpg'),
        ((255, 0, 255), 'purple_square.jpg')
    ]
    
    created_images = []
    for color, name in colors_and_names:
        # 创建纯色图像
        image = Image.new('RGB', (224, 224), color)
        image_path = demo_dir / name
        image.save(image_path)
        created_images.append(str(image_path))
        print(f"创建演示图像: {image_path}")
    
    return created_images

def main():
    """主演示函数"""
    print("=" * 60)
    print("SigLIP模型推理演示")
    print("=" * 60)
    
    # 创建推理对象
    # 注意：这里使用随机权重的模型，实际应用中应该加载训练好的模型
    inference = SigLIPInference(model_size='base')
    
    # 创建演示图像
    print("\n1. 创建演示图像...")
    demo_images = create_demo_images()
    
    # 演示1：零样本图像分类
    print("\n2. 零样本图像分类演示...")
    candidate_labels = ['red object', 'green object', 'blue object', 'yellow object', 'purple object']
    
    for image_path in demo_images[:3]:  # 只测试前3张图像
        print(f"\n分类图像: {Path(image_path).name}")
        similarities = inference.zero_shot_classification(image_path, candidate_labels)
        
        print("分类结果:")
        for label, score in similarities.items():
            print(f"  {label}: {score:.4f}")
        
        # 显示最可能的标签
        best_label = max(similarities, key=similarities.get)
        print(f"  -> 预测标签: {best_label}")
    
    # 演示2：图像-文本相似度计算
    print("\n3. 图像-文本相似度计算演示...")
    test_image = demo_images[0]  # 使用红色图像
    test_texts = [
        "a red square",
        "a blue circle", 
        "a green triangle",
        "red color",
        "something blue"
    ]
    
    print(f"\n测试图像: {Path(test_image).name}")
    for text in test_texts:
        similarity = inference.compute_similarity(test_image, text)
        print(f"  '{text}': {similarity:.4f}")
    
    # 演示3：图像-文本检索
    print("\n4. 图像-文本检索演示...")
    text_queries = [
        "red color",
        "blue color", 
        "green color",
        "bright color",
        "dark color"
    ]
    
    try:
        # 可视化相似度矩阵
        inference.visualize_similarities(
            demo_images, 
            text_queries, 
            save_path='./similarity_matrix.png'
        )
    except Exception as e:
        print(f"可视化时出错: {e}")
        print("跳过可视化步骤...")
    
    # 演示4：批量处理
    print("\n5. 批量相似度计算...")
    results = inference.image_text_retrieval(demo_images, text_queries)
    similarity_matrix = results['similarity_matrix']
    
    print("相似度矩阵形状:", similarity_matrix.shape)
    print("最高相似度:", np.max(similarity_matrix))
    print("最低相似度:", np.min(similarity_matrix))
    
    # 找到每个查询的最佳匹配图像
    print("\n每个查询的最佳匹配:")
    for i, query in enumerate(text_queries):
        best_image_idx = np.argmax(similarity_matrix[i])
        best_image = Path(demo_images[best_image_idx]).name
        best_score = similarity_matrix[i, best_image_idx]
        print(f"  '{query}' -> {best_image} (相似度: {best_score:.4f})")
    
    print("\n=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    # 提供使用说明
    print("\n使用说明:")
    print("1. 要使用训练好的模型，请提供检查点路径:")
    print("   inference = SigLIPInference(model_path='path/to/checkpoint.ckpt')")
    print("2. 要处理真实图像，请将图像路径传递给相应的方法")
    print("3. 要获得更好的结果，请使用预训练的tokenizer和更大的词汇表")

if __name__ == '__main__':
    main()