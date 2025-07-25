import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import os
from typing import List, Tuple
import argparse

# 导入我们自定义的模块
from model import CLIP, create_clip_model
from dataset import SimpleTokenizer
from train import CLIPLightningModule

class CLIPInference:
    """CLIP模型推理类，用于零样本分类和图像-文本匹配"""
    
    def __init__(self, model_path: str = None, model_size: str = "base", device: str = "auto"):
        """
        初始化CLIP推理器
        
        Args:
            model_path: 预训练模型路径（如果为None则使用随机初始化的模型）
            model_size: 模型大小（"base" 或 "large"）
            device: 计算设备（"auto", "cpu", "cuda"）
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            print(f"从 {model_path} 加载预训练模型...")
            # 从Lightning检查点加载模型
            lightning_model = CLIPLightningModule.load_from_checkpoint(model_path)
            self.model = lightning_model.model
        else:
            print(f"创建新的 {model_size} 模型（随机初始化）...")
            self.model = create_clip_model(model_size)
        
        # 将模型移动到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化分词器
        self.tokenizer = SimpleTokenizer()
        
        # 初始化图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(           # 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 应用预处理变换
        image_tensor = self.image_transform(image).unsqueeze(0)  # 添加批次维度
        
        return image_tensor.to(self.device)
    
    def preprocess_text(self, texts: List[str]) -> torch.Tensor:
        """预处理文本"""
        # 对每个文本进行分词
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        
        # 转换为张量
        text_tensor = torch.tensor(tokenized_texts, dtype=torch.long)
        
        return text_tensor.to(self.device)
    
    def zero_shot_classify(self, image_path: str, class_names: List[str], 
                          template: str = "a photo of a {}") -> Tuple[List[float], int]:
        """零样本图像分类"""
        # 预处理图像
        image = self.preprocess_image(image_path)
        
        # 为每个类别创建文本描述
        text_descriptions = [template.format(class_name) for class_name in class_names]
        texts = self.preprocess_text(text_descriptions)
        
        with torch.no_grad():
            # 编码图像和文本
            image_features = self.model.encode_image(image)  # [1, embed_dim]
            text_features = self.model.encode_text(texts)    # [num_classes, embed_dim]
            
            # 计算相似度
            similarities = torch.matmul(image_features, text_features.T)  # [1, num_classes]
            
            # 应用softmax获得概率分布
            probabilities = F.softmax(similarities, dim=-1).squeeze(0)  # [num_classes]
            
            # 获取预测类别
            predicted_class_idx = torch.argmax(probabilities).item()
        
        # 转换为Python列表
        prob_list = probabilities.cpu().numpy().tolist()
        
        return prob_list, predicted_class_idx
    
    def image_text_similarity(self, image_path: str, texts: List[str]) -> List[float]:
        """计算图像与多个文本的相似度"""
        # 预处理输入
        image = self.preprocess_image(image_path)
        text_tensor = self.preprocess_text(texts)
        
        with torch.no_grad():
            # 编码图像和文本
            image_features = self.model.encode_image(image)     # [1, embed_dim]
            text_features = self.model.encode_text(text_tensor) # [num_texts, embed_dim]
            
            # 计算余弦相似度
            similarities = torch.matmul(image_features, text_features.T)  # [1, num_texts]
            similarities = similarities.squeeze(0)  # [num_texts]
        
        return similarities.cpu().numpy().tolist()
    
    def find_best_text_match(self, image_path: str, texts: List[str]) -> Tuple[str, float, int]:
        """找到与图像最匹配的文本"""
        similarities = self.image_text_similarity(image_path, texts)
        
        # 找到最高相似度的索引
        best_idx = np.argmax(similarities)
        best_text = texts[best_idx]
        best_similarity = similarities[best_idx]
        
        return best_text, best_similarity, best_idx
    
    def visualize_classification_results(self, image_path: str, class_names: List[str], 
                                       probabilities: List[float], predicted_idx: int,
                                       save_path: str = None, show: bool = True):
        """可视化分类结果"""
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示图像
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title(f"输入图像\n预测: {class_names[predicted_idx]}")
        ax1.axis('off')
        
        # 显示概率分布
        colors = ['red' if i == predicted_idx else 'blue' for i in range(len(class_names))]
        bars = ax2.bar(range(len(class_names)), probabilities, color=colors)
        ax2.set_xlabel('类别')
        ax2.set_ylabel('概率')
        ax2.set_title('分类概率分布')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

def demo_zero_shot_classification():
    """演示零样本分类功能"""
    print("=" * 50)
    print("CLIP 零样本分类演示")
    print("=" * 50)
    
    # 初始化推理器
    clip_inference = CLIPInference(model_size="base")
    
    # 创建示例图像（如果不存在真实图像）
    demo_dir = "./demo_images"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 创建一个简单的示例图像
    demo_image_path = os.path.join(demo_dir, "demo_image.jpg")
    if not os.path.exists(demo_image_path):
        # 创建一个彩色方块作为示例
        demo_image = Image.new("RGB", (224, 224), color=(100, 150, 200))
        demo_image.save(demo_image_path)
        print(f"创建示例图像: {demo_image_path}")
    
    # 定义分类类别
    class_names = [
        "cat", "dog", "bird", "car", "tree", 
        "house", "person", "flower", "book", "computer"
    ]
    
    print(f"\n对图像 {demo_image_path} 进行分类...")
    print(f"候选类别: {', '.join(class_names)}")
    
    # 执行零样本分类
    probabilities, predicted_idx = clip_inference.zero_shot_classify(
        demo_image_path, class_names
    )
    
    # 显示结果
    print(f"\n分类结果:")
    print(f"预测类别: {class_names[predicted_idx]}")
    print(f"置信度: {probabilities[predicted_idx]:.4f}")
    
    print(f"\n所有类别的概率:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        marker = " ← 预测" if i == predicted_idx else ""
        print(f"  {class_name:10s}: {prob:.4f}{marker}")
    
    # 可视化结果
    output_path = os.path.join(demo_dir, "classification_result.png")
    clip_inference.visualize_classification_results(
        demo_image_path, class_names, probabilities, predicted_idx,
        save_path=output_path, show=False
    )

def demo_image_text_matching():
    """演示图像-文本匹配功能"""
    print("\n" + "=" * 50)
    print("CLIP 图像-文本匹配演示")
    print("=" * 50)
    
    # 初始化推理器
    clip_inference = CLIPInference(model_size="base")
    
    # 使用之前创建的示例图像
    demo_image_path = "./demo_images/demo_image.jpg"
    
    # 定义候选文本描述
    candidate_texts = [
        "a beautiful landscape with mountains",
        "a cute cat sitting on a chair",
        "a red sports car on the road",
        "a colorful abstract painting",
        "a person reading a book",
        "a blue geometric shape",
        "a green tree in the park",
        "a modern building architecture"
    ]
    
    print(f"\n计算图像与文本的相似度...")
    print(f"图像: {demo_image_path}")
    
    # 计算相似度
    similarities = clip_inference.image_text_similarity(demo_image_path, candidate_texts)
    
    # 找到最佳匹配
    best_text, best_similarity, best_idx = clip_inference.find_best_text_match(
        demo_image_path, candidate_texts
    )
    
    print(f"\n最佳匹配文本: \"{best_text}\"")
    print(f"相似度分数: {best_similarity:.4f}")
    
    print(f"\n所有文本的相似度分数:")
    for i, (text, similarity) in enumerate(zip(candidate_texts, similarities)):
        marker = " ← 最佳匹配" if i == best_idx else ""
        print(f"  {similarity:.4f}: {text}{marker}")

def demo_custom_prompts():
    """演示自定义提示词的效果"""
    print("\n" + "=" * 50)
    print("CLIP 自定义提示词演示")
    print("=" * 50)
    
    # 初始化推理器
    clip_inference = CLIPInference(model_size="base")
    
    demo_image_path = "./demo_images/demo_image.jpg"
    class_names = ["cat", "dog", "car", "tree", "abstract art"]
    
    # 测试不同的提示词模板
    templates = [
        "a photo of a {}",
        "a picture showing a {}",
        "an image of a {}",
        "a beautiful {}",
        "a {} in high quality"
    ]
    
    print(f"\n测试不同提示词模板的效果:")
    
    for template in templates:
        probabilities, predicted_idx = clip_inference.zero_shot_classify(
            demo_image_path, class_names, template=template
        )
        
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        print(f"\n模板: \"{template}\"")
        print(f"  预测: {predicted_class} (置信度: {confidence:.4f})")

def demo_batch_processing():
    """演示批量处理功能"""
    print("\n" + "=" * 50)
    print("CLIP 批量处理演示")
    print("=" * 50)
    
    # 初始化推理器
    clip_inference = CLIPInference(model_size="base")
    
    # 创建多个示例图像
    demo_dir = "./demo_images"
    image_paths = []
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    color_names = ["red", "green", "blue", "yellow", "magenta"]
    
    for i, (color, color_name) in enumerate(zip(colors, color_names)):
        image_path = os.path.join(demo_dir, f"demo_{color_name}.jpg")
        if not os.path.exists(image_path):
            demo_image = Image.new("RGB", (224, 224), color=color)
            demo_image.save(image_path)
        image_paths.append(image_path)
    
    # 定义分类类别
    class_names = ["red object", "green object", "blue object", "yellow object", "purple object"]
    
    print(f"\n批量处理 {len(image_paths)} 张图像...")
    
    # 批量分类
    results = []
    for i, image_path in enumerate(image_paths):
        probabilities, predicted_idx = clip_inference.zero_shot_classify(
            image_path, class_names
        )
        
        results.append({
            'image': os.path.basename(image_path),
            'predicted_class': class_names[predicted_idx],
            'confidence': probabilities[predicted_idx],
            'probabilities': probabilities
        })
    
    # 显示结果
    print(f"\n批量分类结果:")
    print(f"{'图像':<20} {'预测类别':<15} {'置信度':<10}")
    print("-" * 45)
    
    for result in results:
        print(f"{result['image']:<20} {result['predicted_class']:<15} {result['confidence']:<10.4f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CLIP模型使用示例")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="模型大小")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--demo", type=str, default="all", 
                       choices=["all", "classification", "matching", "prompts", "batch"],
                       help="运行的演示类型")
    
    args = parser.parse_args()
    
    print("CLIP模型使用示例")
    print(f"模型大小: {args.model_size}")
    print(f"设备: {args.device}")
    if args.model_path:
        print(f"模型路径: {args.model_path}")
    
    # 运行指定的演示
    if args.demo in ["all", "classification"]:
        demo_zero_shot_classification()
    
    if args.demo in ["all", "matching"]:
        demo_image_text_matching()
    
    if args.demo in ["all", "prompts"]:
        demo_custom_prompts()
    
    if args.demo in ["all", "batch"]:
        demo_batch_processing()
    
    print("\n" + "=" * 50)
    print("所有演示完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()