import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import os
import argparse
from pathlib import Path

# 导入自定义模块
from model import build_dbnet
from train import DBNetLightningModule

class DBNetInference:
    """
    DBNet推理类
    
    用于加载训练好的模型并进行文本检测推理
    包含图像预处理、模型推理和后处理功能
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'auto',
                 image_size: int = 640,
                 binary_threshold: float = 0.3,
                 polygon_threshold: float = 0.7,
                 max_candidates: int = 1000):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重文件路径
            device: 推理设备 ('cpu', 'cuda', 'auto')
            image_size: 输入图像尺寸
            binary_threshold: 二值化阈值
            polygon_threshold: 多边形置信度阈值
            max_candidates: 最大候选区域数量
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 保存参数
        self.image_size = image_size
        self.binary_threshold = binary_threshold
        self.polygon_threshold = polygon_threshold
        self.max_candidates = max_candidates
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406])  # ImageNet均值
        self.std = np.array([0.229, 0.224, 0.225])   # ImageNet标准差
    
    def _load_model(self, model_path: str = None) -> torch.nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        if model_path and os.path.exists(model_path):
            print(f"从检查点加载模型: {model_path}")
            try:
                # 尝试加载Lightning检查点
                lightning_model = DBNetLightningModule.load_from_checkpoint(model_path)
                model = lightning_model.model
            except Exception as e:
                print(f"Lightning检查点加载失败: {e}")
                print("尝试加载普通PyTorch模型...")
                # 创建新模型并加载权重
                model = build_dbnet(backbone='resnet18', pretrained=False)
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
        else:
            print("创建新的预训练模型（用于演示）")
            # 创建预训练模型用于演示
            model = build_dbnet(backbone='resnet18', pretrained=True)
        
        # 设置为评估模式并移动到指定设备
        model.eval()
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """
        图像预处理
        
        Args:
            image: 输入图像 (H, W, 3)
            
        Returns:
            预处理后的张量、缩放比例和原始尺寸
        """
        # 保存原始尺寸
        original_height, original_width = image.shape[:2]
        
        # 计算缩放比例，保持宽高比
        scale = min(self.image_size / original_height, self.image_size / original_width)
        
        # 计算新尺寸
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        
        # 调整图像尺寸
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # 创建填充图像
        padded_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        padded_image[:new_height, :new_width] = resized_image
        
        # 归一化
        normalized_image = padded_image.astype(np.float32) / 255.0
        normalized_image = (normalized_image - self.mean) / self.std
        
        # 转换为张量
        tensor_image = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0)
        tensor_image = tensor_image.to(self.device)
        
        return tensor_image, scale, (original_height, original_width)
    
    def postprocess_outputs(self, 
                          outputs: Dict[str, torch.Tensor], 
                          scale: float, 
                          original_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        后处理模型输出
        
        Args:
            outputs: 模型输出字典
            scale: 图像缩放比例
            original_size: 原始图像尺寸
            
        Returns:
            检测到的文本多边形列表
        """
        # 提取二值图
        binary_map = outputs['binary_map'].squeeze().cpu().numpy()  # (H, W)
        
        # 应用阈值
        binary_mask = (binary_map > self.binary_threshold).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        
        for contour in contours:
            # 过滤太小的轮廓
            if cv2.contourArea(contour) < 10:
                continue
            
            # 多边形近似
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 确保至少有4个点
            if len(approx) < 4:
                # 使用最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                approx = box.reshape(-1, 1, 2).astype(np.int32)
            
            # 转换坐标
            polygon = approx.reshape(-1, 2).astype(np.float32)
            
            # 缩放回原始尺寸
            polygon = polygon / scale
            
            # 确保坐标在图像范围内
            polygon[:, 0] = np.clip(polygon[:, 0], 0, original_size[1] - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, original_size[0] - 1)
            
            polygons.append(polygon)
        
        return polygons
    
    def detect_text(self, image: np.ndarray) -> List[np.ndarray]:
        """
        检测图像中的文本
        
        Args:
            image: 输入图像 (H, W, 3)
            
        Returns:
            检测到的文本多边形列表
        """
        with torch.no_grad():
            # 预处理
            tensor_image, scale, original_size = self.preprocess_image(image)
            
            # 模型推理
            outputs = self.model(tensor_image)
            
            # 后处理
            polygons = self.postprocess_outputs(outputs, scale, original_size)
            
            return polygons
    
    def visualize_results(self, 
                         image: np.ndarray, 
                         polygons: List[np.ndarray], 
                         save_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            polygons: 检测到的多边形
            save_path: 保存路径（可选）
            
        Returns:
            可视化图像
        """
        # 复制图像
        vis_image = image.copy()
        
        # 绘制多边形
        for i, polygon in enumerate(polygons):
            # 转换为整数坐标
            pts = polygon.astype(np.int32)
            
            # 绘制多边形边界
            cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
            
            # 绘制顶点
            for point in pts:
                cv2.circle(vis_image, tuple(point), 3, (255, 0, 0), -1)
            
            # 添加编号
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(vis_image, str(i), tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"可视化结果已保存到: {save_path}")
        
        return vis_image


def create_demo_image() -> np.ndarray:
    """
    创建演示图像
    
    Returns:
        包含文本的演示图像
    """
    # 创建白色背景
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 使用PIL绘制文本
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # 绘制多行文本
    texts = [
        "DBNet Text Detection",
        "Real-time Scene Text",
        "Differentiable Binarization",
        "PyTorch Implementation"
    ]
    
    y_positions = [50, 150, 250, 350]
    
    for text, y in zip(texts, y_positions):
        # 计算文本位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (640 - text_width) // 2
        
        # 绘制文本
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    # 转换回numpy数组
    return np.array(pil_image)


def demo_inference(model_path: str = None, image_path: str = None):
    """
    演示推理过程
    
    Args:
        model_path: 模型文件路径
        image_path: 图像文件路径
    """
    print("=" * 50)
    print("DBNet文本检测演示")
    print("=" * 50)
    
    # 创建推理器
    detector = DBNetInference(
        model_path=model_path,
        device='auto',
        image_size=640,
        binary_threshold=0.3
    )
    
    # 加载或创建图像
    if image_path and os.path.exists(image_path):
        print(f"加载图像: {image_path}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print("创建演示图像...")
        image = create_demo_image()
    
    print(f"图像尺寸: {image.shape}")
    
    # 执行文本检测
    print("执行文本检测...")
    polygons = detector.detect_text(image)
    
    print(f"检测到 {len(polygons)} 个文本区域")
    
    # 打印检测结果
    for i, polygon in enumerate(polygons):
        print(f"文本区域 {i}: {len(polygon)} 个顶点")
        print(f"  坐标: {polygon.tolist()}")
    
    # 可视化结果
    print("生成可视化结果...")
    vis_image = detector.visualize_results(image, polygons, 'detection_result.png')
    
    # 使用matplotlib显示结果
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 检测结果
    plt.subplot(1, 3, 2)
    plt.imshow(vis_image)
    plt.title(f'检测结果 ({len(polygons)} 个区域)')
    plt.axis('off')
    
    # 二值图（如果可用）
    plt.subplot(1, 3, 3)
    with torch.no_grad():
        tensor_image, _, _ = detector.preprocess_image(image)
        outputs = detector.model(tensor_image)
        binary_map = outputs['binary_map'].squeeze().cpu().numpy()
    
    plt.imshow(binary_map, cmap='gray')
    plt.title('二值化结果')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n演示完成！")
    print("结果文件:")
    print("- detection_result.png: 检测结果可视化")
    print("- demo_results.png: 完整演示结果")


def batch_inference(image_dir: str, model_path: str = None, output_dir: str = './results'):
    """
    批量推理
    
    Args:
        image_dir: 图像目录路径
        model_path: 模型文件路径
        output_dir: 输出目录路径
    """
    print(f"批量处理图像目录: {image_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建推理器
    detector = DBNetInference(model_path=model_path)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 遍历图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    for i, image_path in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # 加载图像
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 执行检测
            polygons = detector.detect_text(image)
            
            # 保存可视化结果
            output_path = os.path.join(output_dir, f'result_{image_path.stem}.png')
            detector.visualize_results(image, polygons, output_path)
            
            # 保存检测结果（文本格式）
            txt_path = os.path.join(output_dir, f'result_{image_path.stem}.txt')
            with open(txt_path, 'w') as f:
                for j, polygon in enumerate(polygons):
                    coords = ','.join([f'{x:.1f},{y:.1f}' for x, y in polygon])
                    f.write(f'{coords}\n')
            
            print(f"  检测到 {len(polygons)} 个文本区域")
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    print(f"\n批量处理完成！结果保存在: {output_dir}")


def main():
    """
    主函数：解析命令行参数并执行相应功能
    """
    parser = argparse.ArgumentParser(description='DBNet文本检测推理示例')
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'single', 'batch'],
                       help='运行模式')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型文件路径')
    parser.add_argument('--image_path', type=str, default=None,
                       help='单张图像路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像目录路径（批量模式）')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录路径')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='二值化阈值')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # 演示模式
        demo_inference(args.model_path, args.image_path)
    
    elif args.mode == 'single':
        # 单张图像推理
        if not args.image_path:
            print("错误: 单张图像模式需要指定 --image_path")
            return
        
        detector = DBNetInference(
            model_path=args.model_path,
            binary_threshold=args.threshold
        )
        
        # 加载图像
        image = cv2.imread(args.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        polygons = detector.detect_text(image)
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'single_result.png')
        detector.visualize_results(image, polygons, output_path)
        
        print(f"检测完成！结果保存在: {output_path}")
    
    elif args.mode == 'batch':
        # 批量推理模式
        if not args.image_dir:
            print("错误: 批量模式需要指定 --image_dir")
            return
        
        batch_inference(args.image_dir, args.model_path, args.output_dir)


if __name__ == '__main__':
    # 运行主函数
    main()
    
    # 如果直接运行此文件，执行演示
    print("\n直接运行演示模式...")
    demo_inference()
    
    print("\n使用说明:")
    print("1. 演示模式: python example.py --mode demo")
    print("2. 单张图像: python example.py --mode single --image_path /path/to/image.jpg")
    print("3. 批量处理: python example.py --mode batch --image_dir /path/to/images/")
    print("4. 使用训练好的模型: python example.py --model_path /path/to/model.ckpt")