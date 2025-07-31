import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union
import pandas as pd

class SVMDataset(Dataset):
    """
    SVM专用数据集类
    
    这个类封装了SVM训练所需的数据预处理功能，
    包括特征标准化、标签编码等
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 transform: Optional[callable] = None,
                 normalize: bool = True):
        """
        初始化SVM数据集
        
        Args:
            X: 特征数据 [n_samples, n_features]
            y: 标签数据 [n_samples]
            transform: 可选的数据变换函数
            normalize: 是否对特征进行标准化
        """
        # 将numpy数组转换为PyTorch张量
        self.X = torch.FloatTensor(X)  # 特征数据转换为浮点张量
        self.y = torch.LongTensor(y)   # 标签数据转换为长整型张量
        
        self.transform = transform  # 存储数据变换函数
        self.normalize = normalize  # 是否标准化标志
        
        # 如果需要标准化，计算并存储标准化参数
        if self.normalize:
            self.scaler = StandardScaler()  # 创建标准化器
            # 对特征数据进行标准化：(x - mean) / std
            X_normalized = self.scaler.fit_transform(X)
            self.X = torch.FloatTensor(X_normalized)  # 更新为标准化后的数据
        else:
            self.scaler = None  # 不使用标准化器
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集中样本的数量
        """
        return len(self.X)  # 返回特征矩阵的行数
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: (特征, 标签) 元组
        """
        # 获取指定索引的特征和标签
        x = self.X[idx]  # 第idx个样本的特征向量
        y = self.y[idx]  # 第idx个样本的标签
        
        # 如果定义了数据变换函数，应用变换
        if self.transform:
            x = self.transform(x)
        
        return x, y  # 返回特征和标签的元组
    
    def get_feature_names(self) -> list:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        # 生成默认的特征名称
        return [f'feature_{i}' for i in range(self.X.shape[1])]
    
    def get_class_distribution(self) -> dict:
        """
        获取类别分布信息
        
        Returns:
            类别分布字典 {类别: 数量}
        """
        # 统计每个类别的样本数量
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}
    
    def apply_transform(self, X_new: np.ndarray) -> torch.Tensor:
        """
        对新数据应用相同的预处理变换
        
        这个方法用于对测试数据或新数据应用训练时的预处理步骤
        
        Args:
            X_new: 新的特征数据
            
        Returns:
            变换后的数据张量
        """
        if self.scaler is not None:
            # 使用训练时的标准化参数对新数据进行标准化
            X_transformed = self.scaler.transform(X_new)
        else:
            X_transformed = X_new
        
        return torch.FloatTensor(X_transformed)


class SyntheticDataGenerator:
    """
    合成数据生成器
    
    用于生成各种类型的合成数据集，方便测试SVM在不同数据分布下的性能
    """
    
    @staticmethod
    def generate_linear_separable(n_samples: int = 1000, 
                                 n_features: int = 2, 
                                 n_classes: int = 2,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成线性可分的数据集
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            n_classes: 类别数量
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵 [n_samples, n_features]
            y: 标签向量 [n_samples]
        """
        # 使用sklearn生成线性可分的分类数据
        X, y = make_classification(
            n_samples=n_samples,      # 样本数量
            n_features=n_features,    # 特征数量
            n_redundant=0,           # 冗余特征数量（设为0避免特征间线性相关）
            n_informative=n_features, # 有信息量的特征数量
            n_classes=n_classes,     # 类别数量
            n_clusters_per_class=1,  # 每个类别的簇数量
            class_sep=2.0,           # 类别间的分离度（越大越容易分离）
            random_state=random_state # 随机种子确保结果可重现
        )
        
        return X, y
    
    @staticmethod
    def generate_circles(n_samples: int = 1000, 
                        noise: float = 0.1, 
                        factor: float = 0.5,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成同心圆数据集（非线性可分）
        
        这种数据集需要使用非线性核函数才能有效分类
        
        Args:
            n_samples: 样本数量
            noise: 噪声水平
            factor: 内外圆的比例因子
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵 [n_samples, 2]
            y: 标签向量 [n_samples]
        """
        # 生成两个同心圆的数据点
        X, y = make_circles(
            n_samples=n_samples,     # 样本数量
            noise=noise,             # 添加的高斯噪声标准差
            factor=factor,           # 内圆与外圆的比例
            random_state=random_state # 随机种子
        )
        
        return X, y
    
    @staticmethod
    def generate_moons(n_samples: int = 1000, 
                      noise: float = 0.1,
                      random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成月牙形数据集（非线性可分）
        
        Args:
            n_samples: 样本数量
            noise: 噪声水平
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵 [n_samples, 2]
            y: 标签向量 [n_samples]
        """
        # 生成两个交错的半圆形数据点
        X, y = make_moons(
            n_samples=n_samples,     # 样本数量
            noise=noise,             # 添加的高斯噪声标准差
            random_state=random_state # 随机种子
        )
        
        return X, y
    
    @staticmethod
    def generate_xor_data(n_samples: int = 1000,
                         noise: float = 0.1,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成XOR问题数据集（经典的非线性问题）
        
        XOR问题是机器学习中的经典问题，线性分类器无法解决
        
        Args:
            n_samples: 样本数量
            noise: 噪声水平
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵 [n_samples, 2]
            y: 标签向量 [n_samples]
        """
        np.random.seed(random_state)  # 设置随机种子
        
        # 生成四个象限的数据点
        n_per_quadrant = n_samples // 4  # 每个象限的样本数
        
        # 第一象限和第三象限：标签为1
        X1 = np.random.normal([1, 1], noise, (n_per_quadrant, 2))    # 右上角
        X3 = np.random.normal([-1, -1], noise, (n_per_quadrant, 2))  # 左下角
        y1 = np.ones(n_per_quadrant * 2)  # 标签为1
        
        # 第二象限和第四象限：标签为0
        X2 = np.random.normal([-1, 1], noise, (n_per_quadrant, 2))   # 左上角
        X4 = np.random.normal([1, -1], noise, (n_per_quadrant, 2))   # 右下角
        y2 = np.zeros(n_per_quadrant * 2)  # 标签为0
        
        # 合并所有数据
        X = np.vstack([X1, X3, X2, X4])
        y = np.hstack([y1, y2])
        
        # 随机打乱数据顺序
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        return X, y.astype(int)


class RealWorldDataLoader:
    """
    真实世界数据集加载器
    
    提供加载和预处理真实数据集的功能
    """
    
    @staticmethod
    def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        加载乳腺癌数据集（二分类）
        
        这是一个经典的医学诊断数据集，用于预测乳腺肿瘤是良性还是恶性
        
        Returns:
            X: 特征矩阵 [569, 30]
            y: 标签向量 [569] (0: 恶性, 1: 良性)
            feature_names: 特征名称列表
        """
        # 从sklearn加载乳腺癌数据集
        data = load_breast_cancer()
        X, y = data.data, data.target  # 获取特征和标签
        feature_names = data.feature_names.tolist()  # 获取特征名称
        
        print(f"乳腺癌数据集加载完成：")
        print(f"  - 样本数量: {X.shape[0]}")
        print(f"  - 特征数量: {X.shape[1]}")
        print(f"  - 类别分布: 恶性={np.sum(y==0)}, 良性={np.sum(y==1)}")
        
        return X, y, feature_names
    
    @staticmethod
    def load_wine() -> Tuple[np.ndarray, np.ndarray, list]:
        """
        加载红酒数据集（多分类）
        
        这是一个经典的多分类数据集，用于根据化学成分预测红酒类型
        
        Returns:
            X: 特征矩阵 [178, 13]
            y: 标签向量 [178] (0, 1, 2 三个类别)
            feature_names: 特征名称列表
        """
        # 从sklearn加载红酒数据集
        data = load_wine()
        X, y = data.data, data.target  # 获取特征和标签
        feature_names = data.feature_names.tolist()  # 获取特征名称
        
        print(f"红酒数据集加载完成：")
        print(f"  - 样本数量: {X.shape[0]}")
        print(f"  - 特征数量: {X.shape[1]}")
        print(f"  - 类别数量: {len(np.unique(y))}")
        
        # 统计每个类别的样本数量
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  - 类别{cls}: {count}个样本")
        
        return X, y, feature_names
    
    @staticmethod
    def load_from_csv(file_path: str, 
                     target_column: str,
                     feature_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        从CSV文件加载数据
        
        Args:
            file_path: CSV文件路径
            target_column: 目标列名称
            feature_columns: 特征列名称列表，如果为None则使用除目标列外的所有列
            
        Returns:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            print(f"成功加载CSV文件: {file_path}")
            print(f"数据形状: {df.shape}")
            
            # 提取目标列
            if target_column not in df.columns:
                raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
            
            y = df[target_column].values  # 获取标签数据
            
            # 提取特征列
            if feature_columns is None:
                # 使用除目标列外的所有数值列
                feature_columns = [col for col in df.columns if col != target_column]
                # 只保留数值类型的列
                feature_columns = [col for col in feature_columns if df[col].dtype in ['int64', 'float64']]
            
            X = df[feature_columns].values  # 获取特征数据
            
            # 处理标签编码（如果标签是字符串）
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                print(f"标签编码映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
            
            print(f"特征数量: {len(feature_columns)}")
            print(f"样本数量: {len(X)}")
            print(f"类别数量: {len(np.unique(y))}")
            
            return X, y, feature_columns
            
        except Exception as e:
            print(f"加载CSV文件时出错: {e}")
            raise


def create_data_loaders(X: np.ndarray, 
                       y: np.ndarray,
                       test_size: float = 0.2,
                       batch_size: int = 32,
                       normalize: bool = True,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, SVMDataset, SVMDataset]:
    """
    创建训练和测试数据加载器
    
    Args:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        batch_size: 批次大小
        normalize: 是否标准化特征
        random_state: 随机种子
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        train_dataset: 训练数据集
        test_dataset: 测试数据集
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,      # 测试集比例
        random_state=random_state, # 随机种子
        stratify=y                # 分层采样，保持类别比例
    )
    
    print(f"数据集划分完成：")
    print(f"  - 训练集: {X_train.shape[0]} 样本")
    print(f"  - 测试集: {X_test.shape[0]} 样本")
    
    # 创建训练数据集（包含标准化）
    train_dataset = SVMDataset(X_train, y_train, normalize=normalize)
    
    # 创建测试数据集（使用训练集的标准化参数）
    if normalize and train_dataset.scaler is not None:
        # 对测试集应用训练集的标准化参数
        X_test_normalized = train_dataset.scaler.transform(X_test)
        test_dataset = SVMDataset(X_test_normalized, y_test, normalize=False)
        test_dataset.scaler = train_dataset.scaler  # 共享标准化器
    else:
        test_dataset = SVMDataset(X_test, y_test, normalize=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # 批次大小
        shuffle=True,          # 训练时打乱数据
        num_workers=0          # 数据加载进程数（设为0避免多进程问题）
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # 测试时不打乱数据
        num_workers=0
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def visualize_2d_data(X: np.ndarray, 
                     y: np.ndarray, 
                     title: str = "2D Data Visualization",
                     save_path: Optional[str] = None):
    """
    可视化二维数据分布
    
    Args:
        X: 特征数据 [n_samples, 2]
        y: 标签数据 [n_samples]
        title: 图表标题
        save_path: 保存路径（可选）
    """
    if X.shape[1] != 2:
        print(f"警告：数据维度为{X.shape[1]}，无法进行2D可视化")
        return
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的类别
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))  # 生成不同颜色
    
    # 为每个类别绘制散点图
    for i, cls in enumerate(unique_classes):
        mask = y == cls  # 选择属于当前类别的样本
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=[colors[i]], 
                   label=f'Class {cls}', 
                   alpha=0.7,      # 透明度
                   s=50)           # 点的大小
    
    # 设置图表属性
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)  # 添加网格
    plt.tight_layout()         # 自动调整布局
    
    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()


def analyze_dataset(X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
    """
    分析数据集的基本统计信息
    
    Args:
        X: 特征数据
        y: 标签数据
        feature_names: 特征名称列表
    """
    print("\n" + "="*50)
    print("数据集分析报告")
    print("="*50)
    
    # 基本信息
    print(f"\n📊 基本信息:")
    print(f"  - 样本数量: {X.shape[0]:,}")
    print(f"  - 特征数量: {X.shape[1]}")
    print(f"  - 数据类型: {X.dtype}")
    
    # 类别分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n🏷️  类别分布:")
    for cls, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"  - 类别 {cls}: {count:,} 样本 ({percentage:.1f}%)")
    
    # 特征统计
    print(f"\n📈 特征统计:")
    print(f"  - 最小值: {X.min():.4f}")
    print(f"  - 最大值: {X.max():.4f}")
    print(f"  - 均值: {X.mean():.4f}")
    print(f"  - 标准差: {X.std():.4f}")
    
    # 缺失值检查
    missing_count = np.isnan(X).sum()
    print(f"\n❓ 数据质量:")
    print(f"  - 缺失值数量: {missing_count}")
    print(f"  - 数据完整性: {(1 - missing_count/X.size)*100:.2f}%")
    
    # 特征名称（如果提供）
    if feature_names:
        print(f"\n🔤 特征名称:")
        for i, name in enumerate(feature_names[:10]):  # 只显示前10个
            print(f"  - Feature {i}: {name}")
        if len(feature_names) > 10:
            print(f"  - ... 还有 {len(feature_names)-10} 个特征")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    # 演示数据集功能
    print("SVM数据集模块演示")
    print("="*40)
    
    # 1. 生成合成数据
    print("\n1. 生成线性可分数据:")
    X_linear, y_linear = SyntheticDataGenerator.generate_linear_separable(n_samples=200)
    analyze_dataset(X_linear, y_linear)
    
    print("\n2. 生成同心圆数据:")
    X_circles, y_circles = SyntheticDataGenerator.generate_circles(n_samples=200)
    analyze_dataset(X_circles, y_circles)
    
    print("\n3. 生成月牙形数据:")
    X_moons, y_moons = SyntheticDataGenerator.generate_moons(n_samples=200)
    analyze_dataset(X_moons, y_moons)
    
    # 2. 加载真实数据
    print("\n4. 加载乳腺癌数据集:")
    X_cancer, y_cancer, feature_names = RealWorldDataLoader.load_breast_cancer()
    analyze_dataset(X_cancer, y_cancer, feature_names)
    
    # 3. 创建数据加载器
    print("\n5. 创建数据加载器:")
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
        X_cancer, y_cancer, test_size=0.2, batch_size=32
    )
    
    print(f"训练数据加载器: {len(train_loader)} 批次")
    print(f"测试数据加载器: {len(test_loader)} 批次")
    
    # 4. 演示数据加载
    print("\n6. 数据加载演示:")
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}: 特征形状={features.shape}, 标签形状={labels.shape}")
        if batch_idx >= 2:  # 只显示前3个批次
            break
    
    print("\n数据集模块演示完成！")