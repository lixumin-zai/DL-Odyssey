import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import os
from datetime import datetime

# 导入我们自定义的模块
from model import SVMModel, MultiClassSVM
from dataset import (
    SyntheticDataGenerator, 
    RealWorldDataLoader, 
    create_data_loaders,
    visualize_2d_data,
    analyze_dataset
)

class SVMLightningModule(pl.LightningModule):
    """
    SVM的PyTorch Lightning模块
    
    这个类将SVM模型包装为Lightning模块，提供标准化的训练、验证和测试流程
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        初始化SVM Lightning模块
        
        Args:
            kernel: 核函数类型
            C: 正则化参数
            gamma: RBF核参数
            learning_rate: 学习率（虽然SVM不使用梯度下降，但保留用于兼容性）
            **kwargs: 其他SVM参数
        """
        super().__init__()
        
        # 保存超参数到Lightning的hparams
        self.save_hyperparameters()  # 自动保存所有初始化参数
        
        # 创建SVM模型实例
        self.svm_model = SVMModel(
            kernel=kernel,
            C=C, 
            gamma=gamma,
            **kwargs
        )
        
        # 存储训练和验证指标
        self.train_metrics = []  # 训练过程中的指标记录
        self.val_metrics = []    # 验证过程中的指标记录
        
        # 标志位，表示模型是否已经训练
        self.model_fitted = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, n_features]
            
        Returns:
            output: 模型输出 [batch_size]
        """
        if not self.model_fitted:
            # 如果模型还没有训练，返回零向量
            return torch.zeros(x.shape[0], device=x.device)
        
        # 使用训练好的SVM进行预测
        return self.svm_model.decision_function(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        训练步骤
        
        注意：SVM使用SMO算法训练，不是基于梯度的优化，
        所以这里主要用于记录训练过程和指标
        
        Args:
            batch: 一个批次的数据 (features, labels)
            batch_idx: 批次索引
            
        Returns:
            包含损失和指标的字典
        """
        features, labels = batch  # 解包批次数据
        
        # 如果是第一个epoch的第一个batch，进行SVM训练
        if not self.model_fitted and batch_idx == 0 and self.current_epoch == 0:
            print("\n开始SVM模型训练...")
            
            # 收集所有训练数据进行SVM训练
            # 注意：这里我们需要访问完整的训练数据集
            train_dataloader = self.trainer.train_dataloader
            
            # 收集所有训练数据
            all_features = []
            all_labels = []
            
            for batch_data in train_dataloader:
                batch_features, batch_labels = batch_data
                all_features.append(batch_features)
                all_labels.append(batch_labels)
            
            # 合并所有批次的数据
            X_train = torch.cat(all_features, dim=0)  # 拼接所有特征
            y_train = torch.cat(all_labels, dim=0)    # 拼接所有标签
            
            print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
            
            # 训练SVM模型
            self.svm_model.fit(X_train, y_train)
            self.model_fitted = True  # 标记模型已训练
            
            print("SVM模型训练完成！")
        
        # 如果模型已训练，计算预测和指标
        if self.model_fitted:
            # 进行预测
            predictions = self.svm_model.predict(features)
            
            # 计算准确率
            accuracy = (predictions == labels).float().mean()
            
            # 计算一个虚拟的损失（SVM不使用损失函数训练）
            # 这里使用1-accuracy作为损失，仅用于监控
            loss = 1.0 - accuracy
            
            # 记录指标
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
            
            return {
                'loss': loss,
                'accuracy': accuracy,
                'predictions': predictions,
                'targets': labels
            }
        else:
            # 模型未训练时返回虚拟损失
            dummy_loss = torch.tensor(1.0, requires_grad=True)
            return {'loss': dummy_loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        验证步骤
        
        Args:
            batch: 一个批次的验证数据
            batch_idx: 批次索引
            
        Returns:
            包含验证指标的字典
        """
        features, labels = batch
        
        if not self.model_fitted:
            # 如果模型还没训练，返回虚拟指标
            return {
                'val_loss': torch.tensor(1.0),
                'val_accuracy': torch.tensor(0.0)
            }
        
        # 进行预测
        predictions = self.svm_model.predict(features)
        decision_values = self.svm_model.decision_function(features)
        
        # 计算准确率
        accuracy = (predictions == labels).float().mean()
        
        # 计算虚拟损失
        val_loss = 1.0 - accuracy
        
        # 记录验证指标
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'predictions': predictions,
            'targets': labels,
            'decision_values': decision_values
        }
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """
        测试步骤
        
        Args:
            batch: 一个批次的测试数据
            batch_idx: 批次索引
            
        Returns:
            包含测试指标的字典
        """
        features, labels = batch
        
        if not self.model_fitted:
            return {
                'test_loss': torch.tensor(1.0),
                'test_accuracy': torch.tensor(0.0)
            }
        
        # 进行预测
        predictions = self.svm_model.predict(features)
        decision_values = self.svm_model.decision_function(features)
        
        # 计算准确率
        accuracy = (predictions == labels).float().mean()
        test_loss = 1.0 - accuracy
        
        # 记录测试指标
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'predictions': predictions,
            'targets': labels,
            'decision_values': decision_values
        }
    
    def configure_optimizers(self):
        """
        配置优化器
        
        注意：SVM不使用梯度优化，这里返回一个虚拟优化器以满足Lightning要求
        """
        # 创建一个虚拟参数用于优化器
        dummy_param = nn.Parameter(torch.tensor(0.0))
        self.register_parameter('dummy', dummy_param)
        
        # 返回一个虚拟优化器
        optimizer = torch.optim.Adam([dummy_param], lr=self.hparams.learning_rate)
        return optimizer
    
    def on_train_epoch_end(self):
        """
        训练epoch结束时的回调
        """
        if self.model_fitted:
            # 获取当前epoch的训练指标
            train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0)
            train_acc = self.trainer.callback_metrics.get('train_accuracy_epoch', 0)
            
            # 记录到训练指标列表
            self.train_metrics.append({
                'epoch': self.current_epoch,
                'loss': float(train_loss),
                'accuracy': float(train_acc)
            })
    
    def on_validation_epoch_end(self):
        """
        验证epoch结束时的回调
        """
        if self.model_fitted:
            # 获取当前epoch的验证指标
            val_loss = self.trainer.callback_metrics.get('val_loss', 0)
            val_acc = self.trainer.callback_metrics.get('val_accuracy', 0)
            
            # 记录到验证指标列表
            self.val_metrics.append({
                'epoch': self.current_epoch,
                'loss': float(val_loss),
                'accuracy': float(val_acc)
            })
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        获取支持向量信息
        
        Returns:
            包含支持向量信息的字典
        """
        if not self.model_fitted:
            return {"error": "模型尚未训练"}
        
        support_vectors, support_labels, alphas = self.svm_model.get_support_vectors()
        
        return {
            'n_support_vectors': len(support_vectors),
            'support_vectors': support_vectors,
            'support_labels': support_labels,
            'alphas': alphas,
            'support_ratio': len(support_vectors) / len(self.svm_model.X_train) * 100
        }


class SVMTrainer:
    """
    SVM训练器类
    
    提供完整的SVM训练、评估和可视化功能
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 max_epochs: int = 10,
                 accelerator: str = 'auto'):
        """
        初始化SVM训练器
        
        Args:
            kernel: 核函数类型
            C: 正则化参数
            gamma: RBF核参数
            max_epochs: 最大训练轮数
            accelerator: 加速器类型
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        
        # 创建结果保存目录
        self.results_dir = f"svm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"结果将保存到: {self.results_dir}")
    
    def train(self, 
              train_loader, 
              val_loader, 
              test_loader=None) -> SVMLightningModule:
        """
        训练SVM模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
            
        Returns:
            训练好的Lightning模块
        """
        print("\n" + "="*60)
        print("开始SVM模型训练")
        print("="*60)
        
        # 创建Lightning模块
        model = SVMLightningModule(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma
        )
        
        # 配置回调函数
        callbacks = [
            # 模型检查点回调
            ModelCheckpoint(
                dirpath=self.results_dir,
                filename='best_svm_model',
                monitor='val_accuracy',  # 监控验证准确率
                mode='max',             # 最大化准确率
                save_top_k=1,           # 只保存最好的模型
                verbose=True
            ),
            # 早停回调
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,             # 5个epoch没有改善就停止
                mode='max',
                verbose=True
            )
        ]
        
        # 配置日志记录器
        logger = TensorBoardLogger(
            save_dir=self.results_dir,
            name='svm_logs'
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=1
        )
        
        # 开始训练
        print(f"\n使用参数训练SVM:")
        print(f"  - 核函数: {self.kernel}")
        print(f"  - C参数: {self.C}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - 最大轮数: {self.max_epochs}")
        
        trainer.fit(model, train_loader, val_loader)
        
        # 如果有测试数据，进行测试
        if test_loader is not None:
            print("\n开始模型测试...")
            trainer.test(model, test_loader)
        
        print("\n训练完成！")
        return model
    
    def evaluate_model(self, 
                      model: SVMLightningModule, 
                      test_loader,
                      class_names: Optional[list] = None) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            class_names: 类别名称列表
            
        Returns:
            评估结果字典
        """
        print("\n" + "="*50)
        print("模型性能评估")
        print("="*50)
        
        if not model.model_fitted:
            print("错误：模型尚未训练")
            return {}
        
        # 收集所有预测结果
        all_predictions = []
        all_targets = []
        all_decision_values = []
        
        model.eval()  # 设置为评估模式
        
        with torch.no_grad():  # 禁用梯度计算
            for batch in test_loader:
                features, labels = batch
                
                # 进行预测
                predictions = model.svm_model.predict(features)
                decision_values = model.svm_model.decision_function(features)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_decision_values.extend(decision_values.cpu().numpy())
        
        # 转换为numpy数组
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        decision_scores = np.array(all_decision_values)
        
        # 计算各种指标
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n📊 整体性能:")
        print(f"  - 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  - 测试样本数: {len(y_true)}")
        print(f"  - 正确预测数: {np.sum(y_true == y_pred)}")
        
        # 生成分类报告
        if class_names is None:
            class_names = [f'Class {i}' for i in np.unique(y_true)]
        
        print(f"\n📋 详细分类报告:")
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n🔢 混淆矩阵:")
        print(cm)
        
        # 可视化混淆矩阵
        self._plot_confusion_matrix(cm, class_names)
        
        # 获取支持向量信息
        sv_info = model.get_support_vectors_info()
        print(f"\n🎯 支持向量信息:")
        print(f"  - 支持向量数量: {sv_info['n_support_vectors']}")
        print(f"  - 支持向量比例: {sv_info['support_ratio']:.2f}%")
        
        # 保存评估结果
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'support_vectors_info': sv_info,
            'predictions': y_pred,
            'true_labels': y_true,
            'decision_scores': decision_scores
        }
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """
        绘制混淆矩阵热力图
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
        """
        plt.figure(figsize=(8, 6))
        
        # 创建热力图
        sns.heatmap(cm, 
                   annot=True,           # 显示数值
                   fmt='d',              # 整数格式
                   cmap='Blues',         # 颜色映射
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def visualize_decision_boundary(self, 
                                  model: SVMLightningModule,
                                  X: np.ndarray, 
                                  y: np.ndarray,
                                  title: str = "SVM决策边界"):
        """
        可视化SVM的决策边界（仅适用于2D数据）
        
        Args:
            model: 训练好的模型
            X: 特征数据 [n_samples, 2]
            y: 标签数据 [n_samples]
            title: 图表标题
        """
        if X.shape[1] != 2:
            print(f"警告：数据维度为{X.shape[1]}，无法可视化决策边界")
            return
        
        if not model.model_fitted:
            print("错误：模型尚未训练")
            return
        
        print("\n绘制决策边界...")
        
        # 创建网格点
        h = 0.02  # 网格步长
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 对网格点进行预测
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        
        with torch.no_grad():
            Z = model.svm_model.decision_function(grid_points)
            Z = Z.cpu().numpy().reshape(xx.shape)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制决策边界
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='--')
        
        # 绘制数据点
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       label=f'Class {cls}',
                       s=50, alpha=0.9, edgecolors='black')
        
        # 绘制支持向量
        sv_info = model.get_support_vectors_info()
        if 'support_vectors' in sv_info:
            support_vectors = sv_info['support_vectors'].cpu().numpy()
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                       s=200, facecolors='none', edgecolors='black', 
                       linewidths=2, label='Support Vectors')
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.results_dir, 'decision_boundary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"决策边界图已保存到: {save_path}")
        
        plt.show()


def run_svm_experiment(dataset_name: str = 'circles',
                      kernel: str = 'rbf',
                      C: float = 1.0,
                      gamma: str = 'scale'):
    """
    运行完整的SVM实验
    
    Args:
        dataset_name: 数据集名称 ('linear', 'circles', 'moons', 'cancer', 'wine')
        kernel: 核函数类型
        C: 正则化参数
        gamma: RBF核参数
    """
    print("\n" + "="*80)
    print(f"SVM实验: {dataset_name.upper()} 数据集")
    print("="*80)
    
    # 1. 加载数据
    print("\n1️⃣ 加载数据集...")
    
    if dataset_name == 'linear':
        X, y = SyntheticDataGenerator.generate_linear_separable(n_samples=500)
        class_names = ['Class 0', 'Class 1']
    elif dataset_name == 'circles':
        X, y = SyntheticDataGenerator.generate_circles(n_samples=500)
        class_names = ['Inner Circle', 'Outer Circle']
    elif dataset_name == 'moons':
        X, y = SyntheticDataGenerator.generate_moons(n_samples=500)
        class_names = ['Moon 1', 'Moon 2']
    elif dataset_name == 'cancer':
        X, y, feature_names = RealWorldDataLoader.load_breast_cancer()
        class_names = ['Malignant', 'Benign']
    elif dataset_name == 'wine':
        X, y, feature_names = RealWorldDataLoader.load_wine()
        class_names = ['Class 0', 'Class 1', 'Class 2']
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 分析数据集
    analyze_dataset(X, y)
    
    # 可视化2D数据
    if X.shape[1] == 2:
        visualize_2d_data(X, y, title=f"{dataset_name.title()} Dataset")
    
    # 2. 创建数据加载器
    print("\n2️⃣ 创建数据加载器...")
    train_loader, val_loader, train_dataset, test_dataset = create_data_loaders(
        X, y, test_size=0.2, batch_size=64, normalize=True
    )
    
    # 创建测试数据加载器
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. 训练模型
    print("\n3️⃣ 训练SVM模型...")
    trainer = SVMTrainer(
        kernel=kernel,
        C=C,
        gamma=gamma,
        max_epochs=5  # SVM只需要少量epoch
    )
    
    model = trainer.train(train_loader, val_loader, test_loader)
    
    # 4. 评估模型
    print("\n4️⃣ 评估模型性能...")
    results = trainer.evaluate_model(model, test_loader, class_names)
    
    # 5. 可视化决策边界（仅适用于2D数据）
    if X.shape[1] == 2:
        print("\n5️⃣ 可视化决策边界...")
        # 使用测试数据进行可视化
        X_test = test_dataset.X.numpy()
        y_test = test_dataset.y.numpy()
        trainer.visualize_decision_boundary(
            model, X_test, y_test, 
            title=f"SVM Decision Boundary ({dataset_name.title()}, {kernel} kernel)"
        )
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    return model, results


if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("SVM训练模块演示")
    print("="*50)
    
    # 运行不同数据集的实验
    experiments = [
        {'dataset': 'circles', 'kernel': 'rbf', 'C': 1.0},
        {'dataset': 'moons', 'kernel': 'rbf', 'C': 1.0},
        {'dataset': 'linear', 'kernel': 'linear', 'C': 1.0},
    ]
    
    for exp in experiments:
        try:
            print(f"\n🚀 开始实验: {exp}")
            model, results = run_svm_experiment(**exp)
            print(f"✅ 实验完成，准确率: {results['accuracy']:.4f}")
        except Exception as e:
            print(f"❌ 实验失败: {e}")
    
    print("\n🎉 所有实验完成！")