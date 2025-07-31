#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM模型使用示例

这个文件展示了如何使用我们实现的SVM模型进行各种机器学习任务，
包括二分类、多分类、参数调优等。

作者: DL-Odyssey
日期: 2024
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 导入我们自定义的模块
from model import SVMModel, MultiClassSVM, create_svm_model
from dataset import (
    SyntheticDataGenerator, 
    RealWorldDataLoader, 
    create_data_loaders,
    visualize_2d_data,
    analyze_dataset
)
from train import SVMTrainer, run_svm_experiment

def example_1_basic_binary_classification():
    """
    示例1: 基础二分类任务
    
    演示如何使用SVM进行简单的二分类任务
    """
    print("\n" + "="*60)
    print("示例1: 基础二分类任务")
    print("="*60)
    
    # 1. 生成线性可分的数据
    print("\n📊 生成线性可分数据...")
    X, y = SyntheticDataGenerator.generate_linear_separable(
        n_samples=200,    # 200个样本
        n_features=2,     # 2个特征（便于可视化）
        random_state=42   # 固定随机种子
    )
    
    # 分析数据集
    analyze_dataset(X, y)
    
    # 可视化原始数据
    visualize_2d_data(X, y, title="原始数据分布")
    
    # 2. 创建数据加载器
    print("\n🔄 创建数据加载器...")
    train_loader, val_loader, train_dataset, test_dataset = create_data_loaders(
        X, y, 
        test_size=0.3,     # 30%作为测试集
        batch_size=32,     # 批次大小
        normalize=True     # 标准化特征
    )
    
    # 3. 创建并训练SVM模型
    print("\n🤖 创建SVM模型...")
    svm_model = create_svm_model(
        kernel='linear',   # 使用线性核（适合线性可分数据）
        C=1.0             # 正则化参数
    )
    
    # 获取训练数据
    X_train = train_dataset.X
    y_train = train_dataset.y
    
    print("\n🚀 开始训练...")
    svm_model.fit(X_train, y_train)
    
    # 4. 进行预测
    print("\n🔮 进行预测...")
    X_test = test_dataset.X
    y_test = test_dataset.y
    
    predictions = svm_model.predict(X_test)
    decision_values = svm_model.decision_function(X_test)
    
    # 5. 评估性能
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    print(f"\n📈 模型性能:")
    print(f"  - 测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 获取支持向量信息
    support_vectors, support_labels, alphas = svm_model.get_support_vectors()
    print(f"  - 支持向量数量: {len(support_vectors)}")
    print(f"  - 支持向量比例: {len(support_vectors)/len(X_train)*100:.2f}%")
    
    # 6. 可视化结果
    print("\n🎨 可视化决策边界...")
    visualize_decision_boundary_simple(svm_model, X, y, "线性SVM决策边界")
    
    return svm_model, accuracy

def example_2_nonlinear_classification():
    """
    示例2: 非线性分类任务
    
    演示如何使用RBF核处理非线性数据
    """
    print("\n" + "="*60)
    print("示例2: 非线性分类任务")
    print("="*60)
    
    # 1. 生成非线性数据（同心圆）
    print("\n📊 生成同心圆数据...")
    X, y = SyntheticDataGenerator.generate_circles(
        n_samples=300,
        noise=0.1,
        factor=0.5,
        random_state=42
    )
    
    # 可视化原始数据
    visualize_2d_data(X, y, title="同心圆数据分布")
    
    # 2. 比较不同核函数的效果
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    
    for kernel in kernels:
        print(f"\n🔧 测试 {kernel} 核函数...")
        
        # 创建数据加载器
        train_loader, val_loader, train_dataset, test_dataset = create_data_loaders(
            X, y, test_size=0.3, batch_size=32, normalize=True
        )
        
        # 创建SVM模型
        if kernel == 'rbf':
            svm_model = create_svm_model(kernel=kernel, C=1.0, gamma='scale')
        elif kernel == 'poly':
            svm_model = create_svm_model(kernel=kernel, C=1.0, degree=3)
        else:
            svm_model = create_svm_model(kernel=kernel, C=1.0)
        
        # 训练模型
        X_train = train_dataset.X
        y_train = train_dataset.y
        svm_model.fit(X_train, y_train)
        
        # 测试模型
        X_test = test_dataset.X
        y_test = test_dataset.y
        predictions = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        
        results[kernel] = {
            'model': svm_model,
            'accuracy': accuracy
        }
        
        print(f"  - {kernel} 核准确率: {accuracy:.4f}")
    
    # 3. 可视化最佳模型的决策边界
    best_kernel = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_kernel]['model']
    
    print(f"\n🏆 最佳核函数: {best_kernel} (准确率: {results[best_kernel]['accuracy']:.4f})")
    
    visualize_decision_boundary_simple(
        best_model, X, y, 
        f"最佳SVM决策边界 ({best_kernel} 核)"
    )
    
    return results

def example_3_multiclass_classification():
    """
    示例3: 多分类任务
    
    演示如何使用SVM进行多分类
    """
    print("\n" + "="*60)
    print("示例3: 多分类任务")
    print("="*60)
    
    # 1. 加载红酒数据集（3分类）
    print("\n🍷 加载红酒数据集...")
    X, y, feature_names = RealWorldDataLoader.load_wine()
    
    # 分析数据集
    analyze_dataset(X, y, feature_names)
    
    # 2. 创建数据加载器
    train_loader, val_loader, train_dataset, test_dataset = create_data_loaders(
        X, y, test_size=0.3, batch_size=32, normalize=True
    )
    
    # 3. 创建多分类SVM
    print("\n🤖 创建多分类SVM...")
    multi_svm = MultiClassSVM(
        kernel='rbf',
        C=1.0,
        gamma='scale'
    )
    
    # 训练模型
    X_train = train_dataset.X
    y_train = train_dataset.y
    
    print("\n🚀 开始训练多分类SVM...")
    multi_svm.fit(X_train, y_train)
    
    # 4. 进行预测
    X_test = test_dataset.X
    y_test = test_dataset.y
    
    predictions = multi_svm.predict(X_test)
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
    
    print(f"\n📈 多分类SVM性能:")
    print(f"  - 测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 生成详细的分类报告
    class_names = ['Class 0', 'Class 1', 'Class 2']
    report = classification_report(
        y_test.numpy(), 
        predictions.numpy(), 
        target_names=class_names
    )
    print(f"\n📋 详细分类报告:")
    print(report)
    
    return multi_svm, accuracy

def example_4_parameter_tuning():
    """
    示例4: 参数调优
    
    演示如何进行SVM参数调优
    """
    print("\n" + "="*60)
    print("示例4: SVM参数调优")
    print("="*60)
    
    # 1. 生成测试数据
    print("\n📊 生成月牙形数据...")
    X, y = SyntheticDataGenerator.generate_moons(
        n_samples=400,
        noise=0.15,
        random_state=42
    )
    
    # 可视化数据
    visualize_2d_data(X, y, title="月牙形数据分布")
    
    # 2. 创建数据加载器
    train_loader, val_loader, train_dataset, test_dataset = create_data_loaders(
        X, y, test_size=0.3, batch_size=32, normalize=True
    )
    
    X_train = train_dataset.X
    y_train = train_dataset.y
    X_test = test_dataset.X
    y_test = test_dataset.y
    
    # 3. 定义参数网格
    print("\n🔧 定义参数搜索空间...")
    param_grid = {
        'C': [0.1, 1, 10, 100],           # 正则化参数
        'gamma': [0.001, 0.01, 0.1, 1],   # RBF核参数
    }
    
    best_accuracy = 0
    best_params = {}
    best_model = None
    
    print("\n🔍 开始网格搜索...")
    
    # 手动网格搜索（因为我们的SVM实现不直接兼容sklearn的GridSearchCV）
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            print(f"  测试参数: C={C}, gamma={gamma}")
            
            # 创建并训练模型
            svm_model = create_svm_model(
                kernel='rbf',
                C=C,
                gamma=gamma
            )
            
            svm_model.fit(X_train, y_train)
            
            # 评估模型
            predictions = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
            
            print(f"    准确率: {accuracy:.4f}")
            
            # 更新最佳参数
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'C': C, 'gamma': gamma}
                best_model = svm_model
    
    # 4. 报告最佳结果
    print(f"\n🏆 最佳参数:")
    print(f"  - C: {best_params['C']}")
    print(f"  - gamma: {best_params['gamma']}")
    print(f"  - 最佳准确率: {best_accuracy:.4f}")
    
    # 5. 可视化最佳模型
    visualize_decision_boundary_simple(
        best_model, X, y,
        f"调优后的SVM (C={best_params['C']}, γ={best_params['gamma']})"
    )
    
    return best_model, best_params, best_accuracy

def example_5_real_world_application():
    """
    示例5: 真实世界应用
    
    使用乳腺癌数据集演示SVM在医疗诊断中的应用
    """
    print("\n" + "="*60)
    print("示例5: 真实世界应用 - 乳腺癌诊断")
    print("="*60)
    
    # 1. 加载乳腺癌数据集
    print("\n🏥 加载乳腺癌数据集...")
    X, y, feature_names = RealWorldDataLoader.load_breast_cancer()
    
    # 分析数据集
    analyze_dataset(X, y, feature_names)
    
    # 2. 使用完整的训练流程
    print("\n🚀 使用完整训练流程...")
    
    # 运行完整实验
    model, results = run_svm_experiment(
        dataset_name='cancer',
        kernel='rbf',
        C=1.0,
        gamma='scale'
    )
    
    # 3. 分析结果
    print(f"\n🎯 医疗诊断结果分析:")
    print(f"  - 模型准确率: {results['accuracy']:.4f}")
    print(f"  - 这意味着模型能正确诊断 {results['accuracy']*100:.1f}% 的病例")
    
    # 获取支持向量信息
    sv_info = results['support_vectors_info']
    print(f"  - 关键病例数（支持向量）: {sv_info['n_support_vectors']}")
    print(f"  - 关键病例比例: {sv_info['support_ratio']:.2f}%")
    
    print("\n💡 实际应用建议:")
    print("  - 在实际医疗应用中，建议结合多种诊断方法")
    print("  - 模型预测应作为医生诊断的辅助工具")
    print("  - 需要在更大规模的数据集上进一步验证")
    
    return model, results

def visualize_decision_boundary_simple(model, X, y, title):
    """
    简化的决策边界可视化函数
    
    Args:
        model: 训练好的SVM模型
        X: 特征数据
        y: 标签数据
        title: 图表标题
    """
    if X.shape[1] != 2:
        print(f"警告：数据维度为{X.shape[1]}，无法可视化决策边界")
        return
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    Z = model.decision_function(grid_points)
    Z = Z.detach().numpy().reshape(xx.shape)
    
    # 绘制图形
    plt.figure(figsize=(10, 8))
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='--')
    
    # 绘制数据点
    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color, 
                   label=f'Class {i}', s=50, alpha=0.8, edgecolors='black')
    
    # 绘制支持向量
    try:
        support_vectors, _, _ = model.get_support_vectors()
        support_vectors = support_vectors.numpy()
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1],
                   s=200, facecolors='none', edgecolors='black', 
                   linewidths=2, label='Support Vectors')
    except:
        pass
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：运行所有示例
    """
    print("🎉 SVM模型使用示例集合")
    print("="*80)
    
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 示例1：基础二分类
        print("\n🚀 运行示例1...")
        model1, acc1 = example_1_basic_binary_classification()
        print(f"✅ 示例1完成，准确率: {acc1:.4f}")
        
        # 示例2：非线性分类
        print("\n🚀 运行示例2...")
        results2 = example_2_nonlinear_classification()
        best_acc2 = max(r['accuracy'] for r in results2.values())
        print(f"✅ 示例2完成，最佳准确率: {best_acc2:.4f}")
        
        # 示例3：多分类
        print("\n🚀 运行示例3...")
        model3, acc3 = example_3_multiclass_classification()
        print(f"✅ 示例3完成，准确率: {acc3:.4f}")
        
        # 示例4：参数调优
        print("\n🚀 运行示例4...")
        model4, params4, acc4 = example_4_parameter_tuning()
        print(f"✅ 示例4完成，最佳准确率: {acc4:.4f}")
        
        # 示例5：真实世界应用
        print("\n🚀 运行示例5...")
        model5, results5 = example_5_real_world_application()
        print(f"✅ 示例5完成，准确率: {results5['accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("🎊 所有示例运行完成！")
    print("="*80)
    
    print("\n📚 学习要点总结:")
    print("1. 线性核适用于线性可分数据")
    print("2. RBF核能处理复杂的非线性数据")
    print("3. 参数C控制对误分类的容忍度")
    print("4. 参数gamma控制RBF核的影响范围")
    print("5. 支持向量是决定决策边界的关键样本")
    print("6. 数据标准化对SVM性能很重要")
    print("7. 参数调优能显著提升模型性能")

if __name__ == "__main__":
    main()