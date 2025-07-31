import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

class SVMModel(nn.Module):
    """
    支持向量机(SVM)的PyTorch实现
    
    这个实现包含了线性SVM和非线性SVM（通过核函数），
    使用SMO算法进行优化求解
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',  # 核函数类型：'linear', 'rbf', 'poly', 'sigmoid'
                 C: float = 1.0,       # 正则化参数，控制对误分类的惩罚程度
                 gamma: float = 'scale', # RBF核的参数
                 degree: int = 3,      # 多项式核的度数
                 coef0: float = 0.0,   # 多项式核和sigmoid核的独立项
                 tol: float = 1e-3,    # 停止准则的容忍度
                 max_iter: int = 1000): # 最大迭代次数
        """
        初始化SVM模型参数
        
        Args:
            kernel: 核函数类型，决定如何处理非线性数据
            C: 软间隔参数，C越大对误分类惩罚越重
            gamma: RBF核函数的参数，控制单个训练样本的影响范围
            degree: 多项式核函数的度数
            coef0: 核函数中的独立项
            tol: 算法收敛的容忍度
            max_iter: SMO算法的最大迭代次数
        """
        super(SVMModel, self).__init__()
        
        # 存储模型超参数
        self.kernel = kernel  # 核函数类型决定了如何处理特征空间的映射
        self.C = C           # 正则化参数，平衡间隔最大化和分类错误最小化
        self.gamma = gamma   # RBF核参数，影响决策边界的复杂度
        self.degree = degree # 多项式核的度数
        self.coef0 = coef0  # 核函数的偏置项
        self.tol = tol      # 收敛容忍度，控制算法停止条件
        self.max_iter = max_iter  # 防止算法无限循环的最大迭代次数
        
        # 模型训练后的参数（这些参数在训练过程中学习得到）
        self.support_vectors = None      # 支持向量，决定决策边界的关键样本点
        self.support_labels = None       # 支持向量对应的标签
        self.alphas = None              # 拉格朗日乘子，表示每个样本的重要性
        self.b = 0.0                    # 偏置项，决策超平面的截距
        self.n_support = None           # 每个类别的支持向量数量
        
        # 训练数据存储（用于核函数计算）
        self.X_train = None  # 训练数据，用于计算核函数值
        self.y_train = None  # 训练标签，用于SMO算法
        
    def _compute_kernel_matrix(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        计算核函数矩阵K(X1, X2)
        
        核函数的作用是将原始特征空间映射到高维空间，
        使得在原空间线性不可分的数据在高维空间变得线性可分
        
        Args:
            X1: 第一组数据点 [n1, features]
            X2: 第二组数据点 [n2, features]
            
        Returns:
            kernel_matrix: 核函数矩阵 [n1, n2]
        """
        # 将PyTorch张量转换为numpy数组进行核函数计算
        X1_np = X1.detach().cpu().numpy()  # 分离计算图并转移到CPU
        X2_np = X2.detach().cpu().numpy()  # 避免梯度计算的干扰
        
        if self.kernel == 'linear':
            # 线性核：K(xi, xj) = xi^T * xj
            # 这是最简单的核函数，相当于在原始空间进行线性分类
            kernel_matrix = np.dot(X1_np, X2_np.T)
            
        elif self.kernel == 'rbf':
            # 径向基函数核（高斯核）：K(xi, xj) = exp(-gamma * ||xi - xj||^2)
            # gamma参数控制高斯函数的宽度，影响决策边界的复杂度
            if self.gamma == 'scale':
                # 'scale'模式：gamma = 1 / (n_features * X.var())
                gamma_val = 1.0 / (X1.shape[1] * X1_np.var())  # 自适应gamma值
            else:
                gamma_val = self.gamma  # 使用用户指定的gamma值
            
            # 计算所有样本对之间的欧氏距离的平方
            sq_dists = np.sum(X1_np**2, axis=1).reshape(-1, 1) + \
                      np.sum(X2_np**2, axis=1) - 2 * np.dot(X1_np, X2_np.T)
            # 应用高斯核函数
            kernel_matrix = np.exp(-gamma_val * sq_dists)
            
        elif self.kernel == 'poly':
            # 多项式核：K(xi, xj) = (gamma * xi^T * xj + coef0)^degree
            # degree控制多项式的度数，coef0是独立项
            linear_kernel = np.dot(X1_np, X2_np.T)  # 先计算线性核
            kernel_matrix = (self.gamma * linear_kernel + self.coef0) ** self.degree
            
        elif self.kernel == 'sigmoid':
            # Sigmoid核：K(xi, xj) = tanh(gamma * xi^T * xj + coef0)
            # 类似于神经网络中的sigmoid激活函数
            linear_kernel = np.dot(X1_np, X2_np.T)
            kernel_matrix = np.tanh(self.gamma * linear_kernel + self.coef0)
            
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")
        
        # 将numpy数组转换回PyTorch张量
        return torch.tensor(kernel_matrix, dtype=torch.float32, device=X1.device)
    
    def _smo_algorithm(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        序列最小优化(SMO)算法实现
        
        SMO算法是求解SVM对偶问题的高效方法，每次选择两个拉格朗日乘子进行优化，
        其他乘子保持固定，这样可以将复杂的二次规划问题分解为一系列简单的子问题
        
        Args:
            X: 训练数据 [n_samples, n_features]
            y: 训练标签 [n_samples] (必须是+1或-1)
            
        Returns:
            alphas: 优化后的拉格朗日乘子 [n_samples]
            b: 优化后的偏置项
        """
        n_samples = X.shape[0]  # 训练样本数量
        
        # 初始化拉格朗日乘子为零向量
        alphas = torch.zeros(n_samples, dtype=torch.float32, device=X.device)
        b = 0.0  # 初始化偏置项
        
        # 计算核函数矩阵，这是SMO算法的核心数据结构
        K = self._compute_kernel_matrix(X, X)  # [n_samples, n_samples]
        
        # SMO主循环
        for iteration in range(self.max_iter):
            alpha_pairs_changed = 0  # 记录本轮迭代中改变的alpha对数量
            
            # 遍历所有样本，寻找需要优化的alpha对
            for i in range(n_samples):
                # 计算第i个样本的预测误差
                # E_i = f(xi) - yi，其中f(xi)是当前模型对xi的预测值
                Ei = torch.sum(alphas * y * K[i, :]) + b - y[i]
                
                # 检查KKT条件是否被违反
                # KKT条件是SVM优化问题的必要条件
                if ((y[i] * Ei < -self.tol) and (alphas[i] < self.C)) or \
                   ((y[i] * Ei > self.tol) and (alphas[i] > 0)):
                    
                    # 随机选择第二个alpha进行优化
                    j = self._select_j(i, n_samples, Ei, alphas, y, K, b)
                    
                    if j == i:  # 如果没有找到合适的j，跳过
                        continue
                    
                    # 计算第j个样本的预测误差
                    Ej = torch.sum(alphas * y * K[j, :]) + b - y[j]
                    
                    # 保存旧的alpha值，用于后续计算
                    alpha_i_old = alphas[i].clone()
                    alpha_j_old = alphas[j].clone()
                    
                    # 计算alpha_j的取值范围[L, H]
                    # 这个范围由约束条件决定
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:  # 如果范围为空，跳过
                        continue
                    
                    # 计算eta = K(xi,xi) + K(xj,xj) - 2*K(xi,xj)
                    # eta是二次项的系数，用于计算最优步长
                    eta = K[i, i] + K[j, j] - 2 * K[i, j]
                    
                    if eta <= 0:  # 如果eta非正，跳过（理论上不应该发生）
                        continue
                    
                    # 计算新的alpha_j值
                    # 这是SMO算法的核心更新公式
                    alphas[j] = alphas[j] + y[j] * (Ei - Ej) / eta
                    
                    # 将alpha_j限制在[L, H]范围内
                    alphas[j] = torch.clamp(alphas[j], L, H)
                    
                    # 如果alpha_j的变化很小，跳过
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 根据约束条件更新alpha_i
                    # 约束：sum(alpha_i * y_i) = 0
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # 更新偏置项b
                    # b的更新确保支持向量上的函数值满足约束条件
                    b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    
                    # 根据alpha值选择合适的b值
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    alpha_pairs_changed += 1  # 记录成功优化的alpha对
            
            # 如果没有alpha对发生变化，说明算法收敛
            if alpha_pairs_changed == 0:
                break
        
        return alphas, b
    
    def _select_j(self, i: int, n_samples: int, Ei: float, 
                  alphas: torch.Tensor, y: torch.Tensor, 
                  K: torch.Tensor, b: float) -> int:
        """
        选择第二个优化变量j的启发式方法
        
        好的j选择策略可以加速SMO算法的收敛，
        这里使用最大化|Ei - Ej|的启发式方法
        
        Args:
            i: 第一个alpha的索引
            n_samples: 样本总数
            Ei: 第i个样本的预测误差
            alphas: 当前的拉格朗日乘子
            y: 训练标签
            K: 核函数矩阵
            b: 当前偏置项
            
        Returns:
            j: 选择的第二个alpha的索引
        """
        max_delta_E = 0  # 最大误差差值
        max_j = i        # 最优的j索引
        
        # 遍历所有可能的j值
        for j in range(n_samples):
            if j == i:  # 跳过自己
                continue
            
            # 计算第j个样本的预测误差
            Ej = torch.sum(alphas * y * K[j, :]) + b - y[j]
            
            # 计算误差差值的绝对值
            delta_E = abs(Ei - Ej)
            
            # 选择使误差差值最大的j
            # 这样可以使alpha的更新步长最大，加速收敛
            if delta_E > max_delta_E:
                max_delta_E = delta_E
                max_j = j
        
        return max_j
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'SVMModel':
        """
        训练SVM模型
        
        Args:
            X: 训练数据 [n_samples, n_features]
            y: 训练标签 [n_samples]，必须是+1或-1
            
        Returns:
            self: 训练后的模型实例
        """
        # 确保输入数据在正确的设备上
        X = X.to(next(self.parameters()).device) if list(self.parameters()) else X
        
        # 将标签转换为+1/-1格式（SVM的标准格式）
        unique_labels = torch.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM只支持二分类问题")
        
        # 标签映射：将原始标签映射为+1和-1
        y_binary = torch.where(y == unique_labels[0], -1, 1)
        
        # 存储训练数据和标签（用于预测时的核函数计算）
        self.X_train = X.clone()
        self.y_train = y_binary.clone()
        
        # 使用SMO算法求解对偶问题
        print("开始SMO算法优化...")
        alphas, b = self._smo_algorithm(X, y_binary)
        
        # 找出支持向量（alpha > 0的样本点）
        # 支持向量是决定决策边界的关键样本点
        support_mask = alphas > 1e-5  # 使用小的阈值避免数值误差
        
        # 存储支持向量相关信息
        self.support_vectors = X[support_mask].clone()  # 支持向量的特征
        self.support_labels = y_binary[support_mask].clone()  # 支持向量的标签
        self.alphas = alphas[support_mask].clone()  # 支持向量对应的拉格朗日乘子
        self.b = b  # 偏置项
        
        # 统计支持向量数量
        self.n_support = torch.sum(support_mask).item()
        
        print(f"训练完成！找到 {self.n_support} 个支持向量")
        print(f"支持向量占总样本的比例: {self.n_support/len(X)*100:.2f}%")
        
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        使用训练好的SVM模型进行预测
        
        Args:
            X: 待预测数据 [n_samples, n_features]
            
        Returns:
            predictions: 预测结果 [n_samples]，值为+1或-1
        """
        if self.support_vectors is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 确保输入数据在正确的设备上
        X = X.to(self.support_vectors.device)
        
        # 计算待预测样本与支持向量之间的核函数值
        K = self._compute_kernel_matrix(X, self.support_vectors)  # [n_samples, n_support]
        
        # 计算决策函数值：f(x) = sum(alpha_i * y_i * K(x, xi)) + b
        # 这是SVM的核心预测公式
        decision_values = torch.sum(self.alphas * self.support_labels * K, dim=1) + self.b
        
        # 根据决策函数的符号进行分类
        # f(x) > 0 预测为+1类，f(x) < 0 预测为-1类
        predictions = torch.sign(decision_values)
        
        return predictions
    
    def decision_function(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算决策函数值（到决策边界的距离）
        
        决策函数值的绝对值表示样本到决策边界的距离，
        可以用来衡量分类的置信度
        
        Args:
            X: 输入数据 [n_samples, n_features]
            
        Returns:
            decision_values: 决策函数值 [n_samples]
        """
        if self.support_vectors is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        X = X.to(self.support_vectors.device)
        
        # 计算核函数矩阵
        K = self._compute_kernel_matrix(X, self.support_vectors)
        
        # 计算决策函数值
        decision_values = torch.sum(self.alphas * self.support_labels * K, dim=1) + self.b
        
        return decision_values
    
    def get_support_vectors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取支持向量信息
        
        Returns:
            support_vectors: 支持向量 [n_support, n_features]
            support_labels: 支持向量标签 [n_support]
            alphas: 拉格朗日乘子 [n_support]
        """
        if self.support_vectors is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.support_vectors, self.support_labels, self.alphas
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch模型的前向传播方法
        
        Args:
            x: 输入数据 [batch_size, n_features]
            
        Returns:
            output: 决策函数值 [batch_size]
        """
        return self.decision_function(x)


class MultiClassSVM(nn.Module):
    """
    多分类SVM实现
    
    使用一对一(One-vs-One)策略将多分类问题转化为多个二分类问题
    """
    
    def __init__(self, **svm_params):
        """
        初始化多分类SVM
        
        Args:
            **svm_params: SVM模型的参数
        """
        super(MultiClassSVM, self).__init__()
        self.svm_params = svm_params  # 存储SVM参数
        self.classifiers = {}  # 存储二分类器字典
        self.classes = None   # 存储类别标签
        
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'MultiClassSVM':
        """
        训练多分类SVM模型
        
        Args:
            X: 训练数据 [n_samples, n_features]
            y: 训练标签 [n_samples]
            
        Returns:
            self: 训练后的模型实例
        """
        # 获取所有唯一的类别
        self.classes = torch.unique(y)
        n_classes = len(self.classes)
        
        if n_classes <= 2:
            raise ValueError("多分类SVM需要至少3个类别")
        
        print(f"开始训练多分类SVM，共{n_classes}个类别")
        print(f"将训练{n_classes * (n_classes - 1) // 2}个二分类器")
        
        # 为每对类别训练一个二分类器
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i, class_j = self.classes[i], self.classes[j]
                
                # 选择属于这两个类别的样本
                mask = (y == class_i) | (y == class_j)
                X_binary = X[mask]
                y_binary = y[mask]
                
                # 将标签转换为+1/-1格式
                y_binary = torch.where(y_binary == class_i, -1, 1)
                
                # 创建并训练二分类器
                classifier = SVMModel(**self.svm_params)
                classifier.fit(X_binary, y_binary)
                
                # 存储分类器
                self.classifiers[(class_i.item(), class_j.item())] = classifier
                
                print(f"完成类别 {class_i.item()} vs {class_j.item()} 的训练")
        
        print("多分类SVM训练完成！")
        return self
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        使用训练好的多分类SVM进行预测
        
        Args:
            X: 待预测数据 [n_samples, n_features]
            
        Returns:
            predictions: 预测结果 [n_samples]
        """
        if not self.classifiers:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # 投票矩阵：记录每个样本对每个类别的投票数
        votes = torch.zeros(n_samples, n_classes, device=X.device)
        
        # 对每个二分类器进行预测并投票
        for (class_i, class_j), classifier in self.classifiers.items():
            # 获取类别在classes中的索引
            i_idx = (self.classes == class_i).nonzero(as_tuple=True)[0].item()
            j_idx = (self.classes == class_j).nonzero(as_tuple=True)[0].item()
            
            # 进行二分类预测
            binary_pred = classifier.predict(X)
            
            # 根据预测结果进行投票
            # -1表示class_i，+1表示class_j
            votes[binary_pred == -1, i_idx] += 1
            votes[binary_pred == 1, j_idx] += 1
        
        # 选择得票最多的类别作为最终预测
        _, predicted_indices = torch.max(votes, dim=1)
        predictions = self.classes[predicted_indices]
        
        return predictions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch模型的前向传播方法
        
        Args:
            x: 输入数据 [batch_size, n_features]
            
        Returns:
            output: 预测结果 [batch_size]
        """
        return self.predict(x)


def create_svm_model(kernel: str = 'rbf', 
                     C: float = 1.0, 
                     gamma: Union[str, float] = 'scale',
                     **kwargs) -> SVMModel:
    """
    创建SVM模型的工厂函数
    
    Args:
        kernel: 核函数类型
        C: 正则化参数
        gamma: RBF核参数
        **kwargs: 其他SVM参数
        
    Returns:
        svm_model: 创建的SVM模型实例
    """
    return SVMModel(kernel=kernel, C=C, gamma=gamma, **kwargs)


if __name__ == "__main__":
    # 简单的测试代码
    print("SVM模型实现完成！")
    print("支持的核函数：linear, rbf, poly, sigmoid")
    print("支持二分类和多分类任务")
    
    # 创建一个简单的测试数据
    torch.manual_seed(42)  # 设置随机种子确保结果可重现
    X_test = torch.randn(100, 2)  # 100个样本，2个特征
    y_test = torch.randint(0, 2, (100,))  # 二分类标签
    
    # 创建并测试SVM模型
    svm = create_svm_model(kernel='rbf', C=1.0)
    print(f"\n创建的SVM模型参数：")
    print(f"核函数: {svm.kernel}")
    print(f"正则化参数C: {svm.C}")
    print(f"RBF参数gamma: {svm.gamma}")
    
    print("\n模型创建成功，可以开始训练！")