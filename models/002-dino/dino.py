import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import math
from typing import List

# DINO 头，用于将骨干网络的输出投影到用于计算损失的特征空间
class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = False, norm_last_layer: bool = True, nlayers: int = 3, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        # 构建一个包含 nlayers 层的 MLP
        layers = []
        # 第一层：从输入维度到隐藏维度
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU()) # 使用 GELU 激活函数
        # 中间层：隐藏维度到隐藏维度
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        # 瓶颈层：隐藏维度到瓶颈维度
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        # 最后的投影层
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1) # 初始化权重范数
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False # 如果设置，则冻结范数

    def forward(self, x):
        # 通过 MLP
        x = self.mlp(x)
        # L2 归一化
        x = F.normalize(x, dim=-1, p=2)
        # 通过最后的投影层
        x = self.last_layer(x)
        return x

# DINO 损失函数
class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, ncrops: int, student_temp: float, teacher_temp: float, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        # 注册一个持久化的 buffer `center`，用于教师网络的中心化操作
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        计算学生和教师网络输出之间的交叉熵损失。
        student_output: (B * ncrops, D)
        teacher_output: (B * 2, D)  (因为只有全局视图通过教师网络)
        """
        # 将学生和教师的输出分离
        student_out = student_output / self.student_temp
        # 教师的输出需要进行中心化和锐化
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        # 分离出教师对两个全局视图的输出
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        # 遍历学生网络的所有裁剪视图输出
        for iq, q in enumerate(teacher_out):
            for v in range(self.ncrops):
                # 如果学生视图和教师视图是同一个全局视图，则跳过
                if v == iq:
                    continue
                # 计算交叉熵损失
                loss = torch.sum(-q * F.log_softmax(student_out[v::self.ncrops], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        # 更新中心
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        使用指数移动平均 (EMA) 更新中心。
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # 在多 GPU 训练时需要进行 all_reduce
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))

        # EMA 更新
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# 多裁剪视图包装器
class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        # 移除骨干网络的分类头
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        对输入的多个裁剪视图进行前向传播。
        x: 一个包含不同裁剪视图张量的列表。
        """
        # 将所有视图拼接在一起，进行一次批处理前向传播
        n_crops = len(x)
        concatenated_x = torch.cat(x, dim=0)
        # 通过骨干网络
        features = self.backbone(concatenated_x)
        # 通过 DINO 头
        logits = self.head(features)
        # 将输出分割回对应每个视图
        chunks = logits.chunk(n_crops)
        return chunks

# 示例：如何使用 DINO
if __name__ == '__main__':
    # --- 1. 定义超参数 ---
    # 通常 DINO 使用 ViT 作为骨干网络，这里为了简化和独立性，我们使用 ResNet-50
    # 在实际应用中，你应该加载一个 ViT 模型，例如: backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    backbone_arch = 'resnet50'
    out_dim = 65536  # DINO 论文中使用的输出维度
    num_classes = 1000 # 假设的下游任务类别数
    n_global_crops = 2
    n_local_crops = 6
    n_crops = n_global_crops + n_local_crops

    # --- 2. 创建学生和教师网络 ---
    # 学生网络
    student_backbone = torchvision.models.__dict__[backbone_arch]()
    student_head = DINOHead(in_dim=student_backbone.fc.in_features, out_dim=out_dim)
    student = MultiCropWrapper(student_backbone, student_head)

    # 教师网络
    teacher_backbone = torchvision.models.__dict__[backbone_arch]()
    teacher_head = DINOHead(in_dim=teacher_backbone.fc.in_features, out_dim=out_dim)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)

    # 教师网络不进行梯度更新，其权重是学生网络的 EMA
    # 初始时，将学生网络的权重复制给教师网络
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    print("学生和教师网络创建成功。")

    # --- 3. 创建损失函数 ---
    dino_loss = DINOLoss(
        out_dim=out_dim,
        ncrops=n_crops,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9
    )
    print("DINO 损失函数创建成功。")

    # --- 4. 模拟一个训练步骤 ---
    # 创建一些随机的输入数据来模拟多裁剪视图
    # 2 个全局视图 (224x224) 和 6 个局部视图 (96x96)
    global_crops = [torch.randn(4, 3, 224, 224) for _ in range(n_global_crops)]
    local_crops = [torch.randn(4, 3, 96, 96) for _ in range(n_local_crops)]
    all_crops = global_crops + local_crops

    # 前向传播
    teacher_output_chunks = teacher(global_crops) # 教师只看全局视图
    student_output_chunks = student(all_crops)    # 学生看所有视图

    # 拼接输出以计算损失
    teacher_output = torch.cat(teacher_output_chunks)
    student_output = torch.cat(student_output_chunks)

    # 计算损失
    loss = dino_loss(student_output, teacher_output)
    print(f"计算得到的 DINO 损失为: {loss.item()}")

    # --- 5. 模拟教师网络权重更新 ---
    momentum = 0.996 # 动量系数
    student_params = student.state_dict()
    teacher_params = teacher.state_dict()

    for (name, s_p) in student_params.items():
        t_p = teacher_params[name]
        # EMA 更新
        t_p.data.mul_(momentum).add_((1 - momentum) * s_p.data)

    # 验证权重是否更新
    # assert not torch.equal(teacher.backbone.conv1.weight, student.backbone.conv1.weight)
    print("教师网络权重已通过 EMA 更新。")