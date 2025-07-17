import torch
import torch.nn as nn
from typing import Type, Union, List, Optional

# 定义基础残差块 (BasicBlock)，用于 ResNet-18 和 ResNet-34
class BasicBlock(nn.Module):
    # expansion 变量用于控制输出通道数相对于输入通道数的倍增系数
    # 在 BasicBlock 中，输入和输出通道数相同，所以 expansion = 1
    expansion = 1

    def __init__(
        self,
        in_channels: int,          # 输入通道数
        out_channels: int,         # 输出通道数
        stride: int = 1,           # 卷积步长
        downsample: Optional[nn.Module] = None  # 用于处理快捷连接维度不匹配的下采样模块
    ) -> None:
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        # 3x3 卷积核，步长由 stride 决定，padding=1 保证当 stride=1 时，特征图尺寸不变
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 批量归一化层，加速收敛并提高稳定性
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        # 3x3 卷积核，步长为 1，padding=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 批量归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样模块，当输入和输出维度不匹配时使用
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始输入，用于快捷连接
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样模块（即输入输出维度不匹配），则对原始输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 快捷连接：将主路径的输出与（可能经过下采样的）原始输入相加
        out += identity
        # 应用 ReLU 激活函数
        out = self.relu(out)

        return out

# 定义瓶颈残差块 (Bottleneck)，用于 ResNet-50, 101, 152
class Bottleneck(nn.Module):
    # 在 Bottleneck 中，输出通道数是中间层通道数的 4 倍
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        # 第一个卷积层：1x1 卷积，用于降维（从 in_channels 到 out_channels）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层：3x3 卷积，是主要的计算部分
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 第三个卷积层：1x1 卷积，用于升维（从 out_channels 到 out_channels * expansion）
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 快捷连接
        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 主类
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]], # 使用的残差块类型
        layers: List[int],                      # 每个阶段的残差块数量列表
        num_classes: int = 1000,                # 分类任务的类别数
        zero_init_residual: bool = False        # 是否将残差块的最后一个 BN 层初始化为 0
    ) -> None:
        super(ResNet, self).__init__()
        # 初始输入通道数
        self.in_channels = 64

        # 初始卷积层
        # 7x7 卷积，步长为 2，将输入图像尺寸减半
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # 初始最大池化层
        # 3x3 池化，步长为 2，进一步将尺寸减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 创建四个阶段的残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化层，将每个特征图池化为一个值
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，用于分类
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 He 初始化 (Kaiming Normal)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 将 BN 层的权重初始化为 1，偏置为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 如果设置为 True，将每个残差分支的最后一个 BN 层的权重初始化为 0
        # 这可以使得初始时残差分支为 0，模型更接近于恒等映射，有助于训练
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 辅助函数，用于创建单个阶段的残差层
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,          # 该阶段的输出通道数
        blocks: int,            # 该阶段包含的残差块数量
        stride: int = 1         # 该阶段第一个残差块的步长
    ) -> nn.Sequential:
        downsample = None
        # 判断是否需要下采样（维度不匹配）
        # 1. 步长不为 1，导致特征图尺寸变化
        # 2. 输入通道数与期望的输出通道数不符
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        # 添加该阶段的第一个残差块，它可能包含下采样
        layers.append(block(self.in_channels, channels, stride, downsample))
        # 更新 in_channels，为后续的残差块做准备
        self.in_channels = channels * block.expansion
        # 添加该阶段剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个阶段的残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化和全连接层
        x = self.avgpool(x)
        # 将特征图展平为一维向量
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 工厂函数，用于创建不同深度的 ResNet 模型
def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

# 示例：创建一个 ResNet-50 模型
if __name__ == '__main__':
    # 创建一个 ResNet-50 模型实例，用于对 1000 个类别进行分类
    model = resnet50(num_classes=1000)
    # 打印模型结构
    # print(model)

    # 创建一个随机的输入张量来测试模型
    # batch_size=1, channels=3, height=224, width=224
    input_tensor = torch.randn(1, 3, 224, 224)
    # 通过模型进行前向传播
    output = model(input_tensor)
    # 打印输出张量的形状，应为 (1, 1000)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")