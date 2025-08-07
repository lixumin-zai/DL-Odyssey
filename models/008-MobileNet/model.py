import torch
import torch.nn as nn

# 定义倒置残差块 (Inverted Residual Block)
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        """
        初始化倒置残差块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 步长，用于控制下采样
        :param expand_ratio: 扩展因子，用于控制中间层的通道数
        """
        super(InvertedResidual, self).__init__()
        # 记录步长
        self.stride = stride
        # 计算隐藏层的通道数
        hidden_channels = in_channels * expand_ratio
        # 判断是否使用残差连接 (当步长为1且输入输出通道数相同时)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # 第一个1x1卷积，用于升维 (如果扩展因子不为1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU6(inplace=True))
        
        # 3x3深度卷积和1x1逐点卷积
        layers.extend([
            # 3x3 深度卷积 (Depthwise Convolution)
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            # 1x1 逐点卷积 (Pointwise Convolution), 线性输出
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        # 将所有层组合成一个序列
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        # 如果使用残差连接，则将输入与输出相加
        if self.use_res_connect:
            return x + self.conv(x)
        # 否则直接返回卷积结果
        else:
            return self.conv(x)

# 定义 MobileNetV2 模型
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        """
        初始化 MobileNetV2 模型
        :param num_classes: 分类数
        :param width_mult: 宽度乘数，用于调整网络的宽度
        """
        super(MobileNetV2, self).__init__()
        # 定义网络结构参数
        # t: 扩展因子, c: 输出通道数, n: 重复次数, s: 步长
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 构建第一层卷积
        input_channel = int(32 * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )

        # 构建倒置残差块序列
        self.features = []
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

        # 构建最后一层卷积
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, int(1280 * width_mult), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(1280 * width_mult)),
            nn.ReLU6(inplace=True),
        )

        # 构建分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(1280 * width_mult), num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        # 全局平均池化
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """ 初始化网络权重 """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

if __name__ == '__main__':
    # 创建一个模型实例
    model = MobileNetV2()
    # 打印模型结构
    print(model)
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 3, 224, 224)
    # 进行一次前向传播
    output = model(input_tensor)
    # 打印输出张量的形状
    print(output.shape)