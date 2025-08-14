import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), # 第一个3x3卷积，使用padding保持尺寸不变
            nn.BatchNorm2d(mid_channels), # 批量归一化，加速收敛并提高稳定性
            nn.ReLU(inplace=True), # ReLU激活函数，增加非线性
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), # 第二个3x3卷积
            nn.BatchNorm2d(out_channels), # 批量归一化
            nn.ReLU(inplace=True) # ReLU激活函数
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 2x2最大池化层，将特征图尺寸减半
            DoubleConv(in_channels, out_channels) # 应用双卷积模块
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 使用双线性插值进行上采样，将特征图尺寸加倍
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # 使用双卷积调整通道数
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2) # 使用转置卷积进行上采样
            self.conv = DoubleConv(in_channels, out_channels) # 使用双卷积


    def forward(self, x1, x2):
        x1 = self.up(x1) # 对来自解码器路径的特征图进行上采样
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] # 计算编码器特征图和上采样后的解码器特征图的高度差
        diffX = x2.size()[3] - x1.size()[3] # 计算编码器特征图和上采样后的解码器特征图的宽度差

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # 对上采样后的特征图进行填充，使其与编码器特征图尺寸一致
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/blob/master/model.py
        x = torch.cat([x2, x1], dim=1) # 将编码器的特征图（跳跃连接）和解码器的特征图在通道维度上拼接
        return self.conv(x) # 应用双卷积


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # 使用1x1卷积将特征图的通道数映射到最终的类别数

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels # 输入图像的通道数
        self.n_classes = n_classes # 分割的类别数
        self.bilinear = bilinear # 是否使用双线性插值进行上采样

        self.inc = DoubleConv(n_channels, 64) # 初始的双卷积模块
        self.down1 = Down(64, 128) # 第一个下采样模块
        self.down2 = Down(128, 256) # 第二个下采样模块
        self.down3 = Down(256, 512) # 第三个下采样模块
        factor = 2 if bilinear else 1 # 根据是否使用双线性插值确定因子
        self.down4 = Down(512, 1024 // factor) # 第四个下采样模块
        self.up1 = Up(1024, 512 // factor, bilinear) # 第一个上采样模块
        self.up2 = Up(512, 256 // factor, bilinear) # 第二个上采样模块
        self.up3 = Up(256, 128 // factor, bilinear) # 第三个上采样模块
        self.up4 = Up(128, 64, bilinear) # 第四个上采样模块
        self.outc = OutConv(64, n_classes) # 输出卷积模块

    def forward(self, x):
        x1 = self.inc(x) # 初始特征图
        x2 = self.down1(x1) # 第一次下采样后的特征图
        x3 = self.down2(x2) # 第二次下采样后的特征图
        x4 = self.down3(x3) # 第三次下采样后的特征图
        x5 = self.down4(x4) # 第四次下采样后的特征图（瓶颈层）
        x = self.up1(x5, x4) # 第一次上采样并与x4拼接
        x = self.up2(x, x3) # 第二次上采样并与x3拼接
        x = self.up3(x, x2) # 第三次上采样并与x2拼接
        x = self.up4(x, x1) # 第四次上采样并与x1拼接
        logits = self.outc(x) # 输出最终的分割图
        return logits