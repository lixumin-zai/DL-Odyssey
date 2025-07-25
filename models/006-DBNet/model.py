import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple

class DBNet(nn.Module):
    """
    DBNet: Real-time Scene Text Detection with Differentiable Binarization
    
    主要组件:
    1. Backbone: 特征提取网络 (ResNet)
    2. Neck: 特征融合网络 (FPN)
    3. Head: 检测头，输出概率图、阈值图和二值图
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, k=50):
        """
        初始化DBNet模型
        
        Args:
            backbone (str): 骨干网络类型，支持 'resnet18', 'resnet50'
            pretrained (bool): 是否使用预训练权重
            k (int): 可微分二值化的放大因子
        """
        super(DBNet, self).__init__()
        
        # 保存可微分二值化的放大因子k，控制二值化的锐度
        self.k = k
        
        # 初始化骨干网络，用于提取多尺度特征
        self.backbone = self._build_backbone(backbone, pretrained)
        
        # 获取骨干网络各层的输出通道数，用于后续FPN构建
        self.in_channels = self._get_backbone_channels(backbone)
        
        # 构建特征金字塔网络(FPN)，融合不同尺度的特征
        self.neck = FPN(self.in_channels)
        
        # 构建检测头，输出概率图、阈值图和二值图
        self.head = DBHead()
        
        # 初始化网络权重，确保训练稳定性
        self._init_weights()
    
    def _build_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """
        构建骨干网络
        
        Args:
            backbone: 骨干网络类型
            pretrained: 是否使用预训练权重
            
        Returns:
            骨干网络模型
        """
        if backbone == 'resnet18':
            # 加载ResNet-18作为特征提取器
            model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet50':
            # 加载ResNet-50作为特征提取器
            model = models.resnet50(pretrained=pretrained)
        else:
            # 抛出异常，不支持的骨干网络类型
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 移除最后的全连接层和平均池化层，只保留特征提取部分
        return nn.Sequential(
            model.conv1,      # 初始卷积层
            model.bn1,        # 批归一化层
            model.relu,       # ReLU激活函数
            model.maxpool,    # 最大池化层
            model.layer1,     # 第一个残差块组
            model.layer2,     # 第二个残差块组
            model.layer3,     # 第三个残差块组
            model.layer4      # 第四个残差块组
        )
    
    def _get_backbone_channels(self, backbone: str) -> List[int]:
        """
        获取骨干网络各层的输出通道数
        
        Args:
            backbone: 骨干网络类型
            
        Returns:
            各层输出通道数列表 [C2, C3, C4, C5]
        """
        if backbone == 'resnet18':
            # ResNet-18各层输出通道数
            return [64, 128, 256, 512]
        elif backbone == 'resnet50':
            # ResNet-50各层输出通道数
            return [256, 512, 1024, 2048]
        else:
            # 抛出异常，不支持的骨干网络类型
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _init_weights(self):
        """
        初始化网络权重
        使用Xavier初始化确保训练稳定性
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Xavier正态分布初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    # 偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 (B, 3, H, W)
            
        Returns:
            包含概率图、阈值图和二值图的字典
        """
        # 通过骨干网络提取多尺度特征
        features = self._extract_features(x)
        
        # 通过FPN融合多尺度特征
        fused_features = self.neck(features)
        
        # 通过检测头生成概率图和阈值图
        prob_map, threshold_map = self.head(fused_features)
        
        # 应用可微分二值化生成二值图
        binary_map = self.differentiable_binarization(prob_map, threshold_map)
        
        return {
            'prob_map': prob_map,           # 概率图：每个像素属于文本的概率
            'threshold_map': threshold_map,  # 阈值图：每个像素的自适应阈值
            'binary_map': binary_map        # 二值图：最终的文本分割结果
        }
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        从骨干网络提取多尺度特征
        
        Args:
            x: 输入图像张量
            
        Returns:
            多尺度特征列表 [C2, C3, C4, C5]
        """
        features = []
        
        # 通过骨干网络的前几层
        x = self.backbone[0](x)  # conv1
        x = self.backbone[1](x)  # bn1
        x = self.backbone[2](x)  # relu
        x = self.backbone[3](x)  # maxpool
        
        # 提取各个尺度的特征
        x = self.backbone[4](x)  # layer1 -> C2 (1/4分辨率)
        features.append(x)
        
        x = self.backbone[5](x)  # layer2 -> C3 (1/8分辨率)
        features.append(x)
        
        x = self.backbone[6](x)  # layer3 -> C4 (1/16分辨率)
        features.append(x)
        
        x = self.backbone[7](x)  # layer4 -> C5 (1/32分辨率)
        features.append(x)
        
        return features
    
    def differentiable_binarization(self, prob_map: torch.Tensor, 
                                  threshold_map: torch.Tensor) -> torch.Tensor:
        """
        可微分二值化函数
        
        使用sigmoid函数实现可微分的二值化过程：
        DB(x,y) = 1 / (1 + exp(-k * (P(x,y) - T(x,y))))
        
        Args:
            prob_map: 概率图，形状为 (B, 1, H, W)
            threshold_map: 阈值图，形状为 (B, 1, H, W)
            
        Returns:
            二值图，形状为 (B, 1, H, W)
        """
        # 计算概率图与阈值图的差值
        diff = prob_map - threshold_map
        
        # 应用可微分二值化公式
        # 使用sigmoid函数确保输出在[0,1]范围内且处处可导
        binary_map = torch.sigmoid(self.k * diff)
        
        return binary_map


class FPN(nn.Module):
    """
    特征金字塔网络 (Feature Pyramid Network)
    
    用于融合不同尺度的特征，增强多尺度文本检测能力
    采用自顶向下的路径和横向连接进行特征融合
    """
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        """
        初始化FPN
        
        Args:
            in_channels: 输入特征的通道数列表 [C2, C3, C4, C5]
            out_channels: 输出特征的统一通道数
        """
        super(FPN, self).__init__()
        
        # 保存输出通道数
        self.out_channels = out_channels
        
        # 横向连接：将不同通道数的特征统一到相同通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # 输出卷积：对融合后的特征进行平滑处理
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        FPN前向传播
        
        Args:
            features: 多尺度特征列表 [C2, C3, C4, C5]
            
        Returns:
            融合后的特征图
        """
        # 应用横向连接，统一通道数
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下融合特征
        # 从最高层开始，逐层向下融合
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高层特征到当前层分辨率
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            # 将上采样的高层特征与当前层特征相加
            laterals[i] = laterals[i] + upsampled
        
        # 应用输出卷积进行特征平滑
        fpn_outs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # 将所有尺度的特征上采样到最大分辨率并拼接
        target_size = fpn_outs[0].shape[2:]  # 使用C2的分辨率作为目标
        
        # 上采样所有特征到相同分辨率
        upsampled_features = []
        for feat in fpn_outs:
            if feat.shape[2:] != target_size:
                # 上采样到目标分辨率
                feat = F.interpolate(
                    feat, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            upsampled_features.append(feat)
        
        # 在通道维度拼接所有特征
        fused_feature = torch.cat(upsampled_features, dim=1)
        
        return fused_feature


class DBHead(nn.Module):
    """
    DBNet检测头
    
    负责从融合特征中生成概率图和阈值图
    采用轻量级的卷积结构确保实时性能
    """
    
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256):
        """
        初始化检测头
        
        Args:
            in_channels: 输入特征通道数 (FPN输出通道数 * 4)
            hidden_channels: 隐藏层通道数
        """
        super(DBHead, self).__init__()
        
        # 特征降维卷积，减少计算量
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 概率图预测分支
        self.prob_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, hidden_channels // 4, 2, 2),  # 上采样
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 2, 2),  # 再次上采样到原图分辨率
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # 阈值图预测分支
        self.threshold_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, hidden_channels // 4, 2, 2),  # 上采样
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 2, 2),  # 再次上采样到原图分辨率
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测头前向传播
        
        Args:
            x: 融合后的特征图
            
        Returns:
            概率图和阈值图的元组
        """
        # 特征降维，减少后续计算量
        x = self.reduce_conv(x)
        
        # 分别生成概率图和阈值图
        prob_map = self.prob_conv(x)        # 概率图：文本区域的概率
        threshold_map = self.threshold_conv(x)  # 阈值图：自适应二值化阈值
        
        return prob_map, threshold_map


class DBLoss(nn.Module):
    """
    DBNet损失函数
    
    包含三个组件：
    1. 分割损失：约束概率图预测
    2. 二值化损失：约束二值图预测  
    3. 阈值损失：约束阈值图预测
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 10.0):
        """
        初始化损失函数
        
        Args:
            alpha: 二值化损失权重
            beta: 阈值损失权重
        """
        super(DBLoss, self).__init__()
        self.alpha = alpha  # 二值化损失的权重系数
        self.beta = beta    # 阈值损失的权重系数
    
    def forward(self, pred: Dict[str, torch.Tensor], 
                gt_text: torch.Tensor, 
                gt_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            pred: 模型预测结果字典
            gt_text: 文本区域真值标签 (B, 1, H, W)
            gt_mask: 有效区域掩码 (B, 1, H, W)
            
        Returns:
            损失字典
        """
        # 提取预测结果
        prob_map = pred['prob_map']
        threshold_map = pred['threshold_map']
        binary_map = pred['binary_map']
        
        # 计算分割损失 (概率图损失)
        seg_loss = self._segmentation_loss(prob_map, gt_text, gt_mask)
        
        # 计算二值化损失
        bin_loss = self._binary_loss(binary_map, gt_text, gt_mask)
        
        # 计算阈值损失
        thresh_loss = self._threshold_loss(threshold_map, gt_text, gt_mask)
        
        # 计算总损失
        total_loss = seg_loss + self.alpha * bin_loss + self.beta * thresh_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'bin_loss': bin_loss,
            'thresh_loss': thresh_loss
        }
    
    def _segmentation_loss(self, pred: torch.Tensor, 
                          gt: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
        """
        分割损失：BCE + Dice损失
        
        Args:
            pred: 预测概率图
            gt: 真值标签
            mask: 有效区域掩码
            
        Returns:
            分割损失值
        """
        # 应用掩码，只在有效区域计算损失
        pred = pred * mask
        gt = gt * mask
        
        # BCE损失：关注像素级分类准确性
        bce_loss = F.binary_cross_entropy(pred, gt, reduction='none')
        bce_loss = (bce_loss * mask).sum() / mask.sum().clamp(min=1)
        
        # Dice损失：关注区域重叠度
        dice_loss = self._dice_loss(pred, gt, mask)
        
        return bce_loss + dice_loss
    
    def _binary_loss(self, pred: torch.Tensor, 
                    gt: torch.Tensor, 
                    mask: torch.Tensor) -> torch.Tensor:
        """
        二值化损失：约束二值图与真值一致
        
        Args:
            pred: 预测二值图
            gt: 真值标签
            mask: 有效区域掩码
            
        Returns:
            二值化损失值
        """
        # 应用掩码
        pred = pred * mask
        gt = gt * mask
        
        # 使用BCE损失
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    
    def _threshold_loss(self, pred: torch.Tensor, 
                       gt: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        阈值损失：约束阈值图预测
        
        Args:
            pred: 预测阈值图
            gt: 真值标签
            mask: 有效区域掩码
            
        Returns:
            阈值损失值
        """
        # 生成阈值真值标签（在文本边界附近）
        gt_threshold = self._generate_threshold_target(gt)
        
        # 应用掩码
        pred = pred * mask
        gt_threshold = gt_threshold * mask
        
        # 使用L1损失，更平滑
        loss = F.l1_loss(pred, gt_threshold, reduction='none')
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    
    def _dice_loss(self, pred: torch.Tensor, 
                   gt: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
        """
        Dice损失：衡量预测区域与真值区域的重叠度
        
        Args:
            pred: 预测概率图
            gt: 真值标签
            mask: 有效区域掩码
            
        Returns:
            Dice损失值
        """
        # 应用掩码
        pred = pred * mask
        gt = gt * mask
        
        # 计算交集和并集
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        
        # 计算Dice系数，添加平滑项避免除零
        dice = (2 * intersection + 1) / (union + 1)
        
        # 返回Dice损失 (1 - Dice系数)
        return 1 - dice
    
    def _generate_threshold_target(self, gt: torch.Tensor) -> torch.Tensor:
        """
        生成阈值图的真值标签
        
        在文本边界附近设置较小阈值，文本内部设置较大阈值
        
        Args:
            gt: 文本区域真值标签
            
        Returns:
            阈值图真值标签
        """
        # 简化实现：在文本区域内部设置较高阈值
        # 实际应用中可以使用距离变换生成更精确的阈值标签
        threshold_target = gt * 0.7 + 0.3  # 文本区域阈值0.7，背景区域阈值0.3
        
        return threshold_target


def build_dbnet(backbone='resnet18', pretrained=True, k=50):
    """
    构建DBNet模型的工厂函数
    
    Args:
        backbone: 骨干网络类型
        pretrained: 是否使用预训练权重
        k: 可微分二值化放大因子
        
    Returns:
        DBNet模型实例
    """
    return DBNet(backbone=backbone, pretrained=pretrained, k=k)


if __name__ == '__main__':
    # 测试代码：验证模型的前向传播
    
    # 创建模型实例
    model = build_dbnet(backbone='resnet18', pretrained=False)
    
    # 创建随机输入数据 (batch_size=2, channels=3, height=640, width=640)
    x = torch.randn(2, 3, 640, 640)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    # 打印输出形状
    print("模型输出形状:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试损失函数
    criterion = DBLoss()
    
    # 创建模拟的真值标签
    gt_text = torch.randint(0, 2, (2, 1, 640, 640)).float()
    gt_mask = torch.ones(2, 1, 640, 640)
    
    # 计算损失
    losses = criterion(outputs, gt_text, gt_mask)
    
    print(f"\n损失函数测试:")
    for key, value in losses.items():
        print(f"{key}: {value.item():.4f}")