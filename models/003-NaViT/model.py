import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Optional, Tuple
import math

class FactorizedPositionalEmbedding(nn.Module):
    """分解位置编码模块
    
    将2D位置编码分解为高度和宽度两个独立的1D编码，
    这样可以支持任意分辨率的图像输入
    """
    
    def __init__(self, max_height: int, max_width: int, dim: int):
        super().__init__()
        # 初始化高度方向的位置编码参数，维度为 max_height x (dim//2)
        self.height_embed = nn.Parameter(torch.randn(max_height, dim // 2))
        # 初始化宽度方向的位置编码参数，维度为 max_width x (dim//2)
        self.width_embed = nn.Parameter(torch.randn(max_width, dim // 2))
        # 保存最大高度和宽度，用于插值计算
        self.max_height = max_height
        self.max_width = max_width
        
    def forward(self, height: int, width: int) -> torch.Tensor:
        """前向传播函数
        
        Args:
            height: 图像的patch高度数量
            width: 图像的patch宽度数量
            
        Returns:
            位置编码张量，形状为 (height*width, dim)
        """
        # 如果输入尺寸超过预设最大值，需要进行插值
        if height > self.max_height or width > self.max_width:
            # 对高度编码进行插值，从 max_height 插值到 height
            h_embed = F.interpolate(
                self.height_embed.unsqueeze(0).unsqueeze(0),  # 添加batch和channel维度
                size=(height, self.height_embed.shape[1]),    # 目标尺寸
                mode='bilinear',                              # 双线性插值
                align_corners=False                           # 不对齐角点
            ).squeeze(0).squeeze(0)  # 移除添加的维度
            
            # 对宽度编码进行插值，从 max_width 插值到 width
            w_embed = F.interpolate(
                self.width_embed.unsqueeze(0).unsqueeze(0),   # 添加batch和channel维度
                size=(width, self.width_embed.shape[1]),      # 目标尺寸
                mode='bilinear',                              # 双线性插值
                align_corners=False                           # 不对齐角点
            ).squeeze(0).squeeze(0)  # 移除添加的维度
        else:
            # 如果尺寸在预设范围内，直接截取对应部分
            h_embed = self.height_embed[:height]  # 取前height个高度编码
            w_embed = self.width_embed[:width]    # 取前width个宽度编码
        
        # 创建网格坐标，用于组合高度和宽度编码
        # h_indices: (height*width,) 每个位置对应的高度索引
        # w_indices: (height*width,) 每个位置对应的宽度索引
        h_indices = torch.arange(height, device=h_embed.device).repeat_interleave(width)
        w_indices = torch.arange(width, device=w_embed.device).repeat(height)
        
        # 根据索引获取对应的编码并拼接
        # h_embed[h_indices]: (height*width, dim//2) 每个位置的高度编码
        # w_embed[w_indices]: (height*width, dim//2) 每个位置的宽度编码
        pos_embed = torch.cat([
            h_embed[h_indices],  # 高度编码部分
            w_embed[w_indices]   # 宽度编码部分
        ], dim=-1)  # 在最后一个维度拼接，得到完整的位置编码
        
        return pos_embed  # 返回形状为 (height*width, dim) 的位置编码

class PatchEmbedding(nn.Module):
    """图像块嵌入模块
    
    将输入图像分割成patches并转换为embedding向量
    """
    
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        # 保存patch大小，用于后续计算
        self.patch_size = patch_size
        # 使用卷积层将patch转换为embedding
        # kernel_size和stride都等于patch_size，实现无重叠的patch提取
        self.projection = nn.Conv2d(
            in_channels,    # 输入通道数（如RGB图像为3）
            embed_dim,      # 输出embedding维度
            kernel_size=patch_size,  # 卷积核大小等于patch大小
            stride=patch_size        # 步长等于patch大小，确保无重叠
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """前向传播函数
        
        Args:
            x: 输入图像张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            patches: patch embedding张量，形状为 (batch_size, num_patches, embed_dim)
            h_patches: 高度方向的patch数量
            w_patches: 宽度方向的patch数量
        """
        # 通过卷积提取patches
        # 输出形状: (batch_size, embed_dim, h_patches, w_patches)
        patches = self.projection(x)
        
        # 获取patch网格的尺寸
        batch_size, embed_dim, h_patches, w_patches = patches.shape
        
        # 重新排列维度: (batch_size, embed_dim, h_patches, w_patches) 
        # -> (batch_size, h_patches, w_patches, embed_dim)
        patches = patches.permute(0, 2, 3, 1)
        
        # 展平空间维度: (batch_size, h_patches, w_patches, embed_dim)
        # -> (batch_size, h_patches*w_patches, embed_dim)
        patches = patches.reshape(batch_size, h_patches * w_patches, embed_dim)
        
        return patches, h_patches, w_patches

class MultiHeadAttention(nn.Module):
    """多头自注意力模块
    
    实现了带有查询-键归一化的多头自注意力机制
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # 确保embedding维度能被头数整除
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim                    # embedding维度
        self.num_heads = num_heads        # 注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子，用于稳定训练
        
        # 线性变换层，将输入映射为查询、键、值
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # 查询和键的层归一化，提高训练稳定性
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        # dropout层，用于正则化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            mask: 注意力掩码，形状为 (batch_size, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # 计算查询、键、值
        # qkv形状: (batch_size, seq_len, 3*dim)
        qkv = self.qkv(x)
        # 重新排列并分割为q, k, v
        # 每个的形状: (batch_size, num_heads, seq_len, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离查询、键、值
        
        # 对查询和键进行归一化，提高训练稳定性
        q = self.q_norm(q)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.k_norm(k)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 计算注意力分数
        # attn形状: (batch_size, num_heads, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 如果提供了掩码，应用掩码
        if mask is not None:
            # 将掩码中为0的位置设置为负无穷，这样softmax后会变成0
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax获得注意力权重
        attn = F.softmax(attn, dim=-1)
        # 应用dropout进行正则化
        attn = self.dropout(attn)
        
        # 应用注意力权重到值向量
        # out形状: (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn, v)
        
        # 重新排列维度并合并多头
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        out = out.transpose(1, 2)
        # 合并多头: (batch_size, seq_len, num_heads*head_dim) = (batch_size, seq_len, dim)
        out = out.reshape(batch_size, seq_len, dim)
        
        # 通过输出投影层
        out = self.proj(out)
        
        return out

class MLP(nn.Module):
    """多层感知机模块
    
    Transformer中的前馈网络部分
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # 第一个线性层，扩展维度
        self.fc1 = nn.Linear(dim, hidden_dim)
        # 激活函数使用GELU
        self.act = nn.GELU()
        # dropout层用于正则化
        self.dropout = nn.Dropout(dropout)
        # 第二个线性层，恢复原始维度
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        # 第一个线性变换
        x = self.fc1(x)
        # 应用激活函数
        x = self.act(x)
        # 应用dropout
        x = self.dropout(x)
        # 第二个线性变换
        x = self.fc2(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块
    
    包含多头自注意力和前馈网络，使用残差连接和层归一化
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        # 注意力层的层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 多头自注意力层
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        # MLP层的层归一化
        self.norm2 = nn.LayerNorm(dim)
        # MLP层，隐藏层维度通常是输入维度的4倍
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            mask: 注意力掩码，形状为 (batch_size, seq_len, seq_len)
            
        Returns:
            输出张量，形状为 (batch_size, seq_len, dim)
        """
        # 第一个残差连接：注意力层
        # 先进行层归一化，再通过注意力层，最后加上残差连接
        x = x + self.attn(self.norm1(x), mask)
        
        # 第二个残差连接：MLP层
        # 先进行层归一化，再通过MLP层，最后加上残差连接
        x = x + self.mlp(self.norm2(x))
        
        return x

class NaViT(nn.Module):
    """NaViT (Native Resolution Vision Transformer) 模型
    
    支持任意分辨率和宽高比的视觉Transformer模型
    """
    
    def __init__(
        self,
        image_size: int = 224,           # 默认图像尺寸（用于初始化位置编码）
        patch_size: int = 16,            # patch大小
        in_channels: int = 3,            # 输入通道数
        num_classes: int = 1000,         # 分类类别数
        embed_dim: int = 768,            # embedding维度
        depth: int = 12,                 # Transformer层数
        num_heads: int = 12,             # 注意力头数
        mlp_ratio: float = 4.0,          # MLP隐藏层维度比例
        dropout: float = 0.1,            # dropout概率
        emb_dropout: float = 0.1,        # embedding dropout概率
        token_dropout_prob: float = 0.0, # token dropout概率
    ):
        super().__init__()
        
        # 保存配置参数
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.token_dropout_prob = token_dropout_prob
        
        # 计算最大patch数量，用于初始化位置编码
        max_patches_per_side = image_size // patch_size
        
        # 初始化各个模块
        # patch embedding层，将图像转换为patch embeddings
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        
        # 分解位置编码，支持任意分辨率
        self.pos_embed = FactorizedPositionalEmbedding(
            max_patches_per_side, max_patches_per_side, embed_dim
        )
        
        # embedding dropout层
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)  # 创建depth个Transformer块
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """权重初始化函数
        
        Args:
            m: 模型模块
        """
        if isinstance(m, nn.Linear):
            # 线性层使用截断正态分布初始化
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # 偏置初始化为0
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 层归一化的偏置初始化为0，权重初始化为1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 卷积层使用截断正态分布初始化
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def apply_token_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """应用token dropout
        
        在训练时随机丢弃一部分tokens以提高效率和泛化能力
        
        Args:
            x: 输入token张量，形状为 (batch_size, seq_len, embed_dim)
            
        Returns:
            应用dropout后的token张量
        """
        # 只在训练时应用token dropout
        if not self.training or self.token_dropout_prob == 0.0:
            return x
        
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算保留的token数量
        keep_prob = 1.0 - self.token_dropout_prob
        num_keep = int(seq_len * keep_prob)
        
        # 为每个样本随机选择要保留的token索引
        keep_indices = torch.randperm(seq_len, device=x.device)[:num_keep]
        keep_indices = keep_indices.sort()[0]  # 排序以保持相对位置
        
        # 只保留选中的tokens
        x = x[:, keep_indices, :]
        
        return x
    
    def create_attention_mask(self, batch_images: List[List[torch.Tensor]]) -> torch.Tensor:
        """为序列打包创建注意力掩码
        
        Args:
            batch_images: 打包的图像批次
            
        Returns:
            注意力掩码张量
        """
        # 计算每个图像的token数量
        token_counts = []
        for images in batch_images:
            for img in images:
                h, w = img.shape[-2:]
                h_patches = h // self.patch_size
                w_patches = w // self.patch_size
                token_counts.append(h_patches * w_patches)
        
        # 计算总的序列长度
        total_seq_len = sum(token_counts)
        
        # 创建掩码矩阵
        mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool)
        
        # 填充掩码，确保每个图像的tokens只能注意到自己
        start_idx = 0
        for count in token_counts:
            end_idx = start_idx + count
            mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx
        
        return mask
    
    def forward(self, x: Union[torch.Tensor, List[List[torch.Tensor]]]) -> torch.Tensor:
        """前向传播函数
        
        Args:
            x: 输入图像，可以是:
                - 单个张量: (batch_size, channels, height, width)
                - 序列打包的图像列表: List[List[Tensor]]
                
        Returns:
            分类预测结果，形状为 (batch_size, num_classes)
        """
        # 处理不同的输入格式
        if isinstance(x, torch.Tensor):
            # 标准的批次输入
            return self._forward_standard(x)
        else:
            # 序列打包的输入
            return self._forward_packed(x)
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """标准批次的前向传播
        
        Args:
            x: 输入图像张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            分类预测结果，形状为 (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # 将图像转换为patch embeddings
        patches, h_patches, w_patches = self.patch_embed(x)
        
        # 添加位置编码
        pos_embed = self.pos_embed(h_patches, w_patches)  # (num_patches, embed_dim)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展到批次维度
        
        # 将patch embeddings和位置编码相加
        x = patches + pos_embed
        
        # 应用embedding dropout
        x = self.emb_dropout(x)
        
        # 应用token dropout（仅在训练时）
        x = self.apply_token_dropout(x)
        
        # 通过Transformer编码器层
        for block in self.transformer_blocks:
            x = block(x)  # 标准输入不需要注意力掩码
        
        # 最终的层归一化
        x = self.norm(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # 分类头
        x = self.head(x)  # (batch_size, num_classes)
        
        return x
    
    def _forward_packed(self, batch_images: List[List[torch.Tensor]]) -> torch.Tensor:
        """序列打包的前向传播
        
        Args:
            batch_images: 打包的图像批次，每个元素是一个图像列表
            
        Returns:
            分类预测结果，形状为 (total_images, num_classes)
        """
        all_patches = []
        all_pos_embeds = []
        image_boundaries = []  # 记录每个图像的边界，用于后续分离
        
        current_pos = 0
        
        # 处理每个打包的图像组
        for images in batch_images:
            for img in images:
                # 添加批次维度
                img = img.unsqueeze(0)  # (1, channels, height, width)
                
                # 转换为patch embeddings
                patches, h_patches, w_patches = self.patch_embed(img)
                patches = patches.squeeze(0)  # 移除批次维度 (num_patches, embed_dim)
                
                # 获取位置编码
                pos_embed = self.pos_embed(h_patches, w_patches)  # (num_patches, embed_dim)
                
                # 保存patches和位置编码
                all_patches.append(patches)
                all_pos_embeds.append(pos_embed)
                
                # 记录图像边界
                num_patches = patches.shape[0]
                image_boundaries.append((current_pos, current_pos + num_patches))
                current_pos += num_patches
        
        # 拼接所有patches和位置编码
        x = torch.cat(all_patches, dim=0)  # (total_patches, embed_dim)
        pos_embed = torch.cat(all_pos_embeds, dim=0)  # (total_patches, embed_dim)
        
        # 添加位置编码
        x = x + pos_embed
        
        # 添加批次维度
        x = x.unsqueeze(0)  # (1, total_patches, embed_dim)
        
        # 应用embedding dropout
        x = self.emb_dropout(x)
        
        # 创建注意力掩码
        mask = self.create_attention_mask(batch_images)
        mask = mask.unsqueeze(0).to(x.device)  # 添加批次维度
        
        # 通过Transformer编码器层
        for block in self.transformer_blocks:
            x = block(x, mask)  # 使用注意力掩码
        
        # 最终的层归一化
        x = self.norm(x)
        x = x.squeeze(0)  # 移除批次维度 (total_patches, embed_dim)
        
        # 为每个图像进行全局平均池化
        outputs = []
        for start, end in image_boundaries:
            # 对每个图像的patches进行平均池化
            img_features = x[start:end].mean(dim=0)  # (embed_dim,)
            # 通过分类头
            img_output = self.head(img_features)  # (num_classes,)
            outputs.append(img_output)
        
        # 堆叠所有输出
        outputs = torch.stack(outputs)  # (total_images, num_classes)
        
        return outputs

# 辅助函数：创建模型的不同配置
def navit_tiny(num_classes: int = 1000, **kwargs) -> NaViT:
    """创建NaViT-Tiny模型
    
    Args:
        num_classes: 分类类别数
        **kwargs: 其他参数
        
    Returns:
        NaViT-Tiny模型实例
    """
    return NaViT(
        embed_dim=192,      # 较小的embedding维度
        depth=12,           # 12层Transformer
        num_heads=3,        # 3个注意力头
        mlp_ratio=4.0,      # MLP比例
        num_classes=num_classes,
        **kwargs
    )

def navit_small(num_classes: int = 1000, **kwargs) -> NaViT:
    """创建NaViT-Small模型
    
    Args:
        num_classes: 分类类别数
        **kwargs: 其他参数
        
    Returns:
        NaViT-Small模型实例
    """
    return NaViT(
        embed_dim=384,      # 中等的embedding维度
        depth=12,           # 12层Transformer
        num_heads=6,        # 6个注意力头
        mlp_ratio=4.0,      # MLP比例
        num_classes=num_classes,
        **kwargs
    )

def navit_base(num_classes: int = 1000, **kwargs) -> NaViT:
    """创建NaViT-Base模型
    
    Args:
        num_classes: 分类类别数
        **kwargs: 其他参数
        
    Returns:
        NaViT-Base模型实例
    """
    return NaViT(
        embed_dim=768,      # 标准的embedding维度
        depth=12,           # 12层Transformer
        num_heads=12,       # 12个注意力头
        mlp_ratio=4.0,      # MLP比例
        num_classes=num_classes,
        **kwargs
    )

def navit_large(num_classes: int = 1000, **kwargs) -> NaViT:
    """创建NaViT-Large模型
    
    Args:
        num_classes: 分类类别数
        **kwargs: 其他参数
        
    Returns:
        NaViT-Large模型实例
    """
    return NaViT(
        embed_dim=1024,     # 较大的embedding维度
        depth=24,           # 24层Transformer
        num_heads=16,       # 16个注意力头
        mlp_ratio=4.0,      # MLP比例
        num_classes=num_classes,
        **kwargs
    )

# 测试函数
if __name__ == "__main__":
    # 创建模型实例
    model = navit_base(num_classes=1000)
    
    # 测试标准输入
    print("测试标准输入:")
    x_standard = torch.randn(2, 3, 224, 224)  # 批次大小为2的标准输入
    output_standard = model(x_standard)
    print(f"输入形状: {x_standard.shape}")
    print(f"输出形状: {output_standard.shape}")
    
    # 测试序列打包输入
    print("\n测试序列打包输入:")
    x_packed = [
        [torch.randn(3, 224, 224), torch.randn(3, 160, 160)],  # 第一组：2张不同尺寸的图像
        [torch.randn(3, 192, 256)],                            # 第二组：1张图像
        [torch.randn(3, 128, 128), torch.randn(3, 256, 128)]   # 第三组：2张不同尺寸的图像
    ]
    output_packed = model(x_packed)
    print(f"输入: 3组图像，总共5张")
    print(f"输出形状: {output_packed.shape}")  # 应该是 (5, 1000)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")