import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PatchEmbedding(nn.Module):
    """将图像分割为patches并转换为embedding向量"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size  # 输入图像尺寸
        self.patch_size = patch_size  # 每个patch的尺寸
        self.num_patches = (img_size // patch_size) ** 2  # 计算总patch数量
        
        # 使用卷积层将patches转换为embedding，相当于将每个patch线性投影
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        
        # 确保输入图像尺寸正确
        assert H == self.img_size and W == self.img_size, f"输入图像尺寸必须是 {self.img_size}x{self.img_size}"
        
        # 通过卷积将图像分割为patches并投影到embedding空间
        # 输出shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = self.projection(x)
        
        # 重新排列维度: (batch_size, embed_dim, num_patches) -> (batch_size, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        return x

class MultiHeadAttention(nn.Module):
    """多头自注意力机制实现"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim  # embedding维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        
        # 确保embedding维度能被头数整除
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        # 缩放因子，用于稳定训练
        self.scale = self.head_dim ** -0.5
        
        # 线性投影层，用于生成Q、K、V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 输出投影层
        self.proj = nn.Linear(embed_dim, embed_dim)
        # dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape  # batch_size, sequence_length, embed_dim
        
        # 生成Q、K、V矩阵
        # qkv shape: (batch_size, sequence_length, 3 * embed_dim)
        qkv = self.qkv(x)
        
        # 重新排列为多头格式
        # shape: (batch_size, sequence_length, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # 调整维度顺序: (3, batch_size, num_heads, sequence_length, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # 分离Q、K、V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数: Q * K^T
        # shape: (batch_size, num_heads, sequence_length, sequence_length)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 如果提供了mask，应用mask（主要用于文本的因果注意力）
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获得注意力权重
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重到V
        # shape: (batch_size, num_heads, sequence_length, head_dim)
        x = attn @ v
        
        # 重新排列维度并合并多头
        # shape: (batch_size, sequence_length, embed_dim)
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # 通过输出投影层
        x = self.proj(x)
        
        return x

class FeedForward(nn.Module):
    """前馈神经网络层"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # 第一个线性层，扩展维度
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # 激活函数使用GELU
        self.act = nn.GELU()
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 第二个线性层，恢复原始维度
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过第一个线性层和激活函数
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        # 通过第二个线性层
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # 层归一化，在注意力之前应用
        self.norm1 = nn.LayerNorm(embed_dim)
        # 多头自注意力层
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        # 层归一化，在前馈网络之前应用
        self.norm2 = nn.LayerNorm(embed_dim)
        # 前馈神经网络
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力机制的残差连接：x + attention(norm(x))
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # 前馈网络的残差连接：x + ffn(norm(x))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """视觉Transformer编码器"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 12, 
                 hidden_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        # Patch embedding层
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 类别token，用于分类任务
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码，为每个patch和cls_token添加位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化位置编码
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 初始化类别token
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化其他层的权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # batch size
        
        # 将图像转换为patch embeddings
        x = self.patch_embed(x)
        
        # 添加类别token到序列开头
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 通过所有Transformer块
        for block in self.blocks:
            x = block(x)
            
        # 应用最终的层归一化
        x = self.norm(x)
        
        # 返回类别token的表示（用于分类）
        return x[:, 0]

class TextTransformer(nn.Module):
    """文本Transformer编码器"""
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 77, embed_dim: int = 768,
                 num_layers: int = 12, num_heads: int = 12, hidden_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        self.max_length = max_length
        
        # 词嵌入层
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        
        # dropout层
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化位置编码
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化其他层的权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果注意力mask，确保每个位置只能看到之前的位置"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, seq_len = input_ids.shape
        
        # 确保序列长度不超过最大长度
        assert seq_len <= self.max_length, f"序列长度 {seq_len} 超过最大长度 {self.max_length}"
        
        # 词嵌入
        x = self.token_embed(input_ids)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # 创建因果mask（可选，取决于是否需要因果注意力）
        # causal_mask = self.create_causal_mask(seq_len, input_ids.device)
        
        # 通过所有Transformer块
        for block in self.blocks:
            # 这里可以传入attention_mask来处理padding
            x = block(x, mask=attention_mask)
            
        # 应用最终的层归一化
        x = self.norm(x)
        
        # 对于SigLIP，通常使用序列的最后一个非padding token作为文本表示
        # 这里简化为使用最后一个token
        if attention_mask is not None:
            # 找到每个序列的最后一个有效token
            last_token_indices = attention_mask.sum(dim=1) - 1
            text_features = x[torch.arange(B), last_token_indices]
        else:
            # 如果没有attention_mask，使用最后一个token
            text_features = x[:, -1, :]
            
        return text_features

class SigLIPModel(nn.Module):
    """SigLIP主模型，结合视觉和文本编码器"""
    
    def __init__(self, 
                 # 视觉编码器参数
                 img_size: int = 224,
                 patch_size: int = 16,
                 vision_embed_dim: int = 768,
                 vision_num_layers: int = 12,
                 vision_num_heads: int = 12,
                 # 文本编码器参数
                 vocab_size: int = 32000,
                 max_length: int = 77,
                 text_embed_dim: int = 768,
                 text_num_layers: int = 12,
                 text_num_heads: int = 12,
                 # 共同参数
                 projection_dim: int = 512,
                 temperature: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        
        # 视觉编码器
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            num_layers=vision_num_layers,
            num_heads=vision_num_heads,
            hidden_dim=vision_embed_dim * 4,
            dropout=dropout
        )
        
        # 文本编码器
        self.text_encoder = TextTransformer(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=text_embed_dim,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            hidden_dim=text_embed_dim * 4,
            dropout=dropout
        )
        
        # 投影层，将视觉和文本特征投影到共同的空间
        self.vision_projection = nn.Linear(vision_embed_dim, projection_dim)
        self.text_projection = nn.Linear(text_embed_dim, projection_dim)
        
        # 温度参数，用于缩放相似度
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 初始化投影层权重
        self._init_projections()
        
    def _init_projections(self):
        """初始化投影层权重"""
        for projection in [self.vision_projection, self.text_projection]:
            torch.nn.init.trunc_normal_(projection.weight, std=0.02)
            if projection.bias is not None:
                nn.init.constant_(projection.bias, 0)
                
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像并投影到共同空间"""
        # 通过视觉编码器获取图像特征
        image_features = self.vision_encoder(images)
        # 投影到共同空间
        image_features = self.vision_projection(image_features)
        # L2归一化，确保特征在单位球面上
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码文本并投影到共同空间"""
        # 通过文本编码器获取文本特征
        text_features = self.text_encoder(input_ids, attention_mask)
        # 投影到共同空间
        text_features = self.text_projection(text_features)
        # L2归一化，确保特征在单位球面上
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features
    
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播，返回图像特征、文本特征和相似度矩阵"""
        # 编码图像和文本
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 计算相似度矩阵
        # 使用矩阵乘法计算所有图像-文本对的相似度
        similarity_matrix = torch.matmul(image_features, text_features.T) * self.temperature
        
        return image_features, text_features, similarity_matrix

class SigLIPLoss(nn.Module):
    """SigLIP的sigmoid损失函数"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        """计算SigLIP损失
        
        Args:
            similarity_matrix: 相似度矩阵，shape为(batch_size, batch_size)
            
        Returns:
            损失值
        """
        batch_size = similarity_matrix.shape[0]
        
        # 创建标签矩阵：对角线为1（正样本对），其他位置为0（负样本对）
        labels = torch.eye(batch_size, device=similarity_matrix.device, dtype=torch.float32)
        
        # 将相似度矩阵和标签展平为一维
        similarities_flat = similarity_matrix.flatten()
        labels_flat = labels.flatten()
        
        # 使用二元交叉熵损失（带logits）
        # 这等价于sigmoid + binary cross entropy
        loss = F.binary_cross_entropy_with_logits(
            similarities_flat, 
            labels_flat, 
            reduction='mean'
        )
        
        return loss

def create_siglip_model(model_size: str = 'base') -> SigLIPModel:
    """创建不同大小的SigLIP模型
    
    Args:
        model_size: 模型大小，可选 'tiny', 'small', 'base', 'large'
        
    Returns:
        SigLIP模型实例
    """
    
    # 定义不同模型大小的配置
    configs = {
        'tiny': {
            'vision_embed_dim': 192,
            'vision_num_layers': 12,
            'vision_num_heads': 3,
            'text_embed_dim': 192,
            'text_num_layers': 12,
            'text_num_heads': 3,
            'projection_dim': 192
        },
        'small': {
            'vision_embed_dim': 384,
            'vision_num_layers': 12,
            'vision_num_heads': 6,
            'text_embed_dim': 384,
            'text_num_layers': 12,
            'text_num_heads': 6,
            'projection_dim': 384
        },
        'base': {
            'vision_embed_dim': 768,
            'vision_num_layers': 12,
            'vision_num_heads': 12,
            'text_embed_dim': 768,
            'text_num_layers': 12,
            'text_num_heads': 12,
            'projection_dim': 512
        },
        'large': {
            'vision_embed_dim': 1024,
            'vision_num_layers': 24,
            'vision_num_heads': 16,
            'text_embed_dim': 1024,
            'text_num_layers': 24,
            'text_num_heads': 16,
            'projection_dim': 768
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"不支持的模型大小: {model_size}. 支持的大小: {list(configs.keys())}")
    
    config = configs[model_size]
    
    # 创建模型
    model = SigLIPModel(**config)
    
    return model

if __name__ == "__main__":
    # 测试代码
    print("创建SigLIP模型...")
    
    # 创建一个base大小的模型
    model = create_siglip_model('base')
    
    # 创建损失函数
    criterion = SigLIPLoss()
    
    # 创建示例输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # 随机图像
    input_ids = torch.randint(0, 32000, (batch_size, 77))  # 随机文本token
    attention_mask = torch.ones(batch_size, 77)  # 注意力mask
    
    print(f"输入图像shape: {images.shape}")
    print(f"输入文本shape: {input_ids.shape}")
    
    # 前向传播
    with torch.no_grad():
        image_features, text_features, similarity_matrix = model(images, input_ids, attention_mask)
        
    print(f"图像特征shape: {image_features.shape}")
    print(f"文本特征shape: {text_features.shape}")
    print(f"相似度矩阵shape: {similarity_matrix.shape}")
    
    # 计算损失
    loss = criterion(similarity_matrix)
    print(f"损失值: {loss.item():.4f}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    print("模型测试完成！")