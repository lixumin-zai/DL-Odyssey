import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # 确保模型维度能被注意力头数整除
        assert d_model % n_heads == 0
        
        self.d_model = d_model  # 模型维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 定义线性变换层，用于生成Q、K、V
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query投影层
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key投影层
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value投影层
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影层
        
        self.dropout = nn.Dropout(dropout)  # Dropout层，防止过拟合
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape  # 获取输入张量的形状
        
        # 生成Q、K、V矩阵
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 重塑张量以支持多头注意力：[batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数：Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 如果提供了掩码，则应用掩码（用于因果注意力）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将掩码位置设为负无穷
        
        # 应用softmax获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # 应用dropout
        
        # 计算加权后的值
        out = torch.matmul(attention_weights, V)  # [batch_size, n_heads, seq_len, d_k]
        
        # 重新组合多头输出：[batch_size, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # 通过输出投影层
        return self.w_o(out)

class FeedForward(nn.Module):
    """前馈神经网络模块"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 两层线性变换，中间使用ReLU激活
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层：扩展维度
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层：恢复维度
        self.dropout = nn.Dropout(dropout)  # Dropout层
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前馈网络：Linear -> ReLU -> Dropout -> Linear
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer块，包含多头注意力和前馈网络"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)  # 多头注意力层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈网络层
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 多头注意力 + 残差连接 + 层归一化
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))  # 残差连接和层归一化
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))  # 残差连接和层归一化
        
        return x

class TextEncoder(nn.Module):
    """CLIP文本编码器，基于Transformer架构"""
    
    def __init__(self, 
                 vocab_size: int = 49408,  # 词汇表大小
                 max_length: int = 77,     # 最大序列长度
                 d_model: int = 512,       # 模型维度
                 n_heads: int = 8,         # 注意力头数
                 n_layers: int = 12,       # Transformer层数
                 d_ff: int = 2048,         # 前馈网络维度
                 dropout: float = 0.1):    # Dropout概率
        super().__init__()
        
        self.max_length = max_length
        self.d_model = d_model
        
        # 词嵌入层：将token ID转换为向量表示
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置嵌入层：为每个位置学习一个向量表示
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # 多层Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 最终的层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 线性层使用正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # 层归一化的偏置初始化为0，权重初始化为1
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 创建位置索引
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 获取词嵌入和位置嵌入
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, d_model]
        
        # 将词嵌入和位置嵌入相加
        x = token_emb + pos_emb
        
        # 创建因果掩码（下三角矩阵），确保每个位置只能看到之前的位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # 通过所有Transformer块
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
        
        # 应用最终的层归一化
        x = self.ln_final(x)
        
        # 返回序列的最后一个位置的表示（用于分类）
        # 在CLIP中，通常使用EOS token的位置
        return x[torch.arange(batch_size), input_ids.argmax(dim=-1)]  # [batch_size, d_model]

class PatchEmbedding(nn.Module):
    """图像patch嵌入层，将图像分割成patches并转换为向量"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 计算patch数量
        
        # 使用卷积层将patches转换为嵌入向量
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.shape
        
        # 确保输入图像尺寸正确
        assert height == self.img_size and width == self.img_size, \
            f"Input image size ({height}x{width}) doesn't match model ({self.img_size}x{self.img_size})"
        
        # 应用卷积投影：[batch_size, embed_dim, n_patches_h, n_patches_w]
        x = self.projection(x)
        
        # 重塑为序列格式：[batch_size, n_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) 图像编码器"""
    
    def __init__(self, 
                 img_size: int = 224,      # 输入图像尺寸
                 patch_size: int = 16,     # Patch大小
                 in_channels: int = 3,     # 输入通道数
                 embed_dim: int = 768,     # 嵌入维度
                 n_heads: int = 12,        # 注意力头数
                 n_layers: int = 12,       # Transformer层数
                 d_ff: int = 3072,         # 前馈网络维度
                 dropout: float = 0.1):    # Dropout概率
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch嵌入层
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 类别token（CLS token），用于分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 位置嵌入（包括CLS token的位置）
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 最终的层归一化
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # 将图像转换为patch嵌入
        x = self.patch_embedding(x)  # [batch_size, n_patches, embed_dim]
        
        # 添加CLS token到序列开头
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, n_patches+1, embed_dim]
        
        # 添加位置嵌入
        x = x + self.position_embedding
        x = self.dropout(x)
        
        # 通过所有Transformer块
        for block in self.transformer_blocks:
            x = block(x)  # 图像不需要因果掩码
        
        # 应用最终的层归一化
        x = self.ln_final(x)
        
        # 返回CLS token的表示（第一个位置）
        return x[:, 0]  # [batch_size, embed_dim]

class CLIP(nn.Module):
    """CLIP主模型，包含图像编码器和文本编码器"""
    
    def __init__(self, 
                 # 图像编码器参数
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 vision_embed_dim: int = 768,
                 vision_n_heads: int = 12,
                 vision_n_layers: int = 12,
                 vision_d_ff: int = 3072,
                 
                 # 文本编码器参数
                 vocab_size: int = 49408,
                 max_length: int = 77,
                 text_embed_dim: int = 512,
                 text_n_heads: int = 8,
                 text_n_layers: int = 12,
                 text_d_ff: int = 2048,
                 
                 # 共享参数
                 embed_dim: int = 512,     # 最终嵌入维度
                 dropout: float = 0.1,
                 temperature_init: float = 0.07):  # 温度参数初始值
        super().__init__()
        
        # 图像编码器（Vision Transformer）
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vision_embed_dim,
            n_heads=vision_n_heads,
            n_layers=vision_n_layers,
            d_ff=vision_d_ff,
            dropout=dropout
        )
        
        # 文本编码器（Transformer）
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_length,
            d_model=text_embed_dim,
            n_heads=text_n_heads,
            n_layers=text_n_layers,
            d_ff=text_d_ff,
            dropout=dropout
        )
        
        # 投影层：将不同维度的特征投影到共同的嵌入空间
        self.vision_projection = nn.Linear(vision_embed_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim, bias=False)
        
        # 可学习的温度参数，用于缩放相似度
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature_init)))
        
        # 初始化投影层权重
        nn.init.normal_(self.vision_projection.weight, std=vision_embed_dim ** -0.5)
        nn.init.normal_(self.text_projection.weight, std=text_embed_dim ** -0.5)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像，返回归一化的图像特征"""
        # 通过视觉编码器获取图像特征
        image_features = self.vision_encoder(image)  # [batch_size, vision_embed_dim]
        
        # 投影到共同的嵌入空间
        image_features = self.vision_projection(image_features)  # [batch_size, embed_dim]
        
        # L2归一化，使特征位于单位球面上
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本，返回归一化的文本特征"""
        # 通过文本编码器获取文本特征
        text_features = self.text_encoder(text)  # [batch_size, text_embed_dim]
        
        # 投影到共同的嵌入空间
        text_features = self.text_projection(text_features)  # [batch_size, embed_dim]
        
        # L2归一化，使特征位于单位球面上
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回图像-文本相似度矩阵"""
        # 编码图像和文本
        image_features = self.encode_image(image)  # [batch_size, embed_dim]
        text_features = self.encode_text(text)     # [batch_size, embed_dim]
        
        # 计算相似度矩阵
        # logit_scale是可学习的温度参数，用于控制分布的锐度
        logit_scale = self.logit_scale.exp()  # 将对数尺度转换为线性尺度
        
        # 计算余弦相似度并乘以温度参数
        logits_per_image = logit_scale * image_features @ text_features.T  # [batch_size, batch_size]
        logits_per_text = logits_per_image.T  # [batch_size, batch_size]
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """计算CLIP的对比损失函数"""
    batch_size = logits_per_image.shape[0]
    
    # 创建标签：对角线为正样本（相同索引的图像和文本匹配）
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # 计算双向交叉熵损失
    loss_image_to_text = F.cross_entropy(logits_per_image, labels)  # 图像到文本的损失
    loss_text_to_image = F.cross_entropy(logits_per_text, labels)   # 文本到图像的损失
    
    # 返回平均损失
    return (loss_image_to_text + loss_text_to_image) / 2

# 创建模型的便捷函数
def create_clip_model(model_size: str = "base") -> CLIP:
    """创建不同规模的CLIP模型"""
    
    if model_size == "base":
        # 基础模型配置（类似ViT-B/16）
        return CLIP(
            img_size=224,
            patch_size=16,
            vision_embed_dim=768,
            vision_n_heads=12,
            vision_n_layers=12,
            text_embed_dim=512,
            text_n_heads=8,
            text_n_layers=12,
            embed_dim=512
        )
    elif model_size == "large":
        # 大型模型配置（类似ViT-L/14）
        return CLIP(
            img_size=224,
            patch_size=14,
            vision_embed_dim=1024,
            vision_n_heads=16,
            vision_n_layers=24,
            text_embed_dim=768,
            text_n_heads=12,
            text_n_layers=12,
            embed_dim=768
        )
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_clip_model("base").to(device)
    
    # 创建示例输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)  # 随机图像
    texts = torch.randint(0, 49408, (batch_size, 77)).to(device)  # 随机文本token
    
    # 前向传播
    with torch.no_grad():
        logits_per_image, logits_per_text = model(images, texts)
        
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"图像到文本logits形状: {logits_per_image.shape}")
    print(f"文本到图像logits形状: {logits_per_text.shape}")
    
    # 计算损失
    loss = contrastive_loss(logits_per_image, logits_per_text)
    print(f"对比损失: {loss.item():.4f}")
    
    # 测试单独的编码功能
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)
    
    print(f"图像特征形状: {image_features.shape}")
    print(f"文本特征形状: {text_features.shape}")
    print(f"图像特征范数: {torch.norm(image_features, dim=-1).mean():.4f}")
    print(f"文本特征范数: {torch.norm(text_features, dim=-1).mean():.4f}")