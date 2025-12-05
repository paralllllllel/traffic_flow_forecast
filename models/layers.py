"""
Spatial-Temporal Transformer 核心组件
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpatioTemporalEmbedding(nn.Module):
    """
    时空位置编码模块
    - 时间编码: 捕捉时间步的顺序信息和周期性
    - 空间编码: 为每个路段分配唯一的位置向量
    """

    def __init__(
        self,
        d_model: int,
        num_nodes: int,
        input_dim: int,
        max_len: int = 512,
        dropout: float = 0.1,
        use_time_features: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.use_time_features = use_time_features

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 可学习的时空位置编码
        self.temporal_emb = nn.Parameter(torch.randn(1, max_len, 1, d_model) * 0.02)
        self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_nodes, d_model) * 0.02)

        # 时间周期编码 (小时、星期)
        if use_time_features:
            self.hour_emb = nn.Embedding(24, d_model)
            self.weekday_emb = nn.Embedding(7, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        hour: Optional[torch.Tensor] = None,
        weekday: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) 输入特征
            hour: (B, T) 小时信息 [0-23]
            weekday: (B, T) 星期信息 [0-6]

        Returns:
            (B, T, N, D) 编码后的特征
        """
        B, T, N, F = x.shape

        # 特征投影
        x = self.input_proj(x)  # (B, T, N, D)

        # 添加时空位置编码
        x = x + self.temporal_emb[:, :T] + self.spatial_emb[:, :, :N]

        # 添加时间周期编码
        if self.use_time_features and hour is not None:
            hour_emb = self.hour_emb(hour)  # (B, T, D)
            x = x + hour_emb.unsqueeze(2)   # (B, T, 1, D) -> broadcast

        if self.use_time_features and weekday is not None:
            weekday_emb = self.weekday_emb(weekday)  # (B, T, D)
            x = x + weekday_emb.unsqueeze(2)

        x = self.norm(x)
        x = self.dropout(x)

        return x


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    - 在同一时刻，计算不同路段之间的关联关系
    - 可选: 使用邻接矩阵作为attention mask
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, N, D) 输入特征
            adj_mask: (N, N) 邻接矩阵掩码，0表示无连接
            return_attn: 是否返回注意力权重

        Returns:
            output: (B, T, N, D)
            attn_weights: (B*T, n_heads, N, N) 可选
        """
        B, T, N, D = x.shape
        residual = x

        # 重塑为 (B*T, N, D) 进行空间注意力
        x = x.reshape(B * T, N, D)

        # 计算 Q, K, V
        q = self.q_proj(x).reshape(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*T, heads, N, N)

        # 应用邻接矩阵掩码
        if adj_mask is not None:
            # adj_mask: (N, N) -> (1, 1, N, N)
            mask = adj_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, v)  # (B*T, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B * T, N, D)
        attn_output = self.out_proj(attn_output)

        # 残差连接和层归一化
        output = self.norm(residual + self.dropout(attn_output.reshape(B, T, N, D)))

        if return_attn:
            return output, attn_weights
        return output, None


class TemporalAttention(nn.Module):
    """
    时间注意力模块
    - 在同一路段，计算不同时刻之间的依赖关系
    - 使用因果mask防止未来信息泄露
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
        causal: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, N, D) 输入特征

        Returns:
            output: (B, T, N, D)
            attn_weights: (B*N, n_heads, T, T) 可选
        """
        B, T, N, D = x.shape
        residual = x

        # 重塑为 (B*N, T, D) 进行时间注意力
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # 计算 Q, K, V
        q = self.q_proj(x).reshape(B * N, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B * N, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B * N, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*N, heads, T, T)

        # 因果掩码
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, v)  # (B*N, heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B * N, T, D)
        attn_output = self.out_proj(attn_output)

        # 重塑回 (B, T, N, D)
        attn_output = attn_output.reshape(B, N, T, D).permute(0, 2, 1, 3)

        # 残差连接和层归一化
        output = self.norm(residual + self.dropout(attn_output))

        if return_attn:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(
        self,
        d_model: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class STTransformerBlock(nn.Module):
    """
    时空Transformer块
    - 先进行空间注意力，再进行时间注意力
    - 最后通过前馈网络进行特征变换
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        causal: bool = False,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout, causal=causal)
        self.ffn = FeedForward(d_model, ff_dim, dropout, activation)

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, N, D) 输入特征
            adj_mask: (N, N) 邻接矩阵掩码
            return_attn: 是否返回注意力权重

        Returns:
            output: (B, T, N, D)
            spatial_attn: 空间注意力权重
            temporal_attn: 时间注意力权重
        """
        # 空间注意力
        x, spatial_attn = self.spatial_attn(x, adj_mask, return_attn)

        # 时间注意力
        x, temporal_attn = self.temporal_attn(x, return_attn)

        # 前馈网络
        x = self.ffn(x)

        return x, spatial_attn, temporal_attn


class GraphConvolution(nn.Module):
    """图卷积层（可选，用于结合GCN）"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D)
            adj: (N, N) 归一化邻接矩阵

        Returns:
            (B, T, N, D)
        """
        residual = x
        # 图卷积: A * X * W
        x = torch.einsum('btnd,nm->btmd', x, adj)
        x = self.linear(x)
        x = self.dropout(x)
        return self.norm(residual + x)
