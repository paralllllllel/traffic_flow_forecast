"""
交通流量预测 Spatial-Temporal Transformer 模型
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from .layers import (
    SpatioTemporalEmbedding,
    STTransformerBlock,
    GraphConvolution
)


class TrafficSTTransformer(nn.Module):
    """
    交通流量预测 Spatial-Temporal Transformer

    模型架构:
        Input -> Embedding -> [ST-Block x L] -> Output Projection -> Prediction
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
        pred_len: int = 12,
        dropout: float = 0.1,
        use_time_features: bool = True,
        use_graph_conv: bool = False,
        causal: bool = False,
        activation: str = 'gelu'
    ):
        """
        Args:
            num_nodes: 节点/路段数量
            input_dim: 输入特征维度
            d_model: 模型隐藏维度
            n_heads: 注意力头数
            n_layers: Transformer块数量
            ff_dim: 前馈网络隐藏维度
            pred_len: 预测时间步数
            dropout: Dropout比率
            use_time_features: 是否使用时间特征
            use_graph_conv: 是否使用图卷积
            causal: 是否使用因果注意力
            activation: 激活函数 ('gelu', 'relu', 'silu')
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_graph_conv = use_graph_conv

        # 时空嵌入层
        self.embedding = SpatioTemporalEmbedding(
            d_model=d_model,
            num_nodes=num_nodes,
            input_dim=input_dim,
            dropout=dropout,
            use_time_features=use_time_features
        )

        # ST-Transformer 块
        self.blocks = nn.ModuleList([
            STTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                causal=causal,
                activation=activation
            )
            for _ in range(n_layers)
        ])

        # 可选的图卷积层
        if use_graph_conv:
            self.graph_conv = GraphConvolution(d_model, dropout)

        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        hour: Optional[torch.Tensor] = None,
        weekday: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        前向传播

        Args:
            x: (B, T, N, F) 历史交通数据
            adj_matrix: (N, N) 邻接矩阵
            hour: (B, T) 小时信息 [0-23]
            weekday: (B, T) 星期信息 [0-6]
            return_attn: 是否返回注意力权重

        Returns:
            pred: (B, pred_len, N, 1) 预测结果
            attention_maps: 注意力权重列表（可选）
        """
        B, T, N, F = x.shape

        # 时空嵌入
        x = self.embedding(x, hour, weekday)  # (B, T, N, D)

        # 可选的图卷积
        if self.use_graph_conv and adj_matrix is not None:
            x = self.graph_conv(x, adj_matrix)

        # ST-Transformer 块
        attention_maps = [] if return_attn else None

        for block in self.blocks:
            x, spatial_attn, temporal_attn = block(
                x,
                adj_mask=adj_matrix,
                return_attn=return_attn
            )

            if return_attn:
                attention_maps.append({
                    'spatial': spatial_attn,
                    'temporal': temporal_attn
                })

        # 取最后时刻的特征进行预测
        x = x[:, -1, :, :]  # (B, N, D)

        # 输出投影
        pred = self.output_proj(x)  # (B, N, pred_len)

        # 调整形状为 (B, pred_len, N, 1)
        pred = pred.permute(0, 2, 1).unsqueeze(-1)

        return pred, attention_maps

    def predict(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        hour: Optional[torch.Tensor] = None,
        weekday: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        推理模式（不返回注意力权重）

        Args:
            x: (B, T, N, F) 历史交通数据
            adj_matrix: (N, N) 邻接矩阵
            hour: (B, T) 小时信息
            weekday: (B, T) 星期信息

        Returns:
            pred: (B, pred_len, N, 1) 预测结果
        """
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(x, adj_matrix, hour, weekday, return_attn=False)
        return pred

    def get_attention_maps(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        hour: Optional[torch.Tensor] = None,
        weekday: Optional[torch.Tensor] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        获取注意力权重（用于可解释性分析）

        Returns:
            attention_maps: 每层的空间和时间注意力权重
        """
        self.eval()
        with torch.no_grad():
            _, attention_maps = self.forward(
                x, adj_matrix, hour, weekday, return_attn=True
            )
        return attention_maps

    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TrafficSTTransformerEncoder(nn.Module):
    """
    编码器版本 - 用于迁移学习或多任务学习
    只包含编码部分，不包含预测头
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_time_features: bool = True
    ):
        super().__init__()

        self.embedding = SpatioTemporalEmbedding(
            d_model=d_model,
            num_nodes=num_nodes,
            input_dim=input_dim,
            dropout=dropout,
            use_time_features=use_time_features
        )

        self.blocks = nn.ModuleList([
            STTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        hour: Optional[torch.Tensor] = None,
        weekday: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns:
            encoded: (B, T, N, D) 编码后的特征
        """
        x = self.embedding(x, hour, weekday)

        for block in self.blocks:
            x, _, _ = block(x, adj_mask=adj_matrix)

        return x


class MultiHorizonPredictor(nn.Module):
    """
    多步预测头 - 为不同预测时长使用独立的预测器
    """

    def __init__(
        self,
        d_model: int,
        horizons: List[int] = [3, 6, 12],
        dropout: float = 0.1
    ):
        super().__init__()

        self.horizons = horizons
        self.predictors = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, h)
            )
            for h in horizons
        })

    def forward(
        self,
        x: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) 编码特征
            horizon: 预测时长

        Returns:
            pred: (B, horizon, N, 1)
        """
        if str(horizon) not in self.predictors:
            raise ValueError(f"Unsupported horizon: {horizon}")

        pred = self.predictors[str(horizon)](x)  # (B, N, horizon)
        pred = pred.permute(0, 2, 1).unsqueeze(-1)  # (B, horizon, N, 1)

        return pred


def create_model(config: dict) -> TrafficSTTransformer:
    """
    根据配置创建模型

    Args:
        config: 模型配置字典

    Returns:
        TrafficSTTransformer 模型
    """
    return TrafficSTTransformer(
        num_nodes=config.get('num_nodes', 207),
        input_dim=config.get('input_dim', 1),
        d_model=config.get('d_model', 64),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        ff_dim=config.get('ff_dim', 256),
        pred_len=config.get('pred_len', 12),
        dropout=config.get('dropout', 0.1),
        use_time_features=config.get('use_time_features', True),
        use_graph_conv=config.get('use_graph_conv', False),
        causal=config.get('causal', False),
        activation=config.get('activation', 'gelu')
    )
