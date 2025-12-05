"""
可视化工具模块
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Dict, Union, Tuple
import torch

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_spatial_attention(
    attn_weights: Union[np.ndarray, torch.Tensor],
    node_coords: Optional[np.ndarray] = None,
    node_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Spatial Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'hot'
) -> plt.Figure:
    """
    可视化空间注意力权重

    Args:
        attn_weights: 注意力权重矩阵 (N, N)
        node_coords: 节点坐标 (N, 2)，用于网络图可视化
        node_names: 节点名称列表
        save_path: 保存路径
        title: 图标题
        figsize: 图大小
        cmap: 颜色映射

    Returns:
        matplotlib Figure
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2 if node_coords is not None else 1,
                              figsize=figsize)

    if node_coords is None:
        ax = axes
    else:
        ax = axes[0]

    # 热力图
    im = ax.imshow(attn_weights, cmap=cmap, aspect='auto')
    ax.set_title(f'{title} (Heatmap)')
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 如果有坐标，绘制网络图
    if node_coords is not None:
        import networkx as nx

        ax2 = axes[1]
        G = nx.DiGraph()

        n_nodes = len(node_coords)
        for i in range(n_nodes):
            G.add_node(i, pos=node_coords[i])

        # 添加边（只保留权重较高的）
        threshold = np.percentile(attn_weights, 90)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and attn_weights[i, j] > threshold:
                    G.add_edge(i, j, weight=attn_weights[i, j])

        pos = nx.get_node_attributes(G, 'pos')
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]

        nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.7, ax=ax2)
        nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds,
                               width=1, alpha=0.6, arrows=True, ax=ax2)
        ax2.set_title(f'{title} (Network)')
        ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_temporal_attention(
    attn_weights: Union[np.ndarray, torch.Tensor],
    time_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Temporal Attention Weights",
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    可视化时间注意力权重

    Args:
        attn_weights: 注意力权重 (T, T) 或 (N, T, T)
        time_labels: 时间标签
        save_path: 保存路径
        title: 图标题
        figsize: 图大小
        cmap: 颜色映射

    Returns:
        matplotlib Figure
    """
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()

    # 如果是多节点，取平均
    if attn_weights.ndim == 3:
        attn_weights = attn_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attn_weights, cmap=cmap, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Key Time Step')
    ax.set_ylabel('Query Time Step')

    if time_labels:
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels(time_labels, rotation=45)
        ax.set_yticks(range(len(time_labels)))
        ax.set_yticklabels(time_labels)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_predictions(
    true_values: Union[np.ndarray, torch.Tensor],
    predictions: Union[np.ndarray, torch.Tensor],
    node_id: int = 0,
    time_range: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    可视化预测结果对比

    Args:
        true_values: 真实值
        predictions: 预测值
        node_id: 节点ID
        time_range: 时间范围 (start, end)
        save_path: 保存路径
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # 提取单个节点数据
    if true_values.ndim > 1:
        true_values = true_values.flatten()
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    if time_range:
        start, end = time_range
        true_values = true_values[start:end]
        predictions = predictions[start:end]

    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # 主图：预测对比
    ax1 = axes[0]
    x = range(len(true_values))
    ax1.plot(x, true_values, label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(x, predictions, label='Prediction', color='red',
             linestyle='--', linewidth=1.5)
    ax1.fill_between(x, true_values, predictions, alpha=0.2, color='gray')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Traffic Flow')
    ax1.set_title(title or f'Traffic Flow Prediction (Node {node_id})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 副图：误差
    ax2 = axes[1]
    errors = predictions - true_values
    colors = ['red' if e > 0 else 'blue' for e in errors]
    ax2.bar(x, errors, color=colors, alpha=0.6, width=1.0)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_congestion_map(
    traffic_data: Union[np.ndarray, torch.Tensor],
    node_coords: np.ndarray,
    thresholds: Dict[str, Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    title: str = "Traffic Congestion Map",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    可视化拥堵地图

    Args:
        traffic_data: 交通数据 (N,) 归一化后的流量/速度
        node_coords: 节点坐标 (N, 2)
        thresholds: 拥堵等级阈值
        save_path: 保存路径
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    if isinstance(traffic_data, torch.Tensor):
        traffic_data = traffic_data.detach().cpu().numpy()

    if traffic_data.ndim > 1:
        traffic_data = traffic_data.flatten()

    if thresholds is None:
        thresholds = {
            'free_flow': (0, 0.3),
            'slow': (0.3, 0.5),
            'congested': (0.5, 0.7),
            'severe': (0.7, 1.0),
        }

    colors_map = {
        'free_flow': 'green',
        'slow': 'yellow',
        'congested': 'orange',
        'severe': 'red'
    }

    labels_map = {
        'free_flow': 'Free Flow',
        'slow': 'Slow',
        'congested': 'Congested',
        'severe': 'Severe'
    }

    fig, ax = plt.subplots(figsize=figsize)

    # 为每个节点分配颜色
    node_colors = []
    for val in traffic_data:
        color = 'gray'
        for level, (low, high) in thresholds.items():
            if low <= val < high:
                color = colors_map[level]
                break
        node_colors.append(color)

    scatter = ax.scatter(node_coords[:, 0], node_coords[:, 1],
                         c=node_colors, s=100, alpha=0.7, edgecolors='black')

    # 添加图例
    legend_elements = [
        plt.scatter([], [], c=colors_map[level], s=100, label=labels_map[level])
        for level in thresholds.keys()
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    可视化训练历史

    Args:
        history: 训练历史 {'train_loss': [...], 'val_loss': [...], ...}
        save_path: 保存路径
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, history.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over epochs')
        ax.grid(True, alpha=0.3)

        # 标记最佳值
        if 'loss' in metric_name.lower():
            best_idx = np.argmin(values)
            ax.axvline(x=best_idx + 1, color='red', linestyle='--', alpha=0.5)
            ax.scatter([best_idx + 1], [values[best_idx]], color='red', s=100, zorder=5)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    可视化模型对比结果

    Args:
        results: 模型结果 {'model1': {'MAE': x, 'RMSE': y}, ...}
        save_path: 保存路径
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    models = list(results.keys())
    metrics = list(results[models[0]].keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(models))
    width = 0.6

    for ax, metric in zip(axes, metrics):
        values = [results[model][metric] for model in models]
        bars = ax.bar(x, values, width, color='steelblue', alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_horizon_metrics(
    metrics_by_horizon: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Metrics by Prediction Horizon",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    可视化不同预测时长的指标

    Args:
        metrics_by_horizon: 各时长指标 {'15min': {'MAE': x}, '30min': {...}, ...}
        save_path: 保存路径
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    horizons = list(metrics_by_horizon.keys())
    metrics = list(metrics_by_horizon[horizons[0]].keys())

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(horizons))
    width = 0.25
    multiplier = 0

    for metric in metrics:
        values = [metrics_by_horizon[h][metric] for h in horizons]
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=metric)
        multiplier += 1

    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
