"""
评估指标与损失函数模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union, Optional, List


class MaskedMAELoss(nn.Module):
    """带掩码的MAE损失，忽略缺失值"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值
            target: 真实值
            mask: 掩码（1表示有效，0表示忽略）

        Returns:
            损失值
        """
        if mask is None:
            return torch.abs(pred - target).mean()

        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-8)


class MaskedMSELoss(nn.Module):
    """带掩码的MSE损失"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return ((pred - target) ** 2).mean()

        loss = ((pred - target) ** 2) * mask
        return loss.sum() / (mask.sum() + 1e-8)


class HuberLoss(nn.Module):
    """Huber损失，对异常值更鲁棒"""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class QuantileLoss(nn.Module):
    """分位数损失，用于概率预测"""

    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        return sum(losses) / len(losses)


def masked_mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """计算MAE"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if mask is None:
        return np.abs(pred - target).mean()

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    return (np.abs(pred - target) * mask).sum() / (mask.sum() + 1e-8)


def masked_mse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """计算MSE"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if mask is None:
        return ((pred - target) ** 2).mean()

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + 1e-8)


def masked_rmse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """计算RMSE"""
    return np.sqrt(masked_mse(pred, target, mask))


def masked_mape(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    epsilon: float = 1e-8
) -> float:
    """计算MAPE (%)"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # 避免除零
    valid_mask = np.abs(target) > epsilon
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        valid_mask = valid_mask & (mask > 0)

    if valid_mask.sum() == 0:
        return 0.0

    return (np.abs(pred[valid_mask] - target[valid_mask]) /
            np.abs(target[valid_mask])).mean() * 100


def compute_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred: 预测值
        target: 真实值
        mask: 掩码

    Returns:
        Dict: 包含 MAE, RMSE, MAPE 的字典
    """
    return {
        'MAE': masked_mae(pred, target, mask),
        'RMSE': masked_rmse(pred, target, mask),
        'MAPE': masked_mape(pred, target, mask)
    }


def evaluate_by_horizon(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    horizons: List[int] = [3, 6, 12],
    interval_minutes: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    按预测时长分别评估

    Args:
        pred: 预测值 (B, T, N, 1)
        target: 真实值 (B, T, N, 1)
        horizons: 评估的时间步列表
        interval_minutes: 每个时间步的分钟数

    Returns:
        Dict: 各时长的评估指标
    """
    results = {}

    for h in horizons:
        if isinstance(pred, torch.Tensor):
            p = pred[:, :h]
            t = target[:, :h]
        else:
            p = pred[:, :h]
            t = target[:, :h]

        metrics = compute_metrics(p, t)
        results[f'{h * interval_minutes}min'] = metrics

    return results


def compute_congestion_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    计算拥堵预测相关指标

    Args:
        pred: 预测值（归一化后）
        target: 真实值（归一化后）
        threshold: 拥堵阈值

    Returns:
        Dict: 准确率、精确率、召回率、F1
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred_congestion = (pred > threshold).astype(int)
    true_congestion = (target > threshold).astype(int)

    # True Positives, False Positives, False Negatives
    tp = ((pred_congestion == 1) & (true_congestion == 1)).sum()
    fp = ((pred_congestion == 1) & (true_congestion == 0)).sum()
    fn = ((pred_congestion == 0) & (true_congestion == 1)).sum()
    tn = ((pred_congestion == 0) & (true_congestion == 0)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


class MetricTracker:
    """指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * count
            self.counts[key] += count

    def average(self) -> Dict[str, float]:
        return {
            key: self.metrics[key] / (self.counts[key] + 1e-8)
            for key in self.metrics
        }

    def __str__(self) -> str:
        avg = self.average()
        return ' | '.join([f'{k}: {v:.4f}' for k, v in avg.items()])
