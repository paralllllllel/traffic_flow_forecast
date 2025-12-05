"""
交通数据加载与预处理模块
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, Union


class TrafficDataProcessor:
    """交通数据预处理器"""

    def __init__(self):
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None

    def load_data(self, data_path: str) -> np.ndarray:
        """
        加载原始数据

        Args:
            data_path: 数据文件路径，支持 .npz, .npy, .csv, .h5 格式

        Returns:
            np.ndarray: 加载的数据
        """
        ext = os.path.splitext(data_path)[1].lower()

        if ext == '.npz':
            data = np.load(data_path)
            # 尝试常见的键名
            for key in ['data', 'x', 'traffic', 'flow']:
                if key in data.files:
                    return data[key]
            # 返回第一个数组
            return data[data.files[0]]

        elif ext == '.npy':
            return np.load(data_path)

        elif ext == '.csv':
            df = pd.read_csv(data_path)
            return df.values

        elif ext in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(data_path, 'r') as f:
                for key in ['data', 'x', 'traffic', 'flow']:
                    if key in f.keys():
                        return np.array(f[key])
                # 返回第一个数据集
                first_key = list(f.keys())[0]
                return np.array(f[first_key])

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def load_adjacency_matrix(self, adj_path: str) -> np.ndarray:
        """
        加载邻接矩阵

        Args:
            adj_path: 邻接矩阵文件路径

        Returns:
            np.ndarray: 邻接矩阵
        """
        ext = os.path.splitext(adj_path)[1].lower()

        if ext == '.npz':
            data = np.load(adj_path)
            for key in ['adj', 'adjacency', 'A', 'adj_mx']:
                if key in data.files:
                    return data[key]
            return data[data.files[0]]

        elif ext == '.npy':
            return np.load(adj_path)

        elif ext == '.csv':
            return pd.read_csv(adj_path, header=None).values

        elif ext == '.pkl':
            import pickle
            with open(adj_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                if isinstance(data, np.ndarray):
                    return data
                # METR-LA 格式
                if isinstance(data, list) and len(data) >= 3:
                    return data[2]
                return data

        else:
            raise ValueError(f"不支持的邻接矩阵格式: {ext}")

    def handle_missing_values(
        self,
        data: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        缺失值处理

        Args:
            data: 输入数据
            method: 填充方法 ('linear', 'mean', 'forward', 'zero')

        Returns:
            np.ndarray: 处理后的数据
        """
        data = data.copy()

        if method == 'linear':
            # 线性插值
            df = pd.DataFrame(data.reshape(-1, data.shape[-1]))
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            data = df.values.reshape(data.shape)

        elif method == 'mean':
            # 均值填充
            mask = np.isnan(data) | (data == 0)
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[-1]):
                data[mask[..., i], i] = col_means[i] if not np.isnan(col_means[i]) else 0

        elif method == 'forward':
            # 前向填充
            df = pd.DataFrame(data.reshape(-1, data.shape[-1]))
            df = df.fillna(method='ffill').fillna(method='bfill')
            data = df.values.reshape(data.shape)

        elif method == 'zero':
            # 零值填充
            data = np.nan_to_num(data, nan=0.0)

        return data

    def normalize(
        self,
        data: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        数据标准化

        Args:
            data: 输入数据
            method: 标准化方法 ('zscore', 'minmax')

        Returns:
            np.ndarray: 标准化后的数据
        """
        if method == 'zscore':
            self.mean = np.mean(data)
            self.std = np.std(data)
            return (data - self.mean) / (self.std + 1e-8)

        elif method == 'minmax':
            self.min_val = np.min(data)
            self.max_val = np.max(data)
            return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        else:
            raise ValueError(f"不支持的标准化方法: {method}")

    def inverse_normalize(
        self,
        data: Union[np.ndarray, torch.Tensor],
        method: str = 'zscore'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        反标准化

        Args:
            data: 标准化后的数据
            method: 标准化方法

        Returns:
            反标准化后的数据
        """
        if method == 'zscore':
            if self.mean is None or self.std is None:
                raise ValueError("未找到标准化参数，请先调用 normalize()")
            return data * self.std + self.mean

        elif method == 'minmax':
            if self.min_val is None or self.max_val is None:
                raise ValueError("未找到标准化参数，请先调用 normalize()")
            return data * (self.max_val - self.min_val) + self.min_val

        else:
            raise ValueError(f"不支持的标准化方法: {method}")

    def build_adjacency_matrix(
        self,
        distance_matrix: np.ndarray,
        sigma: float = 10.0,
        epsilon: float = 0.5,
        normalize: bool = True
    ) -> np.ndarray:
        """
        基于距离矩阵构建邻接矩阵

        Args:
            distance_matrix: 节点间距离矩阵
            sigma: 高斯核带宽
            epsilon: 边权重阈值
            normalize: 是否归一化

        Returns:
            np.ndarray: 邻接矩阵
        """
        # 高斯核
        adj = np.exp(-distance_matrix ** 2 / (sigma ** 2))

        # 阈值过滤
        adj[adj < epsilon] = 0

        # 添加自环
        np.fill_diagonal(adj, 1)

        if normalize:
            # 对称归一化: D^{-1/2} A D^{-1/2}
            d = np.sum(adj, axis=1)
            d_inv_sqrt = np.power(d, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat = np.diag(d_inv_sqrt)
            adj = d_mat @ adj @ d_mat

        return adj

    def create_dataset(
        self,
        data: np.ndarray,
        hist_len: int = 12,
        pred_len: int = 12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建训练数据集（滑动窗口）

        Args:
            data: 原始时序数据 (T, N, F) 或 (T, N)
            hist_len: 历史时间步数
            pred_len: 预测时间步数

        Returns:
            X: 输入数据 (samples, hist_len, N, F)
            Y: 目标数据 (samples, pred_len, N, 1)
        """
        if data.ndim == 2:
            data = data[..., np.newaxis]

        T, N, F = data.shape
        num_samples = T - hist_len - pred_len + 1

        X = np.zeros((num_samples, hist_len, N, F))
        Y = np.zeros((num_samples, pred_len, N, 1))

        for i in range(num_samples):
            X[i] = data[i:i + hist_len]
            # 预测目标为第一个特征（通常是流量/速度）
            Y[i] = data[i + hist_len:i + hist_len + pred_len, :, :1]

        return X, Y

    def split_dataset(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        数据集划分

        Args:
            X: 输入数据
            Y: 目标数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            Dict: 包含 train, val, test 的数据字典
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            'train': (X[:train_end], Y[:train_end]),
            'val': (X[train_end:val_end], Y[train_end:val_end]),
            'test': (X[val_end:], Y[val_end:])
        }


class TrafficDataset(Dataset):
    """交通数据 PyTorch Dataset"""

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        add_time_features: bool = True
    ):
        """
        Args:
            X: 输入数据 (samples, hist_len, N, F)
            Y: 目标数据 (samples, pred_len, N, 1)
            add_time_features: 是否添加时间特征
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.add_time_features = add_time_features

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.X[idx],
            'y': self.Y[idx]
        }


def create_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    创建 DataLoader

    Args:
        X: 输入数据
        Y: 目标数据
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数

    Returns:
        DataLoader
    """
    dataset = TrafficDataset(X, Y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_data(
    data_path: str,
    adj_path: Optional[str] = None,
    hist_len: int = 12,
    pred_len: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    normalize_method: str = 'zscore'
) -> Tuple[Dict[str, DataLoader], np.ndarray, TrafficDataProcessor]:
    """
    完整的数据准备流程

    Args:
        data_path: 数据文件路径
        adj_path: 邻接矩阵路径（可选）
        hist_len: 历史时间步
        pred_len: 预测时间步
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        normalize_method: 标准化方法

    Returns:
        dataloaders: 数据加载器字典
        adj_matrix: 邻接矩阵
        processor: 数据处理器（用于反标准化）
    """
    processor = TrafficDataProcessor()

    # 加载数据
    data = processor.load_data(data_path)
    print(f"原始数据形状: {data.shape}")

    # 缺失值处理
    data = processor.handle_missing_values(data, method='linear')

    # 标准化
    data = processor.normalize(data, method=normalize_method)

    # 创建数据集
    X, Y = processor.create_dataset(data, hist_len, pred_len)
    print(f"样本数量: {len(X)}, 输入形状: {X.shape}, 目标形状: {Y.shape}")

    # 划分数据集
    splits = processor.split_dataset(X, Y, train_ratio, val_ratio)
    print(f"训练集: {len(splits['train'][0])}, "
          f"验证集: {len(splits['val'][0])}, "
          f"测试集: {len(splits['test'][0])}")

    # 创建 DataLoader
    dataloaders = {
        'train': create_dataloader(
            splits['train'][0], splits['train'][1],
            batch_size=batch_size, shuffle=True
        ),
        'val': create_dataloader(
            splits['val'][0], splits['val'][1],
            batch_size=batch_size, shuffle=False
        ),
        'test': create_dataloader(
            splits['test'][0], splits['test'][1],
            batch_size=batch_size, shuffle=False
        )
    }

    # 加载邻接矩阵
    adj_matrix = None
    if adj_path and os.path.exists(adj_path):
        adj_matrix = processor.load_adjacency_matrix(adj_path)
        print(f"邻接矩阵形状: {adj_matrix.shape}")

    return dataloaders, adj_matrix, processor
