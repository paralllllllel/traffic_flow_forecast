"""
交通流量预测推理脚本
"""
import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficSTTransformer
from utils.data_loader import TrafficDataProcessor
from utils.visualization import visualize_predictions, visualize_congestion_map


# 拥堵等级定义
CONGESTION_LEVELS = {
    'free_flow': (0, 0.3),
    'slow': (0.3, 0.5),
    'congested': (0.5, 0.7),
    'severe': (0.7, 1.0),
}

CONGESTION_NAMES = {
    'free_flow': '畅通',
    'slow': '缓行',
    'congested': '拥堵',
    'severe': '严重拥堵'
}


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    model_config['pred_len'] = config.get('data', {}).get('pred_len', 12)

    model = TrafficSTTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 恢复标准化参数
    processor = TrafficDataProcessor()
    if 'mean' in checkpoint:
        processor.mean = checkpoint['mean']
        processor.std = checkpoint['std']

    return model, config, processor


class TrafficPredictor:
    """交通流量预测器"""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
            device: 计算设备
        """
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )

        # 加载模型
        self.model, self.config, self.processor = load_model(
            checkpoint_path, self.device
        )

        # 加载额外配置
        if config_path and os.path.exists(config_path):
            extra_config = load_config(config_path)
            self.config.update(extra_config)

        self.hist_len = self.config.get('data', {}).get('hist_len', 12)
        self.pred_len = self.config.get('data', {}).get('pred_len', 12)

        # 加载邻接矩阵
        self.adj_matrix = None
        adj_path = self.config.get('data', {}).get('adj_path')
        if adj_path and os.path.exists(adj_path):
            self.adj_matrix = torch.FloatTensor(
                self.processor.load_adjacency_matrix(adj_path)
            ).to(self.device)

        print(f"Model loaded. Device: {self.device}")
        print(f"History length: {self.hist_len}, Prediction length: {self.pred_len}")

    def predict(
        self,
        history_data: np.ndarray,
        return_attention: bool = False
    ) -> dict:
        """
        预测未来交通流量

        Args:
            history_data: 历史数据 (T, N, F) 或 (T, N)
            return_attention: 是否返回注意力权重

        Returns:
            预测结果字典
        """
        # 数据预处理
        if history_data.ndim == 2:
            history_data = history_data[..., np.newaxis]

        # 确保时间步数正确
        if history_data.shape[0] != self.hist_len:
            if history_data.shape[0] > self.hist_len:
                history_data = history_data[-self.hist_len:]
            else:
                raise ValueError(
                    f"History data should have {self.hist_len} time steps, "
                    f"got {history_data.shape[0]}"
                )

        # 标准化
        if self.processor.mean is not None:
            data_normalized = (history_data - self.processor.mean) / (self.processor.std + 1e-8)
        else:
            data_normalized = history_data

        # 转换为tensor
        x = torch.FloatTensor(data_normalized).unsqueeze(0).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            pred, attention_maps = self.model(
                x, self.adj_matrix,
                return_attn=return_attention
            )

        # 反标准化
        pred_np = pred.cpu().numpy()[0]  # (pred_len, N, 1)
        if self.processor.mean is not None:
            pred_denorm = self.processor.inverse_normalize(pred_np)
        else:
            pred_denorm = pred_np

        result = {
            'predictions': pred_denorm,
            'predictions_normalized': pred_np,
            'num_nodes': pred_denorm.shape[1],
            'pred_len': pred_denorm.shape[0],
            'timestamp': datetime.now().isoformat()
        }

        if return_attention:
            result['attention_maps'] = attention_maps

        return result

    def predict_congestion(
        self,
        history_data: np.ndarray,
        capacity: np.ndarray = None
    ) -> dict:
        """
        预测拥堵状态

        Args:
            history_data: 历史数据
            capacity: 各路段容量（可选）

        Returns:
            拥堵预测结果
        """
        # 获取流量预测
        pred_result = self.predict(history_data)
        predictions = pred_result['predictions']

        # 如果没有容量数据，使用标准化值判断
        if capacity is None:
            # 使用标准化后的值
            pred_normalized = pred_result['predictions_normalized']

            # 判断拥堵等级
            congestion_levels = np.zeros(
                (predictions.shape[0], predictions.shape[1]),
                dtype=object
            )

            for t in range(predictions.shape[0]):
                for n in range(predictions.shape[1]):
                    val = pred_normalized[t, n, 0]
                    for level, (low, high) in CONGESTION_LEVELS.items():
                        if low <= val < high:
                            congestion_levels[t, n] = level
                            break
                    else:
                        congestion_levels[t, n] = 'severe'
        else:
            # 使用流量/容量比
            ratios = predictions[..., 0] / (capacity + 1e-8)
            congestion_levels = np.zeros_like(ratios, dtype=object)

            for level, (low, high) in CONGESTION_LEVELS.items():
                mask = (ratios >= low) & (ratios < high)
                congestion_levels[mask] = level

        # 统计各等级占比
        level_counts = {level: 0 for level in CONGESTION_LEVELS}
        total = congestion_levels.size

        for level in CONGESTION_LEVELS:
            level_counts[level] = np.sum(congestion_levels == level)

        level_ratios = {
            level: count / total * 100
            for level, count in level_counts.items()
        }

        return {
            'predictions': predictions,
            'congestion_levels': congestion_levels,
            'level_counts': level_counts,
            'level_ratios': level_ratios,
            'summary': self._generate_congestion_summary(congestion_levels)
        }

    def _generate_congestion_summary(self, congestion_levels: np.ndarray) -> str:
        """生成拥堵摘要"""
        pred_len, num_nodes = congestion_levels.shape

        # 统计最后一个时间步的拥堵情况
        last_step = congestion_levels[-1]

        severe_nodes = np.where(last_step == 'severe')[0]
        congested_nodes = np.where(last_step == 'congested')[0]

        summary_parts = []

        if len(severe_nodes) > 0:
            summary_parts.append(
                f"严重拥堵路段: {len(severe_nodes)}个 "
                f"(节点: {', '.join(map(str, severe_nodes[:5]))}...)"
            )

        if len(congested_nodes) > 0:
            summary_parts.append(
                f"拥堵路段: {len(congested_nodes)}个"
            )

        if len(summary_parts) == 0:
            summary_parts.append("交通状况良好，无明显拥堵")

        return "; ".join(summary_parts)

    def analyze_propagation(
        self,
        history_data: np.ndarray,
        threshold: float = 0.7
    ) -> dict:
        """
        分析拥堵传播

        Args:
            history_data: 历史数据
            threshold: 拥堵阈值

        Returns:
            传播分析结果
        """
        pred_result = self.predict(history_data)
        pred_normalized = pred_result['predictions_normalized'][..., 0]

        pred_len, num_nodes = pred_normalized.shape
        congestion_mask = pred_normalized > threshold

        # 分析每个时间步的拥堵变化
        propagation_events = []

        for t in range(1, pred_len):
            # 新增拥堵节点
            new_congested = congestion_mask[t] & ~congestion_mask[t-1]
            new_nodes = np.where(new_congested)[0]

            for node in new_nodes:
                propagation_events.append({
                    'time_step': t,
                    'node': int(node),
                    'type': 'new_congestion'
                })

            # 拥堵消散节点
            cleared = ~congestion_mask[t] & congestion_mask[t-1]
            cleared_nodes = np.where(cleared)[0]

            for node in cleared_nodes:
                propagation_events.append({
                    'time_step': t,
                    'node': int(node),
                    'type': 'cleared'
                })

        # 统计
        total_congested = congestion_mask.sum(axis=1)

        return {
            'propagation_events': propagation_events,
            'congestion_trend': total_congested.tolist(),
            'peak_congestion_time': int(np.argmax(total_congested)),
            'peak_congestion_nodes': int(np.max(total_congested))
        }


def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction Inference')
    parser.add_argument(
        '--checkpoint', type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--input', type=str,
        required=True,
        help='Input data file (numpy .npy or .npz)'
    )
    parser.add_argument(
        '--output', type=str,
        default='prediction_result.json',
        help='Output file path'
    )
    parser.add_argument(
        '--mode', type=str,
        choices=['predict', 'congestion', 'propagation'],
        default='predict',
        help='Prediction mode'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization'
    )
    parser.add_argument(
        '--gpu', type=int,
        default=0,
        help='GPU device ID'
    )

    args = parser.parse_args()

    # 初始化预测器
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    predictor = TrafficPredictor(
        args.checkpoint,
        args.config,
        device=device
    )

    # 加载输入数据
    print(f"Loading input data from {args.input}...")
    if args.input.endswith('.npz'):
        data = np.load(args.input)
        input_data = data[data.files[0]]
    else:
        input_data = np.load(args.input)

    print(f"Input shape: {input_data.shape}")

    # 执行预测
    if args.mode == 'predict':
        print("Running prediction...")
        result = predictor.predict(input_data)

        output_data = {
            'predictions': result['predictions'].tolist(),
            'num_nodes': result['num_nodes'],
            'pred_len': result['pred_len'],
            'timestamp': result['timestamp']
        }

    elif args.mode == 'congestion':
        print("Running congestion prediction...")
        result = predictor.predict_congestion(input_data)

        output_data = {
            'predictions': result['predictions'].tolist(),
            'congestion_levels': result['congestion_levels'].tolist(),
            'level_ratios': result['level_ratios'],
            'summary': result['summary']
        }

    elif args.mode == 'propagation':
        print("Running propagation analysis...")
        result = predictor.analyze_propagation(input_data)

        output_data = result

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")

    # 可视化
    if args.visualize and 'predictions' in result:
        print("Generating visualization...")
        preds = np.array(result['predictions'])

        # 选择第一个节点可视化
        if input_data.ndim == 3:
            history = input_data[:, 0, 0]
        else:
            history = input_data[:, 0]

        pred_values = preds[:, 0, 0] if preds.ndim == 3 else preds[:, 0]

        # 拼接历史和预测
        full_series = np.concatenate([history, pred_values])

        output_dir = os.path.dirname(args.output) or '.'
        visualize_predictions(
            true_values=history,
            predictions=pred_values,
            node_id=0,
            save_path=os.path.join(output_dir, 'prediction_visualization.png'),
            title='Traffic Flow Prediction'
        )

        print(f"Visualization saved to {output_dir}/prediction_visualization.png")

    # 打印摘要
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)

    if args.mode == 'predict':
        preds = np.array(result['predictions'])
        print(f"Prediction shape: {preds.shape}")
        print(f"Mean predicted value: {preds.mean():.4f}")
        print(f"Max predicted value: {preds.max():.4f}")
        print(f"Min predicted value: {preds.min():.4f}")

    elif args.mode == 'congestion':
        print(f"Summary: {result['summary']}")
        print("\nCongestion level distribution:")
        for level, ratio in result['level_ratios'].items():
            print(f"  {CONGESTION_NAMES.get(level, level)}: {ratio:.1f}%")

    elif args.mode == 'propagation':
        print(f"Peak congestion at time step: {result['peak_congestion_time']}")
        print(f"Peak congested nodes: {result['peak_congestion_nodes']}")
        print(f"Total propagation events: {len(result['propagation_events'])}")


if __name__ == '__main__':
    main()
