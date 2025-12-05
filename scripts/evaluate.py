"""
交通流量预测模型评估脚本
"""
import os
import sys
import argparse
import json

import numpy as np
import torch
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficSTTransformer
from utils.data_loader import prepare_data
from utils.metrics import (
    compute_metrics, evaluate_by_horizon,
    compute_congestion_metrics, MaskedMAELoss
)
from utils.visualization import (
    visualize_predictions,
    visualize_spatial_attention,
    visualize_temporal_attention,
    visualize_horizon_metrics
)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    model_config['pred_len'] = config.get('data', {}).get('pred_len', 12)

    model = TrafficSTTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    processor,
    device: torch.device,
    adj_matrix: torch.Tensor = None,
    config: dict = None
) -> dict:
    """
    完整评估模型

    Returns:
        评估结果字典
    """
    model.eval()
    criterion = MaskedMAELoss()

    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred, _ = model(x, adj_matrix)

            loss = criterion(pred, y)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    # 合并结果
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 反标准化
    preds_denorm = processor.inverse_normalize(all_preds)
    targets_denorm = processor.inverse_normalize(all_targets)

    # 计算指标
    results = {
        'loss': total_loss / num_batches,
        'num_samples': len(all_preds)
    }

    # 整体指标
    results['overall'] = compute_metrics(preds_denorm, targets_denorm)

    # 按时长评估
    horizons = config.get('evaluation', {}).get('horizons', [3, 6, 12])
    interval = config.get('evaluation', {}).get('interval_minutes', 5)
    results['by_horizon'] = evaluate_by_horizon(
        preds_denorm, targets_denorm,
        horizons=horizons,
        interval_minutes=interval
    )

    # 拥堵预测指标（在标准化数据上计算）
    results['congestion'] = compute_congestion_metrics(
        all_preds, all_targets, threshold=0.7
    )

    # 保存原始预测用于可视化
    results['predictions'] = preds_denorm
    results['targets'] = targets_denorm

    return results


def visualize_results(
    results: dict,
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    adj_matrix: torch.Tensor,
    config: dict,
    save_dir: str
):
    """生成可视化结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. 预测结果对比图
    print("Generating prediction plots...")
    preds = results['predictions']
    targets = results['targets']

    # 随机选择几个节点可视化
    num_nodes = preds.shape[2]
    sample_nodes = np.random.choice(num_nodes, min(5, num_nodes), replace=False)

    for node_id in sample_nodes:
        # 取前100个样本的预测
        node_preds = preds[:100, :, node_id, 0].flatten()
        node_targets = targets[:100, :, node_id, 0].flatten()

        visualize_predictions(
            node_targets, node_preds,
            node_id=node_id,
            save_path=os.path.join(save_dir, f'prediction_node_{node_id}.png'),
            title=f'Traffic Flow Prediction (Node {node_id})'
        )

    # 2. 按时长的指标对比图
    print("Generating horizon metrics plot...")
    visualize_horizon_metrics(
        results['by_horizon'],
        save_path=os.path.join(save_dir, 'horizon_metrics.png'),
        title='Metrics by Prediction Horizon'
    )

    # 3. 注意力权重可视化
    if config.get('visualization', {}).get('plot_attention', True):
        print("Generating attention maps...")

        # 获取一个batch的注意力权重
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            x = batch['x'][:1].to(device)  # 只取一个样本
            _, attention_maps = model(x, adj_matrix, return_attn=True)

            if attention_maps and len(attention_maps) > 0:
                # 可视化最后一层的注意力
                last_layer = attention_maps[-1]

                if last_layer['spatial'] is not None:
                    spatial_attn = last_layer['spatial'][0, 0].cpu().numpy()
                    visualize_spatial_attention(
                        spatial_attn,
                        save_path=os.path.join(save_dir, 'spatial_attention.png'),
                        title='Spatial Attention Weights (Last Layer)'
                    )

                if last_layer['temporal'] is not None:
                    temporal_attn = last_layer['temporal'][0, 0].cpu().numpy()
                    visualize_temporal_attention(
                        temporal_attn,
                        save_path=os.path.join(save_dir, 'temporal_attention.png'),
                        title='Temporal Attention Weights (Last Layer)'
                    )

    print(f"Visualizations saved to {save_dir}")


def print_results(results: dict):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nNumber of samples: {results['num_samples']}")
    print(f"Average loss: {results['loss']:.4f}")

    print("\n--- Overall Metrics ---")
    for metric, value in results['overall'].items():
        if metric == 'MAPE':
            print(f"  {metric}: {value:.2f}%")
        else:
            print(f"  {metric}: {value:.4f}")

    print("\n--- Metrics by Horizon ---")
    for horizon, metrics in results['by_horizon'].items():
        print(f"  {horizon}:")
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"    {metric}: {value:.2f}%")
            else:
                print(f"    {metric}: {value:.4f}")

    print("\n--- Congestion Prediction Metrics ---")
    for metric, value in results['congestion'].items():
        print(f"  {metric}: {value:.4f}")

    print("=" * 60)


def save_results(results: dict, save_path: str):
    """保存评估结果到JSON"""
    # 移除numpy数组（不能直接序列化）
    results_to_save = {
        'loss': results['loss'],
        'num_samples': results['num_samples'],
        'overall': results['overall'],
        'by_horizon': results['by_horizon'],
        'congestion': results['congestion']
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction Evaluation')
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
        '--data_path', type=str,
        default=None,
        help='Override data path in config'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--gpu', type=int,
        default=0,
        help='GPU device ID'
    )

    args = parser.parse_args()

    # 设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载配置和模型
    print("Loading model...")
    model, saved_config = load_model(args.checkpoint, device)

    # 合并配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        # 用保存的模型配置覆盖
        if saved_config:
            config['model'] = saved_config.get('model', config.get('model', {}))
    else:
        config = saved_config

    # 数据路径覆盖
    if args.data_path:
        config['data']['data_path'] = args.data_path

    # 准备数据
    print("Loading data...")
    dataloaders, adj_matrix, processor = prepare_data(
        data_path=config['data']['data_path'],
        adj_path=config['data'].get('adj_path'),
        hist_len=config['data']['hist_len'],
        pred_len=config['data']['pred_len'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        batch_size=config['training']['batch_size'],
        normalize_method=config['data']['normalize_method']
    )

    if adj_matrix is not None:
        adj_matrix = torch.FloatTensor(adj_matrix).to(device)

    # 评估
    results = evaluate_model(
        model, dataloaders['test'],
        processor, device,
        adj_matrix, config
    )

    # 打印结果
    print_results(results)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(
        results,
        os.path.join(args.output_dir, 'evaluation_results.json')
    )

    # 可视化
    if args.visualize:
        visualize_results(
            results, model,
            dataloaders['test'],
            device, adj_matrix, config,
            save_dir=os.path.join(args.output_dir, 'figures')
        )


if __name__ == '__main__':
    main()
