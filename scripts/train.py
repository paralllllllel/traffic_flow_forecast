"""
交通流量预测模型训练脚本
"""
import os
import sys
import argparse
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# TensorBoard 可选
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficSTTransformer
from utils.data_loader import prepare_data, TrafficDataProcessor
from utils.metrics import (
    MaskedMAELoss, MaskedMSELoss, HuberLoss,
    compute_metrics, evaluate_by_horizon, MetricTracker
)
from utils.visualization import visualize_training_history


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    """获取计算设备"""
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def get_loss_function(config: dict) -> nn.Module:
    """获取损失函数"""
    loss_type = config['loss']['type']
    if loss_type == 'mae':
        return MaskedMAELoss()
    elif loss_type == 'mse':
        return MaskedMSELoss()
    elif loss_type == 'huber':
        return HuberLoss(delta=config['loss']['huber_delta'])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_scheduler(optimizer, config: dict, num_batches: int):
    """获取学习率调度器"""
    scheduler_type = config['training']['scheduler']
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        scheduler = None

    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    adj_matrix: torch.Tensor = None
) -> float:
    """训练一个epoch"""
    model.train()
    metric_tracker = MetricTracker()

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()

        # 前向传播
        pred, _ = model(x, adj_matrix)

        # 计算损失
        loss = criterion(pred, y)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()

        # 记录指标
        metric_tracker.update({'loss': loss.item()}, count=x.size(0))
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return metric_tracker.average()['loss']


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    processor: TrafficDataProcessor,
    config: dict,
    adj_matrix: torch.Tensor = None
) -> dict:
    """评估模型"""
    model.eval()
    metric_tracker = MetricTracker()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            pred, _ = model(x, adj_matrix)

            loss = criterion(pred, y)
            metric_tracker.update({'loss': loss.item()}, count=x.size(0))

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    # 合并所有预测结果
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 反标准化后计算指标
    preds_denorm = processor.inverse_normalize(all_preds.numpy())
    targets_denorm = processor.inverse_normalize(all_targets.numpy())

    # 计算整体指标
    overall_metrics = compute_metrics(preds_denorm, targets_denorm)

    # 按预测时长计算指标
    horizon_metrics = evaluate_by_horizon(
        preds_denorm,
        targets_denorm,
        horizons=config['evaluation']['horizons'],
        interval_minutes=config['evaluation']['interval_minutes']
    )

    return {
        'loss': metric_tracker.average()['loss'],
        'overall': overall_metrics,
        'by_horizon': horizon_metrics
    }


def train(config: dict):
    """完整训练流程"""
    # 设置随机种子
    set_seed(config['device']['seed'])

    # 获取设备
    device = get_device(config)

    # 创建保存目录
    os.makedirs(config['save']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['save']['log_dir'], exist_ok=True)

    # TensorBoard (可选)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(os.path.join(config['save']['log_dir'], timestamp))

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

    # 邻接矩阵转tensor
    if adj_matrix is not None:
        adj_matrix = torch.FloatTensor(adj_matrix).to(device)

    # 创建模型
    print("Creating model...")
    model_config = config['model'].copy()
    model_config['pred_len'] = config['data']['pred_len']

    model = TrafficSTTransformer(**model_config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # 损失函数和优化器
    criterion = get_loss_function(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    scheduler = get_scheduler(
        optimizer, config,
        len(dataloaders['train'])
    )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(
        config['save']['checkpoint_dir'],
        'best_model.pth'
    )

    print("Starting training...")
    start_time = time.time()

    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_epoch(
            model, dataloaders['train'],
            criterion, optimizer, device, config, adj_matrix
        )

        # 验证
        val_results = evaluate(
            model, dataloaders['val'],
            criterion, device, processor, config, adj_matrix
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['val_mae'].append(val_results['overall']['MAE'])
        history['val_rmse'].append(val_results['overall']['RMSE'])

        # TensorBoard日志
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_results['loss'], epoch)
            writer.add_scalar('Metrics/MAE', val_results['overall']['MAE'], epoch)
            writer.add_scalar('Metrics/RMSE', val_results['overall']['RMSE'], epoch)
            writer.add_scalar('Metrics/MAPE', val_results['overall']['MAPE'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results['loss'])
            else:
                scheduler.step()

        # 打印进度
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"MAE: {val_results['overall']['MAE']:.4f} | "
              f"RMSE: {val_results['overall']['RMSE']:.4f} | "
              f"MAPE: {val_results['overall']['MAPE']:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['loss'],
                'val_metrics': val_results['overall'],
                'config': config
            }, best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        # 定期保存检查点
        if epoch % config['save']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['save']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 测试集评估
    print("\nEvaluating on test set...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate(
        model, dataloaders['test'],
        criterion, device, processor, config, adj_matrix
    )

    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    print(f"Overall - MAE: {test_results['overall']['MAE']:.4f}, "
          f"RMSE: {test_results['overall']['RMSE']:.4f}, "
          f"MAPE: {test_results['overall']['MAPE']:.2f}%")

    print("\nBy Horizon:")
    for horizon, metrics in test_results['by_horizon'].items():
        print(f"  {horizon}: MAE={metrics['MAE']:.4f}, "
              f"RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

    # 保存训练历史图
    if config['visualization']['save_figures']:
        os.makedirs(config['visualization']['figure_dir'], exist_ok=True)
        visualize_training_history(
            history,
            save_path=os.path.join(
                config['visualization']['figure_dir'],
                'training_history.png'
            )
        )

    if writer is not None:
        writer.close()

    return model, history, test_results


def main():
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction Training')
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
        '--epochs', type=int,
        default=None,
        help='Override epochs in config'
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=None,
        help='Override batch size in config'
    )
    parser.add_argument(
        '--lr', type=float,
        default=None,
        help='Override learning rate in config'
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # 开始训练
    train(config)


if __name__ == '__main__':
    main()
