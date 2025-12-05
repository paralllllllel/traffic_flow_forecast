"""
数据集下载脚本

支持的公开数据集:
- METR-LA: 洛杉矶高速公路交通速度数据 (207个传感器, 4个月)
- PEMS-BAY: 旧金山湾区交通速度数据 (325个传感器, 6个月)
- PeMS04/PeMS08: 加州交通流量数据
"""
import os
import sys
import argparse
import urllib.request
import zipfile
import pickle

import numpy as np

# 数据集下载链接 (来自公开的GitHub仓库)
DATASET_URLS = {
    'METR-LA': {
        'data': 'https://github.com/liyaguang/DCRNN/raw/master/data/metr-la.h5',
        'adj': 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl',
        'description': 'Los Angeles highway traffic speed (207 sensors, 4 months, 5-min intervals)'
    },
    'PEMS-BAY': {
        'data': 'https://github.com/liyaguang/DCRNN/raw/master/data/pems-bay.h5',
        'adj': 'https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx_bay.pkl',
        'description': 'San Francisco Bay Area traffic speed (325 sensors, 6 months, 5-min intervals)'
    }
}

# 备用下载源
BACKUP_URLS = {
    'METR-LA': 'https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g',  # 提取码: s]hs
    'PEMS-BAY': 'https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g',
}


def download_file(url: str, save_path: str, desc: str = None):
    """下载文件"""
    if os.path.exists(save_path):
        print(f"文件已存在: {save_path}")
        return True

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"正在下载: {desc or url}")
    print(f"保存到: {save_path}")

    try:
        # 添加请求头避免被拒绝
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = response.headers.get('content-length')
            if total_size:
                total_size = int(total_size)

            downloaded = 0
            chunk_size = 8192

            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size})", end='')

        print("\n下载完成!")
        return True

    except Exception as e:
        print(f"\n下载失败: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def download_dataset(dataset_name: str, save_dir: str = 'data/raw'):
    """下载指定数据集"""
    if dataset_name not in DATASET_URLS:
        print(f"未知数据集: {dataset_name}")
        print(f"可用数据集: {list(DATASET_URLS.keys())}")
        return False

    urls = DATASET_URLS[dataset_name]
    print(f"\n{'='*50}")
    print(f"数据集: {dataset_name}")
    print(f"描述: {urls['description']}")
    print(f"{'='*50}\n")

    success = True

    # 下载数据文件
    data_filename = f"{dataset_name.lower().replace('-', '_')}.h5"
    data_path = os.path.join(save_dir, data_filename)
    if not download_file(urls['data'], data_path, "交通数据"):
        success = False

    # 下载邻接矩阵
    adj_filename = f"{dataset_name.lower().replace('-', '_')}_adj.pkl"
    adj_path = os.path.join(save_dir, adj_filename)
    if not download_file(urls['adj'], adj_path, "邻接矩阵"):
        success = False

    if success:
        print(f"\n数据集 {dataset_name} 下载完成!")
        print(f"数据文件: {data_path}")
        print(f"邻接矩阵: {adj_path}")
    else:
        print(f"\n数据集 {dataset_name} 下载失败!")
        print(f"请尝试手动下载:")
        print(f"  备用链接: {BACKUP_URLS.get(dataset_name, 'N/A')}")

    return success


def generate_synthetic_data(
    num_nodes: int = 50,
    num_timesteps: int = 12 * 24 * 30,  # 30天, 5分钟间隔
    num_features: int = 1,
    save_dir: str = 'data/raw'
):
    """
    生成模拟交通数据用于测试

    生成的数据包含:
    - 日周期性 (早晚高峰)
    - 周周期性 (工作日vs周末)
    - 随机噪声
    - 节点间的空间相关性
    """
    print(f"\n{'='*50}")
    print("生成模拟交通数据")
    print(f"节点数: {num_nodes}")
    print(f"时间步: {num_timesteps} (约{num_timesteps//(12*24)}天)")
    print(f"{'='*50}\n")

    os.makedirs(save_dir, exist_ok=True)

    # 时间索引
    t = np.arange(num_timesteps)

    # 基础流量
    base_flow = 50

    # 日周期 (每天288个时间步, 即5分钟间隔)
    daily_period = 12 * 24  # 288
    # 早高峰 (7-9点) 和晚高峰 (17-19点)
    hour_of_day = (t % daily_period) / 12  # 转换为小时
    morning_peak = 30 * np.exp(-((hour_of_day - 8) ** 2) / 2)
    evening_peak = 35 * np.exp(-((hour_of_day - 18) ** 2) / 2)
    daily_pattern = morning_peak + evening_peak

    # 周周期 (周末流量较低)
    weekly_period = daily_period * 7
    day_of_week = (t // daily_period) % 7
    weekend_factor = np.where((day_of_week == 5) | (day_of_week == 6), 0.7, 1.0)

    # 生成各节点数据
    data = np.zeros((num_timesteps, num_nodes, num_features))

    # 节点特征 (不同节点有不同的基础流量)
    node_base = np.random.uniform(0.8, 1.2, num_nodes)

    for n in range(num_nodes):
        # 基础模式
        flow = base_flow * node_base[n]
        flow = flow + daily_pattern * weekend_factor

        # 添加噪声
        noise = np.random.normal(0, 5, num_timesteps)
        flow = flow + noise

        # 确保非负
        flow = np.maximum(flow, 0)

        data[:, n, 0] = flow

    # 添加空间相关性 (相邻节点的流量相关)
    # 简单起见，使用移动平均
    for n in range(1, num_nodes):
        correlation = 0.3
        data[:, n, 0] = (1 - correlation) * data[:, n, 0] + correlation * data[:, n-1, 0]

    # 保存数据
    data_path = os.path.join(save_dir, 'synthetic_traffic.npz')
    np.savez(data_path, data=data)
    print(f"数据已保存: {data_path}")
    print(f"数据形状: {data.shape}")

    # 生成邻接矩阵 (简单的链式结构)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        adj_matrix[i, i] = 1  # 自环
        if i > 0:
            adj_matrix[i, i-1] = 1
            adj_matrix[i-1, i] = 1
        # 添加一些随机连接
        if i > 2 and np.random.random() > 0.7:
            j = np.random.randint(0, i-1)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    # 归一化
    d = np.sum(adj_matrix, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat = np.diag(d_inv_sqrt)
    adj_normalized = d_mat @ adj_matrix @ d_mat

    adj_path = os.path.join(save_dir, 'synthetic_adj.npy')
    np.save(adj_path, adj_normalized)
    print(f"邻接矩阵已保存: {adj_path}")
    print(f"邻接矩阵形状: {adj_normalized.shape}")

    # 更新配置文件提示
    print(f"\n请更新 configs/default.yaml:")
    print(f"  data_path: {data_path}")
    print(f"  adj_path: {adj_path}")
    print(f"  num_nodes: {num_nodes}")

    return data_path, adj_path


def main():
    parser = argparse.ArgumentParser(description='下载或生成交通数据集')
    parser.add_argument(
        '--dataset', type=str,
        choices=['METR-LA', 'PEMS-BAY', 'synthetic', 'all'],
        default='synthetic',
        help='要下载的数据集名称，或使用 synthetic 生成模拟数据'
    )
    parser.add_argument(
        '--save_dir', type=str,
        default='data/raw',
        help='数据保存目录'
    )
    parser.add_argument(
        '--num_nodes', type=int,
        default=50,
        help='模拟数据的节点数'
    )
    parser.add_argument(
        '--num_days', type=int,
        default=30,
        help='模拟数据的天数'
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    if args.dataset == 'synthetic':
        generate_synthetic_data(
            num_nodes=args.num_nodes,
            num_timesteps=12 * 24 * args.num_days,
            save_dir=args.save_dir
        )

    elif args.dataset == 'all':
        for dataset in DATASET_URLS:
            download_dataset(dataset, args.save_dir)
        generate_synthetic_data(save_dir=args.save_dir)

    else:
        download_dataset(args.dataset, args.save_dir)


if __name__ == '__main__':
    main()
