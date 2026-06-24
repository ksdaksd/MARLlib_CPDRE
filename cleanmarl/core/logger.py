"""
日志记录器 - 跟踪训练指标
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np


class Logger:
    """训练日志记录器"""

    def __init__(self, log_dir: str, experiment_name: str):
        # 与 EpisodeLogger 一致: 一个 run 的所有产物归到 log_dir/experiment_name/ 子目录.
        self.log_dir = Path(log_dir) / experiment_name
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV文件用于记录每个iteration的指标
        self.csv_path = self.log_dir / "progress.csv"
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers = []

        # 当前epoch的指标缓存
        self.current_metrics = {}

    def log_scalar(self, key: str, value: float, step: int):
        """记录标量值"""
        self.current_metrics[key] = value
        self.current_metrics['step'] = step

    def log_dict(self, metrics: Dict[str, Any], step: int):
        """记录一个字典的指标"""
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.current_metrics[key] = float(value)
            elif isinstance(value, torch.Tensor):
                self.current_metrics[key] = value.item()
        self.current_metrics['step'] = step

    def flush(self):
        """将当前指标写入CSV"""
        if not self.current_metrics:
            return

        # 初始化CSV文件
        if self.csv_writer is None:
            self.csv_headers = sorted(self.current_metrics.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_headers)
            self.csv_writer.writeheader()

        # 写入当前行
        row = {k: self.current_metrics.get(k, '') for k in self.csv_headers}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # 清空缓存
        self.current_metrics = {}

    def save_config(self, config: Dict[str, Any]):
        """保存配置"""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def close(self):
        """关闭日志文件"""
        if self.csv_file is not None:
            self.csv_file.close()

    def print_metrics(self, prefix: str = ""):
        """打印当前指标"""
        if not self.current_metrics:
            return

        step = self.current_metrics.get('step', 0)
        print(f"\n{prefix} Step {step}")
        print("=" * 60)

        # 分组打印
        reward_keys = [k for k in self.current_metrics.keys() if 'reward' in k.lower()]
        loss_keys = [k for k in self.current_metrics.keys() if 'loss' in k.lower()]
        vf_keys = [k for k in self.current_metrics.keys() if 'vf_' in k.lower() or 'explained' in k.lower()]
        other_keys = [k for k in self.current_metrics.keys()
                      if k not in reward_keys + loss_keys + vf_keys and k != 'step']

        if reward_keys:
            print("\n📊 Rewards:")
            for k in reward_keys:
                print(f"  {k}: {self.current_metrics[k]:.4f}")

        if vf_keys:
            print("\n🎯 Value Function:")
            for k in vf_keys:
                print(f"  {k}: {self.current_metrics[k]:.4f}")

        if loss_keys:
            print("\n📉 Losses:")
            for k in loss_keys:
                print(f"  {k}: {self.current_metrics[k]:.4f}")

        if other_keys:
            print("\n📈 Other:")
            for k in other_keys:
                val = self.current_metrics[k]
                if isinstance(val, float):
                    print(f"  {k}: {val:.4f}")
                else:
                    print(f"  {k}: {val}")

        print("=" * 60)
