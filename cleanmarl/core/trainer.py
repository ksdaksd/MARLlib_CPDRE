"""
通用训练器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import numpy as np
from cleanmarl.core.logger import Logger
from cleanmarl.core.episode_logger import EpisodeLogger


class Trainer(ABC):
    """训练器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 训练日志（loss / vf_explained_var / reward 汇总）
        self.logger = Logger(
            log_dir=config.get('log_dir', './logs'),
            experiment_name=config.get('experiment_name', 'experiment')
        )
        self.logger.save_config(config)

        # Episode 详细日志（环境指标 + 每 agent reward + 可选步级详情）
        self.episode_logger = EpisodeLogger(
            log_dir=config.get('log_dir', './logs'),
            experiment_name=config.get('experiment_name', 'experiment'),
            log_step_details=config.get('log_step_details', False),
        )

        # 训练状态
        self.current_step = 0
        self.current_epoch = 0

    @abstractmethod
    def collect_trajectories(self, num_steps: int):
        """收集轨迹数据"""
        pass

    @abstractmethod
    def update(self):
        """执行一次参数更新"""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估当前策略"""
        pass

    def train(self, total_timesteps: int, eval_freq: int = 10000):
        """
        主训练循环

        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
        """
        print(f"🚀 Starting training for {total_timesteps} timesteps")
        print(f"   Device: {self.device}")
        print(f"   Log dir: {self.logger.log_dir}")
        print("-" * 60)

        while self.current_step < total_timesteps:
            # 收集数据
            self.collect_trajectories(self.config['rollout_length'])

            # 更新参数
            train_metrics = self.update()

            # 记录指标
            self.logger.log_dict(train_metrics, self.current_step)

            # 打印进度
            if self.current_epoch % self.config.get('log_freq', 1) == 0:
                self.logger.print_metrics(prefix=f"Epoch {self.current_epoch}")

            self.logger.flush()

            # 评估
            if self.current_step % eval_freq == 0:
                eval_metrics = self.evaluate()
                print(f"\n✅ Evaluation at step {self.current_step}:")
                for k, v in eval_metrics.items():
                    print(f"   {k}: {v:.4f}")

            self.current_epoch += 1

        # 最终评估 (用配置的 eval_episodes, 与中途评估一致, 共享固定轨迹)
        print("\n🎉 Training completed!")
        final_eval = self.evaluate(num_episodes=self.config.get('eval_episodes', 64))
        print("\n📊 Final evaluation:")
        for k, v in final_eval.items():
            print(f"   {k}: {v:.4f}")

        self.logger.close()
        self.episode_logger.close()

    def save_checkpoint(self, path: str):
        """保存检查点（含模型权重）"""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'config': self.config,
        }
        ckpt.update(self.state_dict())
        torch.save(ckpt, path)
        print(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint)
        print(f"📂 Checkpoint loaded from {path}")

    def state_dict(self) -> Dict[str, Any]:
        """子类覆盖：返回模型/优化器权重。默认空。"""
        return {}

    def load_state_dict(self, checkpoint: Dict[str, Any]):
        """子类覆盖：从checkpoint恢复权重。默认no-op。"""
        pass
