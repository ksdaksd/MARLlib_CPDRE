"""
CleanMARL: A clean and maintainable MARL framework
Designed to solve HAPPO value network learning issues

v0.3.0:
- 新增 EpisodeLogger，记录每个 episode 的详细指标
- episode_summary.csv：系统级指标聚合（shortage_rate、profit、jain等）
- 可选步级详情：obs/action/reward/info 每步记录
"""

__version__ = "0.3.0"

from cleanmarl.algos.happo import HAPPO
from cleanmarl.algos.mappo import MAPPO
from cleanmarl.algos.ippo import IPPO
from cleanmarl.core.trainer import Trainer
from cleanmarl.core.episode_logger import EpisodeLogger

__all__ = ["HAPPO", "MAPPO", "IPPO", "Trainer", "EpisodeLogger"]
