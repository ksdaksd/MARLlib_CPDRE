"""
Episode Logger - 记录每个 episode 的详细指标

两个级别：
1. episode_summary.csv：每个 episode 一行，环境系统级指标聚合
2. episodes/ep_XXXX_steps.csv（可选）：单 episode 步级详情（obs/action/reward/指标）

设计要点：
- summary 采用固定 schema，避免动态 header 的脆弱性
- 步级详情按 episode 分文件，避免单文件过大
"""
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


# summary 中"取均值"的指标（rate / 状态量）
MEAN_FIELDS = [
    "price", "supply", "shortage_rate", "jain", "unsold",
    "mu_c", "mu_u", "g_u", "g_c",
    "shortage_norm", "fairness_penalty",
    "coal_profit_norm", "power_profit_norm_total",
    # Capacity-path & reciprocity mechanism metrics (model 4.5.1).
    "theta", "lambda1", "ramp_hit", "chi", "chi_hat", "G",
    "total_demand", "total_order", "total_shipment", "total_shortage",
    "own_demand_u1", "own_demand_u2", "own_demand_u3",
    "own_order_u1", "own_order_u2", "own_order_u3",
    "own_shipment_u1", "own_shipment_u2", "own_shipment_u3",
    "own_shortage_u1", "own_shortage_u2", "own_shortage_u3",
    "own_inventory_u1", "own_inventory_u2", "own_inventory_u3",
    "own_fill_rate_u1", "own_fill_rate_u2", "own_fill_rate_u3",
    "own_service_rate_u1", "own_service_rate_u2", "own_service_rate_u3",
]

# summary 中"取求和"的指标（利润类）
SUM_FIELDS = [
    "system_profit", "coal_profit", "power_profit_total",
    "power_profit_u1", "power_profit_u2", "power_profit_u3",
    "power_profit_norm_u1", "power_profit_norm_u2", "power_profit_norm_u3",
]

# summary 中"取标准差"的指标（episode 内波动性: 订货波动、产能路径波动）
# model 4.5.3 实验一报告订货波动与产能利用路径稳定性.
STD_FIELDS = [
    "total_order", "theta", "G", "shortage_rate",
]


class EpisodeLogger:
    """Episode 级别日志记录器"""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        log_step_details: bool = False,
    ):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_step_details = log_step_details

        # Episode summary CSV（固定 schema）
        self.summary_path = self.log_dir / "episode_summary.csv"
        self.summary_writer = None
        self.summary_file = None
        self.episode_count = 0

        # 步级详情目录
        if self.log_step_details:
            self.steps_dir = self.log_dir / "episodes"
            self.steps_dir.mkdir(parents=True, exist_ok=True)

        # 当前 episode 缓存
        self._step_infos: List[Dict[str, float]] = []
        self._step_details: List[Dict[str, Any]] = []
        self._agent_reward_sum: Optional[np.ndarray] = None

    def reset_episode(self, num_agents: int):
        """开始新 episode，清空缓存"""
        self._step_infos = []
        self._step_details = []
        self._agent_reward_sum = np.zeros(num_agents, dtype=np.float64)

    def record_step(
        self,
        step: int,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        step_info: Dict[str, float],
        agent_ids: Optional[List[str]] = None,
    ):
        """记录单步：聚合 step_info、累加 reward、（可选）记录详情"""
        # 缓存 step_info 用于 episode 聚合
        self._step_infos.append(step_info)

        # 累加每个 agent 的 reward
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        self._agent_reward_sum[:len(rewards)] += rewards

        # 可选：步级详情
        if self.log_step_details:
            if agent_ids is None:
                agent_ids = [f"agent_{i}" for i in range(len(rewards))]
            detail = {"step": step}
            for i, aid in enumerate(agent_ids):
                if isinstance(obs, np.ndarray):
                    obs_i = obs[i] if obs.ndim >= 2 else obs
                    for j, v in enumerate(np.atleast_1d(obs_i)):
                        detail[f"obs_{aid}_{j}"] = float(v)
                if isinstance(actions, np.ndarray):
                    act_i = actions[i] if actions.ndim >= 2 else actions
                    for j, v in enumerate(np.atleast_1d(act_i)):
                        detail[f"action_{aid}_{j}"] = float(v)
                detail[f"reward_{aid}"] = float(rewards[i])
            # 系统指标
            for k, v in step_info.items():
                if isinstance(v, (int, float, np.number)):
                    detail[f"info_{k}"] = float(v)
            self._step_details.append(detail)

    def finalize_episode(self, episode_idx: int, phase: str = "train") -> Dict[str, float]:
        """结束 episode：聚合指标，写入 summary，返回汇总 dict"""
        if not self._step_infos:
            return {}

        # 聚合每个字段
        summary: Dict[str, Any] = {
            "episode": episode_idx,
            "phase": phase,
            "episode_length": len(self._step_infos),
            "total_reward": float(self._agent_reward_sum.sum()),
        }

        # 每个 agent 的累计 reward
        for i, r in enumerate(self._agent_reward_sum):
            summary[f"reward_agent_{i}"] = float(r)

        # 聚合 MEAN_FIELDS / SUM_FIELDS
        for field in MEAN_FIELDS:
            vals = [s[field] for s in self._step_infos if field in s]
            if vals:
                summary[f"mean_{field}"] = float(np.mean(vals))

        for field in SUM_FIELDS:
            vals = [s[field] for s in self._step_infos if field in s]
            if vals:
                summary[f"sum_{field}"] = float(np.sum(vals))

        for field in STD_FIELDS:
            vals = [s[field] for s in self._step_infos if field in s]
            if vals:
                summary[f"std_{field}"] = float(np.std(vals))

        # 写入 summary CSV
        self._write_summary_row(summary)

        # 写入步级详情（可选）
        if self.log_step_details and self._step_details:
            self._write_steps_csv(episode_idx, phase)

        self.episode_count += 1
        return summary

    def _write_summary_row(self, row: Dict[str, Any]):
        """写入 summary CSV 一行（动态扩展 header，兼容不同字段）"""
        if self.summary_file is None:
            fields = sorted(row.keys())
            self.summary_file = open(self.summary_path, "w", newline="")
            self.summary_writer = csv.DictWriter(self.summary_file, fieldnames=fields)
            self.summary_writer.writeheader()
        else:
            existing = set(self.summary_writer.fieldnames)
            new = set(row.keys()) - existing
            if new:
                # 重建 writer 以容纳新字段（仅扩展 header 行为）
                all_fields = sorted(existing | new)
                self.summary_file.close()
                # 读取已有内容，用新 header 重写
                with open(self.summary_path, "r", newline="") as f:
                    existing_rows = list(csv.DictReader(f))
                self.summary_file = open(self.summary_path, "w", newline="")
                self.summary_writer = csv.DictWriter(
                    self.summary_file, fieldnames=all_fields
                )
                self.summary_writer.writeheader()
                for r in existing_rows:
                    full = {k: r.get(k, "") for k in all_fields}
                    self.summary_writer.writerow(full)

        full_row = {k: row.get(k, "") for k in self.summary_writer.fieldnames}
        self.summary_writer.writerow(full_row)
        self.summary_file.flush()

    def _write_steps_csv(self, episode_idx: int, phase: str):
        """写入单 episode 步级详情"""
        if not self._step_details:
            return
        steps_path = self.steps_dir / f"{phase}_ep_{episode_idx:05d}_steps.csv"
        headers = sorted(self._step_details[0].keys())
        with open(steps_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in self._step_details:
                writer.writerow({h: row.get(h, "") for h in headers})

    def close(self):
        if self.summary_file:
            self.summary_file.close()
            self.summary_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
