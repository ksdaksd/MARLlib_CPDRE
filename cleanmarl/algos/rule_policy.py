"""
RulePolicy / FixedPolicy - 非学习型基准策略

用于 0611(1).md 中不需要梯度训练的实验组:
  - fixed / rule_a1 : 固定 base-stock + 公平配给 (实验一 A1)
  - rule_a2         : 状态感知 base-stock + 公平配给 (实验一 A2)
  - rule_b1         : 公平规则基准 (实验二 B1)
  - rule_b3         : 规则互惠机制, 基于 chi_hat 判宽松/紧张 (实验二 B3)
  - rule_b4         : 规则版关系互惠 (用 mu 触发, 供对照)

设计要点:
1. 不做梯度训练, train() 只跑评估循环, 与学习型组别共享相同的 eval 轨迹和
   EpisodeLogger 流程, 使日志可比.
2. act() 是纯函数 (obs + 内部规则状态), 无 torch / 无梯度.
3. 煤企输出 2D 动作 (theta, lambda), 电企输出 1D (omega), 与学习型动作空间一致.
"""
from typing import Dict, Any
import numpy as np


class RulePolicy:
    """非学习型规则/固定策略。接口与 Trainer 对齐以便 train.py 统一调用。"""

    def __init__(self, env, config: Dict, variant: str = "fixed"):
        self.env = env
        self.config = config
        self.variant = variant.lower()
        self.num_agents = env.num_agents
        self.max_act_dim = int(env.max_act_dim)
        self.act_dims = list(env.act_dims)
        self.eval_episodes = config.get('eval_episodes', 64)
        self.current_step = 0
        self.current_epoch = 0

        # 复用 Trainer 的 EpisodeLogger 以保证日志 schema 与学习型一致.
        from cleanmarl.core.episode_logger import EpisodeLogger
        from cleanmarl.core.logger import Logger
        self.logger = Logger(
            log_dir=config.get('log_dir', './logs'),
            experiment_name=config.get('experiment_name', 'rule'),
        )
        self.logger.save_config(config)
        self.episode_logger = EpisodeLogger(
            log_dir=config.get('log_dir', './logs'),
            experiment_name=config.get('experiment_name', 'rule'),
            log_step_details=config.get('log_step_details', False),
        )

        # 规则内部状态: chi_hat 的移动平均, 用于 rule_b3 宽松/紧张判断.
        self._chi_ma = 0.0
        self._chi_ma_n = 0

    # ---------------- 规则动作产出 ----------------
    def _coal_action(self, season: int) -> np.ndarray:
        """返回煤企 2D 归一化动作 (theta_raw, lambda_raw)."""
        cfg = self.env.env.config
        variant = self.variant
        # 默认: theta 取季节档 (rule/fixed 基准不学 theta), lambda=1 (公平).
        theta = cfg.theta_offpeak if season == 0 else cfg.theta_peak
        lam = 1.0

        if variant in ("fixed", "rule_a1"):
            theta = cfg.theta_init if season == 0 else cfg.theta_peak
            lam = 1.0
        elif variant == "rule_a2":
            # 状态感知 base-stock: theta 跟季节, lambda 公平.
            theta = cfg.theta_offpeak if season == 0 else cfg.theta_peak
            lam = 1.0
        elif variant == "rule_b1":
            theta = cfg.theta_offpeak if season == 0 else cfg.theta_peak
            lam = 1.0
        elif variant == "rule_b3":
            # 规则互惠: 用 chi_hat 判宽松/紧张.
            # 宽松 (chi_ma<=0): U1 高覆盖已在 _power_action 处理; 煤企 lambda=1.
            # 紧张 (chi_ma>0): 煤企高保供权重 lambda=lambda_max.
            theta = cfg.theta_offpeak if season == 0 else cfg.theta_peak
            lam = float(cfg.lambda_max) if self._chi_ma > 0 else 1.0
        elif variant == "rule_b4":
            # 规则版关系互惠: 用煤企关系记忆 mu_c 触发紧张期保供.
            mu_c = self.env.env.mu_c_scalar
            theta = cfg.theta_offpeak if season == 0 else cfg.theta_peak
            lam = float(cfg.lambda_max) if (season == 1 and mu_c >= cfg.trust_threshold_c) else 1.0

        return self.env.env.coal_action(float(theta), float(lam))

    def _power_omega(self, idx: int, season: int) -> np.ndarray:
        cfg = self.env.env.config
        variant = self.variant
        is_u1 = (idx == 0)
        if variant in ("fixed", "rule_a1"):
            omega = cfg.omega_base
        elif variant in ("rule_a2", "rule_b1"):
            # 状态感知 base-stock: 季节性覆盖周期.
            omega = cfg.omega_base if season == 0 else 2.5
        elif variant == "rule_b3":
            # 规则互惠: 宽松期 U1 高覆盖, 其余 base-stock.
            if is_u1 and self._chi_ma <= 0:
                omega = cfg.omega_high
            else:
                omega = cfg.omega_base if season == 0 else 2.5
        elif variant == "rule_b4":
            mu_u = self.env.env.mu_u_scalar
            if is_u1 and season == 0 and mu_u >= cfg.trust_threshold_u:
                omega = cfg.omega_high
            else:
                omega = cfg.omega_base if season == 0 else 2.5
        else:
            omega = cfg.omega_base
        return self.env.env.normalized_action_from_omega(float(omega))

    def _build_action(self, obs: np.ndarray) -> np.ndarray:
        """从 obs 构造 (num_agents, max_act_dim) 动作矩阵."""
        season = int(self.env.env.current_season)
        acts = np.zeros((self.num_agents, self.max_act_dim), dtype=np.float32)
        acts[0, :int(self.act_dims[0])] = self._coal_action(season)
        for idx in range(self.num_agents - 1):
            a = self._power_omega(idx, season)
            acts[idx + 1, :int(self.act_dims[idx + 1])] = a
        return acts

    def _update_rule_state(self, step_info: Dict[str, float]):
        """更新规则内部状态 (chi 的移动平均)."""
        chi = step_info.get("chi", 0.0)
        if chi is None:
            return
        self._chi_ma_n += 1
        self._chi_ma += (float(chi) - self._chi_ma) / self._chi_ma_n

    # ---------------- 评估 (与学习型 evaluate 同流程) ----------------
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = self.eval_episodes
        rewards_all = []
        for ep_idx in range(num_episodes):
            obs = self.env.eval_reset(ep_idx)
            done = False
            ep_r = 0.0
            ep_step = 0
            self._chi_ma = 0.0
            self._chi_ma_n = 0
            self.episode_logger.reset_episode(self.num_agents)
            while not done:
                acts = self._build_action(obs)
                obs, r, done, step_info = self.env.step(acts)
                self._update_rule_state(step_info)
                ep_r += float(r.sum())
                ep_step += 1
                self.episode_logger.record_step(
                    step=ep_step, obs=obs, actions=acts, rewards=r,
                    step_info=step_info, agent_ids=self.env.agents,
                )
            rewards_all.append(ep_r)
            self.episode_logger.finalize_episode(episode_idx=ep_idx, phase="eval")
        return {
            'eval/reward_mean': float(np.mean(rewards_all)),
            'eval/reward_std': float(np.std(rewards_all)),
        }

    # ---------------- train: 评估循环 (无梯度) ----------------
    def collect_trajectories(self, num_steps: int):
        """规则策略无梯度训练; 为接口兼容执行一次评估并返回空指标."""
        pass

    def update(self) -> Dict[str, float]:
        return {}

    def train(self, total_timesteps: int, eval_freq: int = 10000):
        """规则策略的 train 只是评估循环: 在固定 eval 轨迹上跑评估并打印."""
        print(f"🚀 RulePolicy ({self.variant}): eval-only, {self.eval_episodes} fixed episodes")
        print("-" * 60)
        eval_metrics = self.evaluate()
        print("\n✅ Evaluation:")
        for k, v in eval_metrics.items():
            print(f"   {k}: {v:.4f}")
        self.current_step = total_timesteps
        self.logger.close()
        self.episode_logger.close()

    # ---------------- checkpoint stubs ----------------
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, ckpt: Dict[str, Any]):
        pass

    def save_checkpoint(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        import torch
        torch.save({'config': self.config, 'variant': self.variant, 'rule': True}, path)
        print(f"💾 RulePolicy checkpoint (metadata) saved to {path}")

    def load_checkpoint(self, path: str):
        pass
