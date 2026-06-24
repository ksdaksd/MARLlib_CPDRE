"""
冒烟测试 - 用极小步数验证CleanMARL流程能跑通（形状、梯度、指标）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from cleanmarl.algos import HAPPO, MAPPO
from cleanmarl.envs.cpdre_wrapper import make_cpdre_env


def main():
    env_args = {
        "map_name": "direct_1c3u_b4_s42",
        "num_power": 3, "num_reciprocal": 1, "episode_len": 156, "weeks_per_year": 52,
        "off_peak_weeks": 26, "lead_time": 1, "seed": 42,
        "mechanism_mode": "b4", "allocation_mode": "fair",
        "demand_mode": "deterministic", "price_mode": "seasonal",
        "demand_offpeak": 1.0, "demand_peak": 1.4, "demand_season_amp": 0.20,
        "demand_sigma": 0.15, "demand_ar1_rho": 0.60, "demand_corr": 0.0,
        "theta_min": 0.70, "theta_max": 1.20, "theta_init": 1.0, "rho_max": 0.05,
        "theta_offpeak": 1.15, "theta_peak": 0.80, "supply_sigma": 0.0,
        "price_offpeak": 1.0, "price_peak": 1.2, "price_sigma": 0.0,
        "price_feedback_kappa": 0.10, "price_feedback_psi": 0.70,
        "lambda_max": 2.0, "c_rep": 1.60, "p_base": 1.00,
    }
    config = {
        "episode_len": 156,
        "rollout_length": 312,   # 2 episodes
        "num_sgd_iter": 2,
        "gamma": 0.95, "gae_lambda": 0.95, "clip_param": 0.2,
        "value_clip_param": 10.0, "entropy_coef": 0.01, "value_loss_coef": 0.5,
        "max_grad_norm": 0.5, "actor_lr": 1e-4, "critic_lr": 1e-4,
        "hidden_dim": 128, "eval_episodes": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "./logs/cleanmarl_smoke",
        "experiment_name": "smoke_happo",
    }

    print("=== 创建环境 ===")
    env = make_cpdre_env(env_args)
    print(f"agents={env.num_agents}, obs_dim={env.observation_space.shape[0]}, "
          f"state_dim={env.state_space.shape[0]}, act_dims={env.act_dims}, "
          f"max_act_dim={env.max_act_dim}, continuous={env.continuous}")

    print("\n=== 初始化HAPPO ===")
    trainer = HAPPO(env, config)

    print("\n=== 采样 ===")
    trainer.collect_trajectories(config['rollout_length'])
    for k, v in trainer._buf.items():
        print(f"  {k}: {v.shape}")

    print("\n=== 更新 ===")
    metrics = trainer.update()
    print("  训练指标:")
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.4f}")

    print("\n=== 再跑2个epoch验证稳定性 ===")
    for ep in range(2):
        trainer.collect_trajectories(config['rollout_length'])
        m = trainer.update()
        print(f"  epoch {ep}: reward={m.get('episode_reward_mean', float('nan')):.2f}, "
              f"vf_ev={m['vf_explained_var']:.3f}, "
              f"ploss={m['policy_loss']:.3f}, vloss={m['value_loss']:.3f}")

    print("\n✅ 冒烟测试通过！")
    env.close()


if __name__ == "__main__":
    main()
