import os
import random

import numpy as np

try:
    import torch
except Exception:
    torch = None


import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

print("PROJECT_ROOT =", PROJECT_ROOT)




from marllib import marl


# import marllib
# import marllib.envs.base_env as base_env
#
# print("USING marllib:", marllib.__file__)
# print("USING base_env:", base_env.__file__)
# print("ENV_REGISTRY:", base_env.ENV_REGISTRY.keys())


# 主动 import，确保 cpdre 环境在 MARLlib 的 ENV_REGISTRY 中完成注册
import marllib.envs.base_env.cpdre  # noqa: F401


def set_global_seed(seed: int) -> None:
    """
    同时固定 Python / NumPy / PyTorch 随机种子。
    这里主要用于烟雾训练的一致性；正式多 seed 实验仍然需要外层循环多个 seed。
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def make_cpdre_env(seed: int = 2026, episode_len: int = 20):
    """
    创建 0513 模型版 CPDRE 环境。

    注意：
    1. 这里是 MARLlib 训练烟雾测试，不是正式实验一。
    2. mechanism_mode="none" 表示无互惠，用于检查训练链路是否跑通。
    3. price_mode 默认采用 0513 主模型里的 seasonal；如果实验一需要固定煤价，
       可把 price_mode 改成 "fixed"。
    4. action_low=-1.0/action_high=1.0 对应 0513 文档里的归一化动作设定。
    """
    return marl.make_env(
        environment_name="cpdre",
        map_name="direct_1c3u",

        # Episode and system scale
        num_power=3,
        episode_len=episode_len,
        weeks_per_year=52,
        off_peak_weeks=26,
        lead_time=1,
        seed=seed,

        # Experiment mode
        mechanism_mode="none",
        allocation_mode="fair",
        demand_mode="deterministic",
        price_mode="seasonal",

        # Demand and supply
        demand_offpeak=1.0,
        demand_peak=1.4,
        demand_sigma=0.0,
        demand_ar1_rho=0.60,
        demand_corr=0.0,
        theta_offpeak=1.15,
        theta_peak=0.80,
        supply_sigma=0.0,

        # Price
        price_offpeak=1.0,
        price_peak=1.2,
        price_sigma=0.0,
        price_feedback_kappa=0.10,
        price_feedback_psi=0.70,
        price_min=0.6,
        price_max=1.8,

        # Inventory and order
        omega_min=1.0,
        omega_max=5.0,
        omega_base=2.0,
        omega_high=3.0,
        safe_inventory_weeks=1.5,
        q_max_weeks=6.0,
        lambda_d=0.30,

        # Coal allocation weight
        w_min=0.5,
        w_max=2.0,
        w_high=1.5,

        # Trust and reciprocity
        mu_c_init=0.50,
        mu_u_init=0.50,
        alpha_c=0.05,
        alpha_u=0.05,
        delta_c=0.005,
        delta_u=0.005,
        trust_threshold_c=0.50,
        trust_threshold_u=0.50,
        eta_c=0.30,
        eta_u=0.30,

        # Economic parameters
        coal_unit_cost=0.70,
        unsold_cost=0.10,
        power_unit_revenue=1.80,
        holding_cost=0.03,
        shortage_cost=5.00,

        # Reward shaping
        lambda_coal_shortage=2.00,
        lambda_coal_fairness=0.30,

        # Observation and numerical control
        obs_clip=10.0,
        reward_clip=None,
        initial_inventory_weeks=2.0,
        include_reciprocity_in_obs=True,

        # Fixed policy hooks
        fixed_power_omega=None,
        fixed_coal_weight=None,

        # Normalized action convention
        action_low=-1.0,
        action_high=1.0,
    )


def main():
    seed = 2026
    episode_len = 20
    timesteps_total = 2000

    set_global_seed(seed)

    print("Start MARLlib MAPPO training smoke test on CPDRE model0513...")
    print("seed:", seed)
    print("episode_len:", episode_len)
    print("timesteps_total:", timesteps_total)

    env = make_cpdre_env(seed=seed, episode_len=episode_len)

    # 烟雾测试先用 MAPPO + MLP，目的是确认环境能被训练链路跑通。
    # 正式实验一如果严格采用文档中的 GRU，可在正式训练脚本里改 core_arch。
    mappo = marl.algos.mappo(hyperparam_source="common")

    model = marl.build_model(
        env,
        mappo,
        {
            "core_arch": "mlp",
            "encode_layer": "64-64",
        },
    )

    mappo.fit(
        env,
        model,
        stop={"timesteps_total": timesteps_total},
        local_mode=True,
        num_workers=0,

        # 当前 agent 前缀为 coal_ 和 power_。
        # group 会按 team_prefix 生成煤企组和电企组策略，适合烟雾测试。
        # 正式异质实验可再比较 group / individual。
        share_policy="group",

        checkpoint_freq=1,
        checkpoint_end=True,
    )

    print("MARLlib MAPPO CPDRE model0513 training smoke test finished.")
    print("If the trial status is TERMINATED and exit code is 0, the train check passed.")


if __name__ == "__main__":
    main()
