from marllib import marl

# 主动 import，确保 MARLlib 的 ENV_REGISTRY 完成注册
import marllib.envs.base_env.cpdre  # noqa: F401


def _unwrap_env(env_pack):
    """
    MARLlib 的 marl.make_env() 通常返回 (env_obj, env_config)。
    为了兼容不同版本，这里做一个保险拆包。
    """
    if isinstance(env_pack, tuple):
        return env_pack[0], env_pack[1] if len(env_pack) > 1 else {}
    return env_pack, {}


def main():
    print("Start MARLlib make_env test for CPDRE model0513...")

    env_pack = marl.make_env(
        environment_name="cpdre",
        map_name="direct_1c3u",

        # Episode and system scale
        num_power=3,
        episode_len=20,
        weeks_per_year=52,
        off_peak_weeks=26,
        lead_time=1,
        seed=2026,

        # Experiment mode
        # 这里是 make_env 烟雾测试，先用无互惠环境。
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

    env_obj, env_config = _unwrap_env(env_pack)

    print("marl.make_env CPDRE OK")
    print("env pack type:", type(env_pack))
    print("env object type:", type(env_obj))
    print("agents:", getattr(env_obj, "agent_ids", None))
    print("obs space:", getattr(env_obj, "observation_space", None))
    print("action space:", getattr(env_obj, "action_space", None))

    # 做一个极小 step 检查，确认新字段 mu_c/mu_u/g_u/g_c 能正常返回。
    obs = env_obj.reset()
    print("reset obs keys:", list(obs.keys()))

    action_dict = env_obj.base_stock_action_dict(omega=2.0)
    obs, rewards, dones, infos = env_obj.step(action_dict)

    coal_info = infos[env_obj.coal_agent_id]
    print("one-step rewards:", rewards)
    print("one-step coal info:", {
        "shortage_rate": coal_info["shortage_rate"],
        "system_profit": coal_info["system_profit"],
        "jain": coal_info["jain"],
        "mu_c": coal_info["mu_c"],
        "mu_u": coal_info["mu_u"],
        "g_u": coal_info["g_u"],
        "g_c": coal_info["g_c"],
        "unsold": coal_info["unsold"],
        "total_order": coal_info["total_order"],
        "total_shipment": coal_info["total_shipment"],
    })
    print("done after one step:", dones["__all__"])

    print("CPDRE make_env smoke test finished.")


if __name__ == "__main__":
    main()
