from custom_envs.coal_power_direct_reciprocity_env import CPDRE


def print_obs(obs):
    for agent_id, agent_obs in obs.items():
        print(agent_id, agent_obs["obs"].shape, agent_obs["obs"].dtype)


def run_one_episode(env, use_base_stock=True):
    obs = env.reset()
    print("reset obs:")
    print_obs(obs)

    done = False
    step = 0

    while not done:
        if use_base_stock:
            # 实验一启发式 sanity check：
            # 所有电企按 omega=2 形成目标库存覆盖周期，煤企公平分配
            action_dict = env.base_stock_action_dict(omega=2.0)
        else:
            # 随机动作 sanity check
            action_dict = {
                agent: env.action_space.sample()
                for agent in env.agent_ids
            }

        obs, rewards, dones, infos = env.step(action_dict)

        if step < 5:
            coal_info = infos[env.coal_agent_id]

            print("\nstep:", step)
            print("rewards:", rewards)
            print("coal info:", {
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

            # 顺手看一下互惠电企 U1 的局部结算信息
            u1_info = infos[env.power_agent_ids[0]]
            print("u1 info:", {
                "own_demand": u1_info["own_demand"],
                "own_d_tilde": u1_info["own_d_tilde"],
                "own_order": u1_info["own_order"],
                "own_shipment": u1_info["own_shipment"],
                "own_fair_shipment": u1_info["own_fair_shipment"],
                "own_base_u_shipment": u1_info["own_base_u_shipment"],
                "own_served": u1_info["own_served"],
                "own_shortage": u1_info["own_shortage"],
                "own_inventory": u1_info["own_inventory"],
                "own_fill_rate": u1_info["own_fill_rate"],
            })

        done = dones["__all__"]
        step += 1

    print("\nepisode finished")
    print("metrics:")
    print(env.get_episode_metrics())


def main():
    env = CPDRE({
        "mechanism_mode": "none",
        "allocation_mode": "fair",
        "demand_mode": "deterministic",
        "price_mode": "fixed",
        "episode_len": 20,
        "seed": 2026,
    })

    print("agents:", env.agent_ids)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    print("\n===== base-stock check =====")
    run_one_episode(env, use_base_stock=True)

    print("\n===== random-action check =====")
    run_one_episode(env, use_base_stock=False)

    print("\nCPDRE API check OK")


if __name__ == "__main__":
    main()