try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:
    from ray.rllib.agents.callbacks import DefaultCallbacks


class CPDREMetricsCallback(DefaultCallbacks):
    """
    Put CPDRE environment metrics into RLlib progress.csv.

    After this callback is active, progress.csv will contain columns like:
    custom_metrics/shortage_rate_mean
    custom_metrics/shortage_norm_mean
    custom_metrics/system_profit_ep_mean
    custom_metrics/coal_profit_norm_mean
    custom_metrics/g_u_mean
    custom_metrics/g_c_mean
    """

    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
    ):
        keys = [
            "shortage_rate",
            "shortage_norm",
            "system_profit",
            "coal_profit",
            "coal_profit_norm",
            "power_profit_total",
            "power_profit_norm_total",
            "jain",
            "fairness_penalty",
            "unsold",
            "g_u",
            "g_c",
            "mu_c",
            "mu_u",
            "total_demand",
            "total_order",
            "total_shipment",
            "total_shortage",
        ]
        for key in keys:
            episode.user_data[key] = []

    def on_episode_step(
        self,
        *,
        worker,
        base_env,
        episode,
        env_index,
        **kwargs,
    ):
        info = None

        # common system metrics are copied to all agents' info.
        # Prefer coal_0 to avoid repeated power-agent-specific fields.
        for agent_id in ["coal_0", "power_0", "power_1", "power_2"]:
            try:
                candidate = episode.last_info_for(agent_id)
                if candidate:
                    info = candidate
                    break
            except Exception:
                continue

        if not info:
            return

        for key in episode.user_data.keys():
            if key in info:
                episode.user_data[key].append(float(info[key]))

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
    ):
        def mean_value(key):
            values = episode.user_data.get(key, [])
            if not values:
                return float("nan")
            return float(sum(values) / len(values))

        def sum_value(key):
            values = episode.user_data.get(key, [])
            if not values:
                return float("nan")
            return float(sum(values))

        # step-level mean metrics
        for key in [
            "shortage_rate",
            "shortage_norm",
            "coal_profit_norm",
            "power_profit_norm_total",
            "jain",
            "fairness_penalty",
            "unsold",
            "g_u",
            "g_c",
            "mu_c",
            "mu_u",
            "total_demand",
            "total_order",
            "total_shipment",
            "total_shortage",
        ]:
            episode.custom_metrics[key] = mean_value(key)

        # episode-level accumulated metrics
        episode.custom_metrics["system_profit_ep"] = sum_value("system_profit")
        episode.custom_metrics["coal_profit_ep"] = sum_value("coal_profit")
        episode.custom_metrics["power_profit_total_ep"] = sum_value("power_profit_total")

        # If your environment exposes get_episode_metrics(), also collect it.
        try:
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "get_episode_metrics"):
                metrics = env.get_episode_metrics()
                for k, v in metrics.items():
                    if v == v:  # skip NaN
                        episode.custom_metrics[f"ep_{k}"] = float(v)
        except Exception:
            pass