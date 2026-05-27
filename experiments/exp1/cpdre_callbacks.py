try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:
    from ray.rllib.agents.callbacks import DefaultCallbacks

import csv
import os
from pathlib import Path
from typing import Any, Dict, List


class CPDREMetricsCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self._episode_metrics_written = 0
    """
    1. 把 episode 内 step-level 指标聚合到 progress.csv 的 custom_metrics。
    2. 把每个 episode 的完整指标写入 cpdre_episode_metrics.csv。
    """

    HIST_PREFIX = "cpdre_episode__"

    STEP_KEYS = [
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

    MEAN_KEYS = [
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
    ]

    SUM_KEYS = [
        "system_profit",
        "coal_profit",
        "power_profit_total",
    ]

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
        for key in self.STEP_KEYS:
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
        # Prefer coal_0.
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

        for key in self.STEP_KEYS:
            if key in info:
                try:
                    episode.user_data[key].append(float(info[key]))
                except Exception:
                    pass

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
        def mean_value(key: str) -> float:
            values = episode.user_data.get(key, [])
            if not values:
                return float("nan")
            return float(sum(values) / len(values))

        def sum_value(key: str) -> float:
            values = episode.user_data.get(key, [])
            if not values:
                return float("nan")
            return float(sum(values))

        row: Dict[str, float] = {}

        row["worker_index"] = float(getattr(worker, "worker_index", -1))
        row["env_index"] = float(env_index)
        row["episode_id"] = float(getattr(episode, "episode_id", -1))
        row["episode_len"] = float(getattr(episode, "length", float("nan")))

        # 每步均值
        for key in self.MEAN_KEYS:
            value = mean_value(key)
            row[key] = value
            episode.custom_metrics[key] = value

        # episode 累计值
        for key in self.SUM_KEYS:
            value = sum_value(key)
            row[f"{key}_ep"] = value
            episode.custom_metrics[f"{key}_ep"] = value

        # 尽量从环境 get_episode_metrics() 里拿更完整的 episode 指标
        # 比如 bullwhip_ratio、order_cv、mean_fill_rate、inventory_violation_rate
        try:
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "get_episode_metrics"):
                metrics = env.get_episode_metrics()
                for k, v in metrics.items():
                    try:
                        v = float(v)
                        if v == v:  # skip NaN
                            row[f"ep_{k}"] = v
                            episode.custom_metrics[f"ep_{k}"] = v
                    except Exception:
                        continue
        except Exception:
            pass

        # 把每个 episode 的 row 塞到 hist_data。
        # on_train_result 会在 driver 端统一写 CSV，避免多 worker 同时写文件冲突。
        try:
            for k, v in row.items():
                episode.hist_data[self.HIST_PREFIX + k] = [float(v)]
        except Exception:
            pass

    def on_train_result(
        self,
        *,
        trainer=None,
        algorithm=None,
        result: Dict[str, Any],
        **kwargs,
    ):
        """
        每个 training iteration 结束后，把本轮采样到的 episode 明细写入 CSV。
        这个函数在 driver 端执行，比在 worker 里直接写文件更稳。
        """

        hist_stats = result.get("hist_stats", {})
        if not hist_stats:
            return

        prefix = self.HIST_PREFIX
        metric_keys = [k for k in hist_stats.keys() if k.startswith(prefix)]
        if not metric_keys:
            return

        metric_names = [k[len(prefix):] for k in metric_keys]

        n = 0
        for k in metric_keys:
            try:
                n = max(n, len(hist_stats[k]))
            except Exception:
                pass

        if n <= 0:
            return

        start = int(getattr(self, "_episode_metrics_written", 0))
        end = int(n)

        if start >= end:
            return

        rows = []
        for i in range(start, end):
            row = {
                "training_iteration": result.get("training_iteration", ""),
                "timesteps_total": result.get("timesteps_total", ""),
                "episodes_total": result.get("episodes_total", ""),
                "episode_index_global": i,
                "episode_index_in_iteration": i - start,
            }

            for metric_name in metric_names:
                arr = hist_stats.get(prefix + metric_name, [])
                if i < len(arr):
                    row[metric_name] = arr[i]
                else:
                    row[metric_name] = ""

            rows.append(row)

        self._episode_metrics_written = end

        if not rows:
            return

        custom_logdir = os.environ.get("CPDRE_EPISODE_LOG_DIR", "").strip()

        if custom_logdir:
            logdir = custom_logdir
        else:
            logdir = (
                    result.get("logdir")
                    or getattr(algorithm, "logdir", None)
                    or getattr(trainer, "logdir", None)
                    or "./cpdre_episode_logs"
            )

        logdir = Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        run_name = os.environ.get("CPDRE_RUN_NAME", "cpdre_episode_metrics").strip()

        path = logdir / f"{run_name}.csv"

        fieldnames = list(rows[0].keys())

        # 如果后续 iteration 里字段变多，补齐表头比较麻烦。
        # 所以这里每次都取当前已有字段，正常情况下字段是稳定的。
        file_exists = path.exists()

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)