"""
Enhanced CPDRE callbacks for episode data collection.

This version provides two outputs:
1. Simple aggregate metrics CSV (for quick review)
2. Complete episode history NPZ (for detailed analysis)

Usage:
    from experiments.exp1.cpdre_callbacks_v2 import CPDREMetricsCallback
    algo.fit(..., callbacks=CPDREMetricsCallback)

The complete episode data can be loaded:
    import numpy as np
    data = np.load('episode_123_history.npz')
    print(data.files)  # List all arrays
    demand = data['demand']  # shape: (timesteps, num_agents)
"""

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:
    from ray.rllib.agents.callbacks import DefaultCallbacks

import csv
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class CPDREMetricsCallbackV2(DefaultCallbacks):
    """Enhanced callback for CPDRE episode data collection."""

    def __init__(self):
        super().__init__()
        self._episode_count = 0

    HIST_PREFIX = "cpdre_episode__"

    # Aggregated metrics (same as before, for progress.csv)
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

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Initialize user_data for this episode."""
        for key in self.STEP_KEYS:
            episode.user_data[key] = []
        # Store raw environment history reference
        episode.user_data["_env_history"] = None
        episode.user_data["_env_index"] = env_index

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """Collect step-level metrics from agent info."""
        info = None

        # Prefer coal_0 (common metrics are same for all agents)
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

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """
        At episode end:
        1. Aggregate metrics to custom_metrics and hist_data (for progress.csv)
        2. Extract and store environment's complete history
        """

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

        # Aggregate metrics (mean)
        for key in self.MEAN_KEYS:
            value = mean_value(key)
            row[key] = value
            episode.custom_metrics[key] = value

        # Aggregate metrics (sum for episode totals)
        for key in self.SUM_KEYS:
            value = sum_value(key)
            row[f"{key}_ep"] = value
            episode.custom_metrics[f"{key}_ep"] = value

        # Get complete environment episode metrics
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

        # Store aggregated row for CSV output in on_train_result
        try:
            for k, v in row.items():
                episode.hist_data[self.HIST_PREFIX + k] = [float(v)]
        except Exception:
            pass

        # ============================================================
        # NEW: Extract and store the complete environment history
        # ============================================================
        try:
            env = base_env.get_sub_environments()[env_index]
            if hasattr(env, "history") and env.history:
                # Store reference for on_train_result to process
                episode.user_data["_env_history"] = dict(env.history)
                # Also store episode_id for file naming
                episode.user_data["_episode_id"] = float(getattr(episode, "episode_id", -1))
        except Exception as e:
            print(f"[CPDRE] Warning: Failed to extract environment history: {e}")

    def on_train_result(self, *, trainer=None, algorithm=None, result: Dict[str, Any], **kwargs):
        """
        After each training iteration:
        1. Write aggregated metrics to CSV (existing behavior)
        2. Write complete episode histories to NPZ files (new)
        """

        # ============================================================
        # Part 1: Write aggregated metrics CSV (existing)
        # ============================================================
        hist_stats = result.get("hist_stats", {})
        if hist_stats:
            self._write_aggregated_metrics_csv(result, hist_stats)

    def _write_aggregated_metrics_csv(self, result: Dict[str, Any], hist_stats: Dict[str, Any]):
        """Write aggregated episode metrics to CSV."""
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
        file_exists = path.exists()

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)

        print(f"[CPDRE] Wrote {len(rows)} episode metrics to {path}")


class CPDRECompleteHistoryCallback(DefaultCallbacks):
    """
    Simplified callback focused on saving complete episode histories.

    Saves each episode's complete history dict from env.history to a separate file
    in numpy NPZ format for easy loading and analysis.
    """

    def __init__(self):
        super().__init__()
        self._episode_histories_written = set()

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Save complete episode history to file."""

        episode_id = getattr(episode, "episode_id", None)
        if episode_id is None:
            return

        # Avoid duplicates
        if episode_id in self._episode_histories_written:
            return

        try:
            env = base_env.get_sub_environments()[env_index]
            if not hasattr(env, "history") or not env.history:
                return

            # Determine output directory
            custom_logdir = os.environ.get("CPDRE_HISTORY_LOG_DIR", "").strip()
            if custom_logdir:
                logdir = Path(custom_logdir)
            else:
                logdir = Path("./cpdre_episode_histories")

            logdir.mkdir(parents=True, exist_ok=True)

            run_name = os.environ.get("CPDRE_RUN_NAME", "cpdre").strip()

            # Save as NPZ for easy numpy loading
            output_file = logdir / f"{run_name}_episode_{episode_id:06d}_history.npz"

            # Convert history lists/arrays to numpy arrays
            history_np = {}
            for key, values in env.history.items():
                try:
                    arr = np.asarray(values, dtype=np.float32)
                    history_np[key] = arr
                except Exception:
                    # Skip non-numeric data
                    pass

            # Save
            np.savez_compressed(str(output_file), **history_np)
            self._episode_histories_written.add(episode_id)

            print(
                f"[CPDRE] Saved episode {episode_id} history to {output_file.name} "
                f"with {len(history_np)} arrays"
            )

        except Exception as e:
            print(f"[CPDRE] Warning: Failed to save episode history: {e}")

    @staticmethod
    def load_episode_history(filepath: str) -> Dict[str, np.ndarray]:
        """
        Convenience function to load a saved episode history.

        Args:
            filepath: Path to the .npz file

        Returns:
            Dict of {array_name: numpy_array}

        Example:
            history = CPDRECompleteHistoryCallback.load_episode_history(
                "./cpdre_episode_histories/cpdre_episode_000001_history.npz"
            )
            demand = history["demand"]  # shape: (156, 3)
            orders = history["orders"]
        """
        data = np.load(filepath)
        return {k: data[k] for k in data.files}
