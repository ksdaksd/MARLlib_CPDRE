"""
Self-adaptive CPDRE callback v3 - tailored for Ray 1.8.0 multi-process workers.

Design choices for Ray 1.8:
1. Rollout workers are SEPARATE PROCESSES, not threads. Use per-worker files
   to avoid concurrent-write corruption: each worker writes its own CSV named
   `<run_name>_w<worker_idx>.csv`. The plotting script auto-merges them.
2. Use `get_unwrapped()` (Ray 1.8 API) to access env; fall back to other APIs.
3. NPZ files are per-episode and per-worker, so no collision.
4. Write IMMEDIATELY at episode end - no dependency on on_train_result
   (HAPPO custom trainer may skip it).
5. Auto-detect ALL fields from env.history, including per-agent breakdown.
"""

try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except Exception:
    from ray.rllib.agents.callbacks import DefaultCallbacks

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np


SUM_PATTERNS = ("profit", "_ep", "unsold_total", "_count", "_sum")


def _decide_aggregation(field_name: str) -> str:
    name = field_name.lower()
    for pat in SUM_PATTERNS:
        if pat in name:
            return "sum"
    return "mean"


class CPDREAdaptiveCallback(DefaultCallbacks):
    """Self-adaptive callback, multi-process safe via per-worker files."""

    def __init__(self):
        super().__init__()
        self._saved_episode_ids: Set[int] = set()
        self._known_fields: List[str] = []
        self._csv_path: Path = None  # type: ignore
        self._local_ep_counter = 0
        self._pid = os.getpid()
        print(f"[CPDRE-v3] Callback initialized in PID {self._pid}")

    # ---------- env resolution (Ray 1.8 compatibility) ----------
    @staticmethod
    def _resolve_env(base_env, env_index: int):
        """Return the underlying env, compatible with Ray 1.8 + newer APIs."""
        # Ray 1.8: BaseEnv has .envs and .get_unwrapped()
        if hasattr(base_env, "get_unwrapped"):
            try:
                envs = base_env.get_unwrapped()
                if envs:
                    return envs[env_index] if env_index < len(envs) else envs[0]
            except Exception:
                pass
        if hasattr(base_env, "envs"):
            try:
                envs = base_env.envs
                if envs:
                    return envs[env_index] if env_index < len(envs) else envs[0]
            except Exception:
                pass
        # Newer Ray
        if hasattr(base_env, "get_sub_environments"):
            try:
                envs = base_env.get_sub_environments()
                if envs:
                    return envs[env_index] if env_index < len(envs) else envs[0]
            except Exception:
                pass
        if hasattr(base_env, "env"):
            return base_env.env
        return None

    # ---------- log paths ----------
    def _setup_log_paths(self, worker_idx: int) -> None:
        if self._csv_path is not None:
            return
        custom_logdir = os.environ.get("CPDRE_EPISODE_LOG_DIR", "").strip()
        csv_dir = Path(custom_logdir) if custom_logdir else Path("./cpdre_episode_logs")
        csv_dir.mkdir(parents=True, exist_ok=True)

        run_name = os.environ.get("CPDRE_RUN_NAME", "cpdre").strip()
        # Per-worker filename to avoid multi-process write collisions.
        self._csv_path = csv_dir / f"{run_name}_w{worker_idx}.csv"
        self._run_name = run_name

        custom_npz = os.environ.get("CPDRE_HISTORY_LOG_DIR", "").strip()
        self._npz_dir = Path(custom_npz) if custom_npz else csv_dir.parent / "episode_histories"
        self._npz_dir.mkdir(parents=True, exist_ok=True)

        print(f"[CPDRE-v3] PID {self._pid} worker {worker_idx} writes to:")
        print(f"  CSV: {self._csv_path}")
        print(f"  NPZ dir: {self._npz_dir}")

    # ---------- episode hooks ----------
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        pass

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode_id = int(getattr(episode, "episode_id", -1))
        worker_idx = int(getattr(worker, "worker_index", 0))

        env = self._resolve_env(base_env, env_index)
        if env is None:
            print(f"[CPDRE-v3] PID {self._pid}: cannot resolve env "
                  f"(base_env={type(base_env).__name__}, has_envs={hasattr(base_env, 'envs')})")
            return

        if not hasattr(env, "history") or not env.history:
            print(f"[CPDRE-v3] PID {self._pid}: env.history empty for ep {episode_id}")
            return

        self._setup_log_paths(worker_idx)

        row = self._build_row(env, env.history, episode, worker_idx, env_index, episode_id)

        try:
            self._write_csv_row(row)
        except Exception as e:
            print(f"[CPDRE-v3] PID {self._pid}: CSV write failed for ep {episode_id}: {e}")
            import traceback
            traceback.print_exc()

        if episode_id not in self._saved_episode_ids:
            try:
                self._save_npz_history(env.history, episode_id, worker_idx)
                self._saved_episode_ids.add(episode_id)
            except Exception as e:
                print(f"[CPDRE-v3] PID {self._pid}: NPZ save failed: {e}")

    # ---------- row building ----------
    def _build_row(self, env, history, episode, worker_idx, env_index, episode_id):
        agent_labels = getattr(env, "power_agent_ids", None)

        row: Dict[str, float] = {
            "worker_index": float(worker_idx),
            "env_index": float(env_index),
            "episode_id": float(episode_id),
            "episode_len": float(getattr(episode, "length", float("nan"))),
            "local_ep_idx": float(self._local_ep_counter),
            "pid": float(self._pid),
        }
        self._local_ep_counter += 1

        for field_name, values in history.items():
            if not values:
                continue
            try:
                arr = np.asarray(values, dtype=np.float64)
            except Exception:
                continue
            if arr.size == 0:
                continue

            agg = _decide_aggregation(field_name)

            if arr.ndim >= 2:
                num_agents = arr.shape[1]
                arr_total = arr.sum(axis=tuple(range(1, arr.ndim)))
                if agg == "sum":
                    row[f"{field_name}_total_ep"] = float(np.nansum(arr_total))
                else:
                    row[f"{field_name}_total"] = float(np.nanmean(arr_total))

                for j in range(num_agents):
                    if agent_labels is not None and j < len(agent_labels):
                        label = str(agent_labels[j])
                    else:
                        label = f"power_{j}"
                    agent_arr = arr[:, j]
                    if agg == "sum":
                        row[f"{field_name}_{label}_ep"] = float(np.nansum(agent_arr))
                    else:
                        row[f"{field_name}_{label}"] = float(np.nanmean(agent_arr))
            else:
                if agg == "sum":
                    row[f"{field_name}_ep"] = float(np.nansum(arr))
                else:
                    row[field_name] = float(np.nanmean(arr))

        # episode-level summary
        if hasattr(env, "get_episode_metrics"):
            try:
                metrics = env.get_episode_metrics()
                for k, v in metrics.items():
                    try:
                        fv = float(v)
                        if fv == fv:
                            row[f"ep_{k}"] = fv
                    except Exception:
                        continue
            except Exception:
                pass

        # mirror to RLlib's progress.csv
        try:
            for k, v in row.items():
                episode.custom_metrics[k] = v
        except Exception:
            pass

        return row

    # ---------- writers ----------
    def _write_csv_row(self, row: Dict[str, float]) -> None:
        """Per-worker file: no concurrency, no locks needed."""
        first_write = not self._csv_path.exists()

        if first_write:
            self._known_fields = list(row.keys())
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._known_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
        else:
            # If new keys appeared, add them (rare; happens if history changes per ep)
            new_keys = [k for k in row.keys() if k not in self._known_fields]
            if new_keys:
                self._known_fields = self._known_fields + new_keys
            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._known_fields, extrasaction="ignore")
                full_row = {k: row.get(k, "") for k in self._known_fields}
                writer.writerow(full_row)

        # Throttle logs.
        if self._local_ep_counter <= 3 or self._local_ep_counter % 10 == 0:
            print(f"[CPDRE-v3] PID {self._pid} wrote ep {int(row['episode_id'])} "
                  f"(local #{int(row['local_ep_idx'])}) -> {self._csv_path.name}")

    def _save_npz_history(self, history, episode_id: int, worker_idx: int) -> None:
        output_file = self._npz_dir / f"{self._run_name}_w{worker_idx}_ep{episode_id:012d}.npz"
        if output_file.exists():
            return

        history_np: Dict[str, np.ndarray] = {}
        for key, values in history.items():
            try:
                history_np[key] = np.asarray(values, dtype=np.float32)
            except Exception:
                continue

        np.savez_compressed(str(output_file), **history_np)

        if self._local_ep_counter <= 3 or self._local_ep_counter % 10 == 0:
            print(f"[CPDRE-v3] PID {self._pid} saved {output_file.name} "
                  f"({len(history_np)} arrays)")

    # ---------- convenience ----------
    @staticmethod
    def load_episode_history(filepath: str) -> Dict[str, np.ndarray]:
        data = np.load(filepath)
        return {k: data[k] for k in data.files}


# Backwards-compatible aliases
CPDREMetricsCallbackV2 = CPDREAdaptiveCallback
CPDRECompleteHistoryCallback = CPDREAdaptiveCallback
CPDREMetricsCallback = CPDREAdaptiveCallback

__all__ = [
    "CPDREAdaptiveCallback",
    "CPDREMetricsCallbackV2",
    "CPDRECompleteHistoryCallback",
    "CPDREMetricsCallback",
]
