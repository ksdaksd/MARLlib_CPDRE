"""
Utilities for loading, analyzing, and visualizing CPDRE episode data.

Example usage:
    from experiments.exp1.episode_data_utils import EpisodeDataLoader

    loader = EpisodeDataLoader("./cpdre_episode_histories")

    # Load a single episode
    ep_data = loader.load_episode(episode_id=1)
    demand = ep_data['demand']  # shape: (timesteps, num_agents)

    # Load all episodes from an experiment
    episodes = loader.load_all_episodes()

    # Convert to pandas
    df = loader.to_dataframe(ep_data, agent_names=['coal_0', 'power_0', 'power_1', 'power_2'])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re


class EpisodeData:
    """Container for a single episode's complete history."""

    def __init__(self, episode_id: int, history: Dict[str, np.ndarray]):
        self.episode_id = episode_id
        self.history = history
        self.timesteps = len(next(iter(history.values())))

    def get_array(self, key: str) -> np.ndarray:
        """Get an array by key, with shape validation."""
        if key not in self.history:
            raise KeyError(f"Key '{key}' not found. Available: {list(self.history.keys())}")
        return self.history[key]

    def to_dataframe(self, agent_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert episode history to a pandas DataFrame for easy analysis.

        Args:
            agent_names: Names of agents (for multi-agent arrays). If None, uses indices.

        Returns:
            DataFrame with MultiIndex columns (array_name, agent/timestep)
        """
        rows = []

        for t in range(self.timesteps):
            row = {"timestep": t}

            for key, arr in self.history.items():
                arr = np.asarray(arr, dtype=np.float32)

                if arr.ndim == 1:
                    # Scalar per timestep
                    row[key] = arr[t]
                elif arr.ndim == 2:
                    # Array per timestep (e.g., per agent)
                    num_agents = arr.shape[1]
                    for i in range(num_agents):
                        agent_name = agent_names[i] if agent_names and i < len(agent_names) else f"agent_{i}"
                        row[f"{key}_{agent_name}"] = arr[t, i]
                elif arr.ndim == 3:
                    # 3D array (rare, flatten with indices)
                    shape = arr.shape[1:]
                    for idx in np.ndindex(shape):
                        col_name = f"{key}_{'_'.join(map(str, idx))}"
                        row[col_name] = arr[t, *idx]

            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("timestep", inplace=True)
        return df

    def get_episode_stats(self) -> Dict[str, float]:
        """Compute summary statistics for the episode."""
        stats = {"episode_id": self.episode_id, "timesteps": self.timesteps}

        for key, arr in self.history.items():
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                # Scalar time series
                stats[f"{key}_mean"] = float(np.nanmean(arr))
                stats[f"{key}_sum"] = float(np.nansum(arr))
                stats[f"{key}_std"] = float(np.nanstd(arr))
                stats[f"{key}_min"] = float(np.nanmin(arr))
                stats[f"{key}_max"] = float(np.nanmax(arr))
            elif arr.ndim == 2:
                # Per-agent time series
                for i in range(arr.shape[1]):
                    agent_name = f"agent_{i}"
                    stats[f"{key}_{agent_name}_mean"] = float(np.nanmean(arr[:, i]))
                    stats[f"{key}_{agent_name}_sum"] = float(np.nansum(arr[:, i]))

        return stats

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self.history.keys())[:5])
        if len(self.history) > 5:
            keys += f", ... ({len(self.history)} total)"
        return f"EpisodeData(id={self.episode_id}, t={self.timesteps}, keys=[{keys}])"


class EpisodeDataLoader:
    """Load and manage collections of episode history files."""

    def __init__(self, history_dir: str = "./cpdre_episode_histories"):
        self.history_dir = Path(history_dir)
        if not self.history_dir.exists():
            raise FileNotFoundError(f"History directory not found: {self.history_dir}")

    def list_episodes(self, pattern: Optional[str] = None) -> List[Tuple[int, Path]]:
        """
        List all episode history files, optionally filtered by pattern.

        Args:
            pattern: Regex pattern to match filenames (e.g., "episode_.*" or "A2_.*")

        Returns:
            Sorted list of (episode_id, file_path) tuples
        """
        files = []

        for f in self.history_dir.glob("*_episode_*_history.npz"):
            # Extract episode_id from filename: ..._episode_000001_history.npz
            match = re.search(r"_episode_(\d+)_history\.npz$", f.name)
            if match:
                episode_id = int(match.group(1))
                if pattern is None or re.search(pattern, f.name):
                    files.append((episode_id, f))

        files.sort(key=lambda x: x[0])
        return files

    def load_episode(self, episode_id: int, run_name: Optional[str] = None) -> EpisodeData:
        """
        Load a single episode by ID.

        Args:
            episode_id: Episode ID (e.g., 1, 2, ...)
            run_name: Optional run name filter (e.g., 'A2_mappo'). If None, finds first match.

        Returns:
            EpisodeData object
        """
        pattern = f"{run_name}_episode_{episode_id:06d}" if run_name else f"_episode_{episode_id:06d}"

        files = self.history_dir.glob(f"*{pattern}_history.npz")
        file = next(files, None)

        if file is None:
            raise FileNotFoundError(f"Episode {episode_id} not found in {self.history_dir}")

        data = np.load(str(file), allow_pickle=False)
        history = {k: data[k] for k in data.files}

        return EpisodeData(episode_id, history)

    def load_all_episodes(self, pattern: Optional[str] = None) -> List[EpisodeData]:
        """
        Load all episode histories.

        Args:
            pattern: Regex pattern to filter filenames

        Returns:
            List of EpisodeData objects
        """
        episodes = []

        for episode_id, filepath in self.list_episodes(pattern):
            try:
                data = np.load(str(filepath), allow_pickle=False)
                history = {k: data[k] for k in data.files}
                episodes.append(EpisodeData(episode_id, history))
            except Exception as e:
                print(f"Warning: Failed to load episode {episode_id}: {e}")

        return episodes

    def load_by_run_name(self, run_name: str) -> List[EpisodeData]:
        """
        Load all episodes from a specific training run.

        Args:
            run_name: Run name prefix (e.g., 'A2_mappo_seed42')

        Returns:
            List of EpisodeData objects
        """
        return self.load_all_episodes(pattern=f"^{re.escape(run_name)}_episode")

    def compare_episodes(
        self,
        episode_ids: List[int],
        metric_key: str,
        agent_idx: int = 0,
    ) -> pd.DataFrame:
        """
        Compare a metric across multiple episodes.

        Args:
            episode_ids: List of episode IDs to compare
            metric_key: Key in history (e.g., 'demand', 'system_profit')
            agent_idx: Agent index if metric is per-agent

        Returns:
            DataFrame with episodes as columns, timesteps as rows
        """
        data_dict = {}

        for eid in episode_ids:
            try:
                ep = self.load_episode(eid)
                arr = ep.get_array(metric_key)

                if arr.ndim == 1:
                    data_dict[f"episode_{eid}"] = arr
                elif arr.ndim == 2:
                    data_dict[f"episode_{eid}"] = arr[:, agent_idx]
            except Exception as e:
                print(f"Warning: Failed to load episode {eid}: {e}")

        if not data_dict:
            raise ValueError("No episodes loaded successfully")

        df = pd.DataFrame(data_dict)
        df.index.name = "timestep"
        return df

    def aggregate_episodes(self, run_name: str = None) -> pd.DataFrame:
        """
        Aggregate statistics across all episodes in a run.

        Args:
            run_name: Optional run name filter

        Returns:
            DataFrame with one row per episode, statistics as columns
        """
        episodes = self.load_by_run_name(run_name) if run_name else self.load_all_episodes()

        stats_list = [ep.get_episode_stats() for ep in episodes]
        df = pd.DataFrame(stats_list)
        df.set_index("episode_id", inplace=True)

        return df

    @staticmethod
    def concatenate_runs(
        history_dirs: List[str],
        run_names: List[str],
    ) -> pd.DataFrame:
        """
        Concatenate episode statistics from multiple runs for comparison.

        Args:
            history_dirs: List of history directories
            run_names: List of run names (must match history_dirs length)

        Returns:
            DataFrame with run_name as additional column
        """
        if len(history_dirs) != len(run_names):
            raise ValueError("history_dirs and run_names must have same length")

        dfs = []

        for hdir, rname in zip(history_dirs, run_names):
            loader = EpisodeDataLoader(hdir)
            df = loader.aggregate_episodes(rname)
            df["run_name"] = rname
            dfs.append(df)

        return pd.concat(dfs, ignore_index=False)


# =====================================================================
# Quick analysis functions
# =====================================================================


def quick_load(episode_id: int, history_dir: str = "./cpdre_episode_histories") -> EpisodeData:
    """Quick load shortcut."""
    loader = EpisodeDataLoader(history_dir)
    return loader.load_episode(episode_id)


def quick_plot_metric(
    episode_id: int,
    metric_key: str,
    agent_names: Optional[List[str]] = None,
    history_dir: str = "./cpdre_episode_histories",
):
    """
    Quick plot a metric from an episode.

    Args:
        episode_id: Episode ID
        metric_key: Key to plot (e.g., 'demand', 'shortage')
        agent_names: Optional agent names for legend
        history_dir: Path to history directory

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    ep = quick_load(episode_id, history_dir)
    arr = ep.get_array(metric_key)

    if arr.ndim == 1:
        plt.figure(figsize=(12, 5))
        plt.plot(arr, label=metric_key)
        plt.xlabel("Timestep")
        plt.ylabel(metric_key)
        plt.title(f"Episode {episode_id}: {metric_key}")
        plt.legend()
        plt.grid(True)

    elif arr.ndim == 2:
        plt.figure(figsize=(12, 5))
        for i in range(arr.shape[1]):
            agent_name = agent_names[i] if agent_names and i < len(agent_names) else f"agent_{i}"
            plt.plot(arr[:, i], label=agent_name)
        plt.xlabel("Timestep")
        plt.ylabel(metric_key)
        plt.title(f"Episode {episode_id}: {metric_key}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("EpisodeDataUtils loaded successfully")
    print("\nExample:")
    print("  loader = EpisodeDataLoader('./cpdre_episode_histories')")
    print("  ep = loader.load_episode(episode_id=1)")
    print("  df = ep.to_dataframe()")
    print("  stats = ep.get_episode_stats()")
