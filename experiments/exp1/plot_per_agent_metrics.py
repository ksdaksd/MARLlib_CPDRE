"""
Per-agent plotting script for CPDRE.

After upgrading to cpdre_callbacks_v3.py (the per-agent version),
the CSV will contain columns like:
    demand_power_0, demand_power_1, demand_power_2,
    shipments_power_0, shipments_power_1, shipments_power_2,
    shortage_power_0, shortage_power_1, shortage_power_2,
    ...

This script auto-detects these patterns and groups them by base metric,
plotting one line per agent on each figure.

Usage:
    python plot_per_agent_metrics.py --csv results/episode_metrics/A3_mappo_seed42.csv
    python plot_per_agent_metrics.py --csv data.csv --smooth 5 --metrics shortage shipments
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Pattern to extract per-agent columns: e.g. "shortage_power_0" -> ("shortage", "power_0")
PER_AGENT_RE = re.compile(r"^(.+?)_(power_\d+|coal_\d+)(_ep)?$")


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV. If csv_path is e.g. results/.../A4_happo_seed43.csv but only
    per-worker files exist (A4_happo_seed43_w0.csv, _w1.csv, ...),
    concatenate them automatically.
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        # Look for per-worker files
        stem = csv_path.stem
        parent = csv_path.parent
        worker_files = sorted(parent.glob(f"{stem}_w*.csv"))
        if not worker_files:
            raise FileNotFoundError(
                f"No CSV found at {csv_path} nor per-worker files {stem}_w*.csv "
                f"in {parent}"
            )
        print(f"Merging {len(worker_files)} per-worker files:")
        for wf in worker_files:
            print(f"  {wf.name}")
        dfs = [pd.read_csv(wf) for wf in worker_files]
        df = pd.concat(dfs, ignore_index=True)

    # Sort to get a stable order. Use episode_id (RLlib's unique id) if present.
    sort_keys = []
    for k in ("episode_id", "episode_index_global", "local_ep_idx"):
        if k in df.columns:
            sort_keys.append(k)
            break
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)

    # Add a global episode index for plotting
    df["episode_index_global"] = np.arange(len(df))

    return df


def detect_per_agent_groups(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Scan column names and group per-agent columns by their base metric.

    Returns:
        {base_metric_name: {agent_label: full_column_name, ...}, ...}

    Example:
        {
            "shortage": {"power_0": "shortage_power_0", "power_1": "shortage_power_1", ...},
            "shipments_ep": {"power_0": "shipments_power_0_ep", ...},
        }
    """
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)

    for col in df.columns:
        m = PER_AGENT_RE.match(col)
        if not m:
            continue
        base, agent, ep_suffix = m.group(1), m.group(2), m.group(3) or ""
        # Skip the "_total" pseudo-agent (it's the system aggregate, not a real agent)
        if agent.startswith("total"):
            continue
        # Skip columns like "g_c" -> ("g", "c_0") false positives by checking it has "_0/_1/.."
        key = f"{base}{ep_suffix}"
        groups[key][agent] = col

    # Filter to groups that actually have >= 2 agents (real per-agent breakdown)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_per_agent_grid(
    df: pd.DataFrame,
    groups: Dict[str, Dict[str, str]],
    output_dir: Path,
    smooth_w: int = 5,
    x_col: str = "episode_index_global",
    csv_label: str = "",
    selected_metrics: Optional[List[str]] = None,
):
    """Plot all per-agent metric groups in one grid figure."""
    if selected_metrics:
        # Filter groups by user selection (matches if metric is a prefix of group name)
        filtered = {}
        for k, v in groups.items():
            if any(k == m or k.startswith(m + "_") or k.startswith(m) for m in selected_metrics):
                filtered[k] = v
        groups = filtered

    if not groups:
        print("No per-agent groups detected. Did you train with cpdre_callbacks_v3?")
        return

    print(f"Detected {len(groups)} per-agent metric groups:")
    for k, v in sorted(groups.items()):
        print(f"  {k:30s} -> {len(v)} agents: {sorted(v.keys())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort groups for consistent layout
    group_names = sorted(groups.keys())
    n = len(group_names)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # Use a fixed color for each agent across all subplots
    all_agents = sorted({a for ad in groups.values() for a in ad.keys()})
    agent_colors = {a: plt.cm.tab10(i) for i, a in enumerate(all_agents)}

    for i, group_name in enumerate(group_names):
        ax = axes[i]
        agent_dict = groups[group_name]

        x = df[x_col].values if x_col in df.columns else np.arange(len(df))

        for agent_label in sorted(agent_dict.keys()):
            col = agent_dict[agent_label]
            if col not in df.columns:
                continue
            y = df[col].values
            if pd.isna(y).all():
                continue

            color = agent_colors[agent_label]
            # Light raw line
            if smooth_w > 1:
                ax.plot(x, y, alpha=0.2, color=color, linewidth=0.6)
            # Bold smoothed line
            y_s = smooth(df[col], smooth_w).values
            ax.plot(x, y_s, label=agent_label, color=color, linewidth=2)

        # Highlight power_0 (reciprocal firm U1)
        ax.set_title(group_name, fontsize=11)
        ax.set_xlabel(x_col, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    title = f"Per-Agent Metrics (smooth={smooth_w})"
    if csv_label:
        title = f"{csv_label} - {title}"
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = output_dir / "per_agent_dashboard.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def plot_key_per_agent_separate(
    df: pd.DataFrame,
    groups: Dict[str, Dict[str, str]],
    output_dir: Path,
    smooth_w: int = 5,
    x_col: str = "episode_index_global",
):
    """
    Plot the most important per-agent metrics as larger, separate figures.

    These are: shortage, shipments, demand, fill_rate, inventory, omega, orders
    """
    KEY_METRICS = ["shortage", "shipments", "demand", "orders",
                   "fill_rate", "inventory", "omega", "served"]

    all_agents = sorted({a for ad in groups.values() for a in ad.keys()})
    agent_colors = {a: plt.cm.tab10(i) for i, a in enumerate(all_agents)}

    for metric in KEY_METRICS:
        # Look for both "metric" (mean) and "metric_ep" (sum) variants
        matched_groups = [(k, v) for k, v in groups.items() if k == metric or k == f"{metric}_ep"]
        if not matched_groups:
            continue

        for group_name, agent_dict in matched_groups:
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            x = df[x_col].values if x_col in df.columns else np.arange(len(df))

            for agent_label in sorted(agent_dict.keys()):
                col = agent_dict[agent_label]
                if col not in df.columns:
                    continue
                y = df[col].values
                if pd.isna(y).all():
                    continue

                color = agent_colors[agent_label]
                # Add a marker for power_0 since it's the reciprocal firm
                marker_kwargs = {"marker": "o", "markersize": 4, "markevery": max(1, len(x) // 20)} \
                    if agent_label == "power_0" else {}

                if smooth_w > 1:
                    ax.plot(x, y, alpha=0.2, color=color, linewidth=0.6)
                y_s = smooth(df[col], smooth_w).values
                label = f"{agent_label}" + (" (U1, reciprocal)" if agent_label == "power_0" else "")
                ax.plot(x, y_s, label=label, color=color, linewidth=2, **marker_kwargs)

            ax.set_title(f"{group_name} per agent", fontsize=13)
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(group_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            out_path = output_dir / f"per_agent_{group_name}.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-agent metrics from CPDRE CSV.")
    parser.add_argument("--csv", type=str, required=True, help="CSV file path")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--x_col", type=str, default="episode_index_global",
                        choices=["episode_index_global", "timesteps_total",
                                 "episodes_total", "training_iteration"])
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Base metric names to plot (e.g. shortage shipments demand)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    # Note: don't raise here - load_csv() handles per-worker file merging
    # (e.g. A4_happo_seed43.csv doesn't exist but _w1.csv, _w2.csv, _w3.csv do)

    df = load_csv(csv_path)
    groups = detect_per_agent_groups(df)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"plots_per_agent_{csv_path.stem}")

    if not groups:
        print("\n⚠ No per-agent columns detected.")
        print("Your CSV only has system-aggregated columns (sum across agents).")
        print("To get per-agent breakdown, you need to:")
        print("  1. Replace cpdre_callbacks.py with cpdre_callbacks_v3.py")
        print("  2. Re-run training")
        print("  3. The new CSV will have columns like shortage_power_0, shortage_power_1, ...")
        return

    plot_per_agent_grid(
        df, groups, output_dir,
        smooth_w=args.smooth, x_col=args.x_col,
        csv_label=csv_path.stem,
        selected_metrics=args.metrics,
    )
    plot_key_per_agent_separate(df, groups, output_dir, smooth_w=args.smooth, x_col=args.x_col)

    print(f"\nDone. Open: {output_dir}/")


if __name__ == "__main__":
    main()
