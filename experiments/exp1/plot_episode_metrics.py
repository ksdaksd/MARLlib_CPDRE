"""
Generic plotting script for CPDRE episode metrics CSV.

This script automatically detects available metrics in the CSV file,
so it works even if you change indicators in the environment code.

Usage:
    # Plot a single run
    python plot_episode_metrics.py --csv results/episode_metrics/A3_mappo_seed42.csv

    # Compare multiple runs
    python plot_episode_metrics.py --csv run1.csv run2.csv run3.csv

    # Plot specific metrics only
    python plot_episode_metrics.py --csv data.csv --metrics shortage_rate jain system_profit_ep

    # Smooth with rolling window
    python plot_episode_metrics.py --csv data.csv --smooth 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Metric groups for organized plotting
METRIC_GROUPS: Dict[str, List[str]] = {
    "performance": ["shortage_rate", "shortage_norm", "fairness_penalty"],
    "fairness": ["jain", "fairness_penalty"],
    "profit": ["system_profit_ep", "coal_profit_ep", "power_profit_total_ep",
               "coal_profit_norm", "power_profit_norm_total"],
    "reciprocity": ["g_u", "g_c", "mu_c", "mu_u"],
    "operation": ["total_demand", "total_order", "total_shipment", "total_shortage", "unsold"],
}

# Metrics where "higher is better"
HIGHER_BETTER = {
    "system_profit_ep", "coal_profit_ep", "power_profit_total_ep",
    "coal_profit_norm", "power_profit_norm_total",
    "jain", "g_u", "g_c", "mu_c", "mu_u",
    "total_demand", "total_shipment",
}


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV. Auto-merges per-worker files if main CSV doesn't exist
    (e.g. A4_happo_seed43_w0.csv, _w1.csv, ...).
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        stem = csv_path.stem
        parent = csv_path.parent
        worker_files = sorted(parent.glob(f"{stem}_w*.csv"))
        if not worker_files:
            raise FileNotFoundError(
                f"No CSV at {csv_path} nor per-worker files {stem}_w*.csv in {parent}"
            )
        print(f"Merging {len(worker_files)} per-worker files")
        dfs = [pd.read_csv(wf) for wf in worker_files]
        df = pd.concat(dfs, ignore_index=True)

    sort_keys = []
    for k in ("episode_id", "episode_index_global", "local_ep_idx"):
        if k in df.columns:
            sort_keys.append(k)
            break
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)
    df["episode_index_global"] = np.arange(len(df))
    return df


def detect_metrics(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect available metric columns.

    Excludes meta columns like episode_id, training_iteration, etc.
    """
    exclude = {
        "training_iteration", "timesteps_total", "episodes_total",
        "episode_index_global", "episode_index_in_iteration",
        "worker_index", "env_index", "episode_id", "episode_len",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    """Apply rolling mean smoothing."""
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric_subplot(
    ax: plt.Axes,
    dfs: List[pd.DataFrame],
    labels: List[str],
    metric: str,
    x_col: str = "episode_index_global",
    smooth: int = 1,
    show_raw: bool = True,
):
    """Plot a single metric on the given axis."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))

    for df, label, color in zip(dfs, labels, colors):
        if metric not in df.columns:
            continue

        x = df[x_col].values if x_col in df.columns else np.arange(len(df))
        y = df[metric].values

        # Skip if all NaN
        if pd.isna(y).all():
            continue

        # Raw data (light)
        if show_raw and smooth > 1:
            ax.plot(x, y, alpha=0.2, color=color, linewidth=0.5)

        # Smoothed data
        y_smooth = smooth_series(df[metric], smooth).values
        ax.plot(x, y_smooth, label=label, color=color, linewidth=2)

    ax.set_title(metric, fontsize=11)
    ax.set_xlabel(x_col, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add arrow indicator for "higher is better"
    if metric in HIGHER_BETTER:
        ax.text(0.02, 0.98, "↑ better", transform=ax.transAxes,
                fontsize=8, color="green", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="green", alpha=0.7))
    else:
        ax.text(0.02, 0.98, "↓ better", transform=ax.transAxes,
                fontsize=8, color="red", verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))


def plot_dashboard(
    csv_paths: List[Path],
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    smooth: int = 5,
    x_col: str = "episode_index_global",
):
    """
    Generate a complete dashboard of plots from one or more CSV files.

    Args:
        csv_paths: List of CSV file paths
        output_dir: Where to save figures
        metrics: Specific metrics to plot (None = all auto-detected)
        smooth: Rolling window size for smoothing
        x_col: X-axis column ('episode_index_global' or 'timesteps_total')
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all CSVs
    dfs = [load_csv(p) for p in csv_paths]
    labels = [p.stem for p in csv_paths]

    # Auto-detect metrics if not specified
    if metrics is None:
        all_metrics = set()
        for df in dfs:
            all_metrics.update(detect_metrics(df))
        metrics = sorted(all_metrics)

    print(f"Plotting {len(metrics)} metrics from {len(csv_paths)} run(s)")
    print(f"  Smoothing window: {smooth}")
    print(f"  X-axis: {x_col}")
    print(f"  Output: {output_dir}")

    # ============================================================
    # Figure 1: Main dashboard (all metrics in a grid)
    # ============================================================
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        plot_metric_subplot(axes[i], dfs, labels, metric, x_col=x_col, smooth=smooth)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    # Single legend at the top
    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.02), fontsize=10)

    fig.suptitle(f"CPDRE Episode Metrics Dashboard (smooth={smooth})",
                 fontsize=14, y=1.05)
    plt.tight_layout()

    dashboard_path = output_dir / "dashboard.png"
    plt.savefig(dashboard_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dashboard_path}")

    # ============================================================
    # Figure 2: Grouped plots (by semantic category)
    # ============================================================
    for group_name, group_metrics in METRIC_GROUPS.items():
        # Filter to metrics that exist in data
        available = [m for m in group_metrics if any(m in df.columns for df in dfs)]
        if not available:
            continue

        n = len(available)
        n_cols_g = min(2, n)
        n_rows_g = (n + n_cols_g - 1) // n_cols_g

        fig, axes = plt.subplots(n_rows_g, n_cols_g, figsize=(7 * n_cols_g, 4 * n_rows_g))
        axes = np.atleast_1d(axes).flatten()

        for i, metric in enumerate(available):
            plot_metric_subplot(axes[i], dfs, labels, metric, x_col=x_col, smooth=smooth)

        for i in range(n, len(axes)):
            axes[i].axis("off")

        handles, leg_labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, leg_labels, loc="upper center", ncol=len(labels),
                       bbox_to_anchor=(0.5, 1.02), fontsize=10)

        fig.suptitle(f"{group_name.title()} Metrics (smooth={smooth})",
                     fontsize=13, y=1.03)
        plt.tight_layout()

        group_path = output_dir / f"group_{group_name}.png"
        plt.savefig(group_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {group_path}")

    # ============================================================
    # Figure 3: Learning curves (key metrics aggregated by iteration)
    # ============================================================
    if "training_iteration" in dfs[0].columns:
        key_metrics = [m for m in ["system_profit_ep", "shortage_rate", "jain",
                                    "coal_profit_norm", "power_profit_norm_total"]
                       if any(m in df.columns for df in dfs)]

        if key_metrics:
            n = len(key_metrics)
            n_cols_k = min(3, n)
            n_rows_k = (n + n_cols_k - 1) // n_cols_k

            fig, axes = plt.subplots(n_rows_k, n_cols_k, figsize=(6 * n_cols_k, 4 * n_rows_k))
            axes = np.atleast_1d(axes).flatten()

            for i, metric in enumerate(key_metrics):
                ax = axes[i]
                colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))

                for df, label, color in zip(dfs, labels, colors):
                    if metric not in df.columns:
                        continue
                    # Aggregate by training iteration (mean ± std)
                    grouped = df.groupby("training_iteration")[metric].agg(["mean", "std"])
                    x = grouped.index.values
                    mean = grouped["mean"].values
                    std = grouped["std"].fillna(0).values

                    ax.plot(x, mean, label=label, color=color, linewidth=2, marker="o", markersize=4)
                    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

                ax.set_title(f"{metric} (per iteration)", fontsize=11)
                ax.set_xlabel("training_iteration", fontsize=9)
                ax.grid(True, alpha=0.3)

                if metric in HIGHER_BETTER:
                    ax.text(0.02, 0.98, "↑ better", transform=ax.transAxes,
                            fontsize=8, color="green", verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="green", alpha=0.7))

            for i in range(n, len(axes)):
                axes[i].axis("off")

            handles, leg_labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, leg_labels, loc="upper center", ncol=len(labels),
                           bbox_to_anchor=(0.5, 1.02), fontsize=10)

            fig.suptitle("Learning Curves (mean ± std per iteration)", fontsize=13, y=1.03)
            plt.tight_layout()

            curves_path = output_dir / "learning_curves.png"
            plt.savefig(curves_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {curves_path}")

    # ============================================================
    # Print summary statistics
    # ============================================================
    print("\n" + "=" * 70)
    print("Summary Statistics (last 25% of episodes)")
    print("=" * 70)

    for df, label in zip(dfs, labels):
        n = len(df)
        tail = df.iloc[int(n * 0.75):]
        print(f"\n[{label}] (last {len(tail)}/{n} episodes)")
        for metric in metrics:
            if metric in tail.columns:
                vals = tail[metric].dropna()
                if len(vals) > 0:
                    print(f"  {metric:30s}: {vals.mean():>10.4f} ± {vals.std():>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot CPDRE episode metrics from CSV.")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="One or more CSV file paths (compares multiple runs)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./plots_<csv_stem>)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Specific metrics to plot (default: auto-detect all)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Rolling window for smoothing (default: 5)")
    parser.add_argument("--x_col", type=str, default="episode_index_global",
                        choices=["episode_index_global", "timesteps_total",
                                 "episodes_total", "training_iteration"],
                        help="X-axis column")

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    # Note: don't pre-check existence - load_csv() auto-merges per-worker files
    # if the main file doesn't exist (e.g. _w1.csv, _w2.csv, ...).

    if args.output_dir is None:
        if len(csv_paths) == 1:
            output_dir = Path(f"plots_{csv_paths[0].stem}")
        else:
            output_dir = Path("plots_comparison")
    else:
        output_dir = Path(args.output_dir)

    plot_dashboard(
        csv_paths=csv_paths,
        output_dir=output_dir,
        metrics=args.metrics,
        smooth=args.smooth,
        x_col=args.x_col,
    )

    print(f"\nDone. Open: {output_dir}/")


if __name__ == "__main__":
    main()
