"""
Compare RL algorithms (A2/IPPO, A3/MAPPO, A4/HAPPO, etc.) across multiple seeds.

For each (group, seed), the script:
  1. Locates the per-worker CSV files (e.g. A4_happo_seed43_w1.csv, _w2.csv, ...)
  2. Concatenates them into one run-level dataframe
  3. Aggregates across seeds per algorithm (mean and std over episodes)
  4. Plots one line per algorithm with std-shading

Outputs:
  - algo_comparison_dashboard.png   : key metrics in a grid
  - algo_comparison_summary.csv     : long-form table (algo × metric, mean ± std)
  - algo_comparison_summary_wide.csv: pivoted table for quick reading

Usage:
  # Compare A2/A3/A4 across seeds 42, 43, 44
  python experiments/exp1/plot_algo_comparison.py \
      --results_dir experiments/exp1/results/episode_metrics \
      --groups A2 A3 A4 \
      --seeds 42 43 44 \
      --smooth 5

  # Single-seed quick check (still works, std-shading will be 0)
  python experiments/exp1/plot_algo_comparison.py \
      --groups A3 A4 --seeds 43
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- Group ID → algorithm name mapping --------------------------------------
# Try importing from exp1_common, otherwise fallback to a hardcoded table.
try:
    from experiments.exp1.exp1_common import EXP1_GROUPS  # type: ignore
    GROUP_TO_ALGO_NAME = {
        g: cfg.get("algo") for g, cfg in EXP1_GROUPS.items() if cfg.get("algo")
    }
except Exception:
    GROUP_TO_ALGO_NAME = {
        "A2": "ippo", "A3": "mappo", "A4": "happo",
        "A7": "ippo", "A8": "mappo", "A9": "happo",
    }

# Consistent colors per algorithm so figures are easy to compare across runs.
ALGO_COLORS = {
    "ippo":  "#1f77b4",
    "mappo": "#ff7f0e",
    "happo": "#2ca02c",
}


# ---- Key metrics to plot ----------------------------------------------------
# Each tuple: (csv_column, display_label, direction "↑/↓ better")
KEY_METRICS = [
    ("system_profit_ep",         "System Profit (sum/ep)",           "↑"),
    ("coal_profit_ep",           "Coal Profit (sum/ep)",             "↑"),
    ("power_profit_total_ep",    "Power Profit (sum/ep)",            "↑"),
    ("coal_profit_norm",         "Coal Profit (normalized mean)",    "↑"),
    ("power_profit_norm_total",  "Power Profit (normalized mean)",   "↑"),
    ("shortage_rate",            "Shortage Rate (mean)",             "↓"),
    ("shortage_norm",            "Shortage (normalized mean)",       "↓"),
    ("jain",                     "Jain Fairness (mean)",             "↑"),
    ("fairness_penalty",         "Fairness Penalty (mean)",          "↓"),
    ("unsold",                   "Unsold Coal (mean)",               "↓"),
    ("ep_bullwhip_ratio",        "Bullwhip Ratio (per ep)",          "↓"),
    ("ep_order_cv",              "Order CV (per ep)",                "↓"),
]


# =============================================================================
# Data loading
# =============================================================================
def load_run(results_dir: Path, group_id: str, seed: int) -> Optional[pd.DataFrame]:
    """
    Load one (group, seed) run. Looks for either a merged CSV or per-worker CSVs.
    Returns a single DataFrame sorted by episode order, with `episode_index_global`
    reassigned to 0..N-1 for plotting consistency.
    """
    algo_name = GROUP_TO_ALGO_NAME.get(group_id)
    if algo_name is None:
        print(f"  ! Unknown group_id {group_id}, skipping")
        return None
    stem = f"{group_id}_{algo_name}_seed{seed}"

    # First: try merged main file
    main_csv = results_dir / f"{stem}.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
    else:
        # Fall back to per-worker files
        worker_files = sorted(results_dir.glob(f"{stem}_w*.csv"))
        if not worker_files:
            return None
        dfs = [pd.read_csv(wf) for wf in worker_files]
        df = pd.concat(dfs, ignore_index=True)

    # Sort by episode order for a stable timeline
    for k in ("episode_id", "episode_index_global", "local_ep_idx"):
        if k in df.columns:
            df = df.sort_values(k).reset_index(drop=True)
            break

    df["episode_index_global"] = np.arange(len(df))
    return df


def scan_runs(
    results_dir: Path,
    groups: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Discover and load all (group, seed) combinations in `results_dir`.
    Returns a nested dict: {group_id: {seed: dataframe, ...}, ...}
    """
    runs: Dict[str, Dict[int, pd.DataFrame]] = {}
    for group_id in groups:
        runs[group_id] = {}
        for seed in seeds:
            df = load_run(results_dir, group_id, seed)
            if df is not None:
                runs[group_id][seed] = df
                print(f"  Loaded {group_id} seed={seed}: {len(df)} episodes")
            else:
                print(f"  ! Missing  {group_id} seed={seed}")
    return runs


# =============================================================================
# Aggregation across seeds
# =============================================================================
def aggregate_across_seeds(
    seed_dfs: Dict[int, pd.DataFrame],
    metric: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Stack the chosen metric across all seeds along episode index.
    Different seeds may produce different numbers of episodes;
    we truncate to the common minimum length so all series align.
    """
    if not seed_dfs:
        return None

    arrays = []
    for df in seed_dfs.values():
        if metric in df.columns:
            arrays.append(df[metric].values)
    if not arrays:
        return None

    min_len = min(len(a) for a in arrays)
    if min_len == 0:
        return None

    matrix = np.array([a[:min_len] for a in arrays], dtype=np.float64)
    return {
        "x":       np.arange(min_len),
        "mean":    np.nanmean(matrix, axis=0),
        "std":     np.nanstd(matrix, axis=0),
        "n_seeds": len(arrays),
    }


def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean smoothing (window=1 → no smoothing)."""
    if window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().values


# =============================================================================
# Plotting
# =============================================================================
def plot_one_metric(
    ax: plt.Axes,
    runs: Dict[str, Dict[int, pd.DataFrame]],
    metric: str,
    smooth_w: int,
):
    """One subplot: one line per algorithm with std shading."""
    any_plotted = False
    for group_id, seed_dfs in runs.items():
        agg = aggregate_across_seeds(seed_dfs, metric)
        if agg is None:
            continue
        algo_name = GROUP_TO_ALGO_NAME.get(group_id, group_id)
        color = ALGO_COLORS.get(algo_name, "gray")
        label = f"{group_id}/{algo_name.upper()}  (n={agg['n_seeds']})"

        x = agg["x"]
        m = smooth_series(agg["mean"], smooth_w)
        s = smooth_series(agg["std"], smooth_w)

        ax.plot(x, m, label=label, color=color, linewidth=2)
        if agg["n_seeds"] > 1:
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        any_plotted = True

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("episode_index_global", fontsize=9)
    return any_plotted


def plot_dashboard(
    runs: Dict[str, Dict[int, pd.DataFrame]],
    output_dir: Path,
    smooth_w: int = 5,
):
    """Grid of key metrics, one subplot per metric, one line per algorithm."""
    n = len(KEY_METRICS)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    legend_handles = None
    legend_labels = None

    for i, (metric, label, direction) in enumerate(KEY_METRICS):
        ax = axes[i]
        ok = plot_one_metric(ax, runs, metric, smooth_w)
        title_color = "green" if direction == "↑" else "red"
        ax.set_title(label, fontsize=11)
        ax.text(
            0.02, 0.97, f"{direction} better",
            transform=ax.transAxes, fontsize=9, color=title_color,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=title_color, alpha=0.7),
        )
        if ok and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for i in range(n, len(axes)):
        axes[i].axis("off")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=min(len(legend_handles), 6),
            bbox_to_anchor=(0.5, 1.01), fontsize=11,
        )

    fig.suptitle(
        f"Algorithm Comparison Dashboard  (smooth={smooth_w})",
        fontsize=14, y=1.03,
    )
    plt.tight_layout()

    path = output_dir / "algo_comparison_dashboard.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_learning_curves(
    runs: Dict[str, Dict[int, pd.DataFrame]],
    output_dir: Path,
    smooth_w: int = 5,
):
    """
    Larger focused plot of 4 most informative metrics for paper-quality figures.
    """
    focus_metrics = [
        ("system_profit_ep",        "System Profit",       "↑"),
        ("shortage_rate",           "Shortage Rate",       "↓"),
        ("power_profit_norm_total", "Power Profit (norm)", "↑"),
        ("jain",                    "Jain Fairness",       "↑"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, (metric, label, direction) in enumerate(focus_metrics):
        ax = axes[i]
        ok = plot_one_metric(ax, runs, metric, smooth_w)
        ax.set_title(f"{label}  ({direction} better)", fontsize=13)
        ax.set_ylabel(metric, fontsize=10)
        if ok and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=min(len(legend_handles), 6),
            bbox_to_anchor=(0.5, 1.00), fontsize=11,
        )

    fig.suptitle(
        f"Key Learning Curves  (smooth={smooth_w})",
        fontsize=14, y=1.04,
    )
    plt.tight_layout()

    path = output_dir / "algo_comparison_learning_curves.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Summary tables
# =============================================================================
def write_summary_tables(
    runs: Dict[str, Dict[int, pd.DataFrame]],
    output_dir: Path,
    tail_frac: float = 0.25,
):
    """
    Compute mean ± std for the *last tail_frac* of each run's episodes.
    Inter-seed std reflects between-seed variability of the converged level.
    """
    rows = []
    for group_id, seed_dfs in runs.items():
        algo_name = GROUP_TO_ALGO_NAME.get(group_id, group_id)
        for metric, label, direction in KEY_METRICS:
            seed_means = []
            for _, df in seed_dfs.items():
                if metric not in df.columns:
                    continue
                n = len(df)
                tail = df.iloc[int(n * (1 - tail_frac)):]
                vals = tail[metric].dropna().values
                if len(vals) > 0:
                    seed_means.append(float(np.mean(vals)))
            if seed_means:
                rows.append({
                    "group_id":  group_id,
                    "algorithm": algo_name.upper(),
                    "metric":    metric,
                    "label":     label,
                    "direction": direction,
                    "n_seeds":   len(seed_means),
                    "mean":      float(np.mean(seed_means)),
                    "std":       float(np.std(seed_means)) if len(seed_means) > 1 else 0.0,
                })

    if not rows:
        print("No summary rows to write.")
        return

    df = pd.DataFrame(rows)

    # Long-form CSV
    long_path = output_dir / "algo_comparison_summary.csv"
    df.to_csv(long_path, index=False)
    print(f"Saved: {long_path}")

    # Wide-form CSV (one row per metric, columns per algo)
    pivot = df.pivot_table(
        index=["metric", "label", "direction"],
        columns="algorithm",
        values=["mean", "std"],
        aggfunc="first",
    )
    wide_path = output_dir / "algo_comparison_summary_wide.csv"
    pivot.to_csv(wide_path)
    print(f"Saved: {wide_path}")

    # Console print
    print()
    print("=" * 78)
    print(f"Last {int(tail_frac*100)}% episodes — mean ± std across seeds")
    print("=" * 78)
    for metric, label, direction in KEY_METRICS:
        sub = df[df["metric"] == metric]
        if len(sub) == 0:
            continue
        print(f"\n[{label}] {direction} better:")
        for _, row in sub.iterrows():
            print(
                f"  {row['algorithm']:<6s} "
                f"(n={int(row['n_seeds']):d}): "
                f"{row['mean']:>10.4f} ± {row['std']:>8.4f}"
            )


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare RL algorithms across runs (multi-seed, with std shading)."
    )
    parser.add_argument(
        "--results_dir", type=str,
        default="experiments/exp1/results/episode_metrics",
        help="Directory containing per-worker CSV files",
    )
    parser.add_argument(
        "--groups", nargs="+", default=["A2", "A3", "A4"],
        help="Group IDs to compare (e.g. A2 A3 A4)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, required=True,
        help="Seeds to aggregate over (e.g. 42 43 44)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save figures (default: plots_algo_compare_<groups>_<seeds>)",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Rolling window size for smoothing (default: 5)",
    )
    parser.add_argument(
        "--tail_frac", type=float, default=0.25,
        help="Fraction of trailing episodes used for the summary table (default: 0.25)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    if args.output_dir is None:
        groups_str = "_".join(args.groups)
        seeds_str = "_".join(map(str, args.seeds))
        output_dir = Path(f"plots_algo_compare_{groups_str}_seeds_{seeds_str}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir:  {results_dir.resolve()}")
    print(f"Groups:       {args.groups}")
    print(f"Seeds:        {args.seeds}")
    print(f"Output dir:   {output_dir.resolve()}")
    print()

    runs = scan_runs(results_dir, args.groups, args.seeds)

    total_loaded = sum(len(v) for v in runs.values())
    if total_loaded == 0:
        print("\n! No runs loaded. Check that the CSV files exist with the expected naming:")
        print("    <group>_<algo>_seed<n>_w<worker>.csv")
        print("  e.g. A4_happo_seed43_w1.csv")
        return

    print(f"\nTotal runs loaded: {total_loaded}")
    print()

    plot_dashboard(runs, output_dir, smooth_w=args.smooth)
    plot_learning_curves(runs, output_dir, smooth_w=args.smooth)
    write_summary_tables(runs, output_dir, tail_frac=args.tail_frac)

    print(f"\nDone. Open: {output_dir}/")


if __name__ == "__main__":
    main()
