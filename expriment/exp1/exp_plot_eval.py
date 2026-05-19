from __future__ import annotations

"""
exp_plot_eval.py

Purpose
-------
Plot CPDRE evaluation results from:
    eval_summary.csv
    eval_timeseries.csv

Where to put it
---------------
Put this file under:
    MARLlib/exp_20260513_model_direct/exp_plot_eval.py

Run from MARLlib project root:
    python -m exp_20260513_model_direct.exp_plot_eval \
      --eval_dir exp_20260513_model_direct/eval_outputs \
      --groups A0,A1,A5,A6

This script draws:
1. inventory trajectory
2. order-to-demand ratio trajectory
3. shortage trajectory
4. summary bar charts: system profit, average system cost, shortage rate, bullwhip
5. lead-time sensitivity, if lead_time varies
6. parameter cost bars, if param_name/param_value exist
7. mechanism comparison table, if B0-B3 exist
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_csv_list(x: str) -> Optional[List[str]]:
    if not x:
        return None
    return [item.strip() for item in x.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = eval_dir / "eval_summary.csv"
    ts_path = eval_dir / "eval_timeseries.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing {ts_path}")
    return pd.read_csv(summary_path), pd.read_csv(ts_path)


def filter_groups(df: pd.DataFrame, groups: Optional[List[str]]) -> pd.DataFrame:
    if groups is None:
        return df
    return df[df["group_id"].isin(groups)].copy()


def label_col(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "group_id"


def plot_summary_bar(summary: pd.DataFrame, metric: str, out_dir: Path, title: str = "") -> None:
    if metric not in summary.columns:
        print(f"[skip] {metric} not in eval_summary.csv")
        return

    gcol = label_col(summary)
    stat = summary.groupby(["group_id", gcol], as_index=False)[metric].agg(["mean", "std"]).reset_index()
    stat = stat.sort_values("group_id")

    x = np.arange(len(stat))
    y = stat["mean"].astype(float).to_numpy()
    err = stat["std"].fillna(0).astype(float).to_numpy()

    plt.figure(figsize=(max(7, len(stat) * 1.2), 4.5))
    plt.bar(x, y, yerr=err, capsize=4)
    plt.xticks(x, stat[gcol].astype(str), rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.tight_layout()
    path = out_dir / f"bar_{metric}.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print("Saved:", path)


def choose_representative_episode(summary: pd.DataFrame, group_id: str, seed: Optional[int]) -> int:
    sub = summary[summary["group_id"] == group_id].copy()
    if seed is not None and "seed" in sub.columns:
        sub = sub[sub["seed"] == seed]
    if sub.empty:
        return 0

    # Choose episode with system_profit closest to group mean.
    if "system_profit" in sub.columns:
        mean_profit = sub["system_profit"].mean()
        idx = (sub["system_profit"] - mean_profit).abs().idxmin()
        return int(sub.loc[idx, "eval_episode"])
    return int(sub.iloc[0]["eval_episode"])


def plot_timeseries(
    summary: pd.DataFrame,
    ts: pd.DataFrame,
    group_id: str,
    metric: str,
    out_dir: Path,
    seed: Optional[int] = None,
    eval_episode: Optional[int] = None,
    ylabel: Optional[str] = None,
) -> None:
    if metric not in ts.columns:
        print(f"[skip] {metric} not in eval_timeseries.csv")
        return

    sub = ts[ts["group_id"] == group_id].copy()
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    if sub.empty:
        print(f"[skip] no time-series rows for {group_id}")
        return

    if eval_episode is None:
        eval_episode = choose_representative_episode(summary, group_id, seed)
    sub = sub[sub["eval_episode"] == eval_episode]
    if sub.empty:
        print(f"[skip] no rows for {group_id}, episode {eval_episode}")
        return

    plt.figure(figsize=(8, 4.5))
    for power_id, g in sub.groupby("power_id"):
        g = g.sort_values("t")
        plt.plot(g["t"], g[metric], label=str(power_id))
    plt.xlabel("Time step")
    plt.ylabel(ylabel or metric)
    plt.title(f"{group_id}: {metric}, seed={seed}, eval_episode={eval_episode}")
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{group_id}_{metric}_trajectory.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print("Saved:", path)


def plot_lead_time_sensitivity(summary: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if "lead_time" not in summary.columns or metric not in summary.columns:
        return

    # Only useful if lead_time has more than one value.
    if summary["lead_time"].nunique() <= 1:
        print("[skip] lead_time sensitivity: only one lead_time value")
        return

    stat = summary.groupby(["group_id", "lead_time"], as_index=False)[metric].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(7, 4.5))
    for group_id, g in stat.groupby("group_id"):
        g = g.sort_values("lead_time")
        plt.errorbar(g["lead_time"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=group_id)
    plt.xlabel("Lead time")
    plt.ylabel(metric)
    plt.title(f"Lead-time sensitivity: {metric}")
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"lead_time_sensitivity_{metric}.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print("Saved:", path)


def plot_param_cost_bars(summary: pd.DataFrame, out_dir: Path, metric: str = "avg_system_cost") -> None:
    if "param_name" not in summary.columns or "param_value" not in summary.columns:
        return
    sub = summary[summary["param_name"].fillna("") != ""].copy()
    if sub.empty:
        print("[skip] parameter cost bars: no param_name/param_value")
        return
    if metric not in sub.columns:
        print(f"[skip] {metric} not in summary")
        return

    stat = sub.groupby(["param_name", "param_value", "group_id"], as_index=False)[metric].agg(["mean", "std"]).reset_index()

    for param_name, g0 in stat.groupby("param_name"):
        pivot = g0.pivot(index="param_value", columns="group_id", values="mean").sort_index()
        ax = pivot.plot(kind="bar", figsize=(8, 4.5), yerr=None)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} under different {param_name}")
        plt.tight_layout()
        path = out_dir / f"param_{param_name}_{metric}.png"
        plt.savefig(path, dpi=250)
        plt.close()
        print("Saved:", path)


def make_mechanism_table(summary: pd.DataFrame, out_dir: Path) -> None:
    mechanism_groups = ["B0", "B1", "B2", "B3"]
    sub = summary[summary["group_id"].isin(mechanism_groups)].copy()
    if sub.empty:
        print("[skip] mechanism table: no B0-B3 rows")
        return

    metrics = [
        "system_profit",
        "avg_system_cost",
        "total_shortage_rate",
        "u1_shortage_rate",
        "ordinary_shortage_rate",
        "avg_jain",
        "bullwhip_ratio",
        "mean_mu_c",
        "mean_mu_u",
        "mean_g_u",
        "mean_g_c",
    ]
    metrics = [m for m in metrics if m in sub.columns]

    rows = []
    for (gid, label), g in sub.groupby(["group_id", label_col(sub)]):
        row = {"group_id": gid, "label": label, "n": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("group_id")
    csv_path = out_dir / "mechanism_comparison_table.csv"
    md_path = out_dir / "mechanism_comparison_table.md"
    table.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Mechanism comparison table\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print("Saved:", csv_path)
    print("Saved:", md_path)


def make_summary_table(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "system_profit",
        "system_cost",
        "avg_system_cost",
        "total_shortage_rate",
        "mean_fill_rate",
        "avg_jain",
        "order_cv",
        "demand_cv",
        "bullwhip_ratio",
        "inventory_violation_rate",
        "mean_unsold",
    ]
    metrics = [m for m in metrics if m in summary.columns]

    rows = []
    for (gid, label), g in summary.groupby(["group_id", label_col(summary)]):
        row = {"group_id": gid, "label": label, "n": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std()
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("group_id")
    csv_path = out_dir / "eval_summary_table.csv"
    md_path = out_dir / "eval_summary_table.md"
    table.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation summary table\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print("Saved:", csv_path)
    print("Saved:", md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="exp_20260513_model_direct/eval_outputs")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_episode", type=int, default=None)
    parser.add_argument("--trajectory_group", type=str, default="")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = eval_dir / "figures"
    ensure_dir(out_dir)

    summary, ts = load_data(eval_dir)
    groups = parse_csv_list(args.groups)
    summary = filter_groups(summary, groups)
    ts = filter_groups(ts, groups)

    # Summary bar charts.
    for metric, title in [
        ("system_profit", "System profit"),
        ("avg_system_cost", "Average system cost"),
        ("system_cost", "Total system cost"),
        ("total_shortage_rate", "Total shortage rate"),
        ("bullwhip_ratio", "Bullwhip ratio"),
        ("mean_fill_rate", "Mean fill rate"),
        ("avg_jain", "Allocation fairness"),
    ]:
        plot_summary_bar(summary, metric, out_dir, title)

    # Representative time-series plots.
    trajectory_groups = parse_csv_list(args.trajectory_group) or groups
    if trajectory_groups is None:
        trajectory_groups = sorted(summary["group_id"].dropna().unique().tolist())[:3]

    for gid in trajectory_groups:
        plot_timeseries(summary, ts, gid, "inventory", out_dir, args.seed, args.eval_episode, "Inventory")
        plot_timeseries(summary, ts, gid, "order_to_demand_ratio", out_dir, args.seed, args.eval_episode, "Order-to-demand ratio")
        plot_timeseries(summary, ts, gid, "shortage", out_dir, args.seed, args.eval_episode, "Shortage")

    # Sensitivity plots and tables.
    plot_lead_time_sensitivity(summary, "avg_system_cost", out_dir)
    plot_lead_time_sensitivity(summary, "total_shortage_rate", out_dir)
    plot_lead_time_sensitivity(summary, "bullwhip_ratio", out_dir)

    plot_param_cost_bars(summary, out_dir, "avg_system_cost")
    plot_param_cost_bars(summary, out_dir, "system_profit")

    make_summary_table(summary, out_dir)
    make_mechanism_table(summary, out_dir)


if __name__ == "__main__":
    main()
