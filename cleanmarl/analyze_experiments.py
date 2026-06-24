"""
实验一 / 实验二 结果分析与统计检验 (model 0611(1).md 4.5.3 / 4.5.4).

读取 run_sweep.py 产生的各 run 目录 (每个含 run_meta.json + episode_summary.csv),
抽取**最终评估**的 64 条固定轨迹 per-episode 指标, 汇总成组间对比表, 并对核心机制
对比 (B4 vs B2 / B4 vs B5 / B4 vs C5) 做配对 Wilcoxon 检验 + bootstrap 置信区间.

配对单元 = (seed, episode_idx): 同一 episode_idx 的评估轨迹在所有组别/种子下完全一致
(eval_seed_offset=9000+idx, 与训练种子无关), 因此可直接配对消除轨迹噪声.

用法:
    python analyze_experiments.py --experiment exp1 --logdir logs/cleanmarl
    python analyze_experiments.py --experiment exp2 --logdir logs/cleanmarl
    python analyze_experiments.py --experiment exp2 --pairs B4:B2,B4:B5 --metric system_profit
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


# 实验 -> 期望组别 (与 run_sweep.EXPERIMENTS 对齐).
EXPERIMENT_GROUPS = {
    "exp1": ["A1", "A2", "A3", "A4", "A5", "A6"],
    "exp2": ["B1", "B2", "B3", "B4", "B5"],
    "exp3": ["C1", "C2", "C3", "C4", "C5", "C6"],
}

# 默认配对检验对 (实验二核心: B4 是否优于无互惠 B2 / 非关系动态协调 B5).
DEFAULT_PAIRS = {
    "exp1": [],
    "exp2": [("B4", "B2"), ("B4", "B5"), ("B4", "B3")],
    "exp3": [("C1", "C2"), ("C1", "C3"), ("C1", "C5")],
}

# 排序意义: +1 = 越大越好, -1 = 越小越好.
METRIC_DIRECTION = {
    "system_profit": +1, "coal_profit": +1, "power_profit": +1,
    "SR": -1, "SR_1": -1, "SR_N": -1, "J": +1,
    "theta": 0, "G": 0, "lambda1": 0, "ramp": 0,
    "g_u": +1, "g_c": +1, "mu_c": +1, "mu_u": +1,
    "order_vol": -1, "reward": +1,
}


def discover_runs(logdir: Path):
    """找到所有含 run_meta.json 的 run 目录, 返回 (meta, summary_path) 列表."""
    runs = []
    for meta_path in sorted(logdir.glob("*/run_meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ⚠️ 跳过 {meta_path} (读取失败: {e})")
            continue
        summary = meta_path.parent / "episode_summary.csv"
        if not summary.exists():
            print(f"  ⚠️ {meta_path.parent.name} 缺 episode_summary.csv, 跳过")
            continue
        runs.append((meta, summary))
    return runs


def load_final_eval(summary_path: Path) -> pd.DataFrame:
    """读取最终评估的 per-episode 行.

    episode_summary.csv 含训练(phase=train)与多轮评估(phase=eval)行. 取 phase=eval
    中每个 episode 的最后一次出现 = 最终评估那一轮.
    """
    df = pd.read_csv(summary_path)
    if "phase" not in df.columns:
        return pd.DataFrame()
    ev = df[df["phase"] == "eval"].copy()
    if ev.empty:
        return ev
    ev = ev.drop_duplicates(subset="episode", keep="last")
    return ev


def _col(df, name, default=np.nan):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def derive_metrics(ev: pd.DataFrame) -> pd.DataFrame:
    """把 raw episode_summary 列映射成 spec 报告指标 (per-episode)."""
    out = pd.DataFrame(index=ev.index)
    out["episode"] = _col(ev, "episode")
    out["system_profit"] = _col(ev, "sum_system_profit")
    out["coal_profit"] = _col(ev, "sum_coal_profit")
    out["power_profit"] = _col(ev, "sum_power_profit_total")
    out["SR"] = _col(ev, "mean_shortage_rate")
    out["SR_1"] = 1.0 - _col(ev, "mean_own_service_rate_u1")
    sr_u2 = _col(ev, "mean_own_service_rate_u2")
    sr_u3 = _col(ev, "mean_own_service_rate_u3")
    out["SR_N"] = 1.0 - (sr_u2 + sr_u3) / 2.0
    out["J"] = _col(ev, "mean_jain")
    out["theta"] = _col(ev, "mean_theta")
    out["G"] = _col(ev, "mean_G")
    out["lambda1"] = _col(ev, "mean_lambda1")
    out["ramp"] = _col(ev, "mean_ramp_hit")
    out["g_u"] = _col(ev, "mean_g_u")
    out["g_c"] = _col(ev, "mean_g_c")
    out["mu_c"] = _col(ev, "mean_mu_c")
    out["mu_u"] = _col(ev, "mean_mu_u")
    out["order_vol"] = _col(ev, "std_total_order")
    out["reward"] = _col(ev, "total_reward")
    return out


def build_long(runs) -> pd.DataFrame:
    """汇总所有 run -> 长表: 一行 = (group, algo, seed, episode) + 派生指标."""
    frames = []
    for meta, summary in runs:
        ev = load_final_eval(summary)
        if ev.empty:
            continue
        m = derive_metrics(ev)
        m["group"] = meta.get("group")
        m["algo"] = meta.get("algo")
        m["seed"] = meta.get("seed")
        m["mechanism"] = meta.get("mechanism_mode")
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


METRIC_COLS = ["system_profit", "coal_profit", "power_profit", "SR", "SR_1",
               "SR_N", "J", "theta", "G", "lambda1", "ramp", "g_u", "g_c",
               "mu_c", "mu_u", "order_vol", "reward"]


def group_table(long: pd.DataFrame, groups, metrics):
    """两级聚合: 先 per (group,seed) 对 episode 取均值, 再 per group 对 seed 取 mean±std."""
    seed_mean = (long.groupby(["group", "seed"])[metrics].mean().reset_index())
    rows = []
    for g in groups:
        sub = seed_mean[seed_mean["group"] == g]
        if sub.empty:
            continue
        algo = long[long["group"] == g]["algo"].iloc[0]
        n_seed = sub["seed"].nunique()
        row = {"group": g, "algo": algo, "n_seed": n_seed}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean, std, dec=3):
    if pd.isna(mean):
        return "  -  "
    return f"{mean:.{dec}f}±{std:.{dec}f}"


def print_table(tbl: pd.DataFrame, metrics, title, decimals=None):
    decimals = decimals or {}
    print(f"\n### {title}")
    header = ["group", "algo", "n"] + metrics
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for _, r in tbl.iterrows():
        cells = [str(r["group"]), str(r["algo"]), str(int(r["n_seed"]))]
        for m in metrics:
            cells.append(fmt(r.get(f"{m}_mean", np.nan), r.get(f"{m}_std", np.nan),
                             decimals.get(m, 3)))
        print("| " + " | ".join(cells) + " |")


def paired_arrays(long, ga, gb, metric):
    """按 (seed, episode) 内连接, 返回两组对齐的 metric 数组."""
    a = long[long["group"] == ga][["seed", "episode", metric]].rename(columns={metric: "a"})
    b = long[long["group"] == gb][["seed", "episode", metric]].rename(columns={metric: "b"})
    merged = pd.merge(a, b, on=["seed", "episode"], how="inner").dropna()
    return merged["a"].to_numpy(), merged["b"].to_numpy()


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return (lo, hi)


def paired_test(long, ga, gb, metric):
    a, b = paired_arrays(long, ga, gb, metric)
    n = len(a)
    if n == 0:
        return None
    diff = a - b
    mean_diff = float(diff.mean())
    lo, hi = bootstrap_ci(diff)
    if _scipy_stats is not None and np.any(diff != 0):
        try:
            stat, p = _scipy_stats.wilcoxon(a, b)
            p = float(p)
        except Exception:
            stat, p = np.nan, np.nan
    else:
        stat, p = np.nan, (0.0 if np.any(diff != 0) else 1.0)
    return {
        "metric": metric, "n_pairs": n,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": mean_diff, "ci_lo": float(lo), "ci_hi": float(hi),
        "wilcoxon_p": p,
    }


def print_pairs(long, pairs, metrics):
    print(f"\n### 配对检验 (Wilcoxon signed-rank, 配对单元 = seed×episode)")
    print("| 对比 (A vs B) | 指标 | A 均值 | B 均值 | 均值差 (A−B) | 95% bootstrap CI | Wilcoxon p | 方向 |")
    print("|---|---|---|---|---|---|---|---|")
    for ga, gb in pairs:
        for m in metrics:
            res = paired_test(long, ga, gb, m)
            if res is None:
                continue
            direction = METRIC_DIRECTION.get(m, 0)
            sig = res["wilcoxon_p"] < 0.05 if not pd.isna(res["wilcoxon_p"]) else False
            if direction == 0 or not sig:
                verdict = "≈" if not sig else "≠"
            else:
                better_a = (res["mean_diff"] * direction) > 0
                verdict = f"{ga}更优" if better_a else f"{gb}更优"
            pstr = "n/a" if pd.isna(res["wilcoxon_p"]) else f"{res['wilcoxon_p']:.2e}"
            print(f"| {ga} vs {gb} | {m} | {res['mean_a']:.3f} | {res['mean_b']:.3f} "
                  f"| {res['mean_diff']:+.3f} | [{res['ci_lo']:+.3f}, {res['ci_hi']:+.3f}] "
                  f"| {pstr} | {verdict} |")


def main():
    parser = argparse.ArgumentParser(description="实验一/二结果分析与配对检验")
    parser.add_argument("--logdir", type=str, default="logs/cleanmarl")
    parser.add_argument("--experiment", type=str, default=None,
                        help="exp1 / exp2 / exp3 (决定报告哪些组别和默认配对)")
    parser.add_argument("--groups", type=str, default=None,
                        help="显式指定组别 (逗号分隔), 覆盖 --experiment 的默认列表")
    parser.add_argument("--pairs", type=str, default=None,
                        help="配对检验对, 如 B4:B2,B4:B5")
    parser.add_argument("--metric", type=str, default="system_profit,SR",
                        help="配对检验用的指标 (逗号分隔)")
    parser.add_argument("--csv", type=str, default=None, help="把组间汇总表写到此 CSV")
    args = parser.parse_args()

    # run_sweep 写到 cfg['system']['log_dir'] = "./logs/cleanmarl" (相对 MARLlib 根).
    # 解析顺序: 绝对路径 > 当前工作目录 > MARLlib 根目录.
    logdir = Path(args.logdir)
    if not logdir.is_absolute():
        candidates = [Path.cwd() / args.logdir,
                      Path(__file__).resolve().parent.parent / args.logdir]
        logdir = next((c for c in candidates if c.exists()), candidates[0])
    if not logdir.exists():
        print(f"❌ 日志目录不存在: {logdir}")
        sys.exit(1)

    runs = discover_runs(logdir)
    if not runs:
        print(f"❌ 在 {logdir} 下没找到任何 run (需含 run_meta.json). 先跑 run_sweep.py.")
        sys.exit(1)
    long = build_long(runs)
    if long.empty:
        print("❌ 没有可用的最终评估数据 (phase=eval).")
        sys.exit(1)

    found_groups = list(long["group"].dropna().unique())
    print(f"发现 {len(runs)} 个 run, 组别: {sorted(found_groups)}, "
          f"种子: {sorted(long['seed'].dropna().unique().tolist())}")

    # 决定报告哪些组别.
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    elif args.experiment and args.experiment.lower() in EXPERIMENT_GROUPS:
        groups = [g for g in EXPERIMENT_GROUPS[args.experiment.lower()] if g in found_groups]
    else:
        groups = sorted(found_groups)

    exp = (args.experiment or "").lower()

    if exp == "exp1":
        metrics = ["system_profit", "SR", "J", "theta", "G", "ramp", "order_vol", "reward"]
        print_table(group_table(long, groups, METRIC_COLS), metrics,
                    "实验一: 基准策略与算法选择 (无互惠)",
                    decimals={"system_profit": 1, "reward": 1})
        print("\n说明: order_vol=episode 内订货量标准差(订货波动), ramp=爬坡约束命中率, "
              "reward_std 反映训练/评估稳定性 (跨种子). 主算法选 system_profit 高且 reward_std 小者.")
    elif exp in ("exp2", "exp3"):
        metrics = ["system_profit", "coal_profit", "power_profit", "SR", "SR_1",
                   "SR_N", "J", "theta", "lambda1", "g_u", "g_c", "mu_c", "mu_u"]
        title = ("实验二: 直接互惠机制主实验 (B4 vs B5 为核心)" if exp == "exp2"
                 else "实验三: 机制来源消融 (b4 变体)")
        print_table(group_table(long, groups, METRIC_COLS), metrics, title,
                    decimals={"system_profit": 1, "coal_profit": 1, "power_profit": 1})
    else:
        metrics = ["system_profit", "SR", "J", "theta", "lambda1", "g_u", "g_c"]
        print_table(group_table(long, groups, METRIC_COLS), metrics,
                    "组间汇总", decimals={"system_profit": 1})

    # 配对检验.
    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if ":" in p]
    else:
        pairs = [p for p in DEFAULT_PAIRS.get(exp, []) if p[0] in found_groups and p[1] in found_groups]
    test_metrics = [m.strip() for m in args.metric.split(",") if m.strip()]
    if pairs:
        print_pairs(long, pairs, test_metrics)
    elif exp in ("exp2", "exp3"):
        print("\n(无可用配对组别; 确认 B2/B4/B5 等都已跑.)")

    if args.csv:
        tbl = group_table(long, groups, METRIC_COLS)
        out = Path(args.csv)
        tbl.to_csv(out, index=False)
        print(f"\n💾 组间汇总表已写入 {out}")


if __name__ == "__main__":
    main()
