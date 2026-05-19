from __future__ import annotations

from exp_20260513_model_direct.exp1_common import BASELINE_DIR, PROGRESS_DIR, TABLE_DIR, ensure_dirs, read_csv, to_markdown_table


def main():
    ensure_dirs()

    baseline_summary = read_csv(BASELINE_DIR / "exp1_baselines_summary.csv")
    rl_summary = read_csv(PROGRESS_DIR / "exp1_rllib_progress_last_summary.csv")

    baseline_cols = [
        "group_id", "label", "n",
        "system_profit_mean", "system_profit_std",
        "total_shortage_rate_mean", "total_shortage_rate_std",
        "avg_jain_mean", "avg_jain_std",
        "order_cv_mean", "order_cv_std",
        "bullwhip_ratio_mean", "bullwhip_ratio_std",
    ]

    rl_cols = [
        "group_id", "label", "n",
        "episode_reward_mean_mean", "episode_reward_mean_std",
        "episode_len_mean_mean", "timesteps_total_mean", "time_total_s_mean",
    ]

    md = []
    md.append("# Experiment 1 summary tables\n")
    md.append("## A0/A1/A5/A6 traditional baselines\n")
    md.append(to_markdown_table(baseline_summary, baseline_cols))
    md.append("\n\n## A2-A4/A7-A9 RL final training progress\n")
    md.append(to_markdown_table(rl_summary, rl_cols))

    out_path = TABLE_DIR / "exp1_summary_tables.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
