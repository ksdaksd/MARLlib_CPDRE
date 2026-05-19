from __future__ import annotations

import argparse
import subprocess
import sys

from exp_20260513_model_direct.exp1_common import ensure_dirs


def run(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full formal Experiment 1 pipeline.")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--episode_len", type=int, default=156)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--price_mode", type=str, default="fixed")
    parser.add_argument("--share_policy", type=str, default="individual")
    parser.add_argument("--core_arch", type=str, default="gru")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    run([
        sys.executable, "-m", "exp_20260513_model_direct.exp1_eval_baselines",
        "--seeds", args.seeds,
        "--episode_len", str(args.episode_len),
        "--price_mode", args.price_mode,
    ])

    train_cmd = [
        sys.executable, "-m", "exp_20260513_model_direct.exp1_train_all",
        "--groups", "A2,A3,A4,A7,A8,A9",
        "--seeds", args.seeds,
        "--episode_len", str(args.episode_len),
        "--timesteps", str(args.timesteps),
        "--price_mode", args.price_mode,
        "--share_policy", args.share_policy,
        "--core_arch", args.core_arch,
        "--num_workers", str(args.num_workers),
        "--checkpoint_freq", str(args.checkpoint_freq),
    ]
    if args.continue_on_error:
        train_cmd.append("--continue_on_error")
    run(train_cmd)

    run([sys.executable, "-m", "exp_20260513_model_direct.exp1_collect_progress", "--exp_results_dir", "exp_results"])
    run([sys.executable, "-m", "exp_20260513_model_direct.exp1_make_tables"])
    run([sys.executable, "-m", "exp_20260513_model_direct.exp1_plot_progress", "--exp_results_dir", "exp_results"])


if __name__ == "__main__":
    main()
