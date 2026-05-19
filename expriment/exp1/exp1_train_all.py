from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from exp_20260513_model_direct.exp1_common import OUTPUT_DIR, ensure_dirs, parse_groups, parse_seeds, write_json


def main():
    parser = argparse.ArgumentParser(description="Run all formal Experiment 1 RL groups/seeds.")
    parser.add_argument("--groups", type=str, default="A2,A3,A4,A7,A8,A9")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--episode_len", type=int, default=156)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--price_mode", type=str, default="fixed", choices=["fixed", "seasonal", "feedback"])
    parser.add_argument("--share_policy", type=str, default="individual", choices=["group", "individual"])
    parser.add_argument("--core_arch", type=str, default="gru", choices=["gru", "lstm", "mlp"])
    parser.add_argument("--encode_layer", type=str, default="128-128")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    groups = parse_groups(args.groups, default_groups=["A2", "A3", "A4", "A7", "A8", "A9"])
    seeds = parse_seeds(args.seeds)

    commands = []
    failures = []

    for group_id in groups:
        for seed in seeds:
            cmd = [
                sys.executable, "-m", "exp_20260513_model_direct.exp1_train_single",
                "--group_id", group_id,
                "--seed", str(seed),
                "--episode_len", str(args.episode_len),
                "--timesteps", str(args.timesteps),
                "--price_mode", args.price_mode,
                "--share_policy", args.share_policy,
                "--core_arch", args.core_arch,
                "--encode_layer", args.encode_layer,
                "--num_workers", str(args.num_workers),
                "--checkpoint_freq", str(args.checkpoint_freq),
            ]
            if args.local_mode:
                cmd.append("--local_mode")

            commands.append(" ".join(cmd))
            print("\n" + "=" * 100)
            print("Running:", " ".join(cmd))
            print("=" * 100)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                failure = {"group_id": group_id, "seed": seed, "returncode": exc.returncode}
                failures.append(failure)
                print("FAILED:", failure)
                if not args.continue_on_error:
                    raise

    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "groups": groups,
        "seeds": seeds,
        "commands": commands,
        "failures": failures,
    }
    log_path = OUTPUT_DIR / "exp1_train_all_commands.json"
    write_json(log_path, log)

    print("\nSaved command log:", log_path)
    if failures:
        print("There were failures:", failures)
    else:
        print("All requested RL runs finished.")


if __name__ == "__main__":
    main()
