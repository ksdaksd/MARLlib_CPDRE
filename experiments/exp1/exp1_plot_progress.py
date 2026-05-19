from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from exp_20260513_model_direct.exp1_common import FIGURE_DIR, ensure_dirs


PATTERN = re.compile(r"(A[0-9])_s(\d+)", re.IGNORECASE)


def read_progress(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 1 training curves.")
    parser.add_argument("--exp_results_dir", type=str, default="exp_results")
    parser.add_argument("--metric", type=str, default="episode_reward_mean")
    args = parser.parse_args()

    ensure_dirs()
    root = Path(args.exp_results_dir)

    grouped: Dict[str, List[Path]] = {}
    for progress_path in root.rglob("progress.csv"):
        match = PATTERN.search(str(progress_path))
        if not match:
            continue
        grouped.setdefault(match.group(1).upper(), []).append(progress_path)

    if not grouped:
        print("No formal Experiment-1 progress.csv found.")
        return

    for group_id, paths in grouped.items():
        plt.figure()
        for path in paths:
            rows = read_progress(path)
            xs, ys = [], []
            for row in rows:
                if args.metric not in row:
                    continue
                try:
                    x = float(row.get("timesteps_total", row.get("training_iteration", len(xs))))
                    y = float(row[args.metric])
                except Exception:
                    continue
                xs.append(x)
                ys.append(y)
            if xs:
                seed_match = PATTERN.search(str(path))
                seed = seed_match.group(2) if seed_match else "unknown"
                plt.plot(xs, ys, label=f"seed={seed}")

        plt.xlabel("Timesteps")
        plt.ylabel(args.metric)
        plt.title(f"{group_id} training curve")
        plt.legend()
        plt.tight_layout()
        out = FIGURE_DIR / f"{group_id}_{args.metric}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved:", out)


if __name__ == "__main__":
    main()
