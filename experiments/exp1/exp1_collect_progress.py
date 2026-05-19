from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List

from exp_20260513_model_direct.exp1_common import EXP1_GROUPS, PROGRESS_DIR, ensure_dirs, summarize_numeric, write_csv


PATTERN = re.compile(r"(A[0-9])_s(\d+)", re.IGNORECASE)


def read_last_progress_row(path: Path) -> Dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
    return last or {}


def main():
    parser = argparse.ArgumentParser(description="Collect final progress.csv rows for formal Experiment 1.")
    parser.add_argument("--exp_results_dir", type=str, default="exp_results")
    args = parser.parse_args()

    ensure_dirs()
    root = Path(args.exp_results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find exp_results_dir: {root}")

    rows: List[Dict[str, Any]] = []
    for progress_path in root.rglob("progress.csv"):
        match = PATTERN.search(str(progress_path))
        if not match:
            continue

        group_id = match.group(1).upper()
        seed = int(match.group(2))
        last = read_last_progress_row(progress_path)

        row: Dict[str, Any] = {
            "group_id": group_id,
            "seed": seed,
            "algo": EXP1_GROUPS.get(group_id, {}).get("algo", ""),
            "progress_path": str(progress_path),
        }

        for key in ["episode_reward_mean", "episode_reward_max", "episode_reward_min", "episode_len_mean", "timesteps_total", "training_iteration", "time_total_s"]:
            if key in last and last[key] != "":
                try:
                    row[key] = float(last[key])
                except Exception:
                    row[key] = last[key]
        rows.append(row)

    if not rows:
        print("No matching progress.csv found.")
        print("Formal Experiment-1 runs should have map_name such as direct_1c3u_A3_s2026.")
        return

    raw_path = PROGRESS_DIR / "exp1_rllib_progress_last_raw.csv"
    summary_path = PROGRESS_DIR / "exp1_rllib_progress_last_summary.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, summarize_numeric(rows, group_key="group_id"))

    print("Saved:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
