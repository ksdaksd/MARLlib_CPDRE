from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from custom_envs.coal_power_direct_reciprocity_env import CPDRE
from experiments.exp1.exp1_common import (
    BASELINE_DIR,
    EXP1_GROUPS,
    base_env_args,
    ensure_dirs,
    parse_groups,
    parse_seeds,
    summarize_numeric,
    write_csv,
)


def seasonal_base_stock_action_dict(env: CPDRE, omega_off: float = 2.0, omega_peak: float = 3.0) -> Dict[str, np.ndarray]:
    season = env._season(env.t)
    omega = omega_off if season == 0 else omega_peak
    actions = {env.coal_agent_id: env.normalized_action_from_weight(1.0)}
    for agent in env.power_agent_ids:
        actions[agent] = env.normalized_action_from_omega(omega)
    return actions


def run_baseline(group_id: str, seed: int, episode_len: int, price_mode: str) -> Dict[str, float]:
    group = EXP1_GROUPS[group_id]
    if group["kind"] != "baseline":
        raise ValueError(f"{group_id} is not a baseline group.")

    env = CPDRE(base_env_args(seed=seed, group_id=group_id, episode_len=episode_len, price_mode=price_mode))
    env.reset()

    done = False
    while not done:
        if group["policy"] == "base_stock":
            action_dict = env.base_stock_action_dict(omega=2.0)
        elif group["policy"] == "seasonal_base_stock":
            action_dict = seasonal_base_stock_action_dict(env, omega_off=2.0, omega_peak=3.0)
        else:
            raise ValueError(f"Unknown baseline policy: {group['policy']}")

        _, _, dones, _ = env.step(action_dict)
        done = dones["__all__"]

    metrics = env.get_episode_metrics()
    metrics.update({
        "group_id": group_id,
        "label": group["label"],
        "policy": group["policy"],
        "demand_mode": group["demand_mode"],
        "demand_sigma": group["demand_sigma"],
        "price_mode": price_mode,
        "seed": seed,
        "episode_len": episode_len,
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run formal Experiment 1 baselines: A0/A1/A5/A6.")
    parser.add_argument("--groups", type=str, default="A0,A1,A5,A6")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--episode_len", type=int, default=156)
    parser.add_argument("--price_mode", type=str, default="fixed", choices=["fixed", "seasonal", "feedback"])
    args = parser.parse_args()

    ensure_dirs()
    groups = parse_groups(args.groups, default_groups=["A0", "A1", "A5", "A6"])
    seeds = parse_seeds(args.seeds)

    rows = []
    for group_id in groups:
        for seed in seeds:
            metrics = run_baseline(group_id, seed, args.episode_len, args.price_mode)
            rows.append(metrics)
            print(group_id, seed, metrics)

    raw_path = BASELINE_DIR / "exp1_baselines_raw.csv"
    summary_path = BASELINE_DIR / "exp1_baselines_summary.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, summarize_numeric(rows, group_key="group_id"))

    print("\nSaved baseline results:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
