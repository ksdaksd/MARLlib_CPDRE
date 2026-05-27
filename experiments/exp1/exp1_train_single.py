"""
Updated training script with enhanced episode data collection.

Usage:
    python exp1_train_single_v2.py --group_id A2 --seed 42

This script demonstrates using the new callback system:
- CPDREMetricsCallbackV2: Writes aggregated metrics to CSV + complete history to NPZ
- CPDRECompleteHistoryCallback: Saves only complete episode histories (lightweight)
"""

from __future__ import annotations
import os
from pathlib import Path

import argparse
from datetime import datetime
from marllib import marl

import marllib.envs.base_env.cpdre  # noqa: F401

from experiments.exp1.exp1_common import (
    EXP1_GROUPS,
    RL_META_DIR,
    base_env_args,
    ensure_dirs,
    set_global_seed,
    write_json,
)

# Import the new callback
try:
    from experiments.exp1.cpdre_callbacks_v3 import (
        CPDREMetricsCallbackV2,
        CPDRECompleteHistoryCallback,
    )
except ImportError:
    print("Warning: cpdre_callbacks_v3 not found, falling back to old callback")
    from experiments.exp1.cpdre_callbacks import CPDREMetricsCallback
    CPDREMetricsCallbackV2 = CPDREMetricsCallback
    CPDRECompleteHistoryCallback = None


def build_algo(algo_name: str):
    if algo_name == "ippo":
        return marl.algos.ippo(hyperparam_source="common")
    if algo_name == "mappo":
        return marl.algos.mappo(hyperparam_source="common")
    if algo_name == "happo":
        return marl.algos.happo(hyperparam_source="common")
    raise ValueError(f"Unknown algo: {algo_name}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1 RL training with enhanced data collection.")
    parser.add_argument("--group_id", type=str, required=True, choices=["A2", "A3", "A4", "A7", "A8", "A9"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_len", type=int, default=156)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--price_mode", type=str, default="seasonal", choices=["fixed", "seasonal", "feedback"])
    parser.add_argument("--share_policy", type=str, default="group", choices=["group", "individual"])
    parser.add_argument("--core_arch", type=str, default="gru", choices=["gru", "lstm", "mlp"])
    parser.add_argument("--encode_layer", type=str, default="128-128")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--episode_metrics_dir", type=str, default=None)
    parser.add_argument("--episode_history_dir", type=str, default=None, help="Directory to save complete episode histories")
    parser.add_argument("--callback_mode", type=str, default="v2", choices=["v2", "history_only"],
                        help="v2: Full metrics + history; history_only: History only (lightweight)")
    args = parser.parse_args()

    ensure_dirs()
    group = EXP1_GROUPS[args.group_id]
    if group["kind"] != "rl":
        raise ValueError(f"{args.group_id} is not an RL group.")

    algo_name = group["algo"]

    # Setup directories
    if args.episode_metrics_dir is None:
        episode_metrics_dir = Path("results/episode_metrics")
    else:
        episode_metrics_dir = Path(args.episode_metrics_dir)

    if args.episode_history_dir is None:
        episode_history_dir = Path("results/episode_histories")
    else:
        episode_history_dir = Path(args.episode_history_dir)

    episode_metrics_dir.mkdir(parents=True, exist_ok=True)
    episode_history_dir.mkdir(parents=True, exist_ok=True)

    # Setup environment variables for callbacks
    os.environ["CPDRE_EPISODE_LOG_DIR"] = str(episode_metrics_dir.resolve())
    os.environ["CPDRE_HISTORY_LOG_DIR"] = str(episode_history_dir.resolve())
    run_name = f"{args.group_id}_{algo_name}_seed{args.seed}"
    os.environ["CPDRE_RUN_NAME"] = run_name

    print("=" * 70)
    print("CPDRE Experiment 1 - RL Training with Enhanced Data Collection")
    print("=" * 70)
    print(f"Episode metrics dir: {episode_metrics_dir.resolve()}")
    print(f"Episode history dir: {episode_history_dir.resolve()}")
    print(f"Run name: {run_name}")
    print()

    set_global_seed(args.seed)

    env_args = base_env_args(
        seed=args.seed,
        group_id=args.group_id,
        episode_len=args.episode_len,
        price_mode=args.price_mode,
    )

    print("=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Group ID:    {args.group_id}")
    print(f"Label:       {group['label']}")
    print(f"Algorithm:   {algo_name}")
    print(f"Seed:        {args.seed}")
    print(f"Episode Len: {args.episode_len}")
    print(f"Timesteps:   {args.timesteps}")
    print(f"Core Arch:   {args.core_arch}")
    print(f"Share Policy: {args.share_policy}")
    print(f"Callback Mode: {args.callback_mode}")
    print()

    env = marl.make_env(environment_name="cpdre", **env_args)
    algo = build_algo(algo_name)
    model = marl.build_model(env, algo, {"core_arch": args.core_arch, "encode_layer": args.encode_layer})

    # Metadata
    meta = {
        "group_id": args.group_id,
        "label": group["label"],
        "algo": algo_name,
        "seed": args.seed,
        "episode_len": args.episode_len,
        "timesteps": args.timesteps,
        "price_mode": args.price_mode,
        "share_policy": args.share_policy,
        "core_arch": args.core_arch,
        "encode_layer": args.encode_layer,
        "callback_mode": args.callback_mode,
        "env_args": env_args,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = RL_META_DIR / f"{run_name}_meta.json"
    write_json(meta_path, meta)
    print(f"Saved metadata: {meta_path}")
    print()

    # Select callback
    if args.callback_mode == "v2":
        callback = CPDREMetricsCallbackV2
        print("Using callback: CPDREMetricsCallbackV2 (Aggregated metrics + complete history)")
    elif args.callback_mode == "history_only":
        if CPDRECompleteHistoryCallback is None:
            print("Warning: CPDRECompleteHistoryCallback not available, using v2")
            callback = CPDREMetricsCallbackV2
        else:
            callback = CPDRECompleteHistoryCallback
            print("Using callback: CPDRECompleteHistoryCallback (History only, lightweight)")
    else:
        raise ValueError(f"Unknown callback_mode: {args.callback_mode}")

    print(f"Callback class: {callback.__name__}")
    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()

    # Train
    algo.fit(
        env,
        model,
        stop={"timesteps_total": args.timesteps},
        local_mode=bool(args.local_mode),
        num_workers=args.num_workers,
        share_policy=args.share_policy,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_end=True,
        callbacks=callback,
    )

    print()
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Group ID: {args.group_id}, Seed: {args.seed}")
    print()
    print("Results saved to:")
    print(f"  Metrics CSV: {episode_metrics_dir}/{run_name}.csv")
    print(f"  Histories:   {episode_history_dir}/{run_name}_episode_*.npz")
    print()
    print("Load and analyze data:")
    print("  from experiments.exp1.episode_data_utils import EpisodeDataLoader")
    print(f"  loader = EpisodeDataLoader('{episode_history_dir}')")
    print(f"  episodes = loader.load_by_run_name('{run_name}')")
    print()


if __name__ == "__main__":
    main()
