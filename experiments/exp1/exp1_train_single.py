from __future__ import annotations

import argparse
from datetime import datetime

from marllib import marl

import marllib.envs.base_env.cpdre  # noqa: F401

from exp1.exp1_common import (
    EXP1_GROUPS,
    RL_META_DIR,
    base_env_args,
    ensure_dirs,
    set_global_seed,
    write_json,
)


def build_algo(algo_name: str):
    if algo_name == "ippo":
        return marl.algos.ippo(hyperparam_source="common")
    if algo_name == "mappo":
        return marl.algos.mappo(hyperparam_source="common")
    if algo_name == "happo":
        return marl.algos.happo(hyperparam_source="common")
    raise ValueError(f"Unknown algo: {algo_name}")


def main():
    parser = argparse.ArgumentParser(description="Run one formal Experiment 1 RL group/seed.")
    parser.add_argument("--group_id", type=str, required=True, choices=["A2", "A3", "A4", "A7", "A8", "A9"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--episode_len", type=int, default=156)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--price_mode", type=str, default="fixed", choices=["fixed", "seasonal", "feedback"])
    parser.add_argument("--share_policy", type=str, default="individual", choices=["group", "individual"])
    parser.add_argument("--core_arch", type=str, default="gru", choices=["gru", "lstm", "mlp"])
    parser.add_argument("--encode_layer", type=str, default="128-128")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    args = parser.parse_args()

    ensure_dirs()
    group = EXP1_GROUPS[args.group_id]
    if group["kind"] != "rl":
        raise ValueError(f"{args.group_id} is not an RL group.")

    algo_name = group["algo"]
    set_global_seed(args.seed)

    env_args = base_env_args(seed=args.seed, group_id=args.group_id, episode_len=args.episode_len, price_mode=args.price_mode)


    print("Start formal CPDRE Experiment 1 RL training")
    print("group_id:", args.group_id)
    print("label:", group["label"])
    print("algo:", algo_name)
    print("seed:", args.seed)
    print("episode_len:", args.episode_len)
    print("timesteps:", args.timesteps)
    print("core_arch:", args.core_arch)
    print("share_policy:", args.share_policy)

    env = marl.make_env(environment_name="cpdre", **env_args)
    # HAPPO uses heterogeneous sequential updating in MARLlib.
    # agent_level_batch_update is a MARLlib top-level env config,
    # not a CPDRE env_args parameter, so it must not be passed into marl.make_env().
    if args.group_id in ["A4", "A9"]:
        if isinstance(env, tuple) and len(env) >= 2 and isinstance(env[1], dict):
            env[1]["seed"] = args.seed
            if "env_args" in env[1]:
                env[1]["env_args"]["seed"] = args.seed

            env[1]["agent_level_batch_update"] = True
            print("Set agent_level_batch_update=True for HAPPO.")
        else:
            print("Warning: cannot set agent_level_batch_update because env config is not a tuple/dict.")

    algo = build_algo(algo_name)

    model = marl.build_model(env, algo, {"core_arch": args.core_arch, "encode_layer": args.encode_layer})

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
        "env_args": env_args,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = RL_META_DIR / f"{args.group_id}_{algo_name}_seed{args.seed}_meta.json"
    write_json(meta_path, meta)
    print("Saved run meta:", meta_path)

    algo.fit(
        env,
        model,
        stop={"timesteps_total": args.timesteps},
        local_mode=bool(args.local_mode),
        num_workers=args.num_workers,
        share_policy=args.share_policy,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_end=True,
    )

    print("Finished formal CPDRE Experiment 1 RL training")
    print("group_id:", args.group_id, "seed:", args.seed)


if __name__ == "__main__":
    main()
