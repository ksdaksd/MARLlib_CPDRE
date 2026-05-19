from __future__ import annotations

"""
exp_eval_collect_rl.py

Purpose
-------
Evaluate trained MARLlib/RLlib CPDRE policies from a checkpoint and export:

1. eval_summary.csv
   One row per evaluation episode.

2. eval_timeseries.csv
   One row per power firm per time step.

This file is designed to use the same CSV schema as:
    exp_eval_collect_heuristics.py

Where to put it
---------------
Put this file under:
    MARLlib/exp_20260513_model_direct/exp_eval_collect_rl.py

Example
-------
Run from MARLlib project root:

python -m exp_20260513_model_direct.exp_eval_collect_rl \
  --checkpoint "/home/asus/code/MARLlib/exp_20260513_model_direct/exp_results/happo_gru_direct_1c3u_A4_s42_0.0001_0.0005_APPEND-DATA_seed-2/HAPPOTrainer_cpdre_direct_1c3u_A4_s42_cef22_00000_0_2026-05-18_17-34-46/checkpoint_000007" \
  --group_id A4 \
  --seed 42 \
  --eval_episodes 20 \
  --out_dir exp_20260513_model_direct/eval_outputs_rl_A4_s42

Notes
-----
- The checkpoint argument can be either the checkpoint directory
  `checkpoint_000007` or the actual checkpoint file inside it.
- This script restores the trainer from the trial's params.json.
- It evaluates with `explore=False`.
- It uses CPDRE's raw environment directly so that env.history can be exported.
"""

import argparse
import csv
import json
import pickle
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# Important: register CPDRE before creating/restoring trainers.
import marllib.envs.base_env.cpdre  # noqa: F401
from ray import tune
from ray.rllib.models import ModelCatalog
from marllib import marl

from custom_envs.coal_power_direct_reciprocity_env import CPDRE, EPS

# Reuse the exact CSV schema and metric helpers from the heuristic collector.
from exp_20260513_model_direct.exp_eval_collect_heuristics import (
    SUMMARY_FIELDS,
    TIMESERIES_FIELDS,
    append_rows,
    compute_cost_timeseries,
    compute_power_cost_matrix,
    _safe_array,
)


# ---------------------------------------------------------------------------
# Checkpoint / trainer restore
# ---------------------------------------------------------------------------

def normalize_wsl_path(path_str: str) -> str:
    """Convert a Windows UNC WSL path to a Linux path when needed.

    Example:
        \\wsl$\\Ubuntu\\home\\asus\\code\\MARLlib\\... -> /home/asus/code/MARLlib/...
    """
    p = path_str.strip().strip('"').strip("'")

    # Common pasted Windows UNC form.
    lower = p.lower().replace("/", "\\")
    prefix = "\\\\wsl$\\ubuntu\\"
    if lower.startswith(prefix):
        rest = p[len(prefix):].replace("\\", "/")
        return "/" + rest

    # Also handle escaped single-backslash variants just in case.
    prefix2 = r"\wsl$\Ubuntu\\"
    if p.startswith(prefix2):
        rest = p[len(prefix2):].replace("\\", "/")
        return "/" + rest

    return p.replace("\\", "/") if p.startswith("\\\\wsl$") else p


def find_checkpoint_file(checkpoint: str | Path) -> Path:
    """Return actual RLlib checkpoint file.

    RLlib 1.x usually stores:
        checkpoint_000007/checkpoint-7
    """
    ckpt = Path(normalize_wsl_path(str(checkpoint))).expanduser()

    if ckpt.is_file():
        return ckpt

    if ckpt.is_dir():
        candidates = sorted([p for p in ckpt.iterdir() if p.name.startswith("checkpoint-") and p.is_file()])
        if candidates:
            return candidates[0]

        # Some Ray versions store metadata/checkpoint differently.
        # Fall back to any file containing "checkpoint".
        candidates = sorted([p for p in ckpt.iterdir() if "checkpoint" in p.name and p.is_file()])
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Cannot find checkpoint file under: {ckpt}")


def trial_dir_from_checkpoint_file(checkpoint_file: Path) -> Path:
    # checkpoint file is usually trial_dir/checkpoint_000007/checkpoint-7
    if checkpoint_file.parent.name.startswith("checkpoint_"):
        return checkpoint_file.parent.parent
    return checkpoint_file.parent


def load_trial_config(checkpoint_file: Path) -> Dict[str, Any]:
    """Load the original Ray Tune config.

    Prefer params.pkl because params.json serializes Python objects such as
    policy_mapping_fn and Gym spaces into strings. RLlib needs the original
    callable policy_mapping_fn and real spaces when rebuilding the Trainer.
    """
    trial_dir = trial_dir_from_checkpoint_file(checkpoint_file)

    params_pkl = trial_dir / "params.pkl"
    if params_pkl.exists():
        with params_pkl.open("rb") as f:
            config = pickle.load(f)
        print("Loaded config from:", params_pkl)
        return config

    params_json = trial_dir / "params.json"
    if params_json.exists():
        with params_json.open("r", encoding="utf-8") as f:
            config = json.load(f)
        print("Loaded config from JSON fallback:", params_json)
        return config

    raise FileNotFoundError(
        f"Cannot find params.pkl or params.json near checkpoint directory: {trial_dir}"
    )



def _import_class_from_path(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def _find_model_class_by_scanning(class_name: str):
    """Last-resort search for a MARLlib model class by class name.

    This avoids hard-coding too much because MARLlib forks sometimes place
    the same class under slightly different modules.
    """
    import importlib.util
    import marllib

    marllib_root = Path(marllib.__file__).resolve().parent
    model_root = marllib_root / "marl" / "models"
    if not model_root.exists():
        return None

    for py_file in model_root.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if f"class {class_name}" not in text:
            continue

        rel = py_file.relative_to(marllib_root.parent).with_suffix("")
        module_name = ".".join(rel.parts)
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            print(f"Found {class_name} in {module_name}, but import failed: {e}")

    return None


def build_marl_algo_for_registration(algo_name: str):
    """Build the same MARLlib algo object used in exp1_train_single.py.

    This is used only to call marl.build_model(...), because build_model is the
    safest way to register MARLlib custom models such as Centralized_Critic_Model.
    """
    algo_name = str(algo_name).lower()
    if algo_name == "ippo":
        return marl.algos.ippo(hyperparam_source="test")
    if algo_name == "mappo":
        return marl.algos.mappo(hyperparam_source="test")
    if algo_name == "happo":
        return marl.algos.happo(hyperparam_source="test")
    raise ValueError(f"Unknown algo for MARLlib registration: {algo_name}")


def register_marllib_model_via_build_model(config: Dict[str, Any]) -> bool:
    """Register the exact MARLlib model through marl.build_model().

    Your training script uses:
        env = marl.make_env(...)
        algo = marl.algos.happo(...)
        model = marl.build_model(env, algo, {"core_arch": ..., "encode_layer": ...})

    Calling the same build_model path is more robust than guessing model class
    paths manually, because MARLlib forks place model classes under different
    module names.
    """
    try:
        ccfg = config.get("model", {}).get("custom_model_config", {})
        algo_name = str(ccfg.get("algorithm", "")).lower()
        env_args = dict(ccfg.get("env_args", {}))
        env_args["use_global_state"] = bool(ccfg.get("global_state_flag", False))

        model_arch = ccfg.get("model_arch_args", {})
        core_arch = model_arch.get("core_arch", "gru")
        encode_layer = model_arch.get("encode_layer", "128-128")

        env_pack = marl.make_env(environment_name="cpdre", **env_args)

        # Keep the returned MARLlib env config aligned with the checkpoint config.
        if isinstance(env_pack, tuple) and len(env_pack) >= 2 and isinstance(env_pack[1], dict):
            env_pack[1]["seed"] = int(ccfg.get("seed", env_args.get("seed", 42)))
            env_pack[1]["global_state_flag"] = bool(ccfg.get("global_state_flag", False))
            env_pack[1]["agent_level_batch_update"] = bool(ccfg.get("agent_level_batch_update", False))
            if "env_args" in env_pack[1]:
                env_pack[1]["env_args"]["seed"] = int(env_args.get("seed", 42))
                env_pack[1]["env_args"]["use_global_state"] = bool(ccfg.get("global_state_flag", False))

        algo_obj = build_marl_algo_for_registration(algo_name)
        marl.build_model(env_pack, algo_obj, {"core_arch": core_arch, "encode_layer": encode_layer})

        print(
            "Registered MARLlib custom model via marl.build_model:",
            f"algo={algo_name}, core_arch={core_arch}, encode_layer={encode_layer}"
        )
        return True
    except Exception as e:
        print("marl.build_model registration failed; falling back to manual import.")
        print("Registration error:", repr(e))
        return False


def register_marllib_custom_model(config: Dict[str, Any]) -> None:
    """Register MARLlib custom model before restoring RLlib Trainer.

    First try the exact MARLlib path used during training: marl.build_model().
    If that fails, fall back to manual model-class import.
    """
    if register_marllib_model_via_build_model(config):
        return

    model_cfg = config.get("model", {})
    custom_model_name = model_cfg.get("custom_model")
    if not custom_model_name:
        return

    core_arch = (
        model_cfg.get("custom_model_config", {})
        .get("model_arch_args", {})
        .get("core_arch", "")
    )
    core_arch = str(core_arch).lower()

    candidate_map = {
        "Centralized_Critic_Model": [
            # RNN/GRU centralized critic.
            "marllib.marl.models.zoo.rnn.cc_rnn.Centralized_Critic_Model",
            "marllib.marl.models.zoo.rnn.cc_rnn.CentralizedCriticModel",
            "marllib.marl.models.zoo.rnn.centralized_critic_rnn.Centralized_Critic_Model",
            # MLP centralized critic.
            "marllib.marl.models.zoo.mlp.cc_mlp.Centralized_Critic_Model",
            "marllib.marl.models.zoo.mlp.cc_mlp.CentralizedCriticModel",
        ],
        "Base_Model": [
            # RNN/GRU base model.
            "marllib.marl.models.zoo.rnn.base_rnn.Base_Model",
            "marllib.marl.models.zoo.rnn.base_rnn.BaseModel",
            # MLP base model.
            "marllib.marl.models.zoo.mlp.base_mlp.Base_Model",
            "marllib.marl.models.zoo.mlp.base_mlp.BaseModel",
        ],
    }

    candidates = candidate_map.get(custom_model_name, [])

    # Prefer architecture-matching candidates.
    if core_arch in ["gru", "lstm", "rnn"]:
        candidates = sorted(candidates, key=lambda x: 0 if ".rnn." in x else 1)
    elif core_arch == "mlp":
        candidates = sorted(candidates, key=lambda x: 0 if ".mlp." in x else 1)

    last_error = None
    model_cls = None
    for path in candidates:
        try:
            model_cls = _import_class_from_path(path)
            print(f"Imported custom model: {path}")
            break
        except Exception as e:
            last_error = e

    if model_cls is None:
        model_cls = _find_model_class_by_scanning(custom_model_name)
        if model_cls is not None:
            print(f"Imported custom model by scanning: {model_cls}")

    if model_cls is None:
        raise ImportError(
            f"Cannot import MARLlib custom model {custom_model_name!r}. "
            f"Tried candidates={candidates}. Last error={last_error}. "
            f"Run: grep -R \"class {custom_model_name}\" -n /home/asus/code/MARLlib/marllib/marl/models"
        )

    try:
        ModelCatalog.register_custom_model(custom_model_name, model_cls)
        print(f"Registered custom model with RLlib: {custom_model_name} -> {model_cls}")
    except Exception as e:
        # Duplicate registration is usually harmless in interactive reruns.
        print(f"Model registration warning for {custom_model_name}: {e}")



def repair_config_for_restore(config: Dict[str, Any]) -> Dict[str, Any]:
    """Repair JSON/pickle config so RLlib can instantiate a trainer.

    Ray Tune's params.json stores policy_mapping_fn as a string. Even params.pkl
    can be fragile across sessions. Here we force a fresh callable mapping.
    If the policies came from JSON and their spaces are strings, reconstruct
    spaces from a dummy CPDRE environment.
    """
    ccfg = config.get("model", {}).get("custom_model_config", {})
    agent_names = ccfg.get("agent_name_ls", ["coal_0", "power_0", "power_1", "power_2"])

    ma = config.setdefault("multiagent", {})
    policies = ma.get("policies", {})
    policy_ids = list(policies.keys()) if isinstance(policies, dict) and policies else [
        f"policy_{i}" for i in range(len(agent_names))
    ]

    mapping = {agent: policy_ids[i] for i, agent in enumerate(agent_names) if i < len(policy_ids)}

    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return mapping.get(agent_id, policy_ids[0])

    ma["policy_mapping_fn"] = policy_mapping_fn

    # If policies were loaded from params.json, observation/action spaces are
    # strings. Reconstruct them with a dummy CPDRE env.
    need_space_repair = False
    if isinstance(policies, dict) and policies:
        for spec in policies.values():
            if isinstance(spec, (list, tuple)) and len(spec) >= 3:
                if isinstance(spec[1], str) or isinstance(spec[2], str):
                    need_space_repair = True
                    break

    if need_space_repair:
        env_args = dict(ccfg.get("env_args", {}))
        env_args["use_global_state"] = bool(ccfg.get("global_state_flag", False))
        dummy_env = CPDRE(env_args)
        repaired = {}
        for pid in policy_ids:
            repaired[pid] = (None, dummy_env.observation_space, dummy_env.action_space, {})
        ma["policies"] = repaired
        print("Repaired multiagent policy spaces from dummy CPDRE env.")

    # Register the custom CPDRE trial env name for RLlib restore.
    # In params.pkl, config["env"] is usually a MARLlib-generated string such as
    # "cpdre_direct_1c3u_A4_s42". Gym does not accept this as a native Gym id,
    # so Trainer construction fails unless we register this exact name with Tune.
    env_name = str(config.get("env") or ccfg.get("map_name") or "cpdre_eval_env")
    env_args_for_creator = dict(ccfg.get("env_args", {}))
    env_args_for_creator["use_global_state"] = bool(ccfg.get("global_state_flag", False))

    def _cpdre_env_creator(env_context):
        merged = dict(env_args_for_creator)
        # Allow optional override from env_config, but normally keep original training args.
        if isinstance(env_context, dict):
            merged.update({k: v for k, v in env_context.items() if k in merged or k == "seed"})
        return CPDRE(merged)

    tune.register_env(env_name, _cpdre_env_creator)
    config["env"] = env_name
    config["env_config"] = {}

    # Make evaluation lightweight and avoid worker/env randomness.
    config["num_workers"] = 0
    config["num_gpus_per_worker"] = 0
    config["evaluation_num_workers"] = 0
    config["explore"] = False

    # `local_mode` is a Ray/Tune run option used by MARLlib's fit(),
    # not an RLlib Trainer config key in this Ray version. If it remains
    # at the top level, Trainer(config=...) raises:
    #   Exception: Unknown config parameter `local_mode`
    config.pop("local_mode", None)

    # Other Tune/MARLlib run-only keys may also appear in params.pkl on
    # some versions. Keep custom_model_config untouched, but remove only
    # top-level keys that are known to break RLlib Trainer construction.
    for run_only_key in [
        "checkpoint_freq",
        "checkpoint_end",
        "restore_path",
        "local_dir",
        "stop_iters",
        "stop_timesteps",
        "stop_reward",
    ]:
        config.pop(run_only_key, None)

    print("Registered eval env:", env_name)
    print("Repaired policy_mapping_fn with mapping:", mapping)
    return config


def get_algorithm_name(config: Dict[str, Any], override: str = "") -> str:
    if override:
        return override.lower()

    try:
        algo = config["model"]["custom_model_config"]["algorithm"]
        if algo:
            return str(algo).lower()
    except Exception:
        pass

    env_name = str(config.get("env", "")).lower()
    if "happo" in env_name:
        return "happo"
    if "mappo" in env_name:
        return "mappo"
    if "ippo" in env_name:
        return "ippo"

    raise ValueError("Cannot infer algorithm. Pass --algo ippo/mappo/happo.")


def import_trainer_class(algo: str):
    """Import the real RLlib Trainer class.

    Important:
    MARLlib's `marllib.marl.algos.scripts.*` modules often expose wrapper
    functions or lambda factories, not the actual Ray/RLlib Trainer classes.
    Those wrappers cannot be instantiated as `Trainer(config=...)` and will
    raise errors such as:
        TypeError: <lambda>() got an unexpected keyword argument 'config'

    For restoring checkpoints, we need the real Trainer class from
    `marllib.marl.algos.core.*`.
    """
    algo = algo.lower()

    if algo == "ippo":
        candidates = [
            ("marllib.marl.algos.core.IL.ippo", "IPPOTrainer"),
            ("marllib.marl.algos.core.IL.ppo", "PPOTrainer"),
            ("ray.rllib.agents.ppo", "PPOTrainer"),
        ]
    elif algo == "mappo":
        candidates = [
            ("marllib.marl.algos.core.CC.mappo", "MAPPOTrainer"),
            ("marllib.marl.algos.core.CC.ppo", "CCPPOTrainer"),
            ("ray.rllib.agents.ppo", "PPOTrainer"),
        ]
    elif algo == "happo":
        candidates = [
            ("marllib.marl.algos.core.CC.happo", "HAPPOTrainer"),
            ("marllib.marl.algos.core.CC.mappo", "MAPPOTrainer"),
            ("ray.rllib.agents.ppo", "PPOTrainer"),
        ]
    else:
        raise ValueError(f"Unknown algo: {algo}")

    last_error = None
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            klass = getattr(module, class_name)
            # Exclude obvious factory/lambda objects from scripts modules.
            if getattr(klass, "__name__", "") == "<lambda>":
                last_error = TypeError(f"{module_name}.{class_name} is a lambda factory, not a Trainer class")
                continue
            print(f"Imported trainer: {module_name}.{class_name}")
            return klass
        except Exception as e:
            last_error = e

    raise ImportError(f"Cannot import real trainer class for {algo}. Last error: {last_error}")


def make_eval_trainer(checkpoint_file: Path, algo_override: str = "", force_cpu: bool = True):
    config = load_trial_config(checkpoint_file)
    config = repair_config_for_restore(config)
    register_marllib_custom_model(config)
    algo = get_algorithm_name(config, algo_override)
    trainer_cls = import_trainer_class(algo)

    # Evaluation should be local and deterministic.
    if force_cpu:
        config["num_gpus"] = 0

    # Avoid Tune-only noise if present.
    config.pop("callbacks", None)

    # Filter unknown top-level keys against the Trainer default config. This is
    # necessary because params.pkl can include Tune/MARLlib run-only options.
    try:
        default_keys = set(trainer_cls.get_default_config().keys())
    except Exception:
        default_keys = set(getattr(trainer_cls, "_default_config", {}).keys())

    if default_keys:
        preserved = {}
        for k, v in list(config.items()):
            if k in default_keys:
                preserved[k] = v
        # These are essential and may not appear in some default_config objects.
        for k in ["env", "model", "multiagent", "framework"]:
            if k in config:
                preserved[k] = config[k]
        dropped = sorted(set(config.keys()) - set(preserved.keys()))
        if dropped:
            print("Dropped non-Trainer config keys:", dropped)
        config = preserved

    # Some Ray Trainer classes accept config as a keyword, some older wrappers
    # only accept it positionally. Try both, but the preferred path is keyword.
    try:
        trainer = trainer_cls(config=config)
    except TypeError as e:
        print("Trainer(config=...) failed, retrying Trainer(config) positionally.")
        print("Original TypeError:", e)
        trainer = trainer_cls(config)

    trainer.restore(str(checkpoint_file))
    print("Restored checkpoint:", checkpoint_file)
    print("Algorithm:", algo)
    return trainer, config, algo


# ---------------------------------------------------------------------------
# Policy mapping and action computation
# ---------------------------------------------------------------------------

def build_policy_mapping(config: Dict[str, Any]):
    ccfg = config["model"]["custom_model_config"]
    agent_names = ccfg.get("agent_name_ls", ["coal_0", "power_0", "power_1", "power_2"])
    share_policy = ccfg.get("share_policy", "individual")
    policies = list(config.get("multiagent", {}).get("policies", {}).keys())

    # Most of your formal runs use one_agent_one_policy=True:
    # coal_0 -> policy_0, power_0 -> policy_1, ...
    if share_policy == "individual" or len(policies) == len(agent_names):
        mapping = {agent: f"policy_{i}" for i, agent in enumerate(agent_names)}
    elif share_policy == "group":
        # Common MARLlib grouping: coal and power have different policies.
        mapping = {}
        for agent in agent_names:
            if agent.startswith("coal_"):
                mapping[agent] = "coal_0" if "coal_0" in policies else "policy_0"
            else:
                mapping[agent] = "power_0" if "power_0" in policies else ("policy_1" if "policy_1" in policies else "policy_0")
    else:
        mapping = {agent: f"policy_{i}" for i, agent in enumerate(agent_names)}

    print("Policy mapping used in evaluation:", mapping)
    return mapping


def compute_action_safe(trainer, obs_i, policy_id: str, state: List[Any] | None):
    """Compute one action with compatibility for Ray/RLlib versions."""
    # GRU/LSTM policy.
    if state:
        try:
            action, new_state, _ = trainer.compute_single_action(
                obs_i,
                state=state,
                policy_id=policy_id,
                explore=False,
                full_fetch=True,
            )
            return action, new_state
        except TypeError:
            action, new_state, _ = trainer.compute_single_action(
                obs_i,
                state,
                policy_id=policy_id,
                explore=False,
                full_fetch=True,
            )
            return action, new_state

    # MLP policy.
    try:
        out = trainer.compute_single_action(obs_i, policy_id=policy_id, explore=False, full_fetch=True)
    except TypeError:
        out = trainer.compute_single_action(obs_i, policy_id=policy_id, explore=False)

    if isinstance(out, tuple):
        # Could be (action, state, info) or (action, info)
        action = out[0]
        new_state = out[1] if len(out) > 1 and isinstance(out[1], list) else state
        return action, new_state
    return out, state


def init_policy_states(trainer, policy_mapping: Dict[str, str]) -> Dict[str, List[Any]]:
    states = {}
    for agent, pid in policy_mapping.items():
        try:
            states[agent] = trainer.get_policy(pid).get_initial_state()
        except Exception:
            states[agent] = []
    return states


# ---------------------------------------------------------------------------
# Evaluation metric extraction
# ---------------------------------------------------------------------------

SUMMARY_FIELDS_RL = list(SUMMARY_FIELDS)
TIMESERIES_FIELDS_RL = list(TIMESERIES_FIELDS)


def collect_one_rl_episode(
    trainer,
    config: Dict[str, Any],
    group_id: str,
    algo: str,
    seed: int,
    eval_episode: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ccfg = config["model"]["custom_model_config"]
    env_args = dict(ccfg["env_args"])

    # Keep CTDE mode consistent with training config.
    global_state_flag = bool(ccfg.get("global_state_flag", False))
    env_args["use_global_state"] = global_state_flag

    # Deterministic but different evaluation trace for each episode.
    env_args["seed"] = int(seed) + int(eval_episode) * 100000
    env = CPDRE(env_args)

    policy_mapping = build_policy_mapping(config)
    states = init_policy_states(trainer, policy_mapping)

    obs = env.reset()
    done = False

    while not done:
        action_dict = {}
        for agent_id in env.agent_ids:
            pid = policy_mapping[agent_id]
            action, new_state = compute_action_safe(
                trainer=trainer,
                obs_i=obs[agent_id],
                policy_id=pid,
                state=states.get(agent_id, []),
            )
            states[agent_id] = new_state if new_state is not None else states.get(agent_id, [])
            action_dict[agent_id] = np.asarray(action, dtype=np.float32)

        obs, rewards, dones, infos = env.step(action_dict)
        done = bool(dones["__all__"])

    metrics = env.get_episode_metrics()
    costs = compute_cost_timeseries(env)
    power_cost = compute_power_cost_matrix(env)

    system_cost_total = float(np.sum(costs["system_cost"])) if costs["system_cost"].size else math.nan
    avg_system_cost = float(np.mean(costs["system_cost"])) if costs["system_cost"].size else math.nan

    summary = {
        "group_id": group_id,
        "label": f"{group_id} {algo.upper()}",
        "policy": algo,
        "seed": seed,
        "eval_episode": eval_episode,
        "param_name": "",
        "param_value": "",
        "lead_time": env.config.lead_time,
        "mechanism_mode": env.config.mechanism_mode,
        "demand_mode": env.config.demand_mode,
        "system_cost": system_cost_total,
        "avg_system_cost": avg_system_cost,
        **metrics,
    }

    h = env.history
    demand = _safe_array(h, "demand", dtype=np.float64)
    orders = _safe_array(h, "orders", dtype=np.float64)
    shipments = _safe_array(h, "shipments", dtype=np.float64)
    served = _safe_array(h, "served", dtype=np.float64)
    shortage = _safe_array(h, "shortage", dtype=np.float64)
    inventory = _safe_array(h, "inventory", dtype=np.float64)
    next_inventory = _safe_array(h, "next_inventory", dtype=np.float64)
    fill_rate = _safe_array(h, "fill_rate", dtype=np.float64)
    omega = _safe_array(h, "omega", dtype=np.float64)
    weight = _safe_array(h, "weight", dtype=np.float64)

    price = _safe_array(h, "price", dtype=np.float64)
    supply = _safe_array(h, "supply", dtype=np.float64)
    system_profit = _safe_array(h, "system_profit", dtype=np.float64)
    shortage_rate = _safe_array(h, "shortage_rate", dtype=np.float64)
    jain = _safe_array(h, "jain", dtype=np.float64)
    mu_c = _safe_array(h, "mu_c", dtype=np.float64)
    mu_u = _safe_array(h, "mu_u", dtype=np.float64)
    g_u = _safe_array(h, "g_u", dtype=np.float64)
    g_c = _safe_array(h, "g_c", dtype=np.float64)
    unsold = _safe_array(h, "unsold", dtype=np.float64)

    ts_rows: List[Dict[str, Any]] = []
    T = demand.shape[0]
    for t in range(T):
        season = env._season(t)
        for i in range(env.num_power):
            d = float(demand[t, i])
            q = float(orders[t, i])
            ts_rows.append({
                "group_id": group_id,
                "label": f"{group_id} {algo.upper()}",
                "policy": algo,
                "seed": seed,
                "eval_episode": eval_episode,
                "param_name": "",
                "param_value": "",
                "lead_time": env.config.lead_time,
                "mechanism_mode": env.config.mechanism_mode,
                "demand_mode": env.config.demand_mode,
                "t": t,
                "season": season,
                "price": float(price[t]) if price.size else "",
                "supply": float(supply[t]) if supply.size else "",
                "power_id": f"U{i + 1}",
                "power_idx": i,
                "demand": d,
                "order": q,
                "shipment": float(shipments[t, i]),
                "served": float(served[t, i]),
                "shortage": float(shortage[t, i]),
                "inventory": float(inventory[t, i]),
                "next_inventory": float(next_inventory[t, i]),
                "fill_rate": float(fill_rate[t, i]),
                "omega": float(omega[t, i]),
                "weight": float(weight[t, i]),
                "order_to_demand_ratio": q / (d + EPS),
                "power_internal_cost": float(power_cost[t, i]) if power_cost.size else "",
                "system_profit": float(system_profit[t]) if system_profit.size else "",
                "system_cost": float(costs["system_cost"][t]) if costs["system_cost"].size else "",
                "shortage_rate": float(shortage_rate[t]) if shortage_rate.size else "",
                "jain": float(jain[t]) if jain.size else "",
                "mu_c": float(mu_c[t]) if mu_c.size else "",
                "mu_u": float(mu_u[t]) if mu_u.size else "",
                "g_u": float(g_u[t]) if g_u.size else "",
                "g_c": float(g_c[t]) if g_c.size else "",
                "unsold": float(unsold[t]) if unsold.size else "",
            })

    return summary, ts_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--group_id", type=str, required=True)
    parser.add_argument("--algo", type=str, default="", choices=["", "ippo", "mappo", "happo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="exp_20260513_model_direct/eval_outputs_rl")
    parser.add_argument("--force_cpu", action="store_true", help="Use CPU for restoring/evaluation even if checkpoint was trained with GPU.")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "eval_summary.csv"
    ts_path = out_dir / "eval_timeseries.csv"

    if not args.append:
        if summary_path.exists():
            summary_path.unlink()
        if ts_path.exists():
            ts_path.unlink()

    checkpoint_file = find_checkpoint_file(args.checkpoint)
    trainer, config, algo = make_eval_trainer(checkpoint_file, algo_override=args.algo, force_cpu=args.force_cpu)

    for ep in range(args.eval_episodes):
        summary, ts_rows = collect_one_rl_episode(
            trainer=trainer,
            config=config,
            group_id=args.group_id,
            algo=algo,
            seed=args.seed,
            eval_episode=ep,
        )
        append_rows(summary_path, SUMMARY_FIELDS_RL, [summary])
        append_rows(ts_path, TIMESERIES_FIELDS_RL, ts_rows)

        print(
            f"[OK] group={args.group_id} algo={algo} seed={args.seed} ep={ep} "
            f"profit={summary.get('system_profit'):.3f} "
            f"cost={summary.get('system_cost'):.3f} "
            f"shortage={summary.get('total_shortage_rate'):.4f}"
        )

    try:
        trainer.stop()
    except Exception:
        pass

    print("\nSaved:")
    print("  ", summary_path)
    print("  ", ts_path)


if __name__ == "__main__":
    main()
