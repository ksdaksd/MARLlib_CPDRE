from __future__ import annotations

"""
exp_eval_collect_heuristics.py

Purpose
-------
Collect evaluation data from CPDRE heuristic/baseline policies and parameter
sweeps. This script produces two CSV files:

1. eval_summary.csv
   One row per evaluation episode. Use this for bar charts, box plots,
   mechanism comparison tables, parameter sensitivity tables.

2. eval_timeseries.csv
   One row per power firm per time step. Use this for inventory trajectories,
   order-to-demand ratio trajectories, shortage trajectories, bullwhip plots.

Where to put it
---------------
Put this file under:
    MARLlib/exp_20260513_model_direct/exp_eval_collect_heuristics.py

Run from MARLlib project root:
    python -m exp_20260513_model_direct.exp_eval_collect_heuristics \
      --groups A0,A1,A5,A6 \
      --seeds 2024,2025,2026,2027,2028 \
      --eval_episodes 20 \
      --out_dir exp_20260513_model_direct/eval_outputs

Notes
-----
This script does NOT load trained RL checkpoints. It evaluates heuristic
policies directly through the environment. For trained RL policies, use this
script as the data format reference; the interaction loop is the same, but the
action_dict must come from restored policies instead of env.base_stock_action_dict().
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from custom_envs.coal_power_direct_reciprocity_env import CPDRE, EPS


# ---------------------------------------------------------------------------
# Experiment/group definitions
# ---------------------------------------------------------------------------

def group_config(group_id: str) -> Dict[str, Any]:
    """Return environment arguments and heuristic policy name for a group.

    You can extend this mapping for Experiment 2/3/4.
    """
    gid = group_id.upper()

    common = {
        "map_name": f"direct_1c3u_{gid}",
        "num_power": 3,
        "episode_len": 156,
        "mechanism_mode": "none",
        "allocation_mode": "fair",
        "price_mode": "fixed",
        "demand_mode": "deterministic",
        "demand_sigma": 0.0,
        "seed": 2026,
        "use_global_state": False,
    }

    # Experiment 1 baselines.
    if gid == "A0":
        return {**common, "_policy": "base_stock", "_label": "Base-stock"}
    if gid == "A1":
        return {**common, "_policy": "seasonal_base_stock", "_label": "Non-stationary base-stock"}
    if gid == "A5":
        return {
            **common,
            "demand_mode": "low_noise",
            "demand_sigma": 0.05,
            "_policy": "base_stock",
            "_label": "Base-stock, low-noise demand",
        }
    if gid == "A6":
        return {
            **common,
            "demand_mode": "low_noise",
            "demand_sigma": 0.05,
            "_policy": "seasonal_base_stock",
            "_label": "Non-stationary base-stock, low-noise demand",
        }

    # Mechanism comparison skeletons. These are useful for checking environment
    # mechanics, but final mechanism experiments should normally use trained RL
    # policies under each mechanism.
    if gid == "B0":
        return {**common, "mechanism_mode": "none", "allocation_mode": "fair", "_policy": "seasonal_base_stock", "_label": "B0 no reciprocity"}
    if gid == "B1":
        return {**common, "mechanism_mode": "long_contract", "allocation_mode": "weighted", "_policy": "seasonal_base_stock", "_label": "B1 long-contract priority"}
    if gid == "B2":
        return {**common, "mechanism_mode": "trigger", "allocation_mode": "weighted", "_policy": "seasonal_base_stock", "_label": "B2 trigger reciprocity"}
    if gid == "B3":
        return {**common, "mechanism_mode": "dynamic", "allocation_mode": "weighted", "_policy": "seasonal_base_stock", "_label": "B3 dynamic reciprocity"}

    raise ValueError(f"Unknown group_id: {group_id}")


def apply_sweep_value(env_args: Dict[str, Any], param_name: str, param_value: str) -> Dict[str, Any]:
    """Apply a parameter sweep value to env_args.

    param_value is a string from CLI and will be converted to int/float when possible.
    """
    out = dict(env_args)
    if not param_name:
        out["_param_name"] = ""
        out["_param_value"] = ""
        return out

    # Convert to int/float when possible.
    value: Any
    try:
        if "." in param_value:
            value = float(param_value)
        else:
            value = int(param_value)
    except ValueError:
        value = param_value

    out[param_name] = value
    out["_param_name"] = param_name
    out["_param_value"] = param_value
    return out


# ---------------------------------------------------------------------------
# Action policies
# ---------------------------------------------------------------------------

def heuristic_action(env: CPDRE, policy_name: str) -> Dict[str, np.ndarray]:
    if policy_name == "base_stock":
        return env.base_stock_action_dict(omega=env.config.omega_base)

    if policy_name == "seasonal_base_stock":
        return env.seasonal_base_stock_action_dict(
            omega_offpeak=env.config.omega_base,
            omega_peak=env.config.omega_high,
        )

    if policy_name == "random":
        return {agent: env.action_space.sample() for agent in env.agent_ids}

    raise ValueError(f"Unknown heuristic policy: {policy_name}")


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _safe_array(history: Dict[str, List[Any]], key: str, dtype=float) -> np.ndarray:
    if key not in history or len(history[key]) == 0:
        return np.asarray([], dtype=dtype)
    return np.asarray(history[key], dtype=dtype)


def compute_cost_timeseries(env: CPDRE) -> Dict[str, np.ndarray]:
    """Compute true system cost per period from environment history.

    This treats coal price as an internal transfer and excludes it from system
    true cost. It includes coal production cost, unsold coal cost, holding cost,
    and shortage cost.
    """
    h = env.history
    shipments = _safe_array(h, "shipments", dtype=np.float64)       # T x m
    shortage = _safe_array(h, "shortage", dtype=np.float64)         # T x m
    inventory = _safe_array(h, "inventory", dtype=np.float64)       # T x m, ending usable inventory
    unsold = _safe_array(h, "unsold", dtype=np.float64)             # T

    if shipments.size == 0:
        return {
            "system_cost": np.asarray([], dtype=np.float64),
            "coal_true_cost": np.asarray([], dtype=np.float64),
            "power_operating_cost": np.asarray([], dtype=np.float64),
        }

    coal_true_cost = (
        env.config.coal_unit_cost * shipments.sum(axis=1)
        + env.config.unsold_cost * unsold
    )
    power_operating_cost = (
        env.config.holding_cost * inventory.sum(axis=1)
        + env.config.shortage_cost * shortage.sum(axis=1)
    )
    system_cost = coal_true_cost + power_operating_cost

    return {
        "system_cost": system_cost,
        "coal_true_cost": coal_true_cost,
        "power_operating_cost": power_operating_cost,
    }


def compute_power_cost_matrix(env: CPDRE) -> np.ndarray:
    """Per-power-firm internal operating cost.

    Includes purchase payment to coal firm because this is useful for individual
    power-firm diagnostics. Do not use it as true supply-chain system cost.
    """
    h = env.history
    shipments = _safe_array(h, "shipments", dtype=np.float64)
    shortage = _safe_array(h, "shortage", dtype=np.float64)
    inventory = _safe_array(h, "inventory", dtype=np.float64)
    price = _safe_array(h, "price", dtype=np.float64)

    if shipments.size == 0:
        return np.asarray([], dtype=np.float64)

    return (
        price.reshape(-1, 1) * shipments
        + env.config.holding_cost * inventory
        + env.config.shortage_cost * shortage
    )


def write_header_if_needed(path: Path, fieldnames: List[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    write_header_if_needed(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


SUMMARY_FIELDS = [
    "group_id", "label", "policy", "seed", "eval_episode",
    "param_name", "param_value", "lead_time", "mechanism_mode", "demand_mode",
    "system_profit", "system_cost", "avg_system_cost",
    "total_shortage_rate", "u1_shortage_rate", "ordinary_shortage_rate",
    "mean_fill_rate", "avg_jain", "order_cv", "demand_cv", "bullwhip_ratio",
    "inventory_violation_rate", "mean_unsold", "mean_mu_c", "mean_mu_u",
    "mean_g_u", "mean_g_c",
]

TIMESERIES_FIELDS = [
    "group_id", "label", "policy", "seed", "eval_episode",
    "param_name", "param_value", "lead_time", "mechanism_mode", "demand_mode",
    "t", "season", "price", "supply", "power_id", "power_idx",
    "demand", "order", "shipment", "served", "shortage",
    "inventory", "next_inventory", "fill_rate", "omega", "weight",
    "order_to_demand_ratio", "power_internal_cost",
    "system_profit", "system_cost", "shortage_rate", "jain",
    "mu_c", "mu_u", "g_u", "g_c", "unsold",
]


def collect_one_episode(
    group_id: str,
    env_args: Dict[str, Any],
    policy_name: str,
    seed: int,
    eval_episode: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one evaluation episode and return summary row + time-series rows."""
    env_config = dict(env_args)
    env_config["seed"] = int(seed) + int(eval_episode) * 100000
    env = CPDRE(env_config)

    obs = env.reset()
    done = False

    while not done:
        action_dict = heuristic_action(env, policy_name)
        obs, rewards, dones, infos = env.step(action_dict)
        done = bool(dones["__all__"])

    metrics = env.get_episode_metrics()
    costs = compute_cost_timeseries(env)
    power_cost = compute_power_cost_matrix(env)

    system_cost_total = float(np.sum(costs["system_cost"])) if costs["system_cost"].size else math.nan
    avg_system_cost = float(np.mean(costs["system_cost"])) if costs["system_cost"].size else math.nan

    summary = {
        "group_id": group_id,
        "label": env_args.get("_label", group_id),
        "policy": policy_name,
        "seed": seed,
        "eval_episode": eval_episode,
        "param_name": env_args.get("_param_name", ""),
        "param_value": env_args.get("_param_value", ""),
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
                "label": env_args.get("_label", group_id),
                "policy": policy_name,
                "seed": seed,
                "eval_episode": eval_episode,
                "param_name": env_args.get("_param_name", ""),
                "param_value": env_args.get("_param_value", ""),
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


def parse_csv_list(x: str) -> List[str]:
    return [item.strip() for item in x.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=str, default="A0,A1,A5,A6")
    parser.add_argument("--seeds", type=str, default="2024,2025,2026,2027,2028")
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="exp_20260513_model_direct/eval_outputs")

    # Optional parameter sweep, e.g.
    # --param_name lead_time --param_values 1,2,4,6
    # --param_name theta_peak --param_values 0.7,0.8,0.9
    parser.add_argument("--param_name", type=str, default="")
    parser.add_argument("--param_values", type=str, default="")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "eval_summary.csv"
    ts_path = out_dir / "eval_timeseries.csv"

    # Start fresh each run. Comment these two lines if you want to append.
    if summary_path.exists():
        summary_path.unlink()
    if ts_path.exists():
        ts_path.unlink()

    groups = parse_csv_list(args.groups)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    param_values = parse_csv_list(args.param_values) if args.param_name else [""]

    for group_id in groups:
        base_cfg = group_config(group_id)
        policy_name = base_cfg.pop("_policy")
        label = base_cfg.get("_label", group_id)

        for pv in param_values:
            cfg = apply_sweep_value(base_cfg, args.param_name, pv)
            cfg["_label"] = label

            for seed in seeds:
                for ep in range(args.eval_episodes):
                    summary, ts_rows = collect_one_episode(
                        group_id=group_id,
                        env_args=cfg,
                        policy_name=policy_name,
                        seed=seed,
                        eval_episode=ep,
                    )
                    append_rows(summary_path, SUMMARY_FIELDS, [summary])
                    append_rows(ts_path, TIMESERIES_FIELDS, ts_rows)

                    print(
                        f"[OK] group={group_id} seed={seed} ep={ep} "
                        f"profit={summary.get('system_profit'):.3f} "
                        f"cost={summary.get('system_cost'):.3f} "
                        f"shortage={summary.get('total_shortage_rate'):.4f}"
                    )

    print("\nSaved:")
    print("  ", summary_path)
    print("  ", ts_path)


if __name__ == "__main__":
    main()
