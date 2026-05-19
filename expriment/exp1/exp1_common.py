from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "outputs" / "exp1"
BASELINE_DIR = OUTPUT_DIR / "baselines"
RL_META_DIR = OUTPUT_DIR / "rl_meta"
PROGRESS_DIR = OUTPUT_DIR / "progress_summary"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

DEFAULT_SEEDS = [2024, 2025, 2026, 2027, 2028]

EXP1_GROUPS: Dict[str, Dict[str, Any]] = {
    "A0": {"demand_mode": "deterministic", "demand_sigma": 0.0, "kind": "baseline", "policy": "base_stock", "algo": None, "label": "Base-stock, deterministic seasonal demand"},
    "A1": {"demand_mode": "deterministic", "demand_sigma": 0.0, "kind": "baseline", "policy": "seasonal_base_stock", "algo": None, "label": "Non-stationary base-stock, deterministic seasonal demand"},
    "A2": {"demand_mode": "deterministic", "demand_sigma": 0.0, "kind": "rl", "policy": None, "algo": "ippo", "label": "IPPO-GRU, deterministic seasonal demand"},
    "A3": {"demand_mode": "deterministic", "demand_sigma": 0.0, "kind": "rl", "policy": None, "algo": "mappo", "label": "MAPPO-GRU, deterministic seasonal demand"},
    "A4": {"demand_mode": "deterministic", "demand_sigma": 0.0, "kind": "rl", "policy": None, "algo": "happo", "label": "HAPPO-GRU, deterministic seasonal demand"},
    "A5": {"demand_mode": "low_noise", "demand_sigma": 0.05, "kind": "baseline", "policy": "base_stock", "algo": None, "label": "Base-stock, low-noise seasonal demand"},
    "A6": {"demand_mode": "low_noise", "demand_sigma": 0.05, "kind": "baseline", "policy": "seasonal_base_stock", "algo": None, "label": "Non-stationary base-stock, low-noise seasonal demand"},
    "A7": {"demand_mode": "low_noise", "demand_sigma": 0.05, "kind": "rl", "policy": None, "algo": "ippo", "label": "IPPO-GRU, low-noise seasonal demand"},
    "A8": {"demand_mode": "low_noise", "demand_sigma": 0.05, "kind": "rl", "policy": None, "algo": "mappo", "label": "MAPPO-GRU, low-noise seasonal demand"},
    "A9": {"demand_mode": "low_noise", "demand_sigma": 0.05, "kind": "rl", "policy": None, "algo": "happo", "label": "HAPPO-GRU, low-noise seasonal demand"},
}


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, BASELINE_DIR, RL_META_DIR, PROGRESS_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_seeds(seed_text: Optional[str]) -> List[int]:
    if seed_text is None or seed_text.strip() == "":
        return list(DEFAULT_SEEDS)
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def parse_groups(group_text: Optional[str], default_groups: Iterable[str]) -> List[str]:
    if group_text is None or group_text.strip() == "":
        groups = list(default_groups)
    else:
        groups = [x.strip().upper() for x in group_text.split(",") if x.strip()]
    bad = [g for g in groups if g not in EXP1_GROUPS]
    if bad:
        raise ValueError(f"Unknown group_id(s): {bad}. Valid groups: {sorted(EXP1_GROUPS)}")
    return groups


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def base_env_args(*, seed: int, group_id: str, episode_len: int = 156, price_mode: str = "fixed") -> Dict[str, Any]:
    group = EXP1_GROUPS[group_id]
    return {
        "map_name": f"direct_1c3u_{group_id}_s{seed}",
        "num_power": 3,
        "episode_len": episode_len,
        "weeks_per_year": 52,
        "off_peak_weeks": 26,
        "lead_time": 1,
        "seed": seed,
        "mechanism_mode": "none",
        "allocation_mode": "fair",
        "demand_mode": group["demand_mode"],
        "price_mode": price_mode,
        "demand_offpeak": 1.0,
        "demand_peak": 1.4,
        "demand_sigma": group["demand_sigma"],
        "demand_ar1_rho": 0.60,
        "demand_corr": 0.0,
        "theta_offpeak": 1.15,
        "theta_peak": 0.80,
        "supply_sigma": 0.0,
        "price_offpeak": 1.0,
        "price_peak": 1.2,
        "price_sigma": 0.0,
        "price_feedback_kappa": 0.10,
        "price_feedback_psi": 0.70,
        "price_min": 0.6,
        "price_max": 1.8,
        "omega_min": 1.0,
        "omega_max": 5.0,
        "omega_base": 2.0,
        "omega_high": 3.0,
        "safe_inventory_weeks": 1.5,
        "q_max_weeks": 6.0,
        "lambda_d": 0.30,
        "w_min": 0.5,
        "w_max": 2.0,
        "w_high": 1.5,
        "mu_c_init": 0.50,
        "mu_u_init": 0.50,
        "alpha_c": 0.05,
        "alpha_u": 0.05,
        "delta_c": 0.005,
        "delta_u": 0.005,
        "trust_threshold_c": 0.50,
        "trust_threshold_u": 0.50,
        "eta_c": 0.30,
        "eta_u": 0.30,
        "coal_unit_cost": 0.70,
        "unsold_cost": 0.10,
        "power_unit_revenue": 1.80,
        "holding_cost": 0.03,
        "shortage_cost": 5.00,
        "lambda_coal_shortage": 2.00,
        "lambda_coal_fairness": 0.30,
        "obs_clip": 10.0,
        "reward_clip": None,
        "initial_inventory_weeks": 2.0,
        "include_reciprocity_in_obs": True,
        "fixed_power_omega": None,
        "fixed_coal_weight": None,
        "action_low": -1.0,
        "action_high": 1.0,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def try_float(x: Any) -> Optional[float]:
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def summarize_numeric(rows: List[Dict[str, Any]], group_key: str = "group_id") -> List[Dict[str, Any]]:
    if not rows:
        return []
    groups = sorted({str(row[group_key]) for row in rows})
    out: List[Dict[str, Any]] = []
    for group in groups:
        sub = [row for row in rows if str(row[group_key]) == group]
        result: Dict[str, Any] = {group_key: group, "n": len(sub)}
        if group in EXP1_GROUPS:
            result["label"] = EXP1_GROUPS[group]["label"]
        numeric_keys = []
        for key in sorted({k for row in sub for k in row.keys()}):
            if key == group_key:
                continue
            vals = [try_float(row.get(key)) for row in sub]
            vals = [v for v in vals if v is not None]
            if vals:
                numeric_keys.append(key)
        for key in numeric_keys:
            vals = [try_float(row.get(key)) for row in sub]
            vals = [v for v in vals if v is not None]
            if vals:
                result[f"{key}_mean"] = mean(vals)
                result[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(result)
    return out


def to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return ""
    def fmt(x: Any) -> str:
        if x is None:
            return ""
        val = try_float(x)
        if val is not None:
            return f"{val:.4f}"
        return str(x)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)
