"""
coal_power_direct_reciprocity_env.py

CPDRE: Coal-Power Direct Reciprocity Environment

This version is rewritten according to the 0513 model document.

Key modeling choices:
1. One upstream coal firm C and m downstream power firms U_1,...,U_m.
2. U_1 is the reciprocal power firm; other power firms are ordinary firms.
3. Power-firm inventory is unbounded. Excess inventory is penalized only through
   holding cost.
4. The power-firm action is target inventory coverage omega_i,t.
5. The coal-firm action is the supply-priority weight w_t for U_1.
6. Current orders are generated before current real demand realization, using:
      D_tilde_i = lambda_D * D_{i,t-1} + (1-lambda_D) * Dbar_i(z_t)
      Q_i,t = [omega_i,t * D_tilde_i - IP_i,t]^+
7. Current shipments enter the pipeline and do not satisfy current demand.
8. Reciprocity contributions are based on actual incremental transactions:
      gU_t: off-peak incremental shipment to U_1 relative to baseline order.
      gC_t: peak incremental shipment to U_1 relative to fair allocation.
9. Trust updates use rolling accumulation with natural decay; no per-period
   betrayal penalty is used in the main model.
10. Local observations exclude derived system indicators such as Jain fairness,
    system shortage rate, system total demand, and current reciprocity
    contribution.

For MARLlib integration, copy this file to:
    MARLlib/custom_envs/coal_power_direct_reciprocity_env.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


EPS = 1e-8


@dataclass
class CPDREConfig:
    """Default configuration for CPDRE.

    Quantity variables are normalized by one ordinary power firm's off-peak
    average coal demand. Price is normalized by off-peak coal price.
    """

    # Episode and system scale
    num_power: int = 3
    episode_len: int = 156
    weeks_per_year: int = 52
    off_peak_weeks: int = 26
    lead_time: int = 1
    seed: int = 42

    # Experiment mode
    mechanism_mode: str = "none"  # none, dynamic, long_contract, trigger
    allocation_mode: str = "fair"  # fair, weighted
    demand_mode: str = "deterministic"  # deterministic, low_noise, lognormal, truncated_normal, ar1
    price_mode: str = "seasonal"  # fixed, seasonal, feedback

    # Demand and supply
    demand_offpeak: float = 1.0
    demand_peak: float = 1.4
    demand_sigma: float = 0.15
    demand_ar1_rho: float = 0.60
    demand_corr: float = 0.0
    theta_offpeak: float = 1.15
    theta_peak: float = 0.80
    supply_sigma: float = 0.0

    # Price
    price_offpeak: float = 1.0
    price_peak: float = 1.2
    price_sigma: float = 0.0
    price_feedback_kappa: float = 0.10
    price_feedback_psi: float = 0.70
    price_min: float = 0.6
    price_max: float = 1.8

    # Inventory and order
    omega_min: float = 1.0
    omega_max: float = 5.0
    omega_base: float = 2.0
    omega_high: float = 3.0
    safe_inventory_weeks: float = 1.5
    q_max_weeks: float = 6.0

    # In the model document:
    # D_tilde_i,t = lambda_D * D_i,t-1 + (1-lambda_D) * Dbar_i(z_t)
    lambda_d: float = 0.30

    # Coal allocation weight
    w_min: float = 0.5
    w_max: float = 2.0
    w_high: float = 1.5

    # Trust and reciprocity
    mu_c_init: float = 0.50  # coal firm's trust in U1
    mu_u_init: float = 0.50  # U1's trust in coal firm
    alpha_c: float = 0.05
    alpha_u: float = 0.05
    delta_c: float = 0.005
    delta_u: float = 0.005

    # Thresholds are only used by the trigger-rule baseline.
    trust_threshold_c: float = 0.50
    trust_threshold_u: float = 0.50

    # Reciprocity reward coefficients
    eta_c: float = 0.30
    eta_u: float = 0.30

    # Economic parameters
    coal_unit_cost: float = 0.70
    unsold_cost: float = 0.10
    power_unit_revenue: float = 1.80
    holding_cost: float = 0.03
    shortage_cost: float = 5.00

    # Reward shaping / responsibility assignment
    lambda_coal_shortage: float = 2.00
    lambda_coal_fairness: float = 0.30

    # Observation and numerical control
    obs_clip: float = 10.0
    reward_clip: Optional[float] = None
    initial_inventory_weeks: float = 2.0
    include_reciprocity_in_obs: bool = True
    debug_checks: bool = False

    # If True, expose an explicit global state space and get_state()/state() for
    # MARLlib/RLlib versions that consume global state in centralized critics.
    # Keep False if your local MARLlib expects joint observations instead.
    use_global_state: bool = True

    # Optional manual fixed policy behavior for non-learning baselines.
    fixed_power_omega: Optional[float] = None
    fixed_coal_weight: Optional[float] = None

    # Normalized action convention.
    # MARLlib can handle Box(-1, 1). If old scripts still output [0, 1],
    # set action_low=0 and action_high=1.
    action_low: float = -1.0
    action_high: float = 1.0


class CoalPowerDirectReciprocityEnv(MultiAgentEnv):
    """Coal-power direct reciprocity supply chain environment.

    Agents:
        coal_0: upstream coal firm C.
        power_0: reciprocal power firm U1.
        power_1,...: ordinary power firms.

    This class uses the old RLlib MultiAgentEnv API:
        reset() -> obs_dict
        step(action_dict) -> obs_dict, reward_dict, done_dict, info_dict
    """

    metadata = {"name": "CPDRE"}

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        raw_config = env_config or {}
        if "env_args" in raw_config and isinstance(raw_config["env_args"], dict):
            raw_config = raw_config["env_args"]

        self.config = self._build_config(raw_config)
        self.rng = np.random.default_rng(self.config.seed)

        self.num_power = int(self.config.num_power)
        if self.num_power < 1:
            raise ValueError("num_power must be at least 1")

        self.coal_agent_id = "coal_0"
        self.power_agent_ids = [f"power_{i}" for i in range(self.num_power)]
        self.agent_ids = [self.coal_agent_id] + self.power_agent_ids
        self.agents = list(self.agent_ids)

        # Common normalized continuous action for MARLlib compatibility.
        self.action_space = spaces.Box(
            low=np.array([self.config.action_low], dtype=np.float32),
            high=np.array([self.config.action_high], dtype=np.float32),
            dtype=np.float32,
        )

        # Fixed dimension for MARLlib. The feature vector is role-specific and
        # padded with zeros where a variable is not locally observable.
        self.max_power_obs = max(3, self.num_power)
        self.obs_dim = 24
        self.num_agents = len(self.agent_ids)

        self.obs_box = spaces.Box(
            low=-self.config.obs_clip,
            high=self.config.obs_clip,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Centralized critic state for CTDE:
        # concatenate all agents' local observations in a fixed agent order.
        self.state_dim = self.obs_dim * self.num_agents
        self.state_space = spaces.Box(
            low=-self.config.obs_clip,
            high=self.config.obs_clip,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        if self.config.use_global_state:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.obs_box,
                    "state": self.state_space,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "obs": self.obs_box,
                }
            )

        self.episode_limit = self.config.episode_len
        self.env_name = "coal_power_direct_reciprocity"
        self.map_name = "CPDRE"

        # State variables initialized in reset().
        self.t = 0
        self.pipe_len = max(self.config.lead_time - 1, 0)

        self.inventory = np.zeros(self.num_power, dtype=np.float32)
        self.pipeline = np.zeros((self.pipe_len, self.num_power), dtype=np.float32)
        self.ar1_noise = np.zeros(self.num_power, dtype=np.float32)

        # Current public state generated before each decision epoch.
        # This keeps observations and settlement in step() aligned on the same
        # p_t and A_t, rather than using a previous-period proxy.
        self.current_season = 0
        self.current_base_demand = np.ones(self.num_power, dtype=np.float32)
        self.current_price = float(self.config.price_offpeak)
        self.current_supply = float(self.config.theta_offpeak * self.num_power * self.config.demand_offpeak)
        self.price = float(self.config.price_offpeak)

        # Model notation: mu^C = coal's trust in U1; mu^U = U1's trust in coal.
        self.mu_c = 0.0
        self.mu_u = 0.0

        # Historical variables entering local observations.
        self.last_demand = np.ones(self.num_power, dtype=np.float32)
        self.last_orders = np.zeros(self.num_power, dtype=np.float32)
        self.last_shipments = np.zeros(self.num_power, dtype=np.float32)
        self.last_shortage = np.zeros(self.num_power, dtype=np.float32)
        self.last_served = np.zeros(self.num_power, dtype=np.float32)
        self.last_omega = np.ones(self.num_power, dtype=np.float32) * self.config.omega_base
        self.last_weight = np.ones(self.num_power, dtype=np.float32)

        # Derived variables for metrics and info only.
        self.last_fill_rate = np.ones(self.num_power, dtype=np.float32)
        self.last_jain = 1.0
        self.last_unsold = 0.0
        self.last_system_profit = 0.0
        self.last_shortage_rate = 0.0
        self.last_g_u = 0.0
        self.last_g_c = 0.0

        self.history: Dict[str, List[Any]] = {}
        self.env_info = self.get_env_info()
        self.reset()

    @staticmethod
    def _build_config(raw_config: Dict[str, Any]) -> CPDREConfig:
        cfg = CPDREConfig()
        valid = set(cfg.__dataclass_fields__.keys())
        for key, value in raw_config.items():
            if key in valid:
                setattr(cfg, key, value)

        cfg.num_power = int(cfg.num_power)
        cfg.episode_len = int(cfg.episode_len)
        cfg.weeks_per_year = int(cfg.weeks_per_year)
        cfg.off_peak_weeks = int(cfg.off_peak_weeks)
        cfg.lead_time = max(1, int(cfg.lead_time))

        # Backward compatibility for earlier config keys.
        if "forecast_beta" in raw_config:
            cfg.lambda_d = float(raw_config["forecast_beta"])
        if "trust_init" in raw_config:
            cfg.mu_c_init = float(raw_config["trust_init"])
            cfg.mu_u_init = float(raw_config["trust_init"])
        if "alpha_pos" in raw_config:
            cfg.alpha_c = float(raw_config["alpha_pos"])
            cfg.alpha_u = float(raw_config["alpha_pos"])
        return cfg

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = self.config.seed
        self.config.seed = int(seed)
        self.rng = np.random.default_rng(self.config.seed)
        return [self.config.seed]

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "space_state": self.state_space if self.config.use_global_state else None,
            "mask_flag": False,
            "global_state_flag": bool(self.config.use_global_state),
            "policy_mapping_info": {
                "all_scenario": {
                    "description": "CPDRE with one coal firm and multiple power firms.",
                    "team_prefix": ("coal_", "power_"),
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                }
            },
        }

    def _build_joint_state_from_obs(
        self,
        obs_dict: Dict[str, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Build centralized critic state from all agents' local observations."""
        state_parts = []
        for agent_id in self.agent_ids:
            agent_obs = obs_dict[agent_id]["obs"]
            state_parts.append(np.asarray(agent_obs, dtype=np.float32).reshape(-1))

        state = np.concatenate(state_parts, axis=0).astype(np.float32)
        if state.shape[0] != self.state_dim:
            raise RuntimeError(
                f"Centralized state dimension mismatch: "
                f"expected {self.state_dim}, got {state.shape[0]}"
            )
        return np.clip(
            state,
            -self.config.obs_clip,
            self.config.obs_clip,
        ).astype(np.float32)

    def get_state(self) -> np.ndarray:
        """Return the explicit centralized critic state for CTDE."""
        obs_dict = self._get_obs()
        return self._build_joint_state_from_obs(obs_dict)

    def state(self) -> np.ndarray:
        """Compatibility alias used by some MARLlib/RLlib wrappers."""
        return self.get_state()

    # ------------------------------------------------------------------
    # RLlib old API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        self.t = 0
        self.price = float(self.config.price_offpeak)

        if self._reciprocity_enabled():
            self.mu_c = float(self.config.mu_c_init)
            self.mu_u = float(self.config.mu_u_init)
        else:
            self.mu_c = 0.0
            self.mu_u = 0.0

        base = self._base_demand_vector(self._season(self.t))
        self.inventory = (self.config.initial_inventory_weeks * base).astype(np.float32)
        self.pipeline = np.zeros((self.pipe_len, self.num_power), dtype=np.float32)
        self.ar1_noise = np.zeros(self.num_power, dtype=np.float32)

        self.last_demand = base.astype(np.float32)
        self.last_orders = np.zeros(self.num_power, dtype=np.float32)
        self.last_shipments = np.zeros(self.num_power, dtype=np.float32)
        self.last_shortage = np.zeros(self.num_power, dtype=np.float32)
        self.last_served = np.zeros(self.num_power, dtype=np.float32)
        self.last_omega = np.ones(self.num_power, dtype=np.float32) * self.config.omega_base
        self.last_weight = np.ones(self.num_power, dtype=np.float32)

        self.last_fill_rate = np.ones(self.num_power, dtype=np.float32)
        self.last_jain = 1.0
        self.last_unsold = 0.0
        self.last_system_profit = 0.0
        self.last_shortage_rate = 0.0
        self.last_g_u = 0.0
        self.last_g_c = 0.0

        self.history = {
            "demand": [],
            "orders": [],
            "orders_base_u1": [],
            "shipments": [],
            "shipments_fair": [],
            "shipments_base_u": [],
            "shortage": [],
            "served": [],
            "inventory": [],
            "next_inventory": [],
            "fill_rate": [],
            "omega": [],
            "weight": [],
            "price": [],
            "supply": [],
            "mu_c": [],
            "mu_u": [],
            "g_u": [],
            "g_c": [],
            "jain": [],
            "system_profit": [],
            "coal_profit": [],
            "power_profit_total": [],
            "shortage_rate": [],
            "unsold": [],
        }

        self._refresh_public_state()
        return self._get_obs()

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """One periodic-review transition.

        Model order:
            s_t -> local observations -> simultaneous actions
            -> D_tilde -> Q_t -> Y_t -> D_t
            -> inventory/pipeline/shortage/leftover supply/trust update
            -> s_{t+1}

        Current real demand D_t is not used to form current orders Q_t.
        Current shipments Y_t enter the pipeline and cannot satisfy current
        demand. With L=1, Y_t is added to available inventory at t+1.
        """

        # 1. Use the public state that was generated before this decision epoch.
        # Agents have already observed these same values in _get_obs().
        season = int(self.current_season)
        base_demand = self.current_base_demand.copy().astype(np.float32)
        price = float(self.current_price)
        supply = float(self.current_supply)

        # 2. At the decision epoch, self.inventory is already the usable beginning
        # inventory I_{i,t}. For lead_time=1, last period's shipment has already
        # been moved into inventory during the previous transition.
        available_inventory = self.inventory.copy().astype(np.float32)

        # 3. Parse simultaneous actions.
        omega, weights = self._parse_actions(action_dict, season)
        if self._force_fair_allocation():
            weights = np.ones(self.num_power, dtype=np.float32)

        # 4. Internal demand-rate estimator and order generation.
        d_tilde = self._estimate_demand_rate(base_demand)
        inventory_position = available_inventory + self.pipeline.sum(axis=0)
        target_inventory = omega * d_tilde
        raw_orders = np.maximum(target_inventory - inventory_position, 0.0)

        q_max = self.config.q_max_weeks * base_demand
        orders = np.minimum(raw_orders, q_max).astype(np.float32)

        # Baseline order for U1 under omega_base, used only for reciprocity accounting.
        q_base_u1 = max(self.config.omega_base * float(d_tilde[0]) - float(inventory_position[0]), 0.0)
        q_base_u1 = min(q_base_u1, float(q_max[0]))
        orders_base_u = orders.copy()
        orders_base_u[0] = np.float32(q_base_u1)

        # 5. Coal allocation and counterfactual allocation.
        shipments = self._allocate_supply(orders, float(supply), weights)
        shipments_fair = self._allocate_supply(
            orders, float(supply), np.ones(self.num_power, dtype=np.float32)
        )
        shipments_base_u = self._allocate_supply(orders_base_u, float(supply), weights)

        if self.config.debug_checks:
            self._run_debug_checks(
                orders=orders,
                shipments=shipments,
                supply=float(supply),
                weights=weights,
                stage="after_allocation",
            )

        # 6. Current real coal demand is realized and satisfied only by available inventory.
        demand = self._generate_demand(season, base_demand)
        served = np.minimum(demand, available_inventory).astype(np.float32)
        shortage = np.maximum(demand - available_inventory, 0.0).astype(np.float32)
        ending_inventory = np.maximum(available_inventory - served, 0.0).astype(np.float32)

        # Inventory and in-transit update for the next decision epoch.
        # This implements:
        #   L=1: I_{t+1}=I_end,t+Y_t and P is empty.
        #   L>1: I_{t+1}=I_end,t+P^{(1)}_t, shift the pipeline, and put Y_t at P^{(L-1)}.
        if self.config.lead_time <= 1:
            next_inventory = (ending_inventory + shipments).astype(np.float32)
            next_pipeline = np.zeros((0, self.num_power), dtype=np.float32)
        else:
            arrivals_next = self.pipeline[0].copy()
            next_inventory = (ending_inventory + arrivals_next).astype(np.float32)
            next_pipeline = np.zeros_like(self.pipeline)
            if self.config.lead_time > 2:
                next_pipeline[:-1] = self.pipeline[1:]
            next_pipeline[-1] = shipments.astype(np.float32)

        # 7. Derived settlement metrics.
        fill_rate = np.where(orders > EPS, shipments / (orders + EPS), 1.0).astype(np.float32)
        jain = self._jain_index(fill_rate)
        unsold = max(float(supply) - float(shipments.sum()), 0.0)

        g_u, g_c = self._compute_reciprocity_contributions(
            season=season,
            base_demand=base_demand,
            shipments=shipments,
            shipments_fair=shipments_fair,
            shipments_base_u=shipments_base_u,
        )


        if self.config.debug_checks and self.config.mechanism_mode.lower() == "none":
            if abs(float(g_u)) > 1e-8 or abs(float(g_c)) > 1e-8:
                raise RuntimeError("No-reciprocity mode should have zero reciprocity contributions.")

        # Rewards use beginning-of-period trust mu_t. The current contribution
        # updates trust only for the next state mu_{t+1}.
        mu_c_before = float(self.mu_c)
        mu_u_before = float(self.mu_u)

        rewards, profit_info = self._compute_rewards(
            price=price,
            demand=demand,
            shipments=shipments,
            served=served,
            shortage=shortage,
            ending_inventory=ending_inventory,
            unsold=unsold,
            fill_rate=fill_rate,
            jain=jain,
            g_u=g_u,
            g_c=g_c,
            mu_c_reward=mu_c_before,
            mu_u_reward=mu_u_before,
        )

        # Trust update follows the model: rolling contribution with natural decay.
        self._update_trust(g_u=g_u, g_c=g_c)

        # 8. Commit next state.
        self.inventory = next_inventory.astype(np.float32)
        self.pipeline = next_pipeline.astype(np.float32)
        self.price = float(price)

        self.last_demand = demand.astype(np.float32)
        self.last_orders = orders.astype(np.float32)
        self.last_shipments = shipments.astype(np.float32)
        self.last_shortage = shortage.astype(np.float32)
        self.last_served = served.astype(np.float32)
        self.last_omega = omega.astype(np.float32)
        self.last_weight = weights.astype(np.float32)
        self.last_fill_rate = fill_rate.astype(np.float32)
        self.last_jain = float(jain)
        self.last_unsold = float(unsold)
        self.last_system_profit = float(profit_info["system_profit"])
        self.last_shortage_rate = float(shortage.sum() / (demand.sum() + EPS))
        self.last_g_u = float(g_u)
        self.last_g_c = float(g_c)

        self._record_history(
            demand=demand,
            orders=orders,
            orders_base_u1=np.array([q_base_u1], dtype=np.float32),
            shipments=shipments,
            shipments_fair=shipments_fair,
            shipments_base_u=shipments_base_u,
            shortage=shortage,
            served=served,
            inventory=ending_inventory,
            next_inventory=next_inventory,
            fill_rate=fill_rate,
            omega=omega,
            weight=weights,
            price=price,
            supply=supply,
            mu_c=self.mu_c,
            mu_u=self.mu_u,
            g_u=g_u,
            g_c=g_c,
            jain=jain,
            system_profit=profit_info["system_profit"],
            coal_profit=profit_info["coal_profit"],
            power_profit_total=profit_info["power_profit_total"],
            shortage_rate=self.last_shortage_rate,
            unsold=unsold,
        )

        current_t = self.t
        # 9. Advance time and build outputs.
        self.t += 1
        done = self.t >= self.config.episode_len
        if not done:
            self._refresh_public_state()

        dones = {agent: done for agent in self.agent_ids}
        dones["__all__"] = done

        infos = self._build_infos(
            t=current_t,
            season=season,
            price=price,
            supply=supply,
            demand=demand,
            d_tilde=d_tilde,
            orders=orders,
            orders_base_u=orders_base_u,
            shipments=shipments,
            shipments_fair=shipments_fair,
            shipments_base_u=shipments_base_u,
            served=served,
            shortage=shortage,
            ending_inventory=ending_inventory,
            fill_rate=fill_rate,
            jain=jain,
            unsold=unsold,
            g_u=g_u,
            g_c=g_c,
            profit_info=profit_info,
        )

        return self._get_obs(), rewards, dones, infos

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------
    def _season(self, t: int) -> int:
        week = t % self.config.weeks_per_year
        return 0 if week < self.config.off_peak_weeks else 1

    def _year_position(self, t: Optional[int] = None) -> float:
        if t is None:
            t = self.t
        return float((t % self.config.weeks_per_year) / max(self.config.weeks_per_year, 1))

    def _refresh_public_state(self) -> None:
        """Generate public exogenous state for the current decision epoch.

        The observation returned before action and the subsequent step()
        settlement must use the same season, price, and effective supply.
        This method makes p_t and A_t part of the environment state rather than
        hidden variables sampled after actions are chosen.
        """
        season = self._season(self.t)
        base_demand = self._base_demand_vector(season)
        price = self._generate_price(season)
        supply = self._generate_supply(season, base_demand)

        self.current_season = int(season)
        self.current_base_demand = base_demand.astype(np.float32)
        self.current_price = float(price)
        self.current_supply = float(supply)

    def _base_demand_vector(self, season: int) -> np.ndarray:
        base = self.config.demand_offpeak if season == 0 else self.config.demand_peak
        return np.ones(self.num_power, dtype=np.float32) * float(base)

    def _safe_inventory_vector(self, season: int) -> np.ndarray:
        base = self._base_demand_vector(season)
        return (self.config.safe_inventory_weeks * base).astype(np.float32)

    def _estimate_demand_rate(self, base_demand: np.ndarray) -> np.ndarray:
        return (
            self.config.lambda_d * self.last_demand
            + (1.0 - self.config.lambda_d) * base_demand
        ).astype(np.float32)

    def _generate_demand(self, season: int, base_demand: np.ndarray) -> np.ndarray:
        mode = self.config.demand_mode.lower()
        sigma = float(self.config.demand_sigma)

        if mode == "deterministic":
            eta = np.ones(self.num_power, dtype=np.float32)
        elif mode == "low_noise":
            eta = self._correlated_lognormal(max(sigma, 0.05))
        elif mode == "lognormal":
            eta = self._correlated_lognormal(sigma)
        elif mode == "truncated_normal":
            raw = self.rng.normal(loc=1.0, scale=sigma, size=self.num_power)
            eta = np.clip(raw, 0.05, None).astype(np.float32)
        elif mode == "ar1":
            eps = self.rng.normal(loc=0.0, scale=sigma, size=self.num_power)
            self.ar1_noise = (
                self.config.demand_ar1_rho * self.ar1_noise
                + np.sqrt(max(1.0 - self.config.demand_ar1_rho**2, 0.0)) * eps
            ).astype(np.float32)
            eta = np.clip(1.0 + self.ar1_noise, 0.05, None).astype(np.float32)
        else:
            raise ValueError(f"Unknown demand_mode: {self.config.demand_mode}")

        return np.maximum(base_demand * eta, 0.0).astype(np.float32)

    def _correlated_lognormal(self, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 0.0)
        if sigma <= EPS:
            return np.ones(self.num_power, dtype=np.float32)
        rho = float(np.clip(self.config.demand_corr, 0.0, 0.999))
        common = self.rng.normal()
        indiv = self.rng.normal(size=self.num_power)
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * indiv
        eta = np.exp(-0.5 * sigma**2 + sigma * z)
        return eta.astype(np.float32)

    def _generate_supply(self, season: int, base_demand: np.ndarray) -> float:
        theta = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        mean_supply = float(theta) * float(np.sum(base_demand))
        if self.config.supply_sigma > EPS:
            shock = float(np.exp(-0.5 * self.config.supply_sigma**2 + self.config.supply_sigma * self.rng.normal()))
        else:
            shock = 1.0
        return max(mean_supply * shock, 0.0)

    def _generate_price(self, season: int) -> float:
        mode = self.config.price_mode.lower()
        if mode == "fixed":
            return float(self.config.price_offpeak)

        seasonal_price = self.config.price_offpeak if season == 0 else self.config.price_peak
        if mode == "seasonal":
            price = seasonal_price
        elif mode == "feedback":
            supply_proxy = self._last_supply_proxy()
            tension = (float(self.last_orders.sum()) - supply_proxy) / (supply_proxy + EPS)
            price = (1.0 - self.config.price_feedback_psi) * self.price + self.config.price_feedback_psi * seasonal_price
            price += self.config.price_feedback_kappa * tension
        else:
            raise ValueError(f"Unknown price_mode: {self.config.price_mode}")

        if self.config.price_sigma > EPS:
            price *= float(np.exp(-0.5 * self.config.price_sigma**2 + self.config.price_sigma * self.rng.normal()))
        return float(np.clip(price, self.config.price_min, self.config.price_max))

    def _last_supply_proxy(self) -> float:
        if self.history.get("supply"):
            return float(self.history["supply"][-1])
        return float(self.config.theta_offpeak * self.num_power * self.config.demand_offpeak)

    def _parse_actions(self, action_dict: Dict[str, Any], season: int) -> Tuple[np.ndarray, np.ndarray]:
        omega = np.zeros(self.num_power, dtype=np.float32)
        weights = np.ones(self.num_power, dtype=np.float32)

        coal_a = self._get_scalar_action(action_dict.get(self.coal_agent_id, np.array([0.0], dtype=np.float32)))
        learned_w1 = self._scale_action(coal_a, self.config.w_min, self.config.w_max)

        for idx, agent in enumerate(self.power_agent_ids):
            a = self._get_scalar_action(action_dict.get(agent, np.array([0.0], dtype=np.float32)))
            omega[idx] = self._scale_action(a, self.config.omega_min, self.config.omega_max)

        if self.config.fixed_power_omega is not None:
            omega[:] = float(self.config.fixed_power_omega)
        if self.config.fixed_coal_weight is not None:
            learned_w1 = float(self.config.fixed_coal_weight)

        mode = self.config.mechanism_mode.lower()
        if mode == "none":
            weights[:] = 1.0
        elif mode == "long_contract":
            weights[:] = 1.0
            if season == 0:
                omega[0] = float(self.config.omega_high)
            else:
                weights[0] = float(self.config.w_high)
        elif mode == "trigger":
            weights[:] = 1.0
            if season == 0 and self.mu_u >= self.config.trust_threshold_u:
                omega[0] = float(self.config.omega_high)
            if season == 1 and self.mu_c >= self.config.trust_threshold_c:
                weights[0] = float(self.config.w_high)
        elif mode == "dynamic":
            weights[:] = 1.0
            weights[0] = float(learned_w1)
        else:
            raise ValueError(f"Unknown mechanism_mode: {self.config.mechanism_mode}")

        omega = np.clip(omega, self.config.omega_min, self.config.omega_max).astype(np.float32)
        weights = np.clip(weights, self.config.w_min, self.config.w_max).astype(np.float32)
        return omega, weights

    def _get_scalar_action(self, action: Any) -> float:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            # Center of normalized action range.
            return 0.0 if self.config.action_low < 0 else 0.5
        return float(np.clip(arr[0], self.config.action_low, self.config.action_high))

    def _scale_action(self, x: float, low: float, high: float) -> float:
        # Supports both [-1, 1] and [0, 1] normalized action conventions.
        a_low = float(self.config.action_low)
        a_high = float(self.config.action_high)
        if abs(a_high - a_low) <= EPS:
            normalized = 0.5
        else:
            normalized = (float(x) - a_low) / (a_high - a_low)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        return float(low + normalized * (high - low))

    def _normalized_action(self, actual: float, low: float, high: float) -> np.ndarray:
        ratio = (float(actual) - low) / (high - low + EPS)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        action = self.config.action_low + ratio * (self.config.action_high - self.config.action_low)
        return np.array([action], dtype=np.float32)

    def _force_fair_allocation(self) -> bool:
        if self.config.allocation_mode.lower() == "fair":
            return True
        if self.config.mechanism_mode.lower() == "none":
            return True
        return False

    def _allocate_supply(self, orders: np.ndarray, supply: float, weights: np.ndarray) -> np.ndarray:
        orders = np.maximum(orders.astype(np.float64), 0.0)
        weights = np.maximum(weights.astype(np.float64), EPS)
        supply = max(float(supply), 0.0)

        if orders.sum() <= supply + EPS:
            return orders.astype(np.float32)

        remaining_supply = supply
        remaining = orders.copy()
        shipments = np.zeros_like(orders)
        active = remaining > EPS

        # Iterative capped weighted rationing.
        for _ in range(self.num_power + 2):
            if remaining_supply <= EPS or not np.any(active):
                break
            weighted = weights * remaining * active
            denom = weighted.sum()
            if denom <= EPS:
                break
            tentative = remaining_supply * weighted / denom
            capped = np.minimum(tentative, remaining)
            shipments += capped
            remaining -= capped
            remaining_supply -= float(capped.sum())
            active = remaining > EPS

        return np.minimum(shipments, orders).astype(np.float32)

    def _compute_reciprocity_contributions(
        self,
        season: int,
        base_demand: np.ndarray,
        shipments: np.ndarray,
        shipments_fair: np.ndarray,
        shipments_base_u: np.ndarray,
    ) -> Tuple[float, float]:
        """Actual incremental reciprocity contributions gU and gC.

        gU: off-peak incremental transaction caused by U1's actual order above
            baseline behavior.
        gC: peak incremental guarantee to U1 above fair allocation baseline.
        """
        if not self._reciprocity_enabled():
            return 0.0, 0.0

        denom = float(base_demand[0] + EPS)

        if season == 0:
            g_u = min(max(float(shipments[0] - shipments_base_u[0]), 0.0) / denom, 1.0)
            g_c = 0.0
        else:
            g_u = 0.0
            g_c = min(max(float(shipments[0] - shipments_fair[0]), 0.0) / denom, 1.0)

        return float(g_u), float(g_c)

    def _update_trust(self, g_u: float, g_c: float) -> None:
        if not self._reciprocity_enabled():
            self.mu_c = 0.0
            self.mu_u = 0.0
            return

        self.mu_c = float(np.clip(
            (1.0 - self.config.delta_c) * self.mu_c
            + self.config.alpha_c * float(g_u) * (1.0 - self.mu_c),
            0.0,
            1.0,
        ))
        self.mu_u = float(np.clip(
            (1.0 - self.config.delta_u) * self.mu_u
            + self.config.alpha_u * float(g_c) * (1.0 - self.mu_u),
            0.0,
            1.0,
        ))

    def _reciprocity_enabled(self) -> bool:
        return self.config.mechanism_mode.lower() in {"dynamic", "long_contract", "trigger"}

    def _compute_rewards(
        self,
        price: float,
        demand: np.ndarray,
        shipments: np.ndarray,
        served: np.ndarray,
        shortage: np.ndarray,
        ending_inventory: np.ndarray,
        unsold: float,
        fill_rate: np.ndarray,
        jain: float,
        g_u: float,
        g_c: float,
        mu_c_reward: Optional[float] = None,
        mu_u_reward: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        coal_profit = (
            (price - self.config.coal_unit_cost) * float(shipments.sum())
            - self.config.unsold_cost * float(unsold)
        )
        power_profit_vec = (
            self.config.power_unit_revenue * served
            - price * shipments
            - self.config.holding_cost * ending_inventory
            - self.config.shortage_cost * shortage
        )

        system_profit = float(coal_profit + power_profit_vec.sum())
        shortage_rate = float(shortage.sum() / (demand.sum() + EPS))

        rewards: Dict[str, float] = {}
        if mu_c_reward is None:
            mu_c_reward = self.mu_c
        if mu_u_reward is None:
            mu_u_reward = self.mu_u

        r_coal = float(coal_profit)
        if self._reciprocity_enabled():
            r_coal += self.config.eta_c * float(mu_c_reward) * float(g_c)
        r_coal -= self.config.lambda_coal_shortage * shortage_rate
        r_coal -= self.config.lambda_coal_fairness * (1.0 - float(jain))
        rewards[self.coal_agent_id] = self._maybe_clip_reward(r_coal)

        for idx, agent in enumerate(self.power_agent_ids):
            r = float(power_profit_vec[idx])
            if idx == 0 and self._reciprocity_enabled():
                r += self.config.eta_u * float(mu_u_reward) * float(g_u)
            rewards[agent] = self._maybe_clip_reward(r)

        info = {
            "coal_profit": float(coal_profit),
            "power_profit_total": float(power_profit_vec.sum()),
            "system_profit": float(system_profit),
            "shortage_rate": float(shortage_rate),
        }
        for idx in range(self.num_power):
            info[f"power_profit_u{idx + 1}"] = float(power_profit_vec[idx])
        return rewards, info

    def _maybe_clip_reward(self, reward: float) -> float:
        if self.config.reward_clip is None:
            return float(reward)
        bound = float(self.config.reward_clip)
        return float(np.clip(reward, -bound, bound))

    @staticmethod
    def _jain_index(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        numerator = float(np.square(np.sum(x)))
        denominator = float(len(x) * np.sum(np.square(x)) + EPS)
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Observation, info and metrics
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return role-specific local observations.

        Derived system metrics such as system shortage rate, Jain fairness and
        current reciprocity contributions are deliberately excluded from local
        observations. They are available only through infos/history for
        evaluation.
        """
        obs: Dict[str, Dict[str, np.ndarray]] = {}
        season = int(self.current_season)
        ell = self._year_position(self.t)
        theta_public = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        price_obs = float(self.current_price)
        supply_obs = float(self.current_supply)
        pipeline_sum = self.pipeline.sum(axis=0) if self.pipeline.size else np.zeros(self.num_power, dtype=np.float32)

        # Coal observation:
        # (ell_t, z_t, p_t, A_t, mu^C_t, {Q_i,t-1, Y_i,t-1}_{i=1}^m).
        coal_features = self._make_feature_vector(
            role_coal=1.0,
            role_power=0.0,
            is_u1=0.0,
            power_id_norm=0.0,
            ell=ell,
            season=season,
            price=price_obs,
            theta_public=theta_public,
            supply_info=supply_obs,
            own_inventory=0.0,
            own_pipeline=0.0,
            own_last_demand=0.0,
            own_last_order=0.0,
            own_last_shipment=0.0,
            own_last_shortage=0.0,
            mu_obs=self.mu_c if self.config.include_reciprocity_in_obs else 0.0,
            coal_last_orders=self.last_orders,
            coal_last_shipments=self.last_shipments,
        )
        obs[self.coal_agent_id] = {"obs": coal_features}

        # Power-firm observation:
        # (ell_t, z_t, p_t, I_i,t, P_i,t, D_i,t-1, Q_i,t-1,
        #  Y_i,t-1, S_i,t-1, rho_i, mu_obs_i,t)
        for idx, agent in enumerate(self.power_agent_ids):
            is_u1 = 1.0 if idx == 0 else 0.0
            mu_obs = self.mu_u if (idx == 0 and self.config.include_reciprocity_in_obs) else 0.0

            power_features = self._make_feature_vector(
                role_coal=0.0,
                role_power=1.0,
                is_u1=is_u1,
                power_id_norm=(idx + 1) / max(self.num_power, 1),
                ell=ell,
                season=season,
                price=price_obs,
                theta_public=theta_public,
                supply_info=0.0,  # exact A_t is not observed by power firms
                own_inventory=float(self.inventory[idx]),
                own_pipeline=float(pipeline_sum[idx]),
                own_last_demand=float(self.last_demand[idx]),
                own_last_order=float(self.last_orders[idx]),
                own_last_shipment=float(self.last_shipments[idx]),
                own_last_shortage=float(self.last_shortage[idx]),
                mu_obs=mu_obs,
                coal_last_orders=np.zeros(self.num_power, dtype=np.float32),
                coal_last_shipments=np.zeros(self.num_power, dtype=np.float32),
            )
            obs[agent] = {"obs": power_features}

        if self.config.use_global_state:
            state = self._build_joint_state_from_obs(obs)
            for agent_id in self.agent_ids:
                obs[agent_id]["state"] = state.copy()

        return obs

    def _make_feature_vector(self, **kwargs: Any) -> np.ndarray:
        last_orders = np.asarray(kwargs.get("coal_last_orders"), dtype=np.float32)
        last_shipments = np.asarray(kwargs.get("coal_last_shipments"), dtype=np.float32)

        # Pad or truncate to 3 entries to keep obs_dim fixed at 24.
        def pad3(x: np.ndarray) -> np.ndarray:
            out = np.zeros(3, dtype=np.float32)
            n = min(3, len(x))
            if n > 0:
                out[:n] = x[:n]
            return out

        q3 = pad3(last_orders)
        y3 = pad3(last_shipments)

        values = np.array(
            [
                kwargs["role_coal"],            # 0
                kwargs["role_power"],           # 1
                kwargs["is_u1"],                # 2
                kwargs["power_id_norm"],        # 3
                kwargs["ell"],                  # 4
                float(kwargs["season"]),        # 5
                kwargs["price"],                # 6
                kwargs["theta_public"],         # 7
                kwargs["supply_info"],          # 8, coal only
                kwargs["own_inventory"],        # 9, power only
                kwargs["own_pipeline"],         # 10, power only
                kwargs["own_last_demand"],      # 11, power only
                kwargs["own_last_order"],       # 12, power only
                kwargs["own_last_shipment"],    # 13, power only
                kwargs["own_last_shortage"],    # 14, power only
                kwargs["mu_obs"],               # 15
                q3[0],                          # 16, coal only
                q3[1],                          # 17, coal only
                q3[2],                          # 18, coal only
                y3[0],                          # 19, coal only
                y3[1],                          # 20, coal only
                y3[2],                          # 21, coal only
                0.0,                            # 22 reserved
                0.0,                            # 23 reserved
            ],
            dtype=np.float32,
        )
        return np.clip(values, -self.config.obs_clip, self.config.obs_clip).astype(np.float32)


    def _run_debug_checks(
        self,
        orders: np.ndarray,
        shipments: np.ndarray,
        supply: float,
        weights: np.ndarray,
        stage: str,
    ) -> None:
        if np.any(shipments < -1e-6):
            raise RuntimeError(f"{stage}: negative shipment detected.")
        if np.any(shipments - orders > 1e-6):
            raise RuntimeError(f"{stage}: shipment exceeds order.")
        if float(shipments.sum()) - float(supply) > 1e-6:
            raise RuntimeError(f"{stage}: total shipment exceeds supply.")
        if self.config.mechanism_mode.lower() == "none":
            if not np.allclose(weights, np.ones_like(weights), atol=1e-6):
                raise RuntimeError(f"{stage}: no-reciprocity mode should force weights to 1.")

    def _build_infos(self, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        info_t = int(kwargs.get("t", self.t))
        common = {
            "t":info_t,
            "season": int(kwargs["season"]),
            "ell": self._year_position(info_t),
            "price": float(kwargs["price"]),
            "supply": float(kwargs["supply"]),
            "total_demand": float(np.sum(kwargs["demand"])),
            "total_order": float(np.sum(kwargs["orders"])),
            "total_shipment": float(np.sum(kwargs["shipments"])),
            "total_shortage": float(np.sum(kwargs["shortage"])),
            "shortage_rate": float(np.sum(kwargs["shortage"]) / (np.sum(kwargs["demand"]) + EPS)),
            "jain": float(kwargs["jain"]),
            "unsold": float(kwargs["unsold"]),
            "mu_c": float(self.mu_c),
            "mu_u": float(self.mu_u),
            "g_u": float(kwargs["g_u"]),
            "g_c": float(kwargs["g_c"]),
            **kwargs["profit_info"],
        }
        infos: Dict[str, Dict[str, Any]] = {self.coal_agent_id: dict(common)}

        for idx, agent in enumerate(self.power_agent_ids):
            info = dict(common)
            info.update(
                {
                    "own_demand": float(kwargs["demand"][idx]),
                    "own_d_tilde": float(kwargs["d_tilde"][idx]),
                    "own_order": float(kwargs["orders"][idx]),
                    "own_shipment": float(kwargs["shipments"][idx]),
                    "own_fair_shipment": float(kwargs["shipments_fair"][idx]),
                    "own_base_u_shipment": float(kwargs["shipments_base_u"][idx]),
                    "own_served": float(kwargs["served"][idx]),
                    "own_shortage": float(kwargs["shortage"][idx]),
                    "own_inventory": float(kwargs["ending_inventory"][idx]),
                    "own_fill_rate": float(kwargs["fill_rate"][idx]),
                }
            )
            infos[agent] = info

        return infos

    def _record_history(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            if isinstance(value, np.ndarray):
                self.history[key].append(value.astype(np.float32).copy())
            else:
                self.history[key].append(float(value))

    def get_episode_metrics(self) -> Dict[str, float]:
        if not self.history.get("demand"):
            return {}

        demand = np.asarray(self.history["demand"], dtype=np.float64)
        orders = np.asarray(self.history["orders"], dtype=np.float64)
        shortage = np.asarray(self.history["shortage"], dtype=np.float64)
        fill_rate = np.asarray(self.history["fill_rate"], dtype=np.float64)
        profits = np.asarray(self.history["system_profit"], dtype=np.float64)
        jain = np.asarray(self.history["jain"], dtype=np.float64)

        demand_cv = self._cv(demand.sum(axis=1))
        order_cv = self._cv(orders.sum(axis=1))
        bullwhip = np.nan if demand_cv < 1e-6 else order_cv / demand_cv

        safe_inventory = np.array(
            [self._safe_inventory_vector(self._season(t)) for t in range(len(demand))]
        )
        inventory = np.asarray(self.history["inventory"], dtype=np.float64)
        inventory_violation = float(np.mean(inventory < safe_inventory))

        return {
            "total_shortage_rate": float(shortage.sum() / (demand.sum() + EPS)),
            "system_profit": float(profits.sum()),
            "avg_jain": float(jain.mean()),
            "inventory_violation_rate": inventory_violation,
            "order_cv": float(order_cv),
            "demand_cv": float(demand_cv),
            "bullwhip_ratio": float(bullwhip) if not np.isnan(bullwhip) else np.nan,
            "u1_shortage_rate": float(shortage[:, 0].sum() / (demand[:, 0].sum() + EPS)),
            "ordinary_shortage_rate": float(shortage[:, 1:].sum() / (demand[:, 1:].sum() + EPS)) if self.num_power > 1 else 0.0,
            "mean_mu_c": float(np.mean(self.history["mu_c"])),
            "mean_mu_u": float(np.mean(self.history["mu_u"])),
            "mean_g_u": float(np.mean(self.history["g_u"])),
            "mean_g_c": float(np.mean(self.history["g_c"])),
            "mean_fill_rate": float(fill_rate.mean()),
            "mean_unsold": float(np.mean(self.history["unsold"])),
            # Backward-compatible aliases
            "mean_trust_u": float(np.mean(self.history["mu_c"])),
            "mean_trust_c": float(np.mean(self.history["mu_u"])),
        }

    @staticmethod
    def _cv(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.std(x) / (np.mean(x) + EPS))

    # ------------------------------------------------------------------
    # Heuristic action helpers for non-learning baselines
    # ------------------------------------------------------------------
    def normalized_action_from_weight(self, w: float) -> np.ndarray:
        return self._normalized_action(w, self.config.w_min, self.config.w_max)

    def normalized_action_from_omega(self, omega: float) -> np.ndarray:
        return self._normalized_action(omega, self.config.omega_min, self.config.omega_max)

    def base_stock_action_dict(self, omega: Optional[float] = None) -> Dict[str, np.ndarray]:
        if omega is None:
            omega = self.config.omega_base

        actions = {
            self.coal_agent_id: self.normalized_action_from_weight(1.0)
        }
        for agent in self.power_agent_ids:
            actions[agent] = self.normalized_action_from_omega(omega)
        return actions

    def seasonal_base_stock_action_dict(
        self,
        omega_offpeak: float = 2.0,
        omega_peak: float = 2.5,
    ) -> Dict[str, np.ndarray]:
        """Simple non-stationary base-stock helper for experiment-one baselines."""
        omega = omega_offpeak if int(self.current_season) == 0 else omega_peak
        return self.base_stock_action_dict(omega=omega)


CPDRE = CoalPowerDirectReciprocityEnv


if __name__ == "__main__":
    env = CPDRE(
        {
            "mechanism_mode": "none",
            "allocation_mode": "fair",
            "demand_mode": "deterministic",
            "price_mode": "fixed",
            "episode_len": 10,
            "seed": 2026,
        }
    )
    obs = env.reset()
    print("agents:", env.agent_ids)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)
    for agent, agent_obs in obs.items():
        print(agent, agent_obs["obs"].shape, agent_obs["obs"].dtype)

    for step in range(10):
        action = env.base_stock_action_dict(omega=2.0)
        obs, rewards, dones, infos = env.step(action)
        print(
            step,
            rewards,
            infos[env.coal_agent_id]["shortage_rate"],
            infos[env.coal_agent_id]["system_profit"],
            infos[env.coal_agent_id]["mu_c"],
            infos[env.coal_agent_id]["mu_u"],
        )
        if dones["__all__"]:
            break
    print(env.get_episode_metrics())
