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
    """Default configuration for CPDRE, aligned to model document 0611(1).md.

    Quantity variables are normalized by one ordinary power firm's average
    weekly coal demand D_bar. Price is normalized by the base coal price.

    Conventions (model 4.2 / 4.3 / 4.4):
    - Coal agent action is 2D: (theta_t, lambda_{1,t}).
      theta_t in [theta_min, theta_max] is the capacity-utilization factor;
      G_t = theta_t * K. lambda_{1,t} in [1, lambda_max] is the guarantee
      weight for the reciprocal firm U_1.
    - Power agent action is 1D: target inventory coverage omega_i,t.
    - Capacity adjustment is ramp-limited: |theta_t - theta_{t-1}| <= rho_max.
    - Demand is sinusoidal: s^D_t = 1 + a_D * sin(2*pi*(t - phi_D)/W).
    """

    # Episode and system scale
    num_power: int = 3
    episode_len: int = 156
    weeks_per_year: int = 52
    off_peak_weeks: int = 26          # kept for legacy season-based helpers only
    lead_time: int = 1
    seed: int = 42
    eval_seed_offset: int = 9000      # fixed eval trajectory seed (model 4.3 table)

    # Experiment mode
    # none   : rule/fixed baseline, fair allocation, no relationship state.
    # b2     : learning no-reciprocity baseline (both learn, lambda=1, no memory).
    # b4     : relationship-specific dynamic reciprocity (full: mu in obs,
    #          lambda learnable, theta learnable).
    # b5     : non-relational dynamic coordination (coal learns theta from PUBLIC
    #          state only; lambda=1; no U1 identity, no memory, no per-firm
    #          differentiation in coal obs).
    # dynamic / long_contract / trigger : legacy rule reciprocity modes.
    mechanism_mode: str = "none"
    allocation_mode: str = "fair"     # fair, weighted
    demand_mode: str = "deterministic"  # deterministic, low_noise, lognormal, truncated_normal, ar1
    price_mode: str = "seasonal"      # fixed, seasonal, feedback

    # Demand process (model 4.3.1): D_{i,t} = D_bar * s^D_t * eta_{i,t},
    # s^D_t = 1 + a_D * sin(2*pi*(t - phi_D)/W).
    demand_offpeak: float = 1.0       # = D_bar (mean weekly demand)
    demand_peak: float = 1.4          # legacy two-level peak (kept for legacy modes)
    demand_season_amp: float = 0.20   # a_D (model 4.5.2)
    demand_season_phi: int = 0        # phi_D, phase in weeks
    demand_sigma: float = 0.15        # sigma_D multiplicative noise stdev
    demand_ar1_rho: float = 0.60
    demand_corr: float = 0.0          # cross-firm correlation (model exp4 D0-D3)
    # Per-firm volatility scale (model exp4 D4: U1 higher volatility).
    # If empty, all firms use 1.0. Otherwise demand_sigma * vol_scale[i].
    demand_vol_scale: tuple = ()
    demand_common_pool: bool = False  # model exp4 D3: all firms face same shock

    # Supply process (model 4.3.1): G_t = theta_t * K, K = m * D_bar.
    K_capacity: Optional[float] = None  # K; None -> m * demand_offpeak (model 4.5.2)
    theta_min: float = 0.70           # model 4.5.2
    theta_max: float = 1.20
    theta_init: float = 1.0           # initial capacity utilization (first step)
    rho_max: float = 0.05             # ramp constraint (model 4.5.2)
    # legacy season-based theta (used only by legacy none/rule modes that do not
    # learn theta): kept for backward compatibility of rule baselines.
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

    # Demand-rate estimator (model 4.3.2):
    # D_tilde_i,t = lambda_D * D_{i,t-1} + (1-lambda_D) * D_bar_i
    lambda_d: float = 0.30

    # Coal allocation guarantee weight
    lambda_max: float = 2.0           # model 4.5.2
    w_min: float = 0.5                # legacy (legacy rule modes map to lambda range)
    w_max: float = 2.0
    w_high: float = 1.5
    # Number of reciprocal firms (model exp5 E2/E3); h>=1.
    num_reciprocal: int = 1

    # Trust and reciprocity (model 4.3.5)
    mu_c_init: float = 0.50
    mu_u_init: float = 0.50
    alpha_c: float = 0.05
    alpha_u: float = 0.05
    delta_c: float = 0.005
    delta_u: float = 0.005
    trust_threshold_c: float = 0.40   # rule-reciprocity trigger (model 4.5.2)
    trust_threshold_u: float = 0.40

    # Reciprocity reward coefficients. Model 4.3.5/4.3.6 EXPLICITLY EXCLUDES the
    # eta*mu*g term from the main-model reward. It is kept behind a flag for the
    # extension ablation only (model 4.5.5 note). Default False = main model.
    use_reciprocity_reward: bool = False
    eta_c: float = 0.30
    eta_u: float = 0.30

    # Economic parameters (model 4.5.2 table 4.2)
    p_base: float = 1.00              # base/long-contract coal price
    coal_unit_cost: float = 0.70      # c^C, applied to G_t (model 4.3.6)
    unsold_cost: float = 0.10         # legacy idle cost; NOT used in main model
    power_unit_revenue: float = 1.80  # r
    holding_cost: float = 0.03        # h
    c_rep: float = 1.60               # external emergency purchase price (p<c_rep<r)
    xi_lost: float = 5.00             # lost-load penalty (robustness scenario only)
    shortage_cost: float = 5.00       # legacy alias of xi_lost; kept for old configs

    # Reward shaping / responsibility assignment (model 4.5.2 table 4.2)
    lambda_coal_shortage: float = 0.50   # lambda_S
    lambda_coal_fairness: float = 0.30   # lambda_J

    # Observation and numerical control
    obs_clip: float = 10.0
    reward_clip: Optional[float] = None
    initial_inventory_weeks: float = 2.0
    include_reciprocity_in_obs: bool = True
    debug_checks: bool = False

    # Ablation toggles (model 4.5.5).
    # disable_g_u (C2): zero U1's purchase contribution g^U so it never builds
    #   relationship credit, while keeping g^C and memory updates.
    # freeze_memory (C4): keep mu^C / mu^U at their initial values; the
    #   contribution g^U/g^C is still computed and observed but does not update
    #   the relationship memory.
    disable_g_u: bool = False
    freeze_memory: bool = False

    # If True, expose an explicit global state space and get_state()/state() for
    # MARLlib/RLlib versions that consume global state in centralized critics.
    # Keep False if your local MARLlib expects joint observations instead.
    use_global_state: bool = True

    # C5 ablation (model 4.5.5): freeze coal capacity path to a fixed schedule
    # and only allow the guarantee weight lambda to vary. When True the coal
    # agent's theta action is ignored and theta follows the legacy schedule.
    freeze_theta: bool = False

    # Optional manual fixed policy behavior for non-learning baselines.
    fixed_power_omega: Optional[float] = None
    fixed_coal_weight: Optional[float] = None
    fixed_coal_theta: Optional[float] = None

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

        # Per-agent action spaces (model 4.4.3): coal = 2D (theta, lambda),
        # power = 1D (omega). All normalized to [-1, 1].
        self._coal_act_dim = 2
        self._power_act_dim = 1
        self.act_dims = [self._coal_act_dim] + [self._power_act_dim] * self.num_power
        self.max_act_dim = int(max(self.act_dims))

        lo = float(self.config.action_low)
        hi = float(self.config.action_high)
        self.action_spaces = {
            self.coal_agent_id: spaces.Box(
                low=np.full(self._coal_act_dim, lo, dtype=np.float32),
                high=np.full(self._coal_act_dim, hi, dtype=np.float32),
                dtype=np.float32,
            )
        }
        for aid in self.power_agent_ids:
            self.action_spaces[aid] = spaces.Box(
                low=np.array([lo], dtype=np.float32),
                high=np.array([hi], dtype=np.float32),
                dtype=np.float32,
            )
        # Legacy shared single action space (max dim) for RLlib / old wrappers.
        self.action_space = spaces.Box(
            low=np.full(self.max_act_dim, lo, dtype=np.float32),
            high=np.full(self.max_act_dim, hi, dtype=np.float32),
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
        # Effective supply G_t = theta_t * K (model 4.3.1). theta starts at the
        # configured init value; the coal agent updates it each step.
        self.theta = float(self.config.theta_init)
        self.theta_prev = float(self.config.theta_init)
        self.theta_raw = float(self.config.theta_init)
        self.last_ramp_hit = False
        self.K = float(self.config.K_capacity) if self.config.K_capacity is not None \
            else float(self.num_power) * float(self.config.demand_offpeak)
        self.current_supply = float(self.theta * self.K)
        self.price = float(self.config.price_offpeak)

        # Supply-demand pressure chi_t (model 4.3.1): chi_t = (sum Q - G_t)/(G_t+eps).
        # chi_hat_t = chi_{t-1} is the observable signal for rules / learning.
        self.chi_t = 0.0
        self.chi_prev = 0.0

        # Reciprocal-firm indices (model exp5). power index 0..h-1 are reciprocal.
        self.reciprocal_idx = list(range(min(self.config.num_reciprocal, self.num_power)))
        self.num_reciprocal = len(self.reciprocal_idx)

        # Model notation: mu^C = coal's trust in U1; mu^U = U1's trust in coal.
        # Vectorized over reciprocal firms (h>=1). For h=1 these are scalars
        # accessed via mu_c[0] / mu_u[0] to keep the multi-reciprocal path uniform.
        self.mu_c = np.zeros(self.num_reciprocal, dtype=np.float32)
        self.mu_u = np.zeros(self.num_reciprocal, dtype=np.float32)

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
        self.last_service_rate = np.ones(self.num_power, dtype=np.float32)
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
        cfg.num_reciprocal = max(1, int(cfg.num_reciprocal))
        cfg.demand_season_phi = int(cfg.demand_season_phi)

        # Backward compatibility for earlier config keys.
        if "forecast_beta" in raw_config:
            cfg.lambda_d = float(raw_config["forecast_beta"])
        if "trust_init" in raw_config:
            cfg.mu_c_init = float(raw_config["trust_init"])
            cfg.mu_u_init = float(raw_config["trust_init"])
        if "alpha_pos" in raw_config:
            cfg.alpha_c = float(raw_config["alpha_pos"])
            cfg.alpha_u = float(raw_config["alpha_pos"])
        # Old configs passed a single shortage_cost (now split into c_rep /
        # xi_lost). Map the legacy key onto c_rep unless c_rep was set.
        if "shortage_cost" in raw_config and "c_rep" not in raw_config:
            cfg.c_rep = float(raw_config["shortage_cost"])
        # Normalize demand_vol_scale to a tuple.
        if isinstance(cfg.demand_vol_scale, (list, tuple)):
            cfg.demand_vol_scale = tuple(float(v) for v in cfg.demand_vol_scale)
        else:
            cfg.demand_vol_scale = ()
        return cfg

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = self.config.seed
        self.config.seed = int(seed)
        self.rng = np.random.default_rng(self.config.seed)
        return [self.config.seed]

    @property
    def mu_c_scalar(self) -> float:
        """Scalar coal trust in U_1 (first reciprocal firm), for h=1 paths."""
        return float(self.mu_c[0]) if self.num_reciprocal else 0.0

    @property
    def mu_u_scalar(self) -> float:
        """Scalar U_1 trust in coal (first reciprocal firm), for h=1 paths."""
        return float(self.mu_u[0]) if self.num_reciprocal else 0.0

    def eval_reset(self, episode_idx: int, eval_seed_offset: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Deterministic reset for evaluation (model 4.3 table: s^eval_0=9000).

        Seeding self.rng once before the reset body makes the whole 156-step
        episode deterministic given episode_idx, because the demand / price /
        supply processes all draw from self.rng step-by-step. Does NOT overwrite
        config.seed so training rng can be restored afterwards.
        """
        offset = self.config.eval_seed_offset if eval_seed_offset is None else int(eval_seed_offset)
        self.rng = np.random.default_rng(int(offset) + int(episode_idx))
        return self.reset()

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
            self.mu_c = np.full(self.num_reciprocal, float(self.config.mu_c_init), dtype=np.float32)
            self.mu_u = np.full(self.num_reciprocal, float(self.config.mu_u_init), dtype=np.float32)
        else:
            self.mu_c = np.zeros(self.num_reciprocal, dtype=np.float32)
            self.mu_u = np.zeros(self.num_reciprocal, dtype=np.float32)

        # Capacity utilization and pressure state.
        self.theta = float(self.config.theta_init)
        self.theta_prev = float(self.config.theta_init)
        self.theta_raw = float(self.config.theta_init)
        self.last_ramp_hit = False
        self.chi_t = 0.0
        self.chi_prev = 0.0

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
        self.last_service_rate = np.ones(self.num_power, dtype=np.float32)
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
            "service_rate": [],
            "omega": [],
            "weight": [],
            "price": [],
            "supply": [],
            "theta": [],
            "lambda1": [],
            "ramp_hit": [],
            "chi": [],
            "chi_hat": [],
            "G": [],
            "mu_c": [],
            "mu_u": [],
            "g_u": [],
            "g_c": [],
            "jain": [],
            "system_profit": [],
            "coal_profit": [],
            "coal_profit_norm": [],
            "power_profit_total": [],
            "power_profit_norm_total": [],
            "shortage_rate": [],
            "shortage_norm": [],
            "fairness_penalty": [],
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

        # 2. At the decision epoch, self.inventory is already the usable beginning
        # inventory I_{i,t}. For lead_time=1, last period's shipment has already
        # been moved into inventory during the previous transition.
        available_inventory = self.inventory.copy().astype(np.float32)

        # 3. Parse simultaneous actions. Coal's theta_t is committed here and
        # sets the current effective supply G_t = theta_t * K (model 4.3.1).
        omega, weights, theta_t = self._parse_actions(action_dict, season)
        if self._force_fair_allocation():
            weights = np.ones(self.num_power, dtype=np.float32)

        self.theta = float(theta_t)
        self.theta_prev = float(theta_t)
        supply = float(self.theta) * float(self.K)
        if self.config.supply_sigma > EPS:
            supply *= float(np.exp(-0.5 * self.config.supply_sigma**2 + self.config.supply_sigma * self.rng.normal()))
        supply = float(max(supply, 0.0))
        self.current_supply = supply  # keep obs/step aligned on this G_t

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

        # Supply-demand pressure chi_t (model 4.3.1): chi_t = (sum Q - G_t)/(G_t+eps).
        # Computed after orders and G_t are known; used to gate g^U / g^C and as
        # the next period's observable signal chi_hat = chi_{t-1}.
        self.chi_prev = float(self.chi_t)
        self.chi_t = float((float(orders.sum()) - supply) / (supply + EPS))

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
        # Model 4.3.6: Jain fairness on the internal service rate sigma_{i,t},
        # NOT on the order fill rate. sigma = (D - S)/(D + eps).
        service_rate = (demand - shortage) / (demand + EPS)
        jain = self._jain_index(service_rate)
        unsold = max(float(supply) - float(shipments.sum()), 0.0)

        g_u, g_c = self._compute_reciprocity_contributions(
            season=season,
            base_demand=base_demand,
            supply=float(supply),
            orders=orders,
            q_base_u1=float(q_base_u1),
            shipments=shipments,
            shipments_fair=shipments_fair,
        )


        if self.config.debug_checks and self.config.mechanism_mode.lower() == "none":
            if abs(float(g_u)) > 1e-8 or abs(float(g_c)) > 1e-8:
                raise RuntimeError("No-reciprocity mode should have zero reciprocity contributions.")

        # Rewards use beginning-of-period trust mu_t. The current contribution
        # updates trust only for the next state mu_{t+1}.
        mu_c_before = self.mu_c_scalar
        mu_u_before = self.mu_u_scalar

        rewards, profit_info = self._compute_rewards(
            price=price,
            demand=demand,
            shipments=shipments,
            served=served,
            shortage=shortage,
            ending_inventory=ending_inventory,
            unsold=unsold,
            fill_rate=fill_rate,
            service_rate=service_rate,
            jain=jain,
            g_u=g_u,
            g_c=g_c,
            G_t=float(supply),
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
        self.last_service_rate = service_rate.astype(np.float32)
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
            service_rate=service_rate,
            omega=omega,
            weight=weights,
            price=price,
            supply=supply,
            theta=self.theta,
            lambda1=float(weights[self.reciprocal_idx[0]]) if self.reciprocal_idx else 1.0,
            ramp_hit=self.last_ramp_hit,
            chi=self.chi_t,
            chi_hat=self.chi_prev,
            G=supply,
            mu_c=self.mu_c,
            mu_u=self.mu_u,
            g_u=g_u,
            g_c=g_c,
            jain=jain,
            system_profit=profit_info["system_profit"],
            coal_profit=profit_info["coal_profit"],
            coal_profit_norm=profit_info["coal_profit_norm"],
            power_profit_total=profit_info["power_profit_total"],
            power_profit_norm_total=profit_info["power_profit_norm_total"],
            shortage_rate=self.last_shortage_rate,
            shortage_norm=profit_info["shortage_norm"],
            fairness_penalty=profit_info["fairness_penalty"],
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
            service_rate=service_rate,
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
        # Legacy binary season used only by the old rule reciprocity modes
        # (long_contract / trigger) and the state-aware base-stock helper.
        week = t % self.config.weeks_per_year
        return 0 if week < self.config.off_peak_weeks else 1

    def _seasonal_factor(self, t: int) -> float:
        """Sinusoidal seasonal demand factor s^D_t (model 4.3.1).

        s^D_t = 1 + a_D * sin(2*pi*(t - phi_D)/W). When a_D == 0 the demand is
        flat (no seasonality). The factor is the same for all firms.
        """
        a_D = float(self.config.demand_season_amp)
        if abs(a_D) <= EPS:
            return 1.0
        W = max(int(self.config.weeks_per_year), 1)
        phi = float(self.config.demand_season_phi)
        return float(1.0 + a_D * np.sin(2.0 * np.pi * (float(t) - phi) / float(W)))

    def _year_position(self, t: Optional[int] = None) -> float:
        if t is None:
            t = self.t
        return float((t % self.config.weeks_per_year) / max(self.config.weeks_per_year, 1))

    def _refresh_public_state(self) -> None:
        """Generate public exogenous state for the current decision epoch.

        The observation returned before action and the subsequent step()
        settlement must use the same season, price, and effective supply.
        Effective supply G_t = theta_t * K is set by the coal action in step()
        (or by the legacy schedule for non-learning modes); this method only
        refreshes the seasonal base demand and price.
        """
        season = self._season(self.t)
        base_demand = self._base_demand_vector(season)
        price = self._generate_price(season)
        # G_t is owned by the coal action (D1). Keep current_supply in sync with
        # the committed theta so obs/step settlement agree. theta was committed
        # at the end of the previous step (or initialized in reset).
        supply = float(self.theta) * float(self.K)
        if self.config.supply_sigma > EPS:
            # supply shock is a per-step multiplicative noise around G_t
            supply *= float(np.exp(-0.5 * self.config.supply_sigma**2 + self.config.supply_sigma * self.rng.normal()))

        self.current_season = int(season)
        self.current_base_demand = base_demand.astype(np.float32)
        self.current_price = float(price)
        self.current_supply = float(max(supply, 0.0))

    def _base_demand_vector(self, season: int) -> np.ndarray:
        # Model 4.3.1: base demand = D_bar * s^D_t (sinusoidal). The legacy
        # off-peak/peak two-level scheme is used only by old rule modes that
        # do not learn theta (mechanism_mode in {none, long_contract, trigger}).
        mode = self.config.mechanism_mode.lower()
        if mode in {"b2", "b4", "b5", "dynamic"}:
            sf = self._seasonal_factor(self.t)
            return np.ones(self.num_power, dtype=np.float32) * float(self.config.demand_offpeak) * sf
        base = self.config.demand_offpeak if season == 0 else self.config.demand_peak
        return np.ones(self.num_power, dtype=np.float32) * float(base)

    def _safe_inventory_vector(self, season: int) -> np.ndarray:
        base = self._base_demand_vector(season)
        return (self.config.safe_inventory_weeks * base).astype(np.float32)

    def _estimate_demand_rate(self, base_demand: np.ndarray) -> np.ndarray:
        # Model 4.3.2: D_tilde_i = lambda_D * D_{i,t-1} + (1-lambda_D) * D_bar_i.
        # base_demand here carries the seasonal factor for the current period,
        # serving as the observable D_bar_i(z_t) proxy.
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

        # Model exp4 D3 (common pool): all firms face the SAME shock.
        if self.config.demand_common_pool and not np.allclose(eta, 1.0):
            eta = np.full(self.num_power, float(eta.mean()), dtype=np.float32)

        # Model exp4 D4 (U1 higher volatility): per-firm multiplicative scale.
        if self.config.demand_vol_scale:
            scales = np.ones(self.num_power, dtype=np.float32)
            n = min(len(self.config.demand_vol_scale), self.num_power)
            scales[:n] = np.asarray(self.config.demand_vol_scale[:n], dtype=np.float32)
            # Blend toward the mean so the per-firm mean stays ~1: rescale eta
            # around 1 by the firm's volatility ratio.
            eta = (1.0 + (eta - 1.0) * scales).astype(np.float32)

        return np.maximum(base_demand * eta, 0.0).astype(np.float32)

    def _correlated_lognormal(self, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 0.0)
        if sigma <= EPS:
            return np.ones(self.num_power, dtype=np.float32)
        rho = float(np.clip(self.config.demand_corr, 0.0, 0.999))
        # Model exp4 D3 (common pool): rho -> 1 means identical shocks.
        if self.config.demand_common_pool:
            rho = 0.999
        common = self.rng.normal()
        indiv = self.rng.normal(size=self.num_power)
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * indiv
        eta = np.exp(-0.5 * sigma**2 + sigma * z)
        return eta.astype(np.float32)

    def _generate_supply(self, season: int, base_demand: np.ndarray) -> float:
        """Legacy supply schedule for non-learning rule modes.

        Learning modes (b2/b4/b5/dynamic) drive G_t = theta_t * K from the coal
        action instead, so this is only used by {none, long_contract, trigger}
        baselines that fix theta to the off-peak/peak schedule.
        """
        theta = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        mean_supply = float(theta) * float(self.K)
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

    def _parse_actions(self, action_dict: Dict[str, Any], season: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Parse per-agent actions into (omega, weights, theta_t).

        Coal agent action is 2D (theta_raw, lambda_raw) in [-1,1]. theta_t is
        the capacity-utilization factor (ramp-limited), lambda_{1,t} is the
        guarantee weight for U_1. Power actions are 1D omega in [-1,1].
        Returns the realized (post-ramp) theta_t.
        """
        omega = np.zeros(self.num_power, dtype=np.float32)
        weights = np.ones(self.num_power, dtype=np.float32)

        # Coal action: (theta_raw, lambda_raw). Old single-dim inputs (legacy
        # rule modes) are handled by _get_scalar_action reading arr[0].
        coal_arr = np.asarray(action_dict.get(self.coal_agent_id, np.zeros(self._coal_act_dim, dtype=np.float32)),
                              dtype=np.float32).reshape(-1)
        coal_arr = np.clip(coal_arr, self.config.action_low, self.config.action_high)
        theta_raw = self._scale_action(float(coal_arr[0]) if coal_arr.size else 0.0,
                                       self.config.theta_min, self.config.theta_max)
        lambda_raw = self._scale_action(float(coal_arr[1]) if coal_arr.size > 1 else 0.0,
                                        1.0, self.config.lambda_max)

        for idx, agent in enumerate(self.power_agent_ids):
            a = self._get_scalar_action(action_dict.get(agent, np.array([0.0], dtype=np.float32)))
            omega[idx] = self._scale_action(a, self.config.omega_min, self.config.omega_max)

        if self.config.fixed_power_omega is not None:
            omega[:] = float(self.config.fixed_power_omega)
        if self.config.fixed_coal_weight is not None:
            lambda_raw = float(self.config.fixed_coal_weight)
        if self.config.fixed_coal_theta is not None:
            theta_raw = float(self.config.fixed_coal_theta)

        mode = self.config.mechanism_mode.lower()
        # Learning modes: coal learns theta (unless frozen for the C5 ablation),
        # and learns lambda only in b4.
        if mode in {"b4", "dynamic"}:
            # b4: full relationship-specific reciprocity. weights[0] = lambda.
            weights[:] = 1.0
            weights[self.reciprocal_idx[0]] = float(lambda_raw)
            if self.config.freeze_theta:
                # C5 ablation: theta follows the legacy schedule, only lambda varies.
                theta_t = self.config.theta_offpeak if season == 0 else self.config.theta_peak
            else:
                theta_t = float(theta_raw)
        elif mode == "b5":
            # Non-relational dynamic coordination: coal learns theta from PUBLIC
            # state only. lambda forced to 1 (no exclusive guarantee weight).
            weights[:] = 1.0
            theta_t = float(theta_raw)
        elif mode == "b2":
            # Learning no-reciprocity baseline: both learn, lambda=1, no memory.
            weights[:] = 1.0
            theta_t = float(theta_raw)
        elif mode == "none":
            weights[:] = 1.0
            theta_t = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        elif mode == "long_contract":
            weights[:] = 1.0
            if season == 0:
                omega[0] = float(self.config.omega_high)
            else:
                weights[0] = float(self.config.w_high)
            theta_t = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        elif mode == "trigger":
            weights[:] = 1.0
            if season == 0 and self.mu_u_scalar >= self.config.trust_threshold_u:
                omega[0] = float(self.config.omega_high)
            if season == 1 and self.mu_c_scalar >= self.config.trust_threshold_c:
                weights[0] = float(self.config.w_high)
            theta_t = self.config.theta_offpeak if season == 0 else self.config.theta_peak
        else:
            raise ValueError(f"Unknown mechanism_mode: {self.config.mechanism_mode}")

        # Ramp constraint (model 4.3.1): |theta_t - theta_{t-1}| <= rho_max.
        theta_raw_pre_ramp = float(theta_t)
        theta_t = float(np.clip(theta_t, self.theta_prev - self.config.rho_max,
                                self.theta_prev + self.config.rho_max))
        theta_t = float(np.clip(theta_t, self.config.theta_min, self.config.theta_max))
        ramp_hit = abs(theta_t - theta_raw_pre_ramp) > 1e-9

        omega = np.clip(omega, self.config.omega_min, self.config.omega_max).astype(np.float32)
        weights = np.clip(weights, 1.0, self.config.lambda_max).astype(np.float32)
        self.theta_raw = theta_raw_pre_ramp
        self.last_ramp_hit = bool(ramp_hit)
        return omega, weights, theta_t

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
        # Only force equal weights for groups with NO exclusive guarantee weight.
        # b4 / dynamic learn lambda_{1,t} and must NOT be overridden by fair allocation
        # (otherwise the guarantee return channel g^C never fires).
        mode = self.config.mechanism_mode.lower()
        if mode in {"none", "b2", "b5"}:
            return True
        if self.config.allocation_mode.lower() == "fair" and mode not in {"b4", "dynamic", "long_contract", "trigger"}:
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
        supply: float,
        orders: np.ndarray,
        q_base_u1: float,
        shipments: np.ndarray,
        shipments_fair: np.ndarray,
    ) -> Tuple[float, float]:
        """Actual incremental reciprocity contributions gU and gC (model 4.3.4).

        Both contributions are identified from ex-post realized transactions and
        gated on the supply-demand pressure chi_t (NOT on the calendar season):

        gU (loose, chi_t <= 0): U1's incremental purchase support relative to the
            base-stock counterfactual order:
                gU_t = I(chi_t<=0) * min{ [Q_{1,t} - Q^{base}_{1,t}]^+ / (D_bar + eps), 1 }
            Only counts when the system is loose, so U1 cannot earn fake
            reciprocity by crowding out ordinary firms under rationing.

        gC (tight, chi_t > 0): coal's incremental guarantee to U1 above the fair
            allocation baseline:
                gC_t = I(chi_t>0) * min{ [Y_{1,t} - Y^0_{1,t}]^+ / (D_bar + eps), 1 }
            Only counts under rationing, so loose-period shipments that need no
            competition are not misread as guarantee returns.
        """
        if not self._reciprocity_enabled():
            return 0.0, 0.0

        denom = float(base_demand[0] + EPS)
        chi = float(self.chi_t)

        if chi <= 0.0:
            # Loose: U1 order increment above the base-stock counterfactual.
            # C2 ablation (disable_g_u): zero U1's purchase contribution.
            g_u_raw = min(max(float(orders[0]) - float(q_base_u1), 0.0) / denom, 1.0)
            g_u = 0.0 if self.config.disable_g_u else g_u_raw
            g_c = 0.0
        else:
            # Tight: coal's shipment to U1 above the fair-allocation baseline.
            g_u = 0.0
            g_c = min(max(float(shipments[0] - shipments_fair[0]), 0.0) / denom, 1.0)

        return float(g_u), float(g_c)

    def _update_trust(self, g_u: float, g_c: float) -> None:
        # Model 4.3.5: contribution-decay relationship memory.
        #   mu^C_{t+1} = clip((1-delta_C) mu^C_t + alpha_C g^U_t (1-mu^C_t), 0, 1)
        #   mu^U_{t+1} = clip((1-delta_U) mu^U_t + alpha_U g^C_t (1-mu^U_t), 0, 1)
        # Vectorized over reciprocal firms; for h=1 only index 0 is used.
        if not self._reciprocity_enabled():
            self.mu_c = np.zeros(self.num_reciprocal, dtype=np.float32)
            self.mu_u = np.zeros(self.num_reciprocal, dtype=np.float32)
            return

        # C4 ablation (freeze_memory): contributions are still computed and
        # observed, but the relationship memory does not update.
        if self.config.freeze_memory:
            return

        self.mu_c = np.clip(
            (1.0 - self.config.delta_c) * self.mu_c
            + self.config.alpha_c * float(g_u) * (1.0 - self.mu_c),
            0.0, 1.0,
        ).astype(np.float32)
        self.mu_u = np.clip(
            (1.0 - self.config.delta_u) * self.mu_u
            + self.config.alpha_u * float(g_c) * (1.0 - self.mu_u),
            0.0, 1.0,
        ).astype(np.float32)

    def _reciprocity_enabled(self) -> bool:
        # Only modes that USE relationship memory enable the reciprocity channel.
        # b2/b5/none do not (b2 and b5 are learning baselines without relations).
        return self.config.mechanism_mode.lower() in {"b4", "dynamic", "long_contract", "trigger"}

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
            service_rate: np.ndarray,
            jain: float,
            g_u: float,
            g_c: float,
            G_t: float,
            mu_c_reward: Optional[float] = None,
            mu_u_reward: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        cfg = self.config

        # 1. Economic profits (model 4.3.6).
        # Coal: pi^C_t = p_t * sum(Y) - c^C * G_t. Cost is on the effective
        # supply G_t (produced whether sold or not), NO idle/unsold cost and NO
        # ramp cost in the main model. This is the causal lever the mechanism
        # relies on: U1's loose-period incremental purchase lowers the coal
        # firm's expected loss on the G_t it has already committed to pay for.
        coal_profit = (
                float(price) * float(shipments.sum())
                - cfg.coal_unit_cost * float(G_t)
        )

        # Power: pi^{U_i} = r*D - p*Y - c_rep*S - h*I_end (model 4.3.6).
        # Revenue is on the FULL real demand D_{i,t}: the main model assumes the
        # power firm buys from the external market to cover the internal shortage
        # S_{i,t} and still generates (and earns) on that demand; it only pays the
        # emergency price c_rep on S. Using r*served would treat S as lost load,
        # which the doc (4.3.6) explicitly says is NOT the main-model setting.
        power_profit_vec = (
                cfg.power_unit_revenue * demand
                - float(price) * shipments
                - cfg.c_rep * shortage
                - cfg.holding_cost * ending_inventory
        )

        system_profit = float(coal_profit + power_profit_vec.sum())

        # 2. Normalization reference: one power firm's base weekly purchase value.
        unit_money_ref = max(float(cfg.p_base) * float(cfg.demand_offpeak), EPS)
        system_money_ref = max(float(self.num_power) * unit_money_ref, EPS)

        coal_profit_norm = float(coal_profit / system_money_ref)
        power_profit_norm_vec = power_profit_vec / unit_money_ref

        # 3. Shortage rate normalized by current real demand (model 4.3.6).
        total_demand_ref = max(float(np.sum(demand)), EPS)
        shortage_norm = float(np.sum(shortage) / total_demand_ref)
        shortage_rate = float(np.sum(shortage) / (np.sum(demand) + EPS))

        # 4. Jain fairness on service rate is already dimensionless.
        fairness_penalty = 1.0 - float(jain)

        if mu_c_reward is None:
            mu_c_reward = self.mu_c_scalar
        if mu_u_reward is None:
            mu_u_reward = self.mu_u_scalar

        rewards: Dict[str, float] = {}

        # 5. Coal reward (model 4.3.6): r^C = pi_tilde^C - lambda_S*SR - lambda_J*(1-J).
        # The eta*mu*g term is EXCLUDED from the main model (model 4.3.5); it is
        # only used behind use_reciprocity_reward for the extension ablation.
        r_coal = coal_profit_norm
        if self.config.use_reciprocity_reward and self._reciprocity_enabled():
            r_coal += cfg.eta_c * float(mu_c_reward) * float(g_c)
        r_coal -= cfg.lambda_coal_shortage * shortage_norm
        r_coal -= cfg.lambda_coal_fairness * fairness_penalty
        rewards[self.coal_agent_id] = self._maybe_clip_reward(r_coal)

        # 6. Power reward (model 4.3.6): r^{U_i} = pi_tilde^{U_i}.
        for idx, agent in enumerate(self.power_agent_ids):
            r = float(power_profit_norm_vec[idx])
            if idx in self.reciprocal_idx and self.config.use_reciprocity_reward and self._reciprocity_enabled():
                r += cfg.eta_u * float(mu_u_reward) * float(g_u)
            rewards[agent] = self._maybe_clip_reward(r)

        # 7. info: raw + normalized profits for reporting.
        info = {
            "coal_profit": float(coal_profit),
            "coal_profit_norm": float(coal_profit_norm),
            "power_profit_total": float(power_profit_vec.sum()),
            "power_profit_norm_total": float(power_profit_vec.sum() / unit_money_ref),
            "system_profit": float(system_profit),
            "shortage_rate": float(shortage_rate),
            "shortage_norm": float(shortage_norm),
            "fairness_penalty": float(fairness_penalty),
        }

        for idx in range(self.num_power):
            info[f"power_profit_u{idx + 1}"] = float(power_profit_vec[idx])
            info[f"power_profit_norm_u{idx + 1}"] = float(power_profit_norm_vec[idx])

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
        price_obs = float(self.current_price)
        supply_obs = float(self.current_supply)
        pipeline_sum = self.pipeline.sum(axis=0) if self.pipeline.size else np.zeros(self.num_power, dtype=np.float32)

        mode = self.config.mechanism_mode.lower()
        # b5 cleanliness (remaining_issues sec5): the non-relational coordinator
        # must NOT observe per-firm differentiation that lets it learn an implicit
        # preference. Replace per-firm Q/Y with system totals and drop mu^C.
        b5_mode = (mode == "b5")

        # Coal observation (model 4.4.2):
        #   o^C_t = (theta_{t-1}, K, mu^C_t, {Q_{i,t-1}, Y_{i,t-1}, S^{obs}_{i,t-1}}).
        # We encode theta_prev, the observable pressure chi_hat, and the public
        # supply A_t=G_t. In b5, mu^C and per-firm Q/Y/S are replaced by aggregates.
        coal_last_orders = self.last_orders
        coal_last_shipments = self.last_shipments
        coal_last_shortage = self.last_shortage
        coal_mu_obs = self.mu_c_scalar if self.config.include_reciprocity_in_obs else 0.0
        if b5_mode:
            # Aggregate only: coal sees system totals, not per-firm identity.
            coal_last_orders = np.full(self.num_power, float(self.last_orders.sum()) / max(self.num_power, 1), dtype=np.float32)
            coal_last_shipments = np.full(self.num_power, float(self.last_shipments.sum()) / max(self.num_power, 1), dtype=np.float32)
            coal_last_shortage = np.full(self.num_power, float(self.last_shortage.sum()) / max(self.num_power, 1), dtype=np.float32)
            coal_mu_obs = 0.0

        coal_features = self._make_feature_vector(
            role_coal=1.0,
            role_power=0.0,
            is_u1=0.0,
            power_id_norm=0.0,
            ell=ell,
            season=season,
            price=price_obs,
            theta_prev=float(self.theta_prev),
            chi_hat=float(self.chi_prev),
            supply_info=supply_obs,
            own_inventory=0.0,
            own_pipeline=0.0,
            own_last_demand=0.0,
            own_last_order=0.0,
            own_last_shipment=0.0,
            own_last_shortage=0.0,
            mu_obs=coal_mu_obs,
            coal_last_orders=coal_last_orders,
            coal_last_shipments=coal_last_shipments,
        )
        obs[self.coal_agent_id] = {"obs": coal_features}

        # Power-firm observation (model 4.4.2):
        #   o^{U_i}_t = (I_i, D_{i,t-1}, Q_{i,t-1}, Y_{i,t-1}, S_{i,t-1}, rho_i, mu_obs_i).
        for idx, agent in enumerate(self.power_agent_ids):
            is_u1 = 1.0 if idx in self.reciprocal_idx else 0.0
            mu_obs = self.mu_u_scalar if (idx in self.reciprocal_idx and self.config.include_reciprocity_in_obs) else 0.0

            power_features = self._make_feature_vector(
                role_coal=0.0,
                role_power=1.0,
                is_u1=is_u1,
                power_id_norm=(idx + 1) / max(self.num_power, 1),
                ell=ell,
                season=season,
                price=price_obs,
                theta_prev=0.0,  # power firms do not observe coal's theta
                chi_hat=float(self.chi_prev) if idx in self.reciprocal_idx else 0.0,
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

        # Fixed 24-dim feature layout (model 4.4.2):
        #  idx 0-3 : role / identity flags
        #  idx 4-6 : year position, season, price
        #  idx 7   : theta_{t-1} (coal only; power firms observe 0)
        #  idx 8   : A_t = G_t (coal only)
        #  idx 9-14: power-firm own inventory/pipeline/demand/order/shipment/shortage
        #  idx 15  : mu_obs (coal: mu^C; reciprocal power: mu^U)
        #  idx 16-21: coal's per-firm {Q_{i,t-1}} (16-18), {Y_{i,t-1}} (19-21)
        #  idx 22  : chi_hat (observable supply-demand pressure, chi_{t-1})
        #  idx 23  : reserved
        values = np.array(
            [
                kwargs["role_coal"],            # 0
                kwargs["role_power"],           # 1
                kwargs["is_u1"],                # 2
                kwargs["power_id_norm"],        # 3
                kwargs["ell"],                  # 4
                float(kwargs["season"]),        # 5
                kwargs["price"],                # 6
                kwargs["theta_prev"],           # 7 coal: theta_{t-1}
                kwargs["supply_info"],          # 8, coal only (A_t)
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
                kwargs["chi_hat"],              # 22 observable pressure chi_{t-1}
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
        if self.config.mechanism_mode.lower() in {"none", "b2", "b5"}:
            if not np.allclose(weights, np.ones_like(weights), atol=1e-6):
                raise RuntimeError(f"{stage}: no-exclusive-weight mode should force weights to 1.")

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
            "mu_c": float(self.mu_c_scalar),
            "mu_u": float(self.mu_u_scalar),
            "g_u": float(kwargs["g_u"]),
            "g_c": float(kwargs["g_c"]),
            "theta": float(self.theta),
            "lambda1": float(self.last_weight[self.reciprocal_idx[0]]) if self.reciprocal_idx else 1.0,
            "ramp_hit": float(self.last_ramp_hit),
            "chi": float(self.chi_t),
            "chi_hat": float(self.chi_prev),
            "G": float(kwargs["supply"]),
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
                    "own_served": float(kwargs["served"][idx]),
                    "own_shortage": float(kwargs["shortage"][idx]),
                    "own_inventory": float(kwargs["ending_inventory"][idx]),
                    "own_fill_rate": float(kwargs["fill_rate"][idx]),
                    "own_service_rate": float(kwargs["service_rate"][idx]),
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
        coal_profit_norm = np.asarray(self.history.get("coal_profit_norm", []), dtype=np.float64)
        power_profit_norm_total = np.asarray(self.history.get("power_profit_norm_total", []), dtype=np.float64)
        shortage_norm = np.asarray(self.history.get("shortage_norm", []), dtype=np.float64)
        fairness_penalty = np.asarray(self.history.get("fairness_penalty", []), dtype=np.float64)
        jain = np.asarray(self.history["jain"], dtype=np.float64)

        demand_cv = self._cv(demand.sum(axis=1))
        order_cv = self._cv(orders.sum(axis=1))
        bullwhip = np.nan if demand_cv < 1e-6 else order_cv / demand_cv

        safe_inventory = np.array(
            [self._safe_inventory_vector(self._season(t)) for t in range(len(demand))]
        )
        inventory = np.asarray(self.history["inventory"], dtype=np.float64)
        inventory_violation = float(np.mean(inventory < safe_inventory))

        # mu_c / mu_u history stores per-step arrays (length h). Mean over all
        # entries gives the average relationship-memory level for h=1.
        mu_c_hist = np.asarray(self.history["mu_c"], dtype=np.float64)
        mu_u_hist = np.asarray(self.history["mu_u"], dtype=np.float64)
        theta_hist = np.asarray(self.history.get("theta", []), dtype=np.float64)
        lam_hist = np.asarray(self.history.get("lambda1", []), dtype=np.float64)
        ramp_hist = np.asarray(self.history.get("ramp_hit", []), dtype=np.float64)
        chi_hist = np.asarray(self.history.get("chi", []), dtype=np.float64)
        G_hist = np.asarray(self.history.get("G", []), dtype=np.float64)
        service_rate_hist = np.asarray(self.history.get("service_rate", []), dtype=np.float64)

        return {
            "total_shortage_rate": float(shortage.sum() / (demand.sum() + EPS)),
            "system_profit": float(profits.sum()),
            "coal_profit_norm_sum": float(coal_profit_norm.sum()) if coal_profit_norm.size else np.nan,
            "coal_profit_norm_mean": float(coal_profit_norm.mean()) if coal_profit_norm.size else np.nan,
            "power_profit_norm_total_sum": float(power_profit_norm_total.sum()) if power_profit_norm_total.size else np.nan,
            "power_profit_norm_total_mean": float(power_profit_norm_total.mean()) if power_profit_norm_total.size else np.nan,
            "shortage_norm_mean": float(shortage_norm.mean()) if shortage_norm.size else np.nan,
            "fairness_penalty_mean": float(fairness_penalty.mean()) if fairness_penalty.size else np.nan,
            "avg_jain": float(jain.mean()),
            "inventory_violation_rate": inventory_violation,
            "order_cv": float(order_cv),
            "demand_cv": float(demand_cv),
            "bullwhip_ratio": float(bullwhip) if not np.isnan(bullwhip) else np.nan,
            "u1_shortage_rate": float(shortage[:, 0].sum() / (demand[:, 0].sum() + EPS)),
            "ordinary_shortage_rate": float(shortage[:, 1:].sum() / (demand[:, 1:].sum() + EPS)) if self.num_power > 1 else 0.0,
            "mean_mu_c": float(np.mean(mu_c_hist)) if mu_c_hist.size else 0.0,
            "mean_mu_u": float(np.mean(mu_u_hist)) if mu_u_hist.size else 0.0,
            "mean_g_u": float(np.mean(self.history["g_u"])),
            "mean_g_c": float(np.mean(self.history["g_c"])),
            "mean_fill_rate": float(fill_rate.mean()),
            "mean_service_rate": float(service_rate_hist.mean()) if service_rate_hist.size else np.nan,
            "mean_unsold": float(np.mean(self.history["unsold"])),
            # Capacity-path and reciprocity mechanism metrics (model 4.5.1).
            "mean_theta": float(theta_hist.mean()) if theta_hist.size else np.nan,
            "mean_lambda1": float(lam_hist.mean()) if lam_hist.size else np.nan,
            "ramp_hit_rate": float(ramp_hist.mean()) if ramp_hist.size else np.nan,
            "mean_chi": float(chi_hist.mean()) if chi_hist.size else np.nan,
            "frac_tight": float(np.mean(chi_hist > 0)) if chi_hist.size else np.nan,
            "mean_G": float(G_hist.mean()) if G_hist.size else np.nan,
            "G_utilization": float(1.0 - np.mean(self.history["unsold"]) / (G_hist.mean() + EPS)) if G_hist.size else np.nan,
            # Backward-compatible aliases
            "mean_trust_u": float(np.mean(mu_c_hist)) if mu_c_hist.size else 0.0,
            "mean_trust_c": float(np.mean(mu_u_hist)) if mu_u_hist.size else 0.0,
        }

    @staticmethod
    def _cv(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.std(x) / (np.mean(x) + EPS))

    # ------------------------------------------------------------------
    # Heuristic action helpers for non-learning baselines
    # ------------------------------------------------------------------
    def normalized_action_from_weight(self, w: float) -> np.ndarray:
        return self._normalized_action(w, 1.0, self.config.lambda_max)

    def normalized_action_from_theta(self, theta: float) -> np.ndarray:
        return self._normalized_action(theta, self.config.theta_min, self.config.theta_max)

    def normalized_action_from_omega(self, omega: float) -> np.ndarray:
        return self._normalized_action(omega, self.config.omega_min, self.config.omega_max)

    def coal_action(self, theta: float, lam: float = 1.0) -> np.ndarray:
        """Build the coal agent's 2D normalized action (theta_raw, lambda_raw).

        The raw action is in [-1,1]; _parse_actions rescales it back to the
        physical ranges [theta_min, theta_max] and [1, lambda_max].
        """
        t_raw = self.normalized_action_from_theta(theta)
        l_raw = self.normalized_action_from_weight(lam)
        return np.array([float(t_raw[0]), float(l_raw[0])], dtype=np.float32)

    def base_stock_action_dict(self, omega: Optional[float] = None,
                               theta: Optional[float] = None,
                               lam: float = 1.0) -> Dict[str, np.ndarray]:
        if omega is None:
            omega = self.config.omega_base
        if theta is None:
            season = int(self.current_season)
            theta = self.config.theta_offpeak if season == 0 else self.config.theta_peak

        actions = {
            self.coal_agent_id: self.coal_action(float(theta), float(lam))
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
