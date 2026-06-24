"""
环境适配器 - 将MARLlib的CPDRE环境适配到CleanMARL

要点：
1. 煤企动作是连续 Box([-1,1], (2,)) = (theta_t, lambda_{1,t})；电企动作是 Box([-1,1], (1,)) = omega。
   异质动作维度：self.act_dims = [2, 1, ..., 1]，self.max_act_dim = 2。
2. 观测字典每个agent含 {'obs':(24,), 'state':(96,)}，state为共享全局状态。
3. reset/step 时缓存 state，供 get_state() 返回。
4. eval_reset(episode_idx) 用固定种子生成确定性评估轨迹（model 4.3 table: s^eval_0=9000）。
5. step() 返回扁平化的步级指标字典，供 EpisodeLogger 记录。
"""
from typing import Tuple, Dict
import numpy as np


class CPDREEnvWrapper:
    """CPDRE环境包装器"""

    def __init__(self, marllib_env):
        self.env = marllib_env
        self.agents = marllib_env.agents
        self.num_agents = len(self.agents)

        # Per-agent action spaces (model 4.4.3): coal 2D (theta, lambda),
        # power 1D (omega). act_dims[i] is agent i's real action dimension.
        self.action_spaces = marllib_env.action_spaces
        self.act_dims = list(marllib_env.act_dims)
        self.max_act_dim = int(marllib_env.max_act_dim)
        # Backward-compat single action space (max dim) + scalar act_dim.
        self.action_space = marllib_env.action_space
        self.act_dim = self.max_act_dim
        self.continuous = True

        # observation_space 是共享 Dict: {'obs': Box(24), 'state': Box(96)}
        self.observation_space = marllib_env.observation_space.spaces['obs']
        self.state_space = marllib_env.observation_space.spaces['state']

        self._last_state = np.zeros(self.state_space.shape, dtype=np.float32)

    def reset(self) -> np.ndarray:
        obs_dict = self.env.reset()
        return self._process(obs_dict)

    def eval_reset(self, episode_idx: int) -> np.ndarray:
        """Deterministic eval reset using a fixed seed offset (model 4.3)."""
        obs_dict = self.env.eval_reset(episode_idx)
        return self._process(obs_dict)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        # actions: (num_agents, max_act_dim) -> per-agent slice to real dims, clip.
        action_dict = {}
        for i, agent_id in enumerate(self.agents):
            a = np.asarray(actions[i], dtype=np.float32).reshape(-1)
            space = self.action_spaces[agent_id]
            d = int(self.act_dims[i])
            a = a[:d] if a.size >= d else np.pad(a, (0, d - a.size))
            action_dict[agent_id] = np.clip(a.astype(np.float32), space.low, space.high)

        obs_dict, reward_dict, done_dict, info = self.env.step(action_dict)
        obs = self._process(obs_dict)
        rewards = np.array([reward_dict[a] for a in self.agents], dtype=np.float32)
        done = bool(done_dict.get('__all__', False))
        step_info = self._extract_step_info(info)
        return obs, rewards, done, step_info

    def _extract_step_info(self, info: dict) -> dict:
        """从 RLlib 的 per-agent info 字典中抽取扁平化的步级指标。

        info 是 Dict[agent_id, info_dict]，所有 agent 共享 common 指标（price、
        shortage_rate、system_profit 等），power agent 额外有 own_* 指标。
        这里把它们合并成一个扁平 dict 供 EpisodeLogger 使用。
        """
        if not isinstance(info, dict) or not info:
            return {}

        # coal agent 持有所有 common 指标
        coal_info = info.get(self.agents[0], {})
        flat: Dict[str, float] = {}
        for k, v in coal_info.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                flat[k] = float(v)
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                flat[k] = float(v)

        # power agent 的 own_* 指标按 u1/u2/u3 拆分
        for idx, agent_id in enumerate(self.agents):
            a_info = info.get(agent_id, {})
            if not isinstance(a_info, dict):
                continue
            uid = idx  # power 索引从 1 开始（假设 coal 在前）
            for k, v in a_info.items():
                if k.startswith("own_") and isinstance(v, (int, float, np.integer, np.floating)):
                    flat[f"{k}_u{uid}"] = float(v)
        return flat

    def get_state(self) -> np.ndarray:
        """返回最近一次缓存的全局状态"""
        return self._last_state

    def _process(self, obs_dict: dict) -> np.ndarray:
        """提取每个agent的'obs'，同时缓存共享'state'"""
        obs_list = []
        for agent_id in self.agents:
            agent_obs = obs_dict[agent_id]
            if isinstance(agent_obs, dict):
                obs_list.append(np.asarray(agent_obs['obs'], dtype=np.float32))
                if 'state' in agent_obs:
                    self._last_state = np.asarray(agent_obs['state'], dtype=np.float32)
            else:
                obs_list.append(np.asarray(agent_obs, dtype=np.float32))
        return np.array(obs_list, dtype=np.float32)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


def make_cpdre_env(env_args: dict):
    """创建并包装CPDRE环境（直接构造，不依赖MARLlib）"""
    from custom_envs.coal_power_direct_reciprocity_env import CoalPowerDirectReciprocityEnv
    marllib_env = CoalPowerDirectReciprocityEnv(env_args)
    return CPDREEnvWrapper(marllib_env)
