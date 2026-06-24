"""
经验回放缓冲区 - 存储轨迹数据
"""
import torch
import numpy as np
from typing import Dict, List, Optional


class RolloutBuffer:
    """存储单个epoch的轨迹数据"""

    def __init__(self, buffer_size: int, num_agents: int, obs_dim: int,
                 state_dim: int, act_dim: int, device: str = "cuda"):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.device = device

        # 为每个agent分配存储空间
        self.observations = torch.zeros((buffer_size, num_agents, obs_dim), device=device)
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, num_agents, act_dim), device=device)
        self.rewards = torch.zeros((buffer_size, num_agents), device=device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=device)
        self.log_probs = torch.zeros((buffer_size, num_agents), device=device)
        self.values = torch.zeros((buffer_size, num_agents), device=device)

        # GAE相关
        self.advantages = torch.zeros((buffer_size, num_agents), device=device)
        self.returns = torch.zeros((buffer_size, num_agents), device=device)

        # RNN隐藏状态
        self.rnn_states = {}

        self.ptr = 0
        self.full = False

    def add(self, obs: torch.Tensor, state: torch.Tensor, actions: torch.Tensor,
            rewards: torch.Tensor, dones: torch.Tensor, log_probs: torch.Tensor,
            values: torch.Tensor, rnn_states: Optional[Dict] = None):
        """添加一个timestep的数据"""
        self.observations[self.ptr] = obs
        self.states[self.ptr] = state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values

        if rnn_states is not None:
            for agent_id, rnn_state in rnn_states.items():
                if agent_id not in self.rnn_states:
                    self.rnn_states[agent_id] = []
                self.rnn_states[agent_id].append(rnn_state)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_returns_and_advantages(self, last_values: torch.Tensor,
                                       gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算GAE优势和returns"""
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_agents, device=self.device)

        # 从后往前计算GAE
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = self.values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_values * next_non_terminal.unsqueeze(-1) - self.values[t]

            # GAE: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal.unsqueeze(-1) * last_gae

        # Returns = Advantages + Values
        self.advantages[:self.ptr] = advantages
        self.returns[:self.ptr] = advantages + self.values[:self.ptr]

    def get(self):
        """获取所有数据"""
        indices = slice(0, self.ptr)
        return {
            'observations': self.observations[indices],
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'log_probs': self.log_probs[indices],
            'values': self.values[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'rnn_states': self.rnn_states
        }

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.full = False
        self.rnn_states = {}
