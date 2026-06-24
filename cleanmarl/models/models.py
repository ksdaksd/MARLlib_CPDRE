"""
模型定义 - Actor (策略网络) 和 Centralized Critic (中心化价值网络)

设计要点：
1. 支持连续动作空间（CPDRE使用Box动作）
2. Critic输出每个agent独立的价值，避免共享价值导致explained_var偏低
3. act() 用于采样（单步），evaluate_actions() 用于更新（整段序列）
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUActor(nn.Module):
    """基于GRU的策略网络（支持连续/离散动作）"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128,
                 continuous: bool = True, rnn_layers: int = 1):
        super().__init__()
        self.continuous = continuous
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = int(rnn_layers)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=self.rnn_layers,
                          batch_first=True)

        if continuous:
            self.mean_head = nn.Linear(hidden_dim, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        else:
            self.logits_head = nn.Linear(hidden_dim, act_dim)

    def _dist(self, feat: torch.Tensor):
        if self.continuous:
            mean = self.mean_head(feat)
            std = torch.exp(self.log_std).expand_as(mean)
            return torch.distributions.Normal(mean, std)
        return torch.distributions.Categorical(logits=self.logits_head(feat))

    def forward(self, obs_seq: torch.Tensor, h: Optional[torch.Tensor] = None):
        """obs_seq: (B, T, obs_dim) -> distribution, new_h"""
        x = self.encoder(obs_seq)
        out, h = self.gru(x, h)
        return self._dist(out), h

    @torch.no_grad()
    def act(self, obs: torch.Tensor, h: Optional[torch.Tensor] = None,
            deterministic: bool = False):
        """单步采样。obs: (obs_dim,) -> action (act_dim,), log_prob标量, new_h"""
        dist, h = self.forward(obs.view(1, 1, -1), h)
        if deterministic:
            a = dist.mean if self.continuous else dist.probs.argmax(-1, keepdim=True)
        else:
            a = dist.sample()
        lp = dist.log_prob(a)
        if self.continuous:
            lp = lp.sum(-1)
            a_out = a.reshape(self.act_dim)
        else:
            a_out = a.reshape(1).float()
        return a_out, lp.reshape(()), h

    def evaluate_actions(self, obs_seq: torch.Tensor, actions: torch.Tensor):
        """整段序列评估。obs_seq:(B,T,obs_dim), actions:(B,T,act_dim) -> log_probs(B,T), entropy(B,T)"""
        dist, _ = self.forward(obs_seq)
        if self.continuous:
            lp = dist.log_prob(actions).sum(-1)
            ent = dist.entropy().sum(-1)
        else:
            lp = dist.log_prob(actions.squeeze(-1))
            ent = dist.entropy()
        return lp, ent


class CentralizedCritic(nn.Module):
    """中心化Critic - 接收全局状态，输出每个agent独立的价值"""

    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 128,
                 rnn_layers: int = 1):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.rnn_layers = int(rnn_layers)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=self.rnn_layers,
                          batch_first=True)
        self.value_head = nn.Linear(hidden_dim, num_agents)

    def forward(self, state_seq: torch.Tensor, h: Optional[torch.Tensor] = None):
        """state_seq: (B, T, state_dim) -> values (B, T, num_agents), new_h"""
        x = self.encoder(state_seq)
        out, h = self.gru(x, h)
        return self.value_head(out), h

    @torch.no_grad()
    def get_value(self, state: torch.Tensor, h: Optional[torch.Tensor] = None):
        """单步价值。state: (state_dim,) -> values (num_agents,), new_h"""
        v, h = self.forward(state.view(1, 1, -1), h)
        return v.reshape(self.num_agents), h
