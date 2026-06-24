"""
IPPO算法实现 - Independent PPO（最简单的多智能体PPO）

与HAPPO/MAPPO的区别：
1. 每个agent有独立的policy和独立的critic（不共享）
2. 采样时每个agent单独计算GAE（基于各自的价值函数）
3. 更新时每个agent独立更新（actor + critic）

是最自然的多智能体PPO扩展，也是最稳定的baseline。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from cleanmarl.core.trainer import Trainer
from cleanmarl.models import GRUActor


class IndependentCritic(nn.Module):
    """独立Critic - 每个agent只看自己的观测"""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, rnn_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_layers = int(rnn_layers)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=self.rnn_layers,
                          batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq: torch.Tensor, h: torch.Tensor = None):
        """obs_seq: (B, T, obs_dim) -> values (B, T, 1), new_h"""
        x = self.encoder(obs_seq)
        out, h = self.gru(x, h)
        return self.value_head(out), h

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor, h: torch.Tensor = None):
        """单步价值。obs: (obs_dim,) -> value scalar, new_h"""
        v, h = self.forward(obs.view(1, 1, -1), h)
        return v.reshape(()), h


class IPPO(Trainer):
    """IPPO训练器 - 每个agent完全独立"""

    def __init__(self, env, config: Dict):
        super().__init__(config)
        self.env = env
        self.num_agents = env.num_agents
        self.obs_dim = env.observation_space.shape[0]
        # Per-agent action dims (coal=2, power=1).
        self.act_dims = list(env.act_dims)
        self.max_act_dim = int(env.max_act_dim)
        self.continuous = getattr(env, 'continuous', True)
        self.episode_len = config.get('episode_len', 156)

        # 超参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_clip = config.get('value_clip_param', 10.0)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.vf_coef = config.get('value_loss_coef', 1.0)
        self.max_grad_norm = config.get('max_grad_norm', 10.0)
        self.num_sgd_iter = config.get('num_sgd_iter', 5)
        self.rollout_length = config.get('rollout_length', 1560)
        self.eval_episodes = config.get('eval_episodes', 64)
        hidden = config.get('hidden_dim', 128)
        rnn_layers = int(config.get('rnn_layers', 1))
        self.chunk_len = int(config.get('chunk_len', 0))

        # 模型：每个agent独立的actor和critic
        self.actors = nn.ModuleList([
            GRUActor(self.obs_dim, self.act_dims[i], hidden, self.continuous,
                     rnn_layers=rnn_layers).to(self.device)
            for i in range(self.num_agents)
        ])
        self.critics = nn.ModuleList([
            IndependentCritic(self.obs_dim, hidden, rnn_layers=rnn_layers).to(self.device)
            for _ in range(self.num_agents)
        ])

        # 每个agent有独立的优化器
        self.actor_opts = [
            torch.optim.Adam(a.parameters(), lr=config.get('actor_lr', 5e-4))
            for a in self.actors
        ]
        self.critic_opts = [
            torch.optim.Adam(c.parameters(), lr=config.get('critic_lr', 5e-4))
            for c in self.critics
        ]

        self._buf = {}
        self._ep_reward_hist = []

    # ---------------- 采样 ----------------
    def collect_trajectories(self, num_steps: int):
        dev = self.device
        obs_buf, act_buf = [], []
        rew_buf, logp_buf, val_buf, done_buf = [], [], [], []

        obs = self.env.reset()
        actor_h = [None] * self.num_agents
        critic_h = [None] * self.num_agents
        ep_reward = 0.0
        ep_step = 0
        self.episode_logger.reset_episode(self.num_agents)

        for _ in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)

            actions, logps, values = [], [], []
            new_actor_h, new_critic_h = [], []
            for i in range(self.num_agents):
                a, lp, ha = self.actors[i].act(obs_t[i], actor_h[i])
                v, hc = self.critics[i].get_value(obs_t[i], critic_h[i])
                padded = np.zeros(self.max_act_dim, dtype=np.float32)
                padded[:int(self.act_dims[i])] = a.cpu().numpy().reshape(-1)
                actions.append(torch.as_tensor(padded, device=dev))
                logps.append(lp)
                values.append(v)
                new_actor_h.append(ha)
                new_critic_h.append(hc)

            act_np = torch.stack(actions).cpu().numpy()
            next_obs, rewards, done, step_info = self.env.step(act_np)

            obs_buf.append(obs_t.cpu().numpy())
            act_buf.append(act_np)
            rew_buf.append(rewards)
            logp_buf.append(torch.stack(logps).cpu().numpy())
            val_buf.append(np.array([float(v) for v in values]))
            done_buf.append(done)

            ep_reward += float(rewards.sum())
            ep_step += 1

            self.episode_logger.record_step(
                step=ep_step,
                obs=obs,
                actions=act_np,
                rewards=rewards,
                step_info=step_info,
                agent_ids=self.env.agents,
            )

            obs, actor_h, critic_h = next_obs, new_actor_h, new_critic_h
            self.current_step += 1

            if done:
                self._ep_reward_hist.append(ep_reward)
                self.episode_logger.finalize_episode(
                    episode_idx=len(self._ep_reward_hist),
                    phase="train",
                )
                ep_reward = 0.0
                ep_step = 0
                obs = self.env.reset()
                actor_h = [None] * self.num_agents
                critic_h = [None] * self.num_agents
                self.episode_logger.reset_episode(self.num_agents)

        # bootstrap
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        last_vals = []
        with torch.no_grad():
            for i in range(self.num_agents):
                v, _ = self.critics[i].get_value(obs_t[i], critic_h[i])
                last_vals.append(float(v))
        last_vals = np.array(last_vals)

        self._buf = {
            'obs': np.array(obs_buf, dtype=np.float32),
            'actions': np.array(act_buf, dtype=np.float32),
            'rewards': np.array(rew_buf, dtype=np.float32),
            'log_probs': np.array(logp_buf, dtype=np.float32),
            'values': np.array(val_buf, dtype=np.float32),
            'dones': np.array(done_buf, dtype=np.float32),
        }
        self._compute_gae(last_vals)

    def _compute_gae(self, last_vals):
        T = self._buf['rewards'].shape[0]
        rewards = self._buf['rewards']
        values = self._buf['values']
        dones = self._buf['dones']
        adv = np.zeros_like(rewards)
        for i in range(self.num_agents):
            last_gae = 0.0
            for t in reversed(range(T)):
                next_nonterminal = 1.0 - dones[t]
                next_val = last_vals[i] if t == T - 1 else values[t + 1, i]
                delta = rewards[t, i] + self.gamma * next_val * next_nonterminal - values[t, i]
                last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
                adv[t, i] = last_gae
        self._buf['advantages'] = adv
        self._buf['returns'] = adv + values

    # ---------------- 更新 ----------------
    def _to_episodes(self, arr):
        T = arr.shape[0]
        n_ep = T // self.episode_len
        flat = arr[:n_ep * self.episode_len].reshape(n_ep, self.episode_len, *arr.shape[1:])
        if self.chunk_len and 0 < self.chunk_len < self.episode_len:
            L = self.chunk_len
            n_chunks = self.episode_len // L
            flat = flat[:, :n_chunks * L].reshape(n_ep * n_chunks, L, *arr.shape[1:])
        return flat

    def update(self) -> Dict[str, float]:
        dev = self.device
        b = self._buf
        obs = torch.as_tensor(self._to_episodes(b['obs']), device=dev)
        actions = torch.as_tensor(self._to_episodes(b['actions']), device=dev)
        old_logp = torch.as_tensor(self._to_episodes(b['log_probs']), device=dev)
        old_val = torch.as_tensor(self._to_episodes(b['values']), device=dev)
        returns = torch.as_tensor(self._to_episodes(b['returns']), device=dev)
        adv = torch.as_tensor(self._to_episodes(b['advantages']), device=dev)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        m = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}

        for _ in range(self.num_sgd_iter):
            # ===== 每个agent独立更新 =====
            for i in range(self.num_agents):
                d = int(self.act_dims[i])
                # Actor update
                logp, ent = self.actors[i].evaluate_actions(obs[:, :, i], actions[:, :, i, :d])
                ratio = torch.exp(logp - old_logp[:, :, i])
                s1 = ratio * adv[:, :, i]
                s2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv[:, :, i]
                policy_loss = -torch.min(s1, s2).mean()
                entropy = ent.mean()
                loss = policy_loss - self.entropy_coef * entropy

                self.actor_opts[i].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                self.actor_opts[i].step()

                # Critic update（每个agent独立）
                values, _ = self.critics[i](obs[:, :, i])
                v_clipped = old_val[:, :, i] + torch.clamp(
                    values.squeeze() - old_val[:, :, i],
                    -self.value_clip, self.value_clip
                )
                vl1 = (values.squeeze() - returns[:, :, i]).pow(2)
                vl2 = (v_clipped - returns[:, :, i]).pow(2)
                value_loss = 0.5 * torch.max(vl1, vl2).mean()

                self.critic_opts[i].zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
                self.critic_opts[i].step()

                m['policy_loss'].append(policy_loss.item())
                m['value_loss'].append(value_loss.item())
                m['entropy'].append(entropy.item())
                with torch.no_grad():
                    m['approx_kl'].append((old_logp[:, :, i] - logp).mean().item())

        res = {k: float(np.mean(v)) for k, v in m.items()}
        res.update(self._final_metrics(returns, old_val))
        if self._ep_reward_hist:
            recent = self._ep_reward_hist[-10:]
            res['episode_reward_mean'] = float(np.mean(recent))
            res['episode_reward_max'] = float(np.max(recent))
            res['episode_reward_min'] = float(np.min(recent))
        return res

    def _final_metrics(self, returns, old_val):
        dev = self.device
        b = self._buf
        obs = torch.as_tensor(self._to_episodes(b['obs']), device=dev)
        out = {}
        for i in range(self.num_agents):
            with torch.no_grad():
                values, _ = self.critics[i](obs[:, :, i])
            y_true = returns[:, :, i].reshape(-1)
            y_pred = values.squeeze().reshape(-1)
            var_y = y_true.var()
            ev = 0.0 if var_y < 1e-8 else (1 - (y_true - y_pred).var() / var_y).item()
            out[f'vf_explained_var_agent_{i}'] = ev
        out['vf_explained_var'] = float(np.mean(list(out.values())))
        return out

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        dev = self.device
        if num_episodes is None:
            num_episodes = self.eval_episodes
        rewards_all = []
        for ep_idx in range(num_episodes):
            obs = self.env.eval_reset(ep_idx)
            actor_h = [None] * self.num_agents
            done = False
            ep_r = 0.0
            ep_step = 0
            self.episode_logger.reset_episode(self.num_agents)
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                acts = []
                for i in range(self.num_agents):
                    a, _, actor_h[i] = self.actors[i].act(obs_t[i], actor_h[i], deterministic=True)
                    padded = np.zeros(self.max_act_dim, dtype=np.float32)
                    padded[:int(self.act_dims[i])] = a.cpu().numpy().reshape(-1)
                    acts.append(torch.as_tensor(padded, device=dev))
                act_np = torch.stack(acts).cpu().numpy()
                obs, r, done, step_info = self.env.step(act_np)
                ep_r += float(r.sum())
                ep_step += 1
                self.episode_logger.record_step(
                    step=ep_step,
                    obs=obs,
                    actions=act_np,
                    rewards=r,
                    step_info=step_info,
                    agent_ids=self.env.agents,
                )
            rewards_all.append(ep_r)
            self.episode_logger.finalize_episode(
                episode_idx=ep_idx,
                phase="eval",
            )
        return {
            'eval/reward_mean': float(np.mean(rewards_all)),
            'eval/reward_std': float(np.std(rewards_all)),
        }

    # ---------------- checkpoint ----------------
    def state_dict(self):
        return {
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
            'actor_opts': [o.state_dict() for o in self.actor_opts],
            'critic_opts': [o.state_dict() for o in self.critic_opts],
        }

    def load_state_dict(self, ckpt):
        for a, sd in zip(self.actors, ckpt['actors']):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, ckpt['critics']):
            c.load_state_dict(sd)
        for o, sd in zip(self.actor_opts, ckpt['actor_opts']):
            o.load_state_dict(sd)
        for o, sd in zip(self.critic_opts, ckpt['critic_opts']):
            o.load_state_dict(sd)