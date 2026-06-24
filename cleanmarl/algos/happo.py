"""
HAPPO算法实现 - 脱离RLlib，PyTorch原生训练循环

解决MARLlib中价值网络学不到的核心设计：
1. critic独立优化器，每个agent独立价值输出
2. 按episode重组数据，RNN隐藏状态在episode内连续、episode间重置
3. HAPPO顺序更新：逐个agent更新策略，并迭代更新后续agent的advantage
4. 所有actor更新完后，统一更新共享critic
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from cleanmarl.core.trainer import Trainer
from cleanmarl.models import GRUActor, CentralizedCritic


class HAPPO(Trainer):
    def __init__(self, env, config: Dict):
        super().__init__(config)
        self.env = env
        self.num_agents = env.num_agents
        self.obs_dim = env.observation_space.shape[0]
        self.state_dim = env.state_space.shape[0]
        # Per-agent action dims (model 4.4.3): coal=2, power=1. log_prob is
        # summed over action dims (models.py) so it stays scalar per agent;
        # HAPPO advantage iteration only touches scalar ratios -> heterogeneous
        # action dims are safe.
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
        self.chunk_len = int(config.get('chunk_len', 0))  # T_chunk; 0 = full episode

        # 模型：每个agent独立actor（各自act_dim），共享critic
        self.actors = nn.ModuleList([
            GRUActor(self.obs_dim, self.act_dims[i], hidden, self.continuous,
                     rnn_layers=rnn_layers).to(self.device)
            for i in range(self.num_agents)
        ])
        self.critic = CentralizedCritic(self.state_dim, self.num_agents, hidden,
                                        rnn_layers=rnn_layers).to(self.device)

        self.actor_opts = [
            torch.optim.Adam(a.parameters(), lr=config.get('actor_lr', 5e-4))
            for a in self.actors
        ]
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.get('critic_lr', 5e-4)
        )

        self._buf = {}
        self._ep_reward_hist = []

    # ---------------- 采样 ----------------
    def collect_trajectories(self, num_steps: int):
        dev = self.device
        obs_buf, state_buf, act_buf = [], [], []
        rew_buf, logp_buf, val_buf, done_buf = [], [], [], []
        step_info_buf = []  # 新增：缓存每步的 step_info

        obs = self.env.reset()
        state = self.env.get_state()
        actor_h = [None] * self.num_agents
        critic_h = None
        ep_reward = 0.0
        ep_step = 0  # 新增：当前 episode 内的步数
        self.episode_logger.reset_episode(self.num_agents)  # 新增：重置 episode 缓存

        for _ in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=dev)

            actions, logps = [], []
            new_h = []
            for i in range(self.num_agents):
                a, lp, h = self.actors[i].act(obs_t[i], actor_h[i])
                # pad to max_act_dim so all agents stack into (N, max_act_dim)
                padded = np.zeros(self.max_act_dim, dtype=np.float32)
                padded[:int(self.act_dims[i])] = a.cpu().numpy().reshape(-1)
                actions.append(torch.as_tensor(padded, device=dev))
                logps.append(lp)
                new_h.append(h)
            values, critic_h = self.critic.get_value(state_t, critic_h)

            act_np = torch.stack(actions).cpu().numpy()
            next_obs, rewards, done, step_info = self.env.step(act_np)
            next_state = self.env.get_state()

            obs_buf.append(obs_t.cpu().numpy())
            state_buf.append(state.astype(np.float32))
            act_buf.append(act_np)
            rew_buf.append(rewards)
            logp_buf.append(torch.stack(logps).cpu().numpy())
            val_buf.append(values.cpu().numpy())
            done_buf.append(done)
            step_info_buf.append(step_info)  # 新增

            ep_reward += float(rewards.sum())
            ep_step += 1

            # 新增：记录步级数据到 EpisodeLogger
            self.episode_logger.record_step(
                step=ep_step,
                obs=obs,
                actions=act_np,
                rewards=rewards,
                step_info=step_info,
                agent_ids=self.env.agents,
            )

            obs, state, actor_h = next_obs, next_state, new_h

            if done:
                self._ep_reward_hist.append(ep_reward)
                # 新增：结束 episode，写入 summary
                self.episode_logger.finalize_episode(
                    episode_idx=len(self._ep_reward_hist),
                    phase="train",
                )
                ep_reward = 0.0
                ep_step = 0
                obs = self.env.reset()
                state = self.env.get_state()
                actor_h = [None] * self.num_agents
                critic_h = None
                self.episode_logger.reset_episode(self.num_agents)  # 重置缓存

            self.current_step += 1

        # bootstrap value
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=dev)
            last_val, _ = self.critic.get_value(state_t, critic_h)
        last_val = last_val.cpu().numpy()

        self._buf = {
            'obs': np.array(obs_buf, dtype=np.float32),       # (T, N, obs)
            'state': np.array(state_buf, dtype=np.float32),   # (T, state)
            'actions': np.array(act_buf, dtype=np.float32),   # (T, N, act)
            'rewards': np.array(rew_buf, dtype=np.float32),   # (T, N)
            'log_probs': np.array(logp_buf, dtype=np.float32),# (T, N)
            'values': np.array(val_buf, dtype=np.float32),    # (T, N)
            'dones': np.array(done_buf, dtype=np.float32),    # (T,)
        }
        self._compute_gae(last_val)

    def _compute_gae(self, last_val):
        T = self._buf['rewards'].shape[0]
        rewards = self._buf['rewards']
        values = self._buf['values']
        dones = self._buf['dones']
        adv = np.zeros_like(rewards)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_val = last_val if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        self._buf['advantages'] = adv
        self._buf['returns'] = adv + values

    # ---------------- 更新 ----------------
    def _to_episodes(self, arr):
        """(T, ...) -> (N, L, ...): 按 episode_len 重组, 再按 T_chunk 切分.

        当 chunk_len (T_chunk) > 0 且 < episode_len 时, 每个 episode 被切成多个
        长度为 chunk_len 的独立子序列 (hidden state 在块边界重置, 与
        evaluate_actions/forward 的 h=None 一致). 否则按整 episode 训练.
        """
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
        # 转episode张量: (E, L, ...)
        obs = torch.as_tensor(self._to_episodes(b['obs']), device=dev)        # (E,L,N,obs)
        state = torch.as_tensor(self._to_episodes(b['state']), device=dev)    # (E,L,state)
        actions = torch.as_tensor(self._to_episodes(b['actions']), device=dev)# (E,L,N,act)
        old_logp = torch.as_tensor(self._to_episodes(b['log_probs']), device=dev)  # (E,L,N)
        old_val = torch.as_tensor(self._to_episodes(b['values']), device=dev) # (E,L,N)
        returns = torch.as_tensor(self._to_episodes(b['returns']), device=dev)# (E,L,N)
        adv = torch.as_tensor(self._to_episodes(b['advantages']), device=dev) # (E,L,N)

        # advantage归一化（全局）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        m = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}

        for _ in range(self.num_sgd_iter):
            cur_adv = adv.clone()
            # ===== HAPPO顺序更新策略 =====
            for i in range(self.num_agents):
                d = int(self.act_dims[i])
                pl, ent, kl = self._update_actor(i, obs[:, :, i], actions[:, :, i, :d],
                                                  old_logp[:, :, i], cur_adv[:, :, i])
                m['policy_loss'].append(pl)
                m['entropy'].append(ent)
                m['approx_kl'].append(kl)
                # 迭代更新后续agent的advantage（HAPPO核心）
                if i < self.num_agents - 1:
                    cur_adv = self._iterate_advantage(i, obs[:, :, i], actions[:, :, i, :d],
                                                      old_logp[:, :, i], cur_adv)
            # ===== 统一更新critic =====
            vl = self._update_critic(state, returns, old_val)
            m['value_loss'].append(vl)

        # 指标
        res = {k: float(np.mean(v)) for k, v in m.items()}
        res.update(self._final_metrics(state, returns))
        if self._ep_reward_hist:
            recent = self._ep_reward_hist[-10:]
            res['episode_reward_mean'] = float(np.mean(recent))
            res['episode_reward_max'] = float(np.max(recent))
            res['episode_reward_min'] = float(np.min(recent))
        return res

    def _update_actor(self, i, obs_i, act_i, old_logp_i, adv_i):
        logp, ent = self.actors[i].evaluate_actions(obs_i, act_i)
        ratio = torch.exp(logp - old_logp_i)
        s1 = ratio * adv_i
        s2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_i
        policy_loss = -torch.min(s1, s2).mean()
        entropy = ent.mean()
        loss = policy_loss - self.entropy_coef * entropy

        self.actor_opts[i].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
        self.actor_opts[i].step()

        with torch.no_grad():
            kl = (old_logp_i - logp).mean().item()
        return policy_loss.item(), entropy.item(), kl

    def _iterate_advantage(self, i, obs_i, act_i, old_logp_i, cur_adv):
        """A_{i+1} <- A_{i+1} * ratio_i （用更新后的策略重新评估）"""
        with torch.no_grad():
            logp, _ = self.actors[i].evaluate_actions(obs_i, act_i)
            ratio = torch.exp(logp - old_logp_i)  # (E, L)
            cur_adv = cur_adv.clone()
            cur_adv[:, :, i + 1] = cur_adv[:, :, i + 1] * ratio
        return cur_adv

    def _update_critic(self, state, returns, old_val):
        values, _ = self.critic(state)  # (E, L, N)
        v_clipped = old_val + torch.clamp(values - old_val, -self.value_clip, self.value_clip)
        vl1 = (values - returns).pow(2)
        vl2 = (v_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(vl1, vl2).mean()

        self.critic_opt.zero_grad()
        (self.vf_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()
        return value_loss.item()

    def _final_metrics(self, state, returns):
        with torch.no_grad():
            values, _ = self.critic(state)  # (E,L,N)
        out = {}
        for i in range(self.num_agents):
            y_true = returns[:, :, i].reshape(-1)
            y_pred = values[:, :, i].reshape(-1)
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
                    a, _, actor_h[i] = self.actors[i].act(obs_t[i], actor_h[i],
                                                           deterministic=True)
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
            # 记录评估 episode
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
            'critic': self.critic.state_dict(),
            'actor_opts': [o.state_dict() for o in self.actor_opts],
            'critic_opt': self.critic_opt.state_dict(),
        }

    def load_state_dict(self, ckpt):
        for a, sd in zip(self.actors, ckpt['actors']):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt['critic'])
        for o, sd in zip(self.actor_opts, ckpt['actor_opts']):
            o.load_state_dict(sd)
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
