"""
MAPPO算法 - 作为对比基准

与HAPPO的唯一区别：并行更新所有agent的策略，不迭代advantage。
其余（采样、GAE、critic更新、指标）全部复用HAPPO。
"""
import numpy as np
import torch
from typing import Dict

from cleanmarl.algos.happo import HAPPO


class MAPPO(HAPPO):
    def update(self) -> Dict[str, float]:
        dev = self.device
        b = self._buf
        obs = torch.as_tensor(self._to_episodes(b['obs']), device=dev)
        state = torch.as_tensor(self._to_episodes(b['state']), device=dev)
        actions = torch.as_tensor(self._to_episodes(b['actions']), device=dev)
        old_logp = torch.as_tensor(self._to_episodes(b['log_probs']), device=dev)
        old_val = torch.as_tensor(self._to_episodes(b['values']), device=dev)
        returns = torch.as_tensor(self._to_episodes(b['returns']), device=dev)
        adv = torch.as_tensor(self._to_episodes(b['advantages']), device=dev)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        m = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': []}

        for _ in range(self.num_sgd_iter):
            # ===== 并行更新所有策略（不迭代advantage）=====
            for i in range(self.num_agents):
                d = int(self.act_dims[i])
                pl, ent, kl = self._update_actor(i, obs[:, :, i], actions[:, :, i, :d],
                                                  old_logp[:, :, i], adv[:, :, i])
                m['policy_loss'].append(pl)
                m['entropy'].append(ent)
                m['approx_kl'].append(kl)
            vl = self._update_critic(state, returns, old_val)
            m['value_loss'].append(vl)

        res = {k: float(np.mean(v)) for k, v in m.items()}
        res.update(self._final_metrics(state, returns))
        if self._ep_reward_hist:
            recent = self._ep_reward_hist[-10:]
            res['episode_reward_mean'] = float(np.mean(recent))
            res['episode_reward_max'] = float(np.max(recent))
            res['episode_reward_min'] = float(np.min(recent))
        return res
