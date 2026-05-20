# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Patched heterogeneous updating utilities for MARLlib HAPPO.

Why this patch is needed for CPDRE:
- CPDRE observations are Dict observations: {"obs": Box(24)}.
- In HAPPO's heterogeneous update, MARLlib wraps one global SampleBatch into
  IterTrainBatch for each agent/policy.
- The original local file lost IterTrainBatch.__getitem__ / __contains__, so
  RLlib's model(train_batch) could not find SampleBatch.OBS although the parent
  batch actually contained keys such as "obs" and "global_obs_agent_*".

This patch restores a robust agent-wise key mapping and keeps normal MAPPO/IPPO
behavior untouched.
"""

import random
import re

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import sequence_mask

from marllib.marl.algos.utils.centralized_critic_hetero import (
    get_global_name,
    global_state_name,
)


torch, nn = try_import_torch()


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _policy_name_variants(policy_name):
    """
    Build possible suffixes used by different MARLlib versions.

    In your error log, HAPPO used keys like:
        global_obs_agent_coal_0
        global_actions_agent_power_0

    Depending on the model wrapper, policy_name may be "agent_coal_0",
    "coal_0", "policy_coal_", etc. These variants make the adaptor tolerant.
    """
    variants = [policy_name]

    if isinstance(policy_name, str):
        if policy_name.startswith("policy_"):
            variants.append("agent_" + policy_name[len("policy_"):])
            variants.append(policy_name[len("policy_"):])
        if policy_name.startswith("agent_"):
            variants.append(policy_name[len("agent_"):])
        if not policy_name.startswith("agent_"):
            variants.append("agent_" + policy_name)

    return _dedupe(variants)


def _global_key_candidates(policy_name, key):
    """
    Candidate keys for fields of a specific heterogeneous agent.

    The most important mapping is:
        obs             -> global_obs_agent_xxx
        actions         -> global_actions_agent_xxx
        action_logp     -> global_action_logp_agent_xxx
        action_dist_inputs -> global_action_dist_inputs_agent_xxx

    We also keep get_global_name(...) because it is MARLlib's native helper.
    """
    candidates = []
    key_variants = [key]

    # CPDRE Dict(obs=Box(...)) may be flattened by RLlib/MARLlib in some places.
    if key == SampleBatch.OBS or key == "obs":
        key_variants.extend(["obs_flat", "obs_obs"])

    # RNN keys such as state_in_0/state_out_0.
    if isinstance(key, str) and re.match(r"^state_(in|out)_\d+$", key):
        key_variants.append(key)

    for kv in _dedupe(key_variants):
        try:
            candidates.append(get_global_name(policy_name, kv))
        except Exception:
            pass

        for name in _policy_name_variants(policy_name):
            # Observed MARLlib heterogeneous critic naming convention.
            candidates.append(f"global_{kv}_{name}")
            candidates.append(f"{name}_{kv}")
            candidates.append(f"{name}{kv}")

    return _dedupe(candidates)


def _ensure_obs_key_for_happo(train_batch):
    """
    Ensure SampleBatch.OBS can be resolved before calling RLlib ModelV2.

    For IterTrainBatch, obs is resolved lazily by __getitem__/__contains__;
    do not mutate the wrapper itself.
    """
    if isinstance(train_batch, IterTrainBatch) or hasattr(train_batch, "main_train_batch"):
        return train_batch

    if SampleBatch.OBS in train_batch:
        return train_batch

    # Normal SampleBatch observation aliases.
    for key in ["obs_flat", "obs_obs", "obs"]:
        if key in train_batch:
            train_batch[SampleBatch.OBS] = train_batch[key]
            return train_batch

    raise KeyError(
        "HAPPO train_batch has no obs-like field. "
        f"Available keys: {list(train_batch.keys())}"
    )


def get_mask_and_reduce_mean(model, train_batch, dist_class):
    train_batch = _ensure_obs_key_for_happo(train_batch)
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    return mask, reduce_mean_valid, curr_action_dist


def update_m_advantage(
    iter_model,
    iter_train_batch,
    iter_dist_class,
    iter_prev_action_logp,
    iter_actions,
    m_advantage,
):
    with torch.no_grad():
        iter_model.eval()
        iter_train_batch = _ensure_obs_key_for_happo(iter_train_batch)
        iter_new_logits, _ = iter_model(iter_train_batch)
        try:
            iter_new_action_dist = iter_dist_class(iter_new_logits, iter_model)
            iter_new_logp_ratio = torch.exp(
                iter_new_action_dist.logp(iter_actions) - iter_prev_action_logp
            )
        except ValueError as e:
            # Preserve original MARLlib behavior, but make the next error clearer.
            print(e)
            raise

    m_advantage = iter_new_logp_ratio * m_advantage
    return m_advantage


class IterTrainBatch(SampleBatch):
    """
    Adaptor for HAPPO heterogeneous updating.

    For the self policy, HAPPO uses the normal train_batch directly.
    For another agent/policy, HAPPO creates IterTrainBatch(train_batch,
    policy_name). When a model asks for SampleBatch.OBS, SampleBatch.ACTIONS,
    SampleBatch.ACTION_LOGP, etc., this adaptor redirects the request to the
    corresponding global_*_agent_* field in the parent train_batch.
    """

    _AGENT_SPECIFIC_KEYS = {
        SampleBatch.OBS,
        SampleBatch.NEXT_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.ACTION_LOGP,
        SampleBatch.ACTION_DIST_INPUTS,
        SampleBatch.VF_PREDS,
        "obs",
        "new_obs",
        "actions",
        "action_logp",
        "action_dist_inputs",
        "vf_preds",
    }

    def __init__(self, main_train_batch, policy_name):
        # Do not call SampleBatch.__init__; this wrapper delegates storage.
        self.main_train_batch = main_train_batch
        self.policy_name = policy_name

        self.copy = self.main_train_batch.copy
        self.keys = self.main_train_batch.keys
        self.is_training = self.main_train_batch.is_training

        self.pat = re.compile(r"^state_in_(\d+)")

    def _candidate_keys(self, key):
        candidates = []
        candidates.extend(_global_key_candidates(self.policy_name, key))

        # Fallbacks for common direct keys. These are useful for fields shared
        # across agents, for older MARLlib versions, or for single-policy tests.
        if key in [SampleBatch.OBS, "obs"]:
            candidates.extend([SampleBatch.OBS, "obs_flat", "obs_obs"])
        elif key in [SampleBatch.NEXT_OBS, "new_obs"]:
            candidates.extend([SampleBatch.NEXT_OBS, "new_obs_flat", "new_obs_obs"])
        else:
            candidates.append(key)

        return _dedupe(candidates)

    def __contains__(self, key):
        for candidate in self._candidate_keys(key):
            if candidate in self.main_train_batch:
                return True
        return False

    def __getitem__(self, key):
        # For agent-specific keys, prefer global_<field>_<agent> over direct
        # self-policy fields. This avoids accidentally feeding self obs/actions
        # into another agent's model.
        candidates = self._candidate_keys(key)
        for candidate in candidates:
            if candidate in self.main_train_batch:
                return self.main_train_batch[candidate]

        raise KeyError(
            f"IterTrainBatch cannot find key={key!r}. "
            f"policy_name={self.policy_name!r}. "
            f"candidate_keys={candidates}. "
            f"available_keys={list(self.main_train_batch.keys())}"
        )

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        # Keep direct writes local to the parent batch using the requested key.
        # HAPPO normally does not rely on mutating IterTrainBatch, but this keeps
        # compatibility with SampleBatch-like behavior.
        self.main_train_batch[key] = value


def get_each_agent_train(model, policy, dist_class, train_batch):
    all_policies_with_names = list(model.other_policies.items()) + [("self", policy)]
    random.shuffle(all_policies_with_names)

    for policy_name, iter_policy in all_policies_with_names:
        is_self = policy_name == "self"
        iter_model = [iter_policy.model, model][is_self]
        iter_dist_class = [iter_policy.dist_class, dist_class][is_self]
        iter_train_batch = [IterTrainBatch(train_batch, policy_name), train_batch][is_self]

        iter_mask, iter_reduce_mean, current_action_dist = get_mask_and_reduce_mean(
            iter_model,
            iter_train_batch,
            iter_dist_class,
        )
        iter_actions = iter_train_batch[SampleBatch.ACTIONS]
        iter_prev_action_logp = iter_train_batch[SampleBatch.ACTION_LOGP]

        yield (
            iter_model,
            iter_dist_class,
            iter_train_batch,
            iter_mask,
            iter_reduce_mean,
            iter_actions,
            iter_policy,
            iter_prev_action_logp,
        )
