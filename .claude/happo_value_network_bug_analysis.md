# HAPPO 价值网络无法学习的根本原因分析

**日期**: 2026-06-14  
**问题**: HAPPO 训练100k步后，价值网络的 vf_explained_var 仍然接近0，几乎没有学习

---

## 🔴 **核心问题：价值网络根本没有被更新！**

### 问题1：HAPPO 的致命设计缺陷

#### 📍 位置：`marllib/marl/algos/core/CC/happo.py:124-177`

```python
# 第124-129行：只更新策略网络
for i, iter_train_info in enumerate(get_each_agent_train(...)):
    ...
    iter_model.update_actor(  # ❌ 只更新 Actor
        loss=iter_surrogate_loss + ...,
        lr=current_lr,
        grad_clip=iter_policy.config['grad_clip'],
    )
    # ❌ 没有 update_critic() 调用！

# 第144-177行：计算价值损失但从未使用
if policy.config["use_critic"]:
    ...
    vf_loss = torch.max(vf_loss1, vf_loss2)
    mean_vf_loss = reduce_mean_valid(vf_loss)
    ...
    value_loss = reduce_mean_valid(policy.config['vf_loss_coeff'] * vf_loss)
    
    # 存储统计信息
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(...)
    
    return value_loss  # ❌ 返回了，但谁来用它做反向传播？
```

**关键问题**：
1. `value_loss` 被计算了，但**从未调用 `.backward()` 进行反向传播**
2. HAPPO 在每个 agent 的循环中调用 `update_actor()`，**手动更新策略网络**
3. 但价值网络被完全忽略了！

---

### 问题2：vf_share_layers=True 加剧了问题

#### 📍 位置：`marllib/marl/algos/scripts/happo.py:113`

```python
config = {
    ...
    "model": {
        "custom_model": "Centralized_Critic_Model",
        "max_seq_len": episode_limit,
        "custom_model_config": back_up_config,
        "vf_share_layers": True,  # ❌ 硬编码为 True
    },
}
```

**问题**：
- 价值网络与策略网络**共享特征提取层**
- 策略网络通过 `update_actor()` 更新，修改了共享层的参数
- 价值网络从未更新，但其依赖的共享层被策略网络"劫持"了
- 导致价值网络的特征表示完全被策略网络主导，无法学习自己的价值估计

---

### 问题3：update_actor() 更新了所有参数

#### 📍 位置：`marllib/marl/models/zoo/rnn/cc_rnn.py:89,159`

```python
# 第89行：Actor optimizer 绑定到所有参数
self.actor_optimizer = Adam(params=self.parameters(), lr=self.custom_config['actor_lr'])
# ❌ 注释掉的才是正确的：
# self.actor_optimizer = Adam(params=self.actor_parameters(), lr=...)

# 第155-161行：update_actor 更新所有参数
def update_actor(self, loss, lr, grad_clip):
    CentralizedCriticRNN.update_use_torch_adam(
        loss=(-1 * loss),
        optimizer=self.actor_optimizer,
        parameters=self.parameters(),  # ❌ 更新所有参数，包括价值网络
        grad_clip=grad_clip
    )
```

**问题**：
- `self.actor_optimizer` 被绑定到 `self.parameters()` (所有参数)
- 每次 `update_actor()` 调用时，策略损失会更新**整个模型**的参数
- 包括：
  - `p_encoder` (策略编码器) ✅ 应该更新
  - `rnn` (循环层) ✅ 应该更新
  - `p_branch` (策略头) ✅ 应该更新
  - `cc_vf_encoder` (价值网络编码器) ❌ 不应该被策略损失更新
  - `cc_vf_branch` (价值网络头) ❌ 不应该被策略损失更新

---

## 🔍 **对比：MAPPO 的正确实现**

### MAPPO 如何工作

#### 📍 位置：`marllib/marl/algos/core/CC/mappo.py:40-68`

```python
def central_critic_ppo_loss(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = ppo_surrogate_loss  # ✅ 使用 RLlib 原生 PPO 损失函数
    
    # 临时替换价值函数
    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(...)
    
    loss = func(policy, model, dist_class, train_batch)  # ✅ 自动处理价值网络更新
    
    model.value_function = vf_saved
    return loss
```

**关键差异**：
- MAPPO 调用 RLlib 的 `ppo_surrogate_loss`
- 该函数返回的损失会被 RLlib 的训练循环**自动用于反向传播**
- RLlib 会统一更新策略网络和价值网络

---

## 🎯 **为什么 HAPPO 训练仍然有效果？**

虽然价值网络没有正常学习，但策略网络仍然能够通过试错学习：

1. **GAE 计算**：使用（不准确的）价值估计计算优势函数
2. **策略梯度**：优势函数指导策略更新方向
3. **试错学习**：即使优势估计不准，策略仍能通过大量采样学到一些模式

**结果**：
- ✅ 奖励能提升（-82 → +13，峰值113）
- ❌ 但不稳定（从113掉到13）
- ❌ 价值网络 vf_explained_var ≈ 0
- ❌ 部分 agent 未收敛（power_1/2 仍在亏损）

---

## 🛠️ **修复方案**

### 方案A：修正 HAPPO 的价值网络更新逻辑（推荐）

#### 修改文件：`marllib/marl/algos/core/CC/happo.py`

```python
# 在 happo_surrogate_loss 函数的最后
# 第177行，修改为：

# 计算总损失
total_loss = mean_policy_loss + value_loss

# 对价值网络进行反向传播和更新
model.update_critic(
    loss=value_loss,
    lr=policy.cur_lr,
    grad_clip=policy.config['grad_clip']
)

# 存储统计信息
model.tower_stats["total_loss"] = total_loss
model.tower_stats["mean_policy_loss"] = mean_policy_loss
model.tower_stats["mean_vf_loss"] = mean_vf_loss
model.tower_stats["vf_explained_var"] = explained_variance(...)

return total_loss  # 或者返回 0（因为已经手动更新了）
```

#### 修改文件：`marllib/marl/models/zoo/rnn/cc_rnn.py`

```python
# 第89行，修正 actor_optimizer
self.actor_optimizer = Adam(params=self.actor_parameters(), lr=self.custom_config['actor_lr'])

# 第90行，取消注释并修正
self.critic_optimizer = Adam(params=self.critic_parameters(), lr=self.custom_config['critic_lr'])

# 添加 update_critic 方法（在 update_actor 后面）
def update_critic(self, loss, lr, grad_clip):
    CentralizedCriticRNN.update_use_torch_adam(
        loss=loss,
        optimizer=self.critic_optimizer,
        parameters=self.critic_parameters(),
        grad_clip=grad_clip
    )
```

---

### 方案B：禁用 vf_share_layers（临时缓解）

#### 修改文件：`marllib/marl/algos/scripts/happo.py`

```python
# 第113行
config = {
    ...
    "model": {
        "custom_model": "Centralized_Critic_Model",
        "max_seq_len": episode_limit,
        "custom_model_config": back_up_config,
        "vf_share_layers": False,  # 改为 False
    },
}
```

**效果**：
- 价值网络有独立的特征提取层
- 虽然仍不更新，但至少不会被策略网络干扰
- **注意**：这只是缓解，不能根本解决问题

---

### 方案C：回退到标准 PPO 损失函数（最彻底）

```python
# 在 happo.py 中，参考 MAPPO 的实现
# 使用 RLlib 原生的 ppo_surrogate_loss
# 然后在此基础上实现 HAPPO 的异构更新逻辑
```

---

## 📊 **验证方法**

修复后，重新训练并检查：

```bash
python experiments/exp1/exp1_train_single.py --group_id A4 --seed 42 --timesteps 100000
```

**预期改善**：
1. ✅ `vf_explained_var` 从 0 增长到 0.3-0.7
2. ✅ 奖励曲线更稳定，不会从113掉到13
3. ✅ 所有 agent 都能收敛（power_1/2 不再亏损）
4. ✅ 最终表现超过 MAPPO

---

## 🎓 **总结**

### 根本原因
1. **HAPPO 实现有 bug**：价值损失被计算但从未用于反向传播
2. **update_actor() 更新了全部参数**：包括本不应该被策略损失更新的价值网络
3. **vf_share_layers=True 加剧问题**：共享层被策略网络完全主导

### 为什么 MAPPO 没问题
- MAPPO 使用 RLlib 原生的 `ppo_surrogate_loss`
- RLlib 会自动、统一地更新策略网络和价值网络
- 不需要手动调用 `update_actor()` 或 `update_critic()`

### 为什么训练仍有效果
- 策略网络仍能通过试错学习
- 但由于价值估计不准确，导致训练不稳定、效率低

### 修复优先级
1. **最高优先级**：实现 `update_critic()` 并在损失函数中调用
2. **次高优先级**：修正 `actor_optimizer` 只更新策略参数
3. **可选**：将 `vf_share_layers` 改为可配置，默认 False

---

**结论**：这不是超参数问题，也不是环境问题，而是 **MARLlib 的 HAPPO 实现存在严重 bug**。
