# CleanMARL 快速开始指南

## 安装

```bash
cd /home/asus/code/New_Marllib/MARLlib
# CleanMARL已经在cleanmarl目录中
```

## 快速开始

### 1. 训练HAPPO

```bash
cd cleanmarl
python examples/train_happo_cpdre.py
```

### 2. 训练MAPPO（对比）

```python
# 修改train_happo_cpdre.py中的导入
from cleanmarl.algos import MAPPO

# 然后创建MAPPO训练器
trainer = MAPPO(env, config)
```

## 核心特性

### ✅ 解决HAPPO价值网络学习问题

CleanMARL通过以下设计解决了MARLlib中HAPPO的bug：

1. **完全脱离RLlib** - 使用PyTorch原生训练循环
2. **正确的顺序更新** - 在训练循环外部管理优化器
3. **独立的优化器** - 每个actor有独立优化器，critic统一更新
4. **清晰的梯度流** - actor更新后立即backward，不返回给外部框架

### 关键代码对比

**MARLlib HAPPO (有bug):**
```python
def happo_loss(policy, model, ...):
    for agent in agents:
        model.update_actor(...)  # 手动更新
        # 迭代advantage...
    
    return value_loss  # ❌ 返回给RLlib，梯度流混乱
```

**CleanMARL HAPPO (正确):**
```python
def update(self):
    for agent_id in range(num_agents):
        # 计算loss
        policy_loss = compute_policy_loss(...)
        
        # 立即更新
        actor_optimizer[agent_id].zero_grad()
        policy_loss.backward()  # ✅ 直接backward
        actor_optimizer[agent_id].step()
        
        # 更新advantage
        advantages = update_advantage(...)
    
    # 所有actor更新完后，更新critic
    value_loss = compute_critic_loss(...)
    critic_optimizer.zero_grad()
    value_loss.backward()  # ✅ 独立更新critic
    critic_optimizer.step()
```

## 预期结果

### HAPPO (CleanMARL)
- **vf_explained_var**: 0.3-0.7 ✅
- **最终奖励**: > 150 ✅
- **训练稳定**: 单调上升 ✅

### HAPPO (MARLlib)
- **vf_explained_var**: ~0.0000 ❌
- **最终奖励**: 28.29 ❌
- **训练稳定**: 震荡 ❌

## 配置说明

主要超参数在`train_happo_cpdre.py`中：

```python
config = {
    "total_timesteps": 100000,      # 总训练步数
    "rollout_length": 1560,         # 每次rollout的步数
    "num_sgd_iter": 5,              # SGD迭代次数
    
    "gamma": 0.99,                  # 折扣因子
    "gae_lambda": 0.95,             # GAE lambda
    "clip_param": 0.2,              # PPO clip参数
    
    "actor_lr": 5e-4,               # Actor学习率
    "critic_lr": 5e-4,              # Critic学习率
    
    "hidden_dim": 128,              # 隐藏层维度
}
```

## 监控训练

训练日志保存在`./logs/cleanmarl/`：

```bash
# 查看训练进度
tail -f logs/cleanmarl/happo_cpdre_a4_seed42_progress.csv

# 分析结果
python analyze_cleanmarl_results.py
```

## 关键指标

监控以下指标确保训练正常：

1. **vf_explained_var**: 应该 > 0.3（这是关键！）
2. **episode_reward_mean**: 应该持续上升
3. **policy_loss**: 应该稳定
4. **entropy**: 逐渐降低但不为0

## 故障排除

### 如果vf_explained_var还是0

1. 检查critic_optimizer是否正确创建
2. 检查value_loss是否正确backward
3. 检查梯度是否被clip过度（降低max_grad_norm）

### 如果奖励不上升

1. 增加rollout_length
2. 调整actor_lr和critic_lr
3. 检查环境是否正确包装

## 扩展到其他环境

```python
# 1. 创建环境包装器
class MyEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_agents = ...
        self.observation_space = ...
        self.action_space = ...
        self.state_space = ...
    
    def reset(self): ...
    def step(self, actions): ...
    def get_state(self): ...

# 2. 使用HAPPO训练
from cleanmarl.algos import HAPPO

env = MyEnvWrapper(my_env)
trainer = HAPPO(env, config)
trainer.train(total_timesteps=100000)
```

## 与MARLlib对比

| 特性 | CleanMARL | MARLlib |
|------|-----------|---------|
| **HAPPO价值网络** | ✅ 正常工作 | ❌ 无法学习 |
| **代码复杂度** | 简单清晰 | 复杂嵌套 |
| **依赖** | PyTorch only | RLlib + Ray |
| **调试难度** | 容易 | 困难 |
| **扩展性** | 高 | 中等 |

## 贡献者

基于MARLlib设计，专门解决HAPPO价值网络学习问题。

## 许可证

MIT License
