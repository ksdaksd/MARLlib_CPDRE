# CleanMARL v0.3.0 更新说明

## 新增功能

### EpisodeLogger：记录每个 episode 的详细数据

之前只记录训练指标（loss、vf_explained_var、reward 汇总），现在新增两层 episode 级日志：

**1. episode_summary.csv（默认开启）**

每个 episode 一行，包含：
- `episode`、`phase`(train/eval)、`episode_length`
- `total_reward` — 团队总 reward
- `reward_agent_0/1/2/3` — 每个 agent 的累计 reward
- 聚合的环境指标（对整个 episode 取 mean 或 sum）：
  - `mean_shortage_rate`、`mean_fairness_penalty`、`mean_jain`
  - `mean_price`、`mean_supply`、`mean_unsold`
  - `mean_own_demand/order/shipment/shortage/inventory/fill_rate_u1/u2/u3`
  - `sum_system_profit`、`sum_coal_profit`
  - `sum_power_profit_u1/u2/u3`、`sum_power_profit_total`

**2. 步级详情（可选，默认关闭）**

在 YAML 中设置 `train.log_step_details: true` 开启，会为每个 episode 单独生成：
```
episodes/train_ep_XXXXX_steps.csv
episodes/eval_ep_XXXXX_steps.csv
```
每步记录：每个 agent 的 obs 各维、action 各维、reward，以及环境指标（price、shortage_rate、profit 等）。

⚠️ 步级详情文件较大（每个 episode ~156 行 × 数十列），仅在需要分析 agent 行为时开启。

### 用法

```bash
# 默认（只记 summary）
python cleanmarl/train.py --config configs/happo_cpdre.yaml

# 启用步级详情
# 编辑 configs/happo_cpdre.yaml -> train.log_step_details: true
python cleanmarl/train.py --config configs/happo_cpdre.yaml
```

### 输出目录结构

```
logs/cleanmarl/<experiment>/
├── <experiment>_progress.csv      # 训练进度（每 epoch 一行）
├── <experiment>_config.json       # 配置快照
├── episode_summary.csv            # 每 episode 一行（汇总）← 新增
└── episodes/                      # 步级详情（可选）← 新增
    ├── train_ep_00001_steps.csv
    ├── train_ep_00002_steps.csv
    └── eval_ep_00000_steps.csv
```

### 改动文件

| 文件 | 改动 |
|------|------|
| `core/episode_logger.py` | 新建 |
| `core/trainer.py` | 集成 EpisodeLogger |
| `envs/cpdre_wrapper.py` | step() 返回扁平化 step_info |
| `algos/happo.py` | collect_trajectories + evaluate 记录 episode |
| `algos/ippo.py` | collect_trajectories + evaluate 记录 episode |
| `algos/mappo.py` | 继承 HAPPO，自动生效 |
| `train.py` | 读取 log_step_details 配置 |
| `configs/*.yaml` | 添加 log_step_details 字段 |

### 测试结果

- ✅ HAPPO：episode_summary.csv 正常生成（72KB，32 episodes）
- ✅ MAPPO：episode_summary.csv + 步级详情正常（80 个 step CSV 文件）
- ✅ IPPO：episode_summary.csv 正常生成（72KB）

---

# CleanMARL v0.2.1 更新说明

## 新增功能

### 1. 三种算法 YAML 配置文件完整

为三种算法都提供了独立的 YAML 配置文件：

| 算法 | 配置文件 | 特点 |
|------|---------|------|
| HAPPO | `configs/happo_cpdre.yaml` | 顺序更新策略 + 共享critic |
| MAPPO | `configs/mappo_cpdre.yaml` | 并行更新策略 + 共享critic |
| IPPO | `configs/ippo_cpdre.yaml` | 完全独立训练 |

### 2. 算法实现检查完成

所有算法实现正确，冒烟测试通过：

```bash
# 测试命令
python cleanmarl/train.py --config configs/happo_cpdre.yaml --timesteps 5000 --device cpu
python cleanmarl/train.py --config configs/mappo_cpdre.yaml --timesteps 5000 --device cpu
python cleanmarl/train.py --config configs/ippo_cpdre.yaml --timesteps 5000 --device cpu
```

所有算法：
- ✅ 正常启动训练
- ✅ vf_explained_var 在初始epoch接近0（正常，训练初期）
- ✅ episode_reward_mean 逐步上升（策略在学习）
- ✅ value_loss 逐步下降（critic在学习）
- ✅ entropy 保持稳定（策略不过早收敛）

---

## 文件结构更新

```
cleanmarl/
├── train.py                  ← 通用训练入口
├── configs/                  ← YAML配置文件目录
│   ├── default.yaml          # 默认配置模板
│   ├── happo_cpdre.yaml      # HAPPO 配置
│   ├── mappo_cpdre.yaml      # MAPPO 配置 ← 新增
│   └── ippo_cpdre.yaml       # IPPO 配置  ← 新增
│
├── algos/
│   ├── happo.py              # HAPPO（顺序更新策略）
│   ├── mappo.py              # MAPPO（并行更新策略）
│   └── ippo.py               # IPPO（完全独立训练）
│
└── requirements.txt          ← PyYAML 依赖
```

---

# CleanMARL v0.2.0 更新说明

## 新增功能

### 1. IPPO 算法

新增 `IPPO`（Independent PPO）算法，每个agent有独立的policy和独立的critic：
- **最简单的多智能体PPO扩展**
- **完全独立训练**，每个agent不共享任何参数
- **最稳定的baseline**

文件：`cleanmarl/algos/ippo.py`

### 2. YAML 配置系统

支持用 YAML 文件配置所有超参数和环境参数：
- **统一配置文件**：`configs/default.yaml`
- **按场景配置**：`configs/happo_cpdre.yaml`、`configs/mappo_cpdre.yaml` 等
- **命令行覆盖**：`--algo`, `--seed`, `--device`, `--timesteps`

文件：`cleanmarl/train.py`（通用训练入口）

### 3. 通用训练脚本

一个脚本支持所有算法：

```bash
# HAPPO
python cleanmarl/train.py --config configs/happo_cpdre.yaml

# MAPPO
python cleanmarl/train.py --config configs/mappo_cpdre.yaml --algo mappo

# IPPO
python cleanmarl/train.py --config configs/default.yaml --algo ippo

# 命令行覆盖
python cleanmarl/train.py --config configs/happo_cpdre.yaml --seed 123 --timesteps 200000 --device cpu
```

---

## 文件结构更新

```
cleanmarl/
├── train.py                  ← 通用训练入口（从YAML读取配置）
├── configs/                  ← YAML配置文件目录
│   ├── default.yaml          # 默认配置模板
│   └── happo_cpdre.yaml      # HAPPO CPDRE A4场景配置
│
├── algos/
│   ├── happo.py              # HAPPO（顺序更新策略）
│   ├── mappo.py              # MAPPO（并行更新策略）
│   └── ippo.py               # IPPO（完全独立训练）← 新增
│
└── requirements.txt          ← 新增 PyYAML 依赖
```

---

## YAML 配置说明

配置文件结构：

```yaml
# 环境参数
env:
  map_name: "direct_1c3u_A4_s42"
  num_power: 3
  seed: 42
  ...

# 训练参数
train:
  total_timesteps: 100000
  rollout_length: 1560
  num_sgd_iter: 5

# PPO 参数
ppo:
  gamma: 0.99
  gae_lambda: 0.95
  clip_param: 0.2

# 学习率
lr:
  actor_lr: 0.0005
  critic_lr: 0.0005

# 网络架构
model:
  hidden_dim: 128

# 系统配置
system:
  device: "cuda"
  log_dir: "./logs/cleanmarl"

# 实验配置
experiment:
  name: "happo_cpdre_a4_seed42"
  algo: "happo"  # happo / mappo / ippo
```

---

## 快速开始（更新版）

```bash
# 1. 安装依赖
pip install -r cleanmarl/requirements.txt

# 2. 使用YAML配置训练
python cleanmarl/train.py --config configs/happo_cpdre.yaml

# 3. 或使用命令行快速切换算法
python cleanmarl/train.py --config configs/default.yaml --algo ippo --seed 100
```

---

## 算法对比

| 算法 | Critic | 更新方式 | 适用场景 |
|------|--------|----------|---------|
| **HAPPO** | 共享 | 顺序更新策略 | 异构agent、需要精细协调 |
| **MAPPO** | 共享 | 并行更新策略 | 同构agent、高协作场景 |
| **IPPO** | 独立 | 完全独立训练 | 简单场景、快速baseline |

---

## 迁移清单（v0.2.0）

新增文件：
- `cleanmarl/train.py`
- `cleanmarl/configs/default.yaml`
- `cleanmarl/configs/happo_cpdre.yaml`
- `cleanmarl/algos/ippo.py`

更新文件：
- `cleanmarl/__init__.py`（新增IPPO导出）
- `cleanmarl/requirements.txt`（新增PyYAML）