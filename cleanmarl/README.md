# CleanMARL

CleanMARL 是当前仓库中用于 CPDRE 煤电直接互惠环境的轻量级多智能体强化学习实现。它绕开 RLlib 的训练循环，直接用 PyTorch 管理采样、GAE、PPO/HAPPO 更新、日志和 checkpoint，主要服务于 `model/0611(1).md` 中的实验复现。

当前代码重点支持 `1` 个煤企 + `3` 个电企、`1` 个互惠电企的 B4 主线实验。环境核心已经比旧版更接近 0611 文档，但完整实验编排、统计检验和多互惠扩展还在补齐中。

## 当前能力

- 算法：`HAPPO`、`MAPPO`、`IPPO`、若干规则策略基线。
- 环境：直接构造 `custom_envs/coal_power_direct_reciprocity_env.py`，不依赖 MARLlib 注册环境。
- 动作：煤企 2 维动作 `(theta, lambda)`，电企 1 维动作 `omega`，wrapper 支持异质动作维度。
- 训练：episode rollout、GAE、PPO clip、GRU actor/critic、checkpoint 保存/加载。
- 日志：训练进度 CSV、episode summary、可选 step-level 详情。
- 配置：YAML 入口支持 B4 主实验参数、固定评估轨迹、多 seed sweep。

## 运行前提

本目录目前不是标准 Python package，没有 `setup.py` 或 `pyproject.toml`。请从仓库根目录运行脚本，让 Python 直接通过源码路径导入 `cleanmarl` 和 `custom_envs`。

运行环境需要已经具备这些依赖：

```text
torch
numpy
gym
ray[rllib]
pyyaml
```

依赖声明在 `cleanmarl/requirements.txt`，但本 README 不自动安装依赖；请使用你自己的 Python 环境管理方式。

## 当前环境解释器地址
\\wsl.localhost\Ubuntu\home\asus\miniconda3\envs\marllib_torchtest\bin
也就是在ubuntu的虚拟环境


## 快速开始

从仓库根目录执行：

```bash
cd /home/asus/code/New_Marllib/MARLlib
python3 cleanmarl/train.py --config configs/default.yaml
```

常用入口：

```bash
# B4 HAPPO 主配置
python3 cleanmarl/train.py --config configs/happo_cpdre.yaml

# B4 IPPO 对照
python3 cleanmarl/train.py --config configs/ippo_cpdre.yaml

# 覆盖 seed / 步数 / 设备
python3 cleanmarl/train.py --config configs/default.yaml --seed 40 --timesteps 100000 --device cpu

# 多 seed 扫描
python3 cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --seeds 40,41,42,43,44

# 极小步数流程检查
python3 cleanmarl/smoke_test.py
```

日志默认写入：

```text
logs/cleanmarl/
logs/cleanmarl/<experiment_name>/episode_summary.csv
```

checkpoint 默认写入：

```text
checkpoints/
```

## 主要配置

`cleanmarl/configs/default.yaml` 是目前最完整的 B4 配置，包含：

- `mechanism_mode: "b4"`
- `demand_mode: "lognormal"`
- `eval_episodes: 64`
- `chunk_len: 10`
- `model.rnn_layers: 2`
- PPO 表 4.3 对齐参数，如 `gamma=0.95`、`actor_lr=1e-4`、`critic_lr=1e-4`

`cleanmarl/configs/happo_cpdre.yaml` 和 `cleanmarl/configs/ippo_cpdre.yaml` 已经切到 B4/weighted/lognormal，但还没有显式写入 `chunk_len` 和 `rnn_layers`。如果用这两个配置训练，当前代码会回退到默认的 `chunk_len=0`、`rnn_layers=1`。

`cleanmarl/configs/mappo_cpdre.yaml` 仍偏旧，还是 `mechanism_mode: "none"` 和 deterministic demand，不适合作为当前 B4 主实验的直接 MAPPO 对照。

## 代码结构

```text
cleanmarl/
├── algos/
│   ├── happo.py          # HAPPO 训练器
│   ├── mappo.py          # 复用 HAPPO 采样/critic 的 MAPPO
│   ├── ippo.py           # independent actor/critic PPO
│   └── rule_policy.py    # 固定/规则策略基线
├── configs/
│   ├── default.yaml
│   ├── happo_cpdre.yaml
│   ├── ippo_cpdre.yaml
│   └── mappo_cpdre.yaml
├── core/
│   ├── trainer.py        # 通用训练循环
│   ├── logger.py         # progress CSV
│   └── episode_logger.py # episode/step 指标
├── envs/
│   └── cpdre_wrapper.py  # CPDRE numpy wrapper
├── models/
│   └── models.py         # GRUActor / CentralizedCritic
├── train.py              # YAML 训练入口
├── run_sweep.py          # 多 seed 入口
└── smoke_test.py         # 小规模流程检查
```

环境实现位于：

```text
custom_envs/coal_power_direct_reciprocity_env.py
```

## 与 0611 文档的对齐状态

已经对齐或基本对齐：

- B4 下 `lambda_{1,t}` 不再被 fair allocation 抹掉。
- 电企利润使用完整需求收入 `r * D`，短缺通过外部采购成本 `c_rep * S` 体现。
- 主配置使用季节性 lognormal 需求扰动。
- 支持煤企产能 `theta`、保供权重 `lambda`、互惠记忆 `mu`、贡献 `g_u/g_c`。
- 支持 `rnn_layers` 和 `chunk_len` 训练路径。
- B5 观测中已做 aggregate-only 处理，减少非关系动态协调组的信息泄漏。

仍需注意：

- HAPPO 的 advantage 迭代仍是相邻 agent 简化传播，不是完整联合 ratio 形式。
- 连续动作目前是无界 Normal 采样后在 wrapper 中 clip，严格 PPO 需要改成 bounded/squashed distribution。
- `run_sweep.py` 还没有 `--group` 参数，不能自动生成 B1/B2/B3/B4/B5/C 系列实验。
- 多互惠 `h>1` 和多电企 `m>3` 还不完整，观测和日志仍有 `u1/u2/u3`、`pad3` 等固定写法。
- 统计检验、配对 bootstrap、论文表格导出脚本还没有体系化。
- `examples/`、`QUICKSTART.md`、部分分析脚本仍偏旧，优先以 `train.py` 和 `configs/default.yaml` 为准。

更完整的问题清单见：

```text
model/cleanmarl_0611_issues.md
```

## 开发建议

1. 优先使用 `cleanmarl/train.py` 和 YAML 配置运行，不建议继续依赖 `examples/` 里的旧硬编码脚本。
2. 正式实验前先统一 `default/happo/ippo/mappo` 配置，确保算法对照只改变算法本身。
3. 跑论文结果前补齐 `run_sweep.py` 的组别编排和统计检验脚本。
4. 若要扩展到 `m>3` 或 `h>1`，先改观测维度、贡献计算和 EpisodeLogger schema。
