# Experiment 1 使用文档

> CPDRE（煤电直接互惠环境）实验一的训练与可视化流程

---

## 一、目录文件清单

清理后 `experiments/exp1/` 目录下只保留以下 5 个 py 文件 + 本文档：

```
experiments/exp1/
├── __init__.py                  # 包标识，无内容
├── exp1_common.py               # 实验配置和公共工具
├── cpdre_callbacks_v3.py        # 数据收集回调（核心）
├── exp1_train_single.py         # 训练入口（核心）
├── plot_per_agent_metrics.py    # Per-agent 视角画图
├── plot_episode_metrics.py      # 系统级视角画图
└── README.md                    # 本文档
```

### 1.1 每个文件的作用

| 文件 | 作用 | 何时使用 |
|------|------|---------|
| **`exp1_common.py`** | 定义 9 个实验组（A0-A9）、`base_env_args()` 生成环境参数、CSV/JSON 工具、目录路径常量 | 被其他脚本 import |
| **`cpdre_callbacks_v3.py`** | RLlib 回调，每个 episode 结束时自动从 `env.history` 收集数据，写到 CSV（per-worker）和 NPZ（per-episode） | 训练时由 RLlib 自动调用，**不直接运行** |
| **`exp1_train_single.py`** | 训练入口脚本，按指定的实验组 + seed 跑一次 RL 训练 | 跑训练时直接执行 |
| **`plot_per_agent_metrics.py`** | 读 per-worker CSV → 合并 → 检测 per-agent 列 → 画每个智能体的对比图（缺货、发货、需求等） | 训练结束后画图 |
| **`plot_episode_metrics.py`** | 读 per-worker CSV → 合并 → 画系统级指标曲线（系统利润、公平性、信任、学习曲线等） | 训练结束后画图 |

### 1.2 已删除的 13 个文件（及原因）

| 文件 | 删除原因 |
|------|---------|
| `cpdre_callbacks.py` | 旧版 v1，硬编码字段，已被 v3 替代 |
| `cpdre_callbacks_v2.py` | 中间版 v2，部分硬编码，已被 v3 替代 |
| `exp1_train_single_v2.py` | 与 `exp1_train_single.py` 重复 |
| `exp1_train_all.py` | 批量训练编排器，非核心 |
| `exp1_eval_baselines.py` | Baseline 策略评估，本次实验未使用 |
| `exp1_collect_progress.py` | **broken**：import `exp_20260513_model_direct`（不存在的模块） |
| `exp1_make_tables.py` | **broken**：同上 |
| `exp1_plot_progress.py` | **broken**：同上 |
| `exp1_run_full_pipeline.py` | **broken**：同上 |
| `exp_eval_collect_heuristics.py` | **broken**：同上，且非实验一核心 |
| `exp_plot_eval.py` | **broken**：同上，且非实验一核心 |
| `episode_data_utils.py` | NPZ 加载辅助工具，非核心 |
| `test_episode_data.py` | 测试脚本，非核心 |

外加 2 个旧 MD 文档（`CPDRE_DATA_GUIDE.md`、`EPISODE_DATA_GUIDE.md`）也已删除，本 README 取代它们。

---

## 二、实验组定义

`exp1_common.py` 中的 `EXP1_GROUPS`：

| Group | 算法 | 需求模式 | demand_sigma | 备注 |
|-------|------|---------|--------------|------|
| A0 | base_stock（baseline） | 确定性 | 0.0 | 不需要 RL，本套代码不跑 |
| A1 | seasonal_base_stock（baseline） | 确定性 | 0.0 | 同上 |
| **A2** | **IPPO-GRU** | **确定性** | **0.0** | **RL 可跑** |
| **A3** | **MAPPO-GRU** | **确定性** | **0.0** | **RL 可跑** |
| **A4** | **HAPPO-GRU** | **确定性** | **0.0** | **RL 可跑** |
| A5 | base_stock | 低噪声 | 0.05 | 同 A0 |
| A6 | seasonal_base_stock | 低噪声 | 0.05 | 同 A1 |
| **A7** | **IPPO-GRU** | **低噪声** | **0.05** | **RL 可跑** |
| **A8** | **MAPPO-GRU** | **低噪声** | **0.05** | **RL 可跑** |
| **A9** | **HAPPO-GRU** | **低噪声** | **0.05** | **RL 可跑** |

`exp1_train_single.py` 只支持 **A2、A3、A4、A7、A8、A9** 6 个 RL 组。

---

## 三、跑实验一的完整步骤

### 步骤 0：环境准备

```bash
# 在 WSL Ubuntu 终端中
cd ~/code/New_Marllib/MARLlib

# 激活 conda 环境
conda activate marllib_torchtest

# 设置 PYTHONPATH（如果通过命令行跑而不是 PyCharm Run）
export PYTHONPATH=/home/asus/code/New_Marllib:/home/asus/code/New_Marllib/MARLlib:$PYTHONPATH
```

### 步骤 1：清理旧数据（**重要！每次新训练前都要做**）

> CSV 写入用 append 模式，旧文件残留会导致新旧数据混合。

```bash
cd ~/code/New_Marllib/MARLlib/experiments/exp1

# 假设要跑 A4, seed=43，清理相关文件
rm -f results/episode_metrics/A4_happo_seed43*.csv
rm -rf results/episode_histories
mkdir -p results/episode_histories
```

### 步骤 2：跑训练

```bash
cd ~/code/New_Marllib/MARLlib

python experiments/exp1/exp1_train_single.py \
  --group_id A4 \
  --seed 43 \
  --timesteps 100000 \
  --episode_len 156 \
  --core_arch gru \
  --share_policy individual \
  --num_workers 3
```

#### 参数说明

| 参数 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `--group_id` | ✅ | - | 实验组 ID，选 `A2/A3/A4/A7/A8/A9` |
| `--seed` | | 42 | 随机种子 |
| `--timesteps` | | 100000 | 训练 timesteps 总数 |
| `--episode_len` | | 156 | 每个 episode 长度（156 周 = 3 年）|
| `--core_arch` | | gru | 网络结构（`gru`/`lstm`/`mlp`）|
| `--share_policy` | | group | 策略共享方式（`group`/`individual`）|
| `--num_workers` | | 0 | RLlib rollout worker 数量 |
| `--price_mode` | | fixed | 价格模式（`fixed`/`seasonal`/`feedback`）|

#### 训练时关键日志（确认 callback 正常工作）

训练开始应当依次看到：

```
[CPDRE] run_cc.py loaded CPDREAdaptiveCallback (v3)
[CPDRE] RLlib callbacks attached: <class '...CPDREAdaptiveCallback'>
[CPDRE-v3] Callback initialized in PID xxxxx
(RolloutWorker pid=xxxx) [CPDRE-v3] PID xxxx worker 1 writes to:
(RolloutWorker pid=xxxx)   CSV: .../A4_happo_seed43_w1.csv
(RolloutWorker pid=xxxx) [CPDRE-v3] PID xxxx wrote ep XXX (local #0) -> A4_happo_seed43_w1.csv
```

如果看不到 `[CPDRE-v3] wrote ep ...`，文件不会生成，需要排查（见末尾「故障排查」）。

### 步骤 3：检查训练产物

```bash
cd ~/code/New_Marllib/MARLlib/experiments/exp1

# 1. CSV：应当有 num_workers 个 per-worker 文件
ls -la results/episode_metrics/
# 示例输出：
# A4_happo_seed43_w1.csv
# A4_happo_seed43_w2.csv
# A4_happo_seed43_w3.csv

# 2. 验证 per-agent 列已生成（应当看到 power_0/power_1/power_2）
head -1 results/episode_metrics/A4_happo_seed43_w1.csv | tr ',' '\n' | grep power_ | head -10

# 3. NPZ：每个 episode 一个文件
ls results/episode_histories/ | wc -l
ls results/episode_histories/ | head -5
```

### 步骤 4：画图

#### 4.1 Per-agent 视角（每个电企的指标对比）

```bash
cd ~/code/New_Marllib/MARLlib

python experiments/exp1/plot_per_agent_metrics.py \
  --csv experiments/exp1/results/episode_metrics/A4_happo_seed43.csv \
  --smooth 5
```

> **注意**：`--csv` 传的是主文件名（**不带** `_w`），脚本会自动找 `_w1.csv`、`_w2.csv`、`_w3.csv` 并合并。

输出（默认在 `plots_per_agent_A4_happo_seed43/`）：

| 文件 | 内容 |
|------|------|
| `per_agent_dashboard.png` | 所有 per-agent 指标网格图 |
| `per_agent_shortage.png` | 每个电企的缺货量对比 |
| `per_agent_shipments.png` | 煤企发给每个电企的发货量 |
| `per_agent_demand.png` | 每个电企的需求量 |
| `per_agent_orders.png` | 每个电企的订单量 |
| `per_agent_inventory.png` | 每个电企的库存 |
| `per_agent_fill_rate.png` | 每个电企的订单满足率 |
| `per_agent_omega.png` | 每个电企的目标库存系数（动作）|

#### 4.2 系统级视角

```bash
python experiments/exp1/plot_episode_metrics.py \
  --csv experiments/exp1/results/episode_metrics/A4_happo_seed43.csv \
  --smooth 5
```

输出（默认在 `plots_A4_happo_seed43/`）：

| 文件 | 内容 |
|------|------|
| `dashboard.png` | 所有系统级指标网格图 |
| `group_performance.png` | 性能（shortage_rate、fairness_penalty）|
| `group_fairness.png` | 公平（jain）|
| `group_profit.png` | 利润（system/coal/power profit）|
| `group_reciprocity.png` | 互惠（g_u、g_c、mu_c、mu_u）|
| `group_operation.png` | 运营（demand、order、shipment）|
| `learning_curves.png` | 关键指标学习曲线（按 iteration 聚合 + std 阴影）|

#### 4.3 多 run 对比（不同算法或 seed）

```bash
python experiments/exp1/plot_episode_metrics.py \
  --csv experiments/exp1/results/episode_metrics/A2_ippo_seed43.csv \
        experiments/exp1/results/episode_metrics/A3_mappo_seed43.csv \
        experiments/exp1/results/episode_metrics/A4_happo_seed43.csv \
  --smooth 5
```

#### 4.4 画图参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | 必需 | CSV 路径（不带 `_w`，自动合并）|
| `--smooth N` | 5 | 滚动平均窗口大小 |
| `--metrics A B C` | 全部 | 只画指定的指标 |
| `--x_col COL` | `episode_index_global` | X 轴列（可选 `timesteps_total`、`training_iteration`）|
| `--output_dir DIR` | 自动 | 输出目录 |

---

## 四、跑多个实验

### 4.1 一次一个

```bash
# A2 + 3 个 seed
for seed in 42 43 44; do
  cd ~/code/New_Marllib/MARLlib/experiments/exp1
  rm -f results/episode_metrics/A2_ippo_seed${seed}*.csv
  cd ~/code/New_Marllib/MARLlib
  python experiments/exp1/exp1_train_single.py \
    --group_id A2 --seed $seed --timesteps 100000 \
    --episode_len 156 --core_arch gru \
    --share_policy individual --num_workers 3
done
```

### 4.2 多算法 × 多 seed

```bash
for algo in A2 A3 A4; do
  for seed in 42 43 44; do
    cd ~/code/New_Marllib/MARLlib/experiments/exp1
    rm -f results/episode_metrics/${algo}_*_seed${seed}*.csv
    cd ~/code/New_Marllib/MARLlib
    python experiments/exp1/exp1_train_single.py \
      --group_id $algo --seed $seed --timesteps 100000 \
      --episode_len 156 --core_arch gru \
      --share_policy individual --num_workers 3
  done
done
```

---

## 五、数据格式

### 5.1 CSV（每行 = 1 个 episode）

文件名格式：`<group>_<algo>_seed<n>_w<worker_idx>.csv`

#### 元数据列
- `worker_index`、`env_index`、`episode_id`、`episode_len`
- `local_ep_idx`（worker 进程内的 episode 序号）
- `pid`

#### 系统级指标（标量字段聚合）
- `system_profit_ep`、`coal_profit_ep`、`power_profit_total_ep`（episode 累计）
- `coal_profit_norm`、`power_profit_norm_total`（每步均值）
- `jain`、`fairness_penalty`、`mu_c`、`mu_u`、`g_u`、`g_c`
- `shortage_rate`、`shortage_norm`、`unsold`

#### Per-agent 指标（每个多维字段 → 4 列）
- `<field>_total` 或 `_total_ep`（跨 agent 求和）
- `<field>_power_0`（U1 互惠电企）
- `<field>_power_1`、`<field>_power_2`（普通电企）

字段来自 `env.history`，包括：`demand`、`orders`、`shipments`、`shortage`、`served`、`inventory`、`fill_rate`、`omega`、`weight` 等。

### 5.2 NPZ（每个文件 = 1 个 episode 的完整时间序列）

文件名格式：`<group>_<algo>_seed<n>_w<worker_idx>_ep<episode_id>.npz`

```python
import numpy as np

data = np.load("results/episode_histories/A4_happo_seed43_w1_ep000000123456.npz")
print(data.files)
# ['demand', 'orders', 'shipments', 'shortage', ...]

# Per-agent shape = (T, num_power) = (156, 3)
data['shortage'].shape  # (156, 3)
data['shortage'][:, 0]  # power_0 (U1) 各时间步缺货量

# Scalar shape = (T,)
data['mu_c'].shape  # (156,)
```

---

## 六、扩展：环境里加新指标

在 `custom_envs/coal_power_direct_reciprocity_env.py` 中：

### 6.1 加 scalar 字段

```python
# reset() 中
self.history = {
    ...
    "my_new_scalar": [],  # 新增
}

# step() 末尾的 _record_history() 调用中
self._record_history(
    ...,
    my_new_scalar=some_scalar_value,  # 新增
)
```

**自动结果**：CSV 多出 `my_new_scalar` 列，NPZ 多出对应 array，画图自动包含。

### 6.2 加 per-agent 字段

```python
self.history["my_per_agent_metric"] = []
self._record_history(..., my_per_agent_metric=numpy_array_of_shape_3)
```

**自动结果**：CSV 多出 4 列（`my_per_agent_metric_total` + 3 个 `_power_N`），NPZ 多出 `(T, 3)` array，画图自动识别为 per-agent 组。

---

## 七、故障排查

### 7.1 训练完没有 CSV 文件

检查训练日志是否有 `[CPDRE-v3] Cannot get env: ...`。如果有，说明 callback 拿不到环境实例——确认 `cpdre_callbacks_v3.py` 包含 `_resolve_env()` 静态方法（已修复 Ray 1.8 兼容性）。

### 7.2 CSV 没有 per-agent 列

确认训练日志开头有：
```
[CPDRE] run_cc.py loaded CPDREAdaptiveCallback (v3)
Callback class: CPDREAdaptiveCallback
```

如果显示的是其他名字（如 `CPDREMetricsCallback`），说明 import 跌回了旧版。直接测试：

```bash
cd ~/code/New_Marllib/MARLlib
python -c "
from experiments.exp1.cpdre_callbacks_v3 import CPDREMetricsCallbackV2
print(CPDREMetricsCallbackV2.__name__)
"
# 应该输出: CPDREAdaptiveCallback
```

### 7.3 画图报 `FileNotFoundError: ...A4_happo_seed43.csv`

确认画图脚本是最新版（`main()` 中已移除早期文件存在性检查）。`load_csv()` 会自动找 `_w*.csv` 合并。

### 7.4 PyCharm 跑出现 `unrecognized arguments: 2>&1 | tee`

PyCharm 用 wsl.exe 调用 Python，shell 重定向失效。改为在 WSL 终端直接跑，或者在 PyCharm Run 配置里不要写重定向。

### 7.5 新训练 CSV 行数异常多

旧 CSV 没删，新数据 append 进去了。每次新训练前**必须**清理：

```bash
rm -f results/episode_metrics/<group>_<algo>_seed<n>*.csv
rm -rf results/episode_histories
mkdir -p results/episode_histories
```

### 7.6 互惠指标 `g_u`、`g_c`、`mu_c`、`mu_u` 全为 0

`exp1_common.py` 中默认 `mechanism_mode="none"`，互惠机制未启用。要看互惠效果，改为 `"dynamic"`、`"long_contract"` 或 `"trigger"`。

---

## 八、最简版完整流程速查

```bash
# 0. 准备
cd ~/code/New_Marllib/MARLlib
conda activate marllib_torchtest
export PYTHONPATH=/home/asus/code/New_Marllib:/home/asus/code/New_Marllib/MARLlib:$PYTHONPATH

# 1. 清理
cd experiments/exp1
rm -f results/episode_metrics/A4_happo_seed43*.csv
rm -rf results/episode_histories && mkdir -p results/episode_histories
cd ../..

# 2. 训练
python experiments/exp1/exp1_train_single.py \
  --group_id A4 --seed 43 --timesteps 100000 \
  --episode_len 156 --core_arch gru \
  --share_policy individual --num_workers 3

# 3. 画 per-agent
python experiments/exp1/plot_per_agent_metrics.py \
  --csv experiments/exp1/results/episode_metrics/A4_happo_seed43.csv \
  --smooth 5

# 4. 画系统级
python experiments/exp1/plot_episode_metrics.py \
  --csv experiments/exp1/results/episode_metrics/A4_happo_seed43.csv \
  --smooth 5
```

---

**文档版本**：1.0（清理后版本）
**最后更新**：2026-05-26
