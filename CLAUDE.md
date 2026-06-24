# CLAUDE.md

本仓库是煤电供应链**直接互惠**(CPDRE)的多智能体强化学习(MARL)研究代码。
建模文档 `model/0611(1).md` 是**权威且不可改**的规格;一切代码以它为准对齐。

**核心研究问题**:关系专属动态直接互惠(B4)能否优于非关系型动态协调(B5)?
即区分"一般状态感知协调 / 关系专属互惠 / 单纯配给偏置"三类机制来源。

只用项目根下的自研框架 **`cleanmarl/`**(PyTorch 原生:采样 + GAE + PPO/HAPPO 更新 +
GRU + 集中 critic)。当前主线:1 个煤企 + 3 个电企 + 1 个互惠电企(U1)的 B4 实验。

> **本文件是跨机器/跨会话的唯一可靠交接渠道。** Claude Code 的"本机记忆"
> (`~/.claude/...`)和对话历史**不随项目走**;只有提交进 git 的东西(本文件、`model/`
> 规格、代码、注释)才会跟到新机器。重要信息一律写进这里并提交。

---

## ⚙️ 运行环境(★换机器必须按本机改★)

代码在 **WSL Ubuntu** 文件系统里(本机 = `/home/asus/code/New_Marllib/MARLlib`,
对应 Windows 侧 `\\wsl.localhost\Ubuntu\...`,同一份文件)。

**训练/分析必须用装了 torch 的 conda 环境**:
- 本机该环境叫 **`marllib_torchtest`**(torch 1.9.0+cu111 + CUDA,scipy 1.10.1,pandas 2.0.3)。
- 另一个环境 `marllib` **只有 numpy、没有 torch**,跑不了训练。
- **换电脑后**:conda 环境名 / Python 绝对路径 / 是否有 WSL / 是否有 GPU 都可能不同。
  先确认 torch 装在哪个环境(`conda env list` + `python -c "import torch"`),把下面命令里的
  解释器路径替换成你本机的。有 GPU 用 `device: cuda`,没有就 `--device cpu`。

本机命令模板(从 Windows 调 WSL;若已在 WSL 终端内则去掉 `wsl.exe -e bash -lc` 外壳):
```bash
wsl.exe -e bash -lc 'cd /home/asus/code/New_Marllib/MARLlib && \
  /home/asus/miniconda3/envs/marllib_torchtest/bin/python cleanmarl/run_sweep.py ...'
```
不想写长路径就先 `conda activate marllib_torchtest` 再用 `python`。

---

## 实验流程(实验一 + 实验二)

目前只完成了实验一与实验二的代码，也可以自行检查代码的完善程度

入口:`cleanmarl/run_sweep.py`(跑实验)+ `cleanmarl/analyze_experiments.py`(出表)。
一条命令跑完一个实验的所有组别 × 所有种子,结果落在 `logs/cleanmarl/{组}_{算法}_seed{种子}/`
(每个 run 一个文件夹,含 `config.json / progress.csv / episode_summary.csv / run_meta.json`)。

### 实验一:基准策略与算法选择(选主算法)
```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp1 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp1
```
- A1/A2 规则基线(固定 / 状态感知 base-stock),A3=IPPO、A4=MAPPO、A5=HAPPO、
  A6=HAPPO+低需求扰动,均**无互惠**。
- 看 `system_profit` 高且跨种子 `reward_std` 小者 → 定主算法(通常 HAPPO)。
- 退出标准:学习型要稳定优于规则基线;否则先调环境/算法,别进实验二。

### 实验二:直接互惠机制主实验(核心结论)
```bash
python cleanmarl/run_sweep.py --config configs/happo_cpdre.yaml --experiment exp2 --seeds 40,41,42,43,44
python cleanmarl/analyze_experiments.py --experiment exp2
```
- B1–B5。**B4 vs B5 是识别"关系专属互惠价值"的关键对比**。
- 分析脚本自动出组间表 + 按 (seed, episode) 配对的 **Wilcoxon + bootstrap CI**(B4 vs B2/B5/B3)。
- 判读(spec 4.5.4):
  - B4>B2 但 B4≈B5 → 改善来自一般动态协调,**不是**关系专属互惠。
  - B4>B5>B2 **且** 普通电企 SR_N、公平 J 未明显恶化 → 关系专属互惠有额外价值。
  - 若 B4 只改善 SR_1 却显著抬高 SR_N → **不能**算系统协调改善(表已并列 SR_1/SR_N 供审计)。

### 实验三:机制来源消融(可选)
```bash
python cleanmarl/run_sweep.py --experiment exp3 --seeds 40,41,42,43,44   # C1-C6
python cleanmarl/analyze_experiments.py --experiment exp3
```

### 常用参数
- `--group B4` 只跑单组;`--seeds 42` 单种子;`--timesteps 1560` 覆盖步数(冒烟,1 rollout)。
- analyze:`--groups B2,B4,B5` 覆盖默认组别;`--pairs B4:B5` 自定义配对;
  `--metric system_profit,SR` 选检验指标;`--csv out.csv` 导出汇总表。
- **冒烟测试(1 rollout)只验管线,数值无意义**;正式结论必须跑满 config 的
  `total_timesteps`(1e5)× 5 种子。

---

## 固定评估轨迹(为什么能做配对检验)

**训练随机性与评估随机性故意分开**:
- 训练种子(`--seeds 40..44`)→ `cfg['env']['seed']`,决定训练时的随机流。5 个种子 = 方法重复 5 次。
- 评估走 `env.eval_reset(episode_idx)`,内部固定 `rng = default_rng(9000 + episode_idx)`,
  **与训练种子无关**。所以不论哪个训练种子、哪个组别,评估都在**同一批 64 条固定轨迹**
  (episode 0..63 ↔ 种子 9000..9063)上跑,整条 156 周需求/价格/供给冲击被钉死。

→ `B4-seed40-ep5` 和 `B5-seed40-ep5` 面对**完全相同的需求实现**,差异只来自机制本身。
这正是 `analyze_experiments.py` 按 (seed, episode) 配对做 Wilcoxon 的前提,能消掉"剧本难易"
这一最大噪声源,大幅提升统计功效。

---

## 组别 → 环境机制映射(`run_sweep.GROUP_ENV_MAP`)

| 组 | mechanism | allocation | 算法/策略 | 备注 |
|---|---|---|---|---|
| A1/A2 | none | fair | rule_a1/rule_a2 | 规则基线,θ 用固定季节排程 |
| A3/A4/A5 | **b2** | fair | ippo/mappo/happo | 学习无互惠,煤企**学 θ**(产能路径可学) |
| A6 | **b2** | fair | happo + low_noise | 低扰动稳定性 |
| B1 | none | fair | rule_b1 | 公平规则基准 |
| B2 | b2 | weighted | (YAML 算法) | 无互惠学习基准,λ=1,无记忆 |
| B3 | none | weighted | rule_b3 | 规则互惠(用 χ̂ 判宽松/紧张) |
| B4 | b4 | **weighted** | (YAML 算法) | 完整关系专属互惠,煤企学 θ 和 λ |
| B5 | b5 | weighted | (YAML 算法) | 非关系动态协调,学 θ,λ=1,无 U1 身份/记忆 |
| C1–C6 | b4 变体 | weighted | (YAML 算法) | C2=disable_g_u, C3=fixed_lambda, C4=freeze_memory, C5=freeze_theta, C6=去关系观测 |

`--group A4` 等会**强制覆盖算法**(force_algo),所以 `--config` 用哪个 yaml 都行(只取它的环境+超参)。

---

## 关键环境语义(容易踩的坑)

环境:`custom_envs/coal_power_direct_reciprocity_env.py`。煤企动作 2D `(θ,λ)`,电企 1D `(ω)`。

1. **`none` 模式煤企不学 θ** —— θ 钉死成季节排程。只有 `b2/b4/b5/dynamic` 让煤企学 θ。
   所以"无互惠学习"组要用 **b2 不是 none**。
2. **b4 的 λ 必须配 `allocation_mode: weighted`** —— 用 fair 会被 `_force_fair_allocation()`
   把权重重置为 1,保供闭环断掉。none/b2/b5 强制 fair(λ=1)。
3. **电企利润用真实需求**:`r·D − p·Y − c_rep·S − h·I`(外部应急补货满足当期发电),不是 `r·served`。
4. **互惠效用奖励 ημg 不在主模型**(`use_reciprocity_reward=False`);主模型奖励
   = 归一化利润 − λ_S·缺煤 − λ_J·(1−Jain)。
5. **SR 派生**:`service_rate=(D−S)/D` 是满足率(越大越好);分析里 `SR_i = 1 − own_service_rate_ui`,
   系统 `SR = mean_shortage_rate`。
6. **异质动作**:`act_dims=[2,1,1,1]`,buffer pad 到 `max_act_dim=2`;log_prob 按动作维求和成标量,
   **不破坏 HAPPO** 顺序更新。

---

## 决策背景(为什么是 cleanmarl;一个要注意的现象)

- **为什么自研 cleanmarl**:旧的 MARLlib/RLlib 里 **HAPPO 价值网络学不动**
  (`vf_explained_var≈0`,各 agent 价值函数完全不学)。试过改 optimizer 绑定/手动更新 critic 等
  多种修法都失败(崩溃或报 grad 错)。根因是 HAPPO 的逐 agent 异构更新和 RLlib 优化流程冲突。
  于是绕开 RLlib,用 PyTorch 原生重写 `cleanmarl/`,HAPPO 价值学习恢复正常。关键设计:
  critic 独立优化器 + 每 agent 独立价值输出、按 episode 重组数据(RNN 隐藏态 episode 内连续/
  间重置)、loss 直接 backward、HAPPO 顺序更新(更新后重评 ratio 迭代后续 advantage)。
  **结论:用 CPDRE 跑 HAPPO 一律用 cleanmarl,别用 MARLlib 的 HAPPO。**
- **要注意的现象:`vf_explained_var ≈ 0` 不一定是 bug。** 对齐 spec 后奖励是"绝对利润量",
  被一大坨**几乎恒定的基础利润**主导(reward ~328±8,相对波动才 ~2.5%)。于是 `value_loss`
  能降(学会了那个大常数量级),但 `vf_explained_var` 上不去(可被状态解释的方差极小,剩下大多是
  不可预测的需求噪声)。后果:advantage 信号弱、训练曲线偏平。**判断机制差异要看组间
  `shortage_rate / system_profit / SR_1-SR_N` 的差(跑满 5 种子后),不是单组训练曲线。**
  若以后想让 RL 学得更动,杠杆是奖励/价值归一化(running mean-std 标准化回报,或 PopArt,
  或把利润项中心化),把"大常数地板"减掉。

---

## 文件地图

- `model/0611(1).md` —— 权威规格(**勿改**)。`model/cleanmarl_0611_issues.md` 代码差距清单;
  `model/0611_remaining_issues.md` 文档自身待完善点。
- `cleanmarl/run_sweep.py` —— 多种子/多组别扫描 + `--experiment` 预设 + `GROUP_ENV_MAP` + `run_meta.json`。
- `cleanmarl/analyze_experiments.py` —— 出实验一/二/三对比表 + 配对 Wilcoxon/bootstrap。
- `cleanmarl/train.py` —— 单次训练入口(`build_policy` 路由 happo/mappo/ippo + rule_*)。
- `cleanmarl/algos/{happo,mappo,ippo}.py` —— 算法;`rule_policy.py` 规则/固定基线(无梯度)。
- `cleanmarl/models/models.py` —— GRUActor + CentralizedCritic(支持 `rnn_layers`)。
- `cleanmarl/core/{trainer,episode_logger,logger}.py` —— 训练基类 + 每 episode 指标(`episode_summary.csv`,
  MEAN/SUM/STD_FIELDS)+ 训练进度(`progress.csv`)。
- `cleanmarl/configs/happo_cpdre.yaml` —— 主配置(对齐 spec 表4.3:γ0.95, lr1e-4, rnn_layers2,
  chunk_len10, eval_episodes64)。mappo/ippo 同构。`default.yaml` 是 train.py 的默认。
- `custom_envs/coal_power_direct_reciprocity_env.py` —— CPDRE 环境本体。
- 产物:`logs/cleanmarl/<run>/`、`checkpoints/`。

---

## 已知尚未完成(按需推进)

- **#4** 连续动作未 squash(采样可能越界,clip 后执行但用 clip 前动作算 log_prob)。
- **#6** HAPPO 顺序更新只修相邻 agent,非标准 Kuba 2022(已声明不依赖单调性)。
- **#9/#10** h>1 完全向量化、m>3 obs 不截断 —— 扩展实验(E 系列)才需要。
- **中途评估从不触发**:trainer.py 用 `current_step % eval_freq == 0`,而 step 永远是
  rollout_length 的整数倍,跟 eval_freq 对不上 → 只有训练结束时的 final eval 写 `phase=eval`。
  后果:中途 Ctrl+C 的 run 一行 eval 都没有(白跑)。要改成"距上次 eval ≥ eval_freq 就评估"
  + 中断时也补跑 final eval。
- 文档侧:训练并行环境数/rollout 步数未交代、煤价固定、λ_S/λ_J 取值来源等
  (见 `model/0611_remaining_issues.md`)。

---

## 验证现状

实验一(A1–A6)、实验二(B1–B5)均已端到端跑通(冒烟:1 种子 1 rollout),表格 + 配对检验
正确输出。B4 冒烟即见机制激活(λ1≈1.48, g_u/g_c>0, μ≈0.52),B2/B5 干净(λ=1, g=0),且分析
正确暴露 B4 的 SR_1≪SR_N(spec 4.5.4 要审计的模式)。**正式结论需跑满 1e5 步 × 5 种子。**
