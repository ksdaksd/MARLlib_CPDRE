# CleanMARL 新手上手指南：VSCode + WSL + Claude Code

> 适用场景：在一台新电脑上从零开始搭建开发环境、运行训练、用 Claude Code 辅助开发。

---

## 一、环境准备

### 1. 安装 WSL（Windows Subsystem for Linux）

```powershell
# 在 Windows PowerShell（管理员）中运行：
wsl --install -d Ubuntu
```

安装后重启电脑，首次进入 Ubuntu 会提示创建用户名和密码。

### 2. 安装 Miniconda（Python 环境管理）

在 WSL Ubuntu 终端中：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示操作，安装到默认路径 ~/miniconda3
```

安装完后关闭终端重新打开，或执行 `source ~/.bashrc`。

### 3. 创建项目环境

```bash
# 创建 conda 环境（推荐 Python 3.10+）
conda create -n marllib python=3.10 -y
conda activate marllib

# 安装依赖
cd ~/你的项目目录/
pip install -r cleanmarl/requirements.txt
```

---

## 二、VSCode 远程连接 WSL

### 打开项目

| 操作 | 方法 |
|------|------|
| **从 WSL 终端打开** | 在 WSL 终端中进入项目目录，输入 `code .` |
| **从 VSCode 界面打开** | 按 `Ctrl+Shift+P` → 输入 `Remote-WSL: Open Folder in WSL` → 选择项目路径（如 `\\wsl.localhost\Ubuntu\home\你的用户名\你的项目`） |

首次使用会提示安装 **Remote - WSL** 扩展，点击安装即可。

### 打开终端

VSCode 内打开终端的方式：

| 操作 | 快捷键/方法 |
|------|------------|
| **新建终端** | `Ctrl + `` ` （反引号，ESC 下方） |
| **切换终端类型** | 终端面板右侧下拉框 → 选择 `bash` 或 `wsl` |
| **新建 WSL 终端** | `Ctrl+Shift+P` → `Terminal: Create New Terminal (WSL)` |

> 如果终端显示的是 Windows PowerShell，点击终端面板右侧的 `+` 旁边的 `˅`，选择 `WSL` 或 `bash`。

### 关键 VSCode 扩展

建议安装以下扩展（在 VSCode 左侧 Extensions 面板中搜索安装）：

| 扩展名 | 用途 |
|--------|------|
| **Remote - WSL** | 从 VSCode 连接 WSL 文件系统 |
| **Python** | Python 语法高亮、调试、环境选择 |
| **Claude Code** | 在 VSCode 内使用 Claude 辅助编程 |

---

## 三、安装和配置 Claude Code

### 安装

```bash
# 在 WSL 终端中安装 Claude Code CLI（新版）
npm install -g @anthropic-ai/claude-code

# 或者使用 npx 直接运行（无需全局安装）
npx @anthropic-ai/claude-code
```

安装后在 VSCode 终端中直接输入 `claude` 即可进入对话模式。

> **注意**：新版 Claude Code 使用方式有所变化，详见下方"VSCode 内使用"部分。

### VSCode 内使用

在 VSCode 中按 `Ctrl+Shift+P`，输入 `Claude Code: Open Chat` 即可在侧边栏打开对话面板。

也可以直接在终端中使用 `claude` 命令进入交互模式。

### Claude Code 项目配置文件

Claude Code 会读取项目根目录下的 `.claude/` 目录中的配置。可以创建以下文件来自定义 Claude Code 的行为：

- `.claude/CLAUDE.md` - 项目级别的 Claude Code 系统提示词
- `.claude/settings.json` - Claude Code 的设置（如权限、环境变量等）

### 第一次使用：让 Claude Code 了解你的项目

进入项目目录后，在终端中运行：

```bash
claude "请先浏览 cleanmarl/ 目录下的所有文件，了解这个框架的结构"
```

或者使用 `/init` 命令让 Claude Code 自动生成项目文档：

```bash
claude /init
```

---

## 四、Claude Code 常用提示词模板

### 运行训练

```
帮我启动 HAPPO 训练：
1. 先激活 conda 环境：source ~/miniconda3/etc/profile.d/conda.sh && conda activate marllib
2. 设置 PYTHONPATH：export PYTHONPATH=$(pwd):$PYTHONPATH
3. 运行 python cleanmarl/examples/train_happo_cpdre.py
训练过程保持在后台运行，完成后帮我分析日志。
```

### 运行 MAPPO 对比实验

```
用 MAPPO 跑一遍同样的训练做对比：
运行 python cleanmarl/examples/train_mappo_cpdre.py
等训练完成后，和上次 HAPPO 的结果对比 reward、vf_explained_var。
```

### 分析训练结果

```
训练跑完了，帮我分析 cleanmarl/logs/ 下面的 progress.csv：
1. 绘制 reward 和 vf_explained_var 曲线
2. 检查每个 agent 的价值网络学习情况
3. 和之前 MARLlib 的结果对比
```

### 修改超参数重新训练

```
把 clip_param 改成 0.3，actor_lr 改成 3e-4，重新跑一遍 HAPPO 训练
```

### 调试问题

```
训练报错了，帮我看看错误日志，分析原因并修复代码
```

### 代码审查/改进

```
帮我审查 cleanmarl/algos/happo.py 的代码，检查有没有潜在的 bug 或性能问题
```

### 扩展新环境

```
我想把另一个环境 MyEnv 适配到 CleanMARL，参考 cleanmarl/envs/cpdre_wrapper.py 的模式，帮我写一个 myenv_wrapper.py
```

---

## 五、完整启动流程速查

在新电脑上从零开始，按顺序执行：

```bash
# === 1. WSL 终端中 ===
conda create -n marllib python=3.10 -y
conda activate marllib
cd ~/你的项目目录
pip install -r cleanmarl/requirements.txt

# === 2. VSCode ===
# 在 WSL 终端中输入：
code .

# === 3. VSCode 终端中（Ctrl+`） ===
# 确认在 WSL bash 终端中（不是 PowerShell）
export PYTHONPATH=$(pwd):$PYTHONPATH

# 冒烟测试（验证环境正确）
python cleanmarl/smoke_test.py

# 启动训练
python cleanmarl/examples/train_happo_cpdre.py

# === 4. Claude Code 对话 ===
# Ctrl+Shift+P → Claude Code: Chat
# 或在终端中：claude "帮我启动HAPPO训练并监控进度"
```

---

## 六、常见问题

### Q: VSCode 终端显示 "python: command not found"

**A:** 检查是否激活了 conda 环境。在终端中运行 `conda activate marllib`。如果终端默认是 PowerShell 而非 bash，点击终端面板的 `+˅` → `Select Default Profile` → 选择 `WSL (Ubuntu)`。

### Q: 提示 `ModuleNotFoundError: No module named 'cleanmarl'`

**A:** 需要设置 PYTHONPATH。运行：
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```
或者在 VSCode 的 `.vscode/settings.json` 中添加：
```json
{
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
    }
}
```

### Q: 提示 `ModuleNotFoundError: No module named 'custom_envs'`

**A:** 确保 `custom_envs/` 文件夹和 `cleanmarl/` 在同一个项目根目录下，且 PYTHONPATH 指向项目根目录。

### Q: Claude Code 连不上

**A:** 确保在 WSL 终端中正确安装了 claude-code：
```bash
npm install -g @anthropic-ai/claude-code
```
如果 npm 命令不可用，先安装 Node.js：
```bash
# 安装 Node.js（推荐使用 nvm）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install --lts
```
然后重新安装 Claude Code。确保网络可以访问 Anthropic API。

### Q: GPU 不可用

**A:** 在训练脚本中将 `"device": "cuda"` 改为 `"device": "cpu"`。训练会慢很多但能跑。

---

## 七、推荐的 VSCode 配置文件

建议在项目根目录创建 `.vscode/` 目录，包含以下配置文件：

### settings.json（项目设置）

```json
{
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
    },
    "python.defaultInterpreterPath": "~/miniconda3/envs/marllib/bin/python",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### launch.json（调试配置）

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Train HAPPO",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/cleanmarl/examples/train_happo_cpdre.py",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH": "${workspaceFolder}"},
            "justMyCode": false
        },
        {
            "name": "Python: Smoke Test",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/cleanmarl/smoke_test.py",
            "console": "integratedTerminal",
            "env": {"PYTHONPATH": "${workspaceFolder}"},
            "justMyCode": false
        }
    ]
}
```

这样可以直接在 VSCode 中按 `F5` 启动调试，或在 Run and Debug 面板中选择配置运行。
