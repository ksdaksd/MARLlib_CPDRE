# human-write

中文优先、可交付导向的写作 skill，用于把中文文档写得更自然、更具体、更少 AI 腔，并且更容易直接交付或转为 Word/PPT。

A delivery-oriented writing skill for Chinese formal documents. It helps produce natural, specific, less AI-sounding output that is easier to deliver directly or convert to Word/PPT.

## 适用场景 / When to Use

- 报告、方案、政策咨询、会议纪要、研究计划、申报材料
- 需要“去 AI 味”、结构清晰、可核查来源、可直接交付的正式文本
- Formal Chinese materials that require clear structure, traceability, and human-like tone

## 核心能力 / What It Enforces

- 先判断文档类型，再套通用规则和类型专项规则
- 约束空话、套话、模板腔，优先具体事实与可执行表达
- 交付前执行结构、语言、排版一致性检查

## 安装 / Installation

### 方式 A：本地目录安装（推荐）

```bash
cp -R ./human-write "${CODEX_HOME:-$HOME/.codex}/skills/"
```

### 方式 B：从 GitHub 克隆

```bash
git clone https://github.com/OscarLishe/human-write.git
cp -R ./human-write "${CODEX_HOME:-$HOME/.codex}/skills/"
```

### 验证是否生效

1. 新开一个 Codex 会话。
2. 输入 `$human-write`。
3. 若能被识别并按该 skill 输出，说明安装成功。

## 跨 AI 通用安装与启用 / Cross-AI Setup

并非所有 AI 都支持 Codex 的 `SKILL.md` 机制。通用做法是：把规则当作“长期系统指令 + 会话激活提示词”。

### 通用三步

1. 准备规则源：以 `SKILL.md` + `references/writing-standard-zh.md` 为主。
2. 放入平台：
   - Codex: 放到 `~/.codex/skills/human-write`。
   - ChatGPT: 放到 Custom Instructions / GPT Instructions。
   - Claude: 放到 Project Instructions。
   - Gemini: 放到 Gem Instructions。
3. 会话启用：在新对话第一条消息显式要求启用 human-write 规则。

### 会话激活模板（可直接复制）

```text
请作为 human-write 写作助手工作，并严格执行以下要求：
1) 使用正式、自然、直接的中文，避免 AI 腔和空泛套话；
2) 先判断文档类型，再按对应结构写作；
3) 结论必须有依据，建议必须可执行；
4) 保留可核查信息，不编造来源；
5) 输出结构可直接转为 Word/PPT。
如果我的指令与上述规则冲突，请先指出冲突并给出可执行改写方案。
```

### 平台差异说明

- Codex: 支持 `$human-write` 直接触发。
- 其他平台: 通常不支持“安装 skill”，但支持“保存长期指令 + 每次会话激活”。

## 快速使用 / Quick Start

```text
$human-write
请把下面这份研究方案改写成正式中文版本，要求：
1) 去掉明显 AI 腔
2) 保留关键事实和证据链
3) 输出可直接转 Word 的结构
```

## 目录结构 / Structure

- `SKILL.md`: 触发与执行流程定义 / Triggering and workflow
- `references/writing-standard-zh.md`: 中文写作与排版细则 / Full Chinese standards
- `agents/openai.yaml`: 展示元数据 / UI metadata

## 说明 / Notes

- 本 skill 主要面向中文写作，因此规则重点在中文表达与正式文档风格。
- This skill is optimized for Chinese formal writing tasks.
