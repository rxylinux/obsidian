---
title: Agent 全面解析
date: 2026-03-01
tags:
  - Agent
  - 智能体
  - LLM应用
  - 多Agent协作
cssclass: main-page
status: active
---

# Agent 全面解析

> [!info] 核心概念
> **Agent（智能体）** 是一种能够自主感知环境、推理规划、调用工具并执行行动的 AI 系统。Agent = LLM + 规划能力 + 工具调用 + 记忆 + 反思

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **学习路线**：[[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线
> - **框架详解**：
>   - [[AI研究/AI学习/03-实战应用/LangChain全面解析]] - LangChain 框架教程
>   - [[AI研究/AI学习/03-实战应用/LangGraph全面解析]] - LangGraph 状态机教程
>   - [[AI研究/AI学习/03-实战应用/LangChain vs LangGraph 对比分析]] - 框架选择对比
> - **相关笔记**：[[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 检索增强生成

---

## 📑 目录

### 核心概念
- [[#一、什么是 Agent]]
- [[#二、Agent vs 传统 LLM vs RAG]]
- [[#三、Agent 的发展历程]]
- [[#四、Agent 解决的问题]]

### 架构原理
- [[#五、Agent 完整架构]]
- [[#六、四大核心组件]] ← 重点
  - [[#组件一：Planner（规划器）]]
  - [[#组件二：Tool Use（工具使用）]]
  - [[#组件三：Memory（记忆系统）]]
  - [[#组件四：Reflection（反思机制）]]

### 推理模式
- [[#七、ReAct 模式]]
- [[#八、CoT（Chain of Thought）]]
- [[#九、Plan-and-Execute]]
- [[#十、其他推理模式]]

### 框架选型
- [[#十一、Agent 框架对比]]
- [[#十二、LangChain Agent]]
- [[#十三、LangGraph]]
- [[#十四、CrewAI & AutoGen]]

### Agent 类型
- [[#十五、单 Agent vs 多 Agent]]
- [[#十六、Multi-Agent 协作模式]]
- [[#十七、Autonomous Agent]]

### 实战应用
- [[#十八、应用场景]]
- [[#十九、最佳实践]]
- [[#二十、评估与调试]]

---

## 一、什么是 Agent

### 1.1 核心定义

**Agent（智能体）** = 能够自主感知、推理、行动的 AI 系统

```
传统 LLM:
用户 → [LLM] → 回答
被动回答，一次交互

Agent:
用户 → [Agent] → 感知 → 规划 → 行动 → 反思 → 结果
主动决策，多步交互
```

### 1.2 本质理解

```
Agent = 雇了一个智能助手

类比人类助手：
✅ 理解目标：知道你要做什么
✅ 制定计划：分解任务为步骤
✅ 使用工具：搜索、计算、写代码等
✅ 记住信息：不用重复交代背景
✅ 自我反思：发现错误并改进
✅ 持续行动：直到任务完成

核心公式：
Agent = LLM（大脑）+ 工具（手脚）+ 记忆（经验）+ 规划（策略）
```

### 1.3 最小示例

```python
# 最简单的 Agent 示例
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool

# 1. 定义工具
def search_tool(query: str) -> str:
    return f"搜索结果：{query}"

tools = [
    Tool(name="搜索", func=search_tool, description="搜索互联网信息")
]

# 2. 创建 Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# 3. 执行
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "搜索最新的AI进展"})

# Agent 会自动：
# 1. 理解需要搜索
# 2. 调用搜索工具
# 3. 整理结果回答
```

---

## 二、Agent vs 传统 LLM vs RAG

### 2.1 对比表格

| 维度 | 传统 LLM | RAG | Agent |
|:-----|:--------|:-----|:------|
| **核心能力** | 理解和生成 | 检索 + 生成 | 规划 + 执行 |
| **主动性** | ❌ 被动 | ❌ 被动 | ✅ 主动 |
| **工具使用** | ❌ | ❌ | ✅ |
| **多步推理** | ⚠️ 有限 | ⚠️ 有限 | ✅ |
| **记忆能力** | ❌ | ⚠️ 通过检索 | ✅ 原生支持 |
| **任务执行** | ❌ | ❌ | ✅ |
| **复杂问题** | ❌ | ⚠️ | ✅✅✅ |

### 2.2 能力对比图

```
┌─────────────────────────────────────────────────────────┐
│                    能力对比                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  传统 LLM:                                              │
│  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     │
│  理解生成                                               │
│                                                         │
│  RAG:                                                  │
│  ████████████████████████████████████░░░░░░░░░░░░░░░   │
│  理解生成 + 知识检索                                    │
│                                                         │
│  Agent:                                                │
│  ████████████████████████████████████████████████████  │
│  理解生成 + 规划 + 执行 + 记忆 + 反思                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.3 选择决策

```
你的需求是？

├─ 简单问答、内容生成
│   └─ 传统 LLM ✅
│
├─ 知识库问答、文档分析
│   └─ RAG ✅
│
├─ 需要调用API、操作数据库
│   └─ Agent（工具调用）✅
│
├─ 多步骤任务（如写报告）
│   └─ Agent（规划能力）✅
│
├─ 需要记忆对话历史
│   └─ Agent（记忆系统）✅
│
└─ 复杂自主任务（如研究助手）
    └─ Multi-Agent ✅
```

---

## 三、Agent 的发展历程

```
Agent 进化史：

┌─────────────────────────────────────────────────────────┐
│  2022 年前                                              │
│  • 符号主义 Agent（规则系统）                            │
│  • 强化学习 Agent（游戏 AI）                             │
│  • 缺乏语言理解和推理能力                                │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2022 年末 - 2023 初                                    │
│  • ReAct 论文发布                                       │
│  • "Reasoning + Acting" 范式                            │
│  • LangChain Agent 框架兴起                              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2023 年中                                               │
│  • AutoGPT、BabyAGI 爆发                                 │
│  • 自主 Agent 概念普及                                   │
│  • 工具调用成为 LLM 标配                                  │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2023 年末 - 2024 初                                    │
│  • Multi-Agent 系统兴起                                  │
│  • CrewAI、AutoGen 框架                                  │
│  • Agent 协作模式成熟                                    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2024 年中                                               │
│  • LangGraph 发布（状态机 Agent）                        │
│  • 可控性成为重点                                        │
│  • 生产级应用落地                                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2024 年末 - 2025                                       │
│  • OpenAI o1（原生推理）                                 │
│  • Claude Computer Use（操作电脑）                       │
│  • 推理与 Agent 融合                                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2025 - 2026  成熟与融合                                │
│  • Agent + RAG 2.0 深度融合                              │
│  • Multi-Agent 编排成熟                                  │
│  • 企业级 Agent 平台                                     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2025.9  Claude Agent SDK 发布                          │
│  • Python + TypeScript 双版本                           │
│  • "给Claude一台电脑"理念                                │
│  • 原生 MCP 集成                                         │
│  • 自动上下文压缩                                        │
│  • 支持超30小时自主编码任务                               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2025末 - 2026初  MCP 协议成为标准                     │
│  • "AI 应用的 USB-C 接口"                                │
│  • 微软、谷歌、OpenAI 纷纷支持                            │
│  • MCP 注册表服务器接近 2000 个                          │
│  • 统一 Agent 工具连接标准                                │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2026.2  Xcode 原生集成 Claude Agent                   │
│  • 苹果与 Anthropic 合作                                 │
│  • Agentic Coding（智能体编程）                          │
│  • 视觉闭环：Agent "看得见"界面                           │
│  • 异步长时任务支持                                       │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  2026  四大技术趋势                                      │
│  • MCP - 统一连接层                                      │
│  • GraphRAG - 知识响应                                   │
│  • AgentDevOps - 可控可靠                                │
│  • RaaS - 结果即服务                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 四、Agent 解决的问题

### 4.1 LLM 的局限性

```
LLM 的三大限制：

1. 只能对话，不能行动
   → 无法调用 API、操作数据库、执行代码

2. 只能单步，不能规划
   → 复杂任务需要多次交互才能完成

3. 只能短期，不能记忆
   → 无法记住之前的对话和经验
```

### 4.2 Agent 的解决方案

| 问题 | Agent 解决方案 |
|:-----|:-------------|
| **无法行动** | ✅ 工具调用（API、数据库、代码等） |
| **无法规划** | ✅ 任务分解、逐步执行 |
| **无法记忆** | ✅ 短期记忆（上下文）+ 长期记忆（向量库） |
| **容易出错** | ✅ 反思机制、自我修正 |

---

## 五、Agent 完整架构

### 5.1 架构全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent 完整架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                               │
│  │   用户目标   │                                               │
│  │ "写一份报告" │                                               │
│  └──────┬──────┘                                               │
│         │                                                      │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    核心循环                              │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │   │
│  │  │  Observer   │───▶│   Planner    │───▶│ Executor  │  │   │
│  │  │  (观察器)    │    │  (规划器)     │    │ (执行器)   │  │   │
│  │  │             │    │              │    │           │  │   │
│  │  │ 理解目标     │    │ 分解任务      │    │ 调用工具   │  │   │
│  │  │ 分析状态     │    │ 制定计划      │    │ 执行动作   │  │   │
│  │  └─────────────┘    └──────────────┘    └─────┬─────┘  │   │
│  │                                              │         │   │
│  │                                              ▼         │   │
│  │                                    ┌──────────────┐    │   │
│  │                                    │   Tools      │    │   │
│  │                                    │  ├─ 搜索     │    │   │
│  │                                    │  ├─ 代码     │    │   │
│  │                                    │  ├─ API      │    │   │
│  │                                    │  └─ 数据库   │    │   │
│  │                                    └──────────────┘    │   │
│  │                                              │         │   │
│  │                                              ▼         │   │
│  │                                    ┌──────────────┐    │   │
│  │                                    │   Memory     │    │   │
│  │                                    │  ├─ 短期记忆  │    │   │
│  │                                    │  └─ 长期记忆  │    │   │
│  │                                    └──────────────┘    │   │
│  │                                              │         │   │
│  │                                              ▼         │   │
│  │                                    ┌──────────────┐    │   │
│  │                                    │  Reflector   │    │   │
│  │                                    │  (反思器)     │    │   │
│  │                                    │  ├─ 检查结果  │    │   │
│  │                                    │  ├─ 发现错误  │    │   │
│  │                                    │  └─ 调整计划  │    │   │
│  │                                    └──────┬───────┘    │   │
│  │                                           │            │   │
│  │                                           │ 完成了？    │   │
│  │                         ┌─────────────────┴────────┐   │   │
│  │                         │                          │   │   │
│  │                         ▼ Yes                     ▼ No  │   │
│  │                    ┌──────────┐              ┌─────────┐│   │
│  │                    │ 最终结果  │              │ 继续循环 ││   │
│  │                    └──────────┘              └─────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 简化版（最小 Agent）

```
最小 Agent 流程：

目标 → [思考] → [行动] → [观察] → [完成了吗？] → 否：重复 / 是：结束

这就是 ReAct 范式的核心循环
```

---

## 六、四大核心组件

> [!info] 组件概览
> Agent 的四大核心组件是：
> 1. **Planner（规划器）** - 如何分解任务、制定计划
> 2. **Tool Use（工具使用）** - 如何调用外部工具和 API
> 3. **Memory（记忆系统）** - 如何存储和检索信息
> 4. **Reflection（反思机制）** - 如何自我评估和改进

### 四大组件关系图

```
┌─────────────────────────────────────────────────────────┐
│                 四大组件协作关系                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    ┌──────────────┐                     │
│                    │    Planner   │                     │
│                    │   (规划器)    │                     │
│                    │              │                     │
│                    │  分解任务     │                     │
│                    │  制定计划     │                     │
│                    └──────┬───────┘                     │
│                           │                             │
│                           ▼                             │
│                    ┌──────────────┐                     │
│                    │   Tool Use   │                     │
│                    │  (工具使用)   │                     │
│                    │              │                     │
│                    │  调用工具     │                     │
│                    │  执行动作     │                     │
│                    └──────┬───────┘                     │
│                           │                             │
│         ┌─────────────────┼─────────────────┐           │
│         ▼                 ▼                 ▼           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Memory    │  │ Reflection  │  │    Result   │     │
│  │  (记忆系统)  │  │  (反思机制)  │  │   (结果)     │     │
│  │             │  │             │  │             │     │
│  │ 存储信息     │  │ 评估结果     │  │ 任务完成     │     │
│  │ 检索历史     │  │ 发现错误     │  │ 返回用户     │     │
│  │             │  │ 改进计划     │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│       │                 │                             │
│       └─────────────────┴─────────────────┐             │
│                         │                 │             │
│                         ▼                 ▼             │
│                  反馈给 Planner   继续下一步              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 组件一：Planner（规划器）

> [!quote] 核心思想
> **"Given a goal, think about what to do."** - 给定目标，思考该做什么

### 1.1 规划器的作用

```
Planner = Agent 的大脑前额叶

功能：
✅ 理解用户目标
✅ 分解复杂任务为子任务
✅ 制定执行计划
✅ 根据反馈调整计划

核心问题：
"我该按什么步骤来完成这个目标？"
```

### 1.2 规划策略对比

| 策略 | 原理 | 优点 | 缺点 | 适用场景 |
|:-----|:-----|:-----|:-----|:---------|
| **ReAct** | 推理→行动→观察 | 简单、灵活 | 可能陷入循环 | 通用场景 |
| **CoT** | 逐步思考推理 | 逻辑清晰 | Token 消耗大 | 复杂推理 |
| **Plan-and-Solve** | 先规划再执行 | 可控性强 | 不够灵活 | 结构化任务 |
| **ReWOO** | 先规划再执行无推理 | 效率高 | 缺少中间推理 | 明确任务 |
| **Tree of Thoughts** | 多分支探索 | 创造性强 | 成本高 | 创意任务 |

### 1.2.1 三种核心规划策略（互斥选择）

> [!warning] 重要理解
> 以下三种策略是**互斥的**，同一个 Agent 在同一时间只能选择其中一种，不能混用。

#### 策略对比总览

```
┌─────────────────────────────────────────────────────────┐
│           三种核心规划策略（三选一）                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  策略 1：即时规划 (ReAct)                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │  无计划，边走边想                                   │   │
│  │                                                   │   │
│  │  每一步：                                          │   │
│  │    1. 思考现在该做什么                             │   │
│  │    2. 做这个动作                                   │   │
│  │    3. 观察结果                                     │   │
│  │    4. 回到步骤 1                                   │   │
│  │                                                   │   │
│  │  特点：没有预设计划，完全动态决策                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  策略 2：事前规划 (Plan-and-Execute)                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  先计划，后执行，不调整                             │   │
│  │                                                   │   │
│  │  阶段 1：规划                                      │   │
│  │    → 制定完整计划                                  │   │
│  │                                                   │   │
│  │  阶段 2：执行                                      │   │
│  │    → 按计划逐步执行                                │   │
│  │    → 不改变计划                                    │   │
│  │                                                   │   │
│  │  特点：有计划但僵化，执行时不调整                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  策略 3：动态规划 (Adaptive)                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  有计划，执行时动态调整                             │   │
│  │                                                   │   │
│  │  初始：制定计划                                     │   │
│  │  执行：每步评估                                    │   │
│  │  调整：发现问题就修改计划                           │   │
│  │                                                   │   │
│  │  特点：有计划且灵活，结合两者优势                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 关键区别对比

| 维度 | ReAct | Plan-and-Execute | Adaptive |
|:-----|:------|:-----------------|:---------|
| **是否有计划** | ❌ 无 | ✅ 有 | ✅ 有 |
| **是否会调整** | ➖ 每步重新决定 | ❌ 不调整 | ✅ 会调整 |
| **LLM 调用次数** | N 次（每步一次） | 1 次（规划）+ N 次（可选）| 1 次（规划）+ K 次（调整） |
| **适用场景** | 探索性任务 | 固定流程任务 | 复杂动态任务 |

#### 成本分析

```
Token 成本对比：

ReAct（即时规划）:
  - LLM 调用：N 次（N = 步数）
  - 每次：短思考（约 100-200 Token）
  - 总成本：N × 短思考

Plan-and-Execute（事前规划）:
  - LLM 调用：1 次（规划阶段）
  - 每次：长计划（可能 500-2000 Token）
  - 总成本：1 × 长计划
  - 优势：减少 N-1 次 API 调用的网络开销

Adaptive（动态规划）:
  - LLM 调用：1 次（初始规划）+ K 次（重新规划）
  - 每次：中等长度
  - 总成本：取决于需要调整的次数


实际成本取决于任务复杂度：

简单任务（3-5 步）:
  ReAct:         5 × 100 = 500 Token
  Plan-and-Exec: 1 × 300 = 300 Token        → 更省

复杂任务（20+ 步）:
  ReAct:         20 × 150 = 3000 Token
  Plan-and-Exec: 1 × 2000 = 2000 Token      → 更省

超复杂任务：
  ReAct:         30 × 100 = 3000 Token
  Plan-and-Exec: 1 × 5000 = 5000 Token      → ReAct 更省！
```

#### 如何选择策略？

```
决策树：

你的任务特点？

├─ 不确定需要做什么（探索性）
│   └─ ReAct ✅
│      例如：研究一个问题、探索未知领域
│
├─ 步骤明确、流程固定
│   └─ Plan-and-Execute ✅
│      例如：写报告的固定流程、数据分析流水线
│
└─ 复杂任务，中间可能遇到问题
    └─ Adaptive ✅
       例如：自动化编程、复杂问题求解
```

### 1.3 ReAct 模式详解

```python
# ReAct = Reasoning + Acting

def react_agent(goal, tools, max_steps=10):
    """
    ReAct 循环：
    Thought → Action → Observation → Thought → ...
    """
    trajectory = []  # 记录轨迹

    for step in range(max_steps):
        # 1. 思考 (Thought)
        thought = llm.generate(
            f"""目标：{goal}

历史轨迹：
{format_trajectory(trajectory)

当前状态：{get_current_state()}

思考：我下一步该做什么？"""
        )

        # 2. 判断是否完成
        if is_finished(thought):
            return extract_answer(thought)

        # 3. 决定行动 (Action)
        action = parse_action(thought)  # 解析出要调用的工具

        # 4. 执行行动 (Observation)
        observation = tools[action.name](**action.args)

        # 5. 记录轨迹
        trajectory.append({
            "thought": thought,
            "action": action,
            "observation": observation
        })

        # 6. 循环继续
        print(f"[Step {step}] {thought}\n  → {action}\n  → {observation}")
```

#### ReAct 实际运行示例

```
目标：查找 2024 年最好的 AI 编程工具并总结

[Step 1] Thought: 我需要搜索 2024 年 AI 编程工具的信息
  Action: 搜索("2024 年最好的 AI 编程工具")
  Observation: 找到 Cursor、Copilot、Codium 等工具...

[Step 2] Thought: 搜索结果不够详细，我需要深入了解 Cursor
  Action: 搜索("Cursor AI 编辑器功能特点")
  Observation: Cursor 是基于 VS Code 的 AI 编辑器...

[Step 3] Thought: 现在我需要了解其他工具的对比
  Action: 搜索("Cursor vs Copilot vs Codium 对比")
  Observation: 三者对比结果...

[Step 4] Thought: 我已经收集了足够信息，现在可以总结
  Action: FINISH
  Answer: 2024 年最好的 AI 编程工具是...
```

### 1.4 CoT（Chain of Thought）

```python
# Chain of Thought - 让模型展示思考过程

def cot_agent(goal):
    prompt = f"""
目标：{goal}

请一步步思考并回答：

思考过程：
1. 首先，我需要...
2. 然后，我应该...
3. 接着，我可以...
4. 最后，得出结论...

答案："""

    return llm.generate(prompt)

# 示例
"""
目标：小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？

思考过程：
1. 最初，小明有 5 个苹果
2. 吃了 2 个后，剩下 5 - 2 = 3 个
3. 又买了 3 个后，现在有 3 + 3 = 6 个

答案：小明现在有 6 个苹果
"""
```

### 1.5 Plan-and-Solve

```python
# 先规划，再执行

def plan_and_solve_agent(goal):
    # 阶段 1：规划
    plan = llm.generate(f"""
目标：{goal}

请制定一个详细的执行计划：
1. 第一步：
2. 第二步：
3. 第三步：
...""")

    # 阶段 2：执行
    results = []
    for step in parse_steps(plan):
        result = execute_step(step)
        results.append(result)

    # 阶段 3：综合
    return synthesize(results)
```

### 1.6 规划器实现模式

```python
# 模式 1：零样本规划（Zero-shot）

def zero_shot_planner(goal):
    """直接让 LLM 生成计划"""
    plan = llm.generate(f"""
请为以下目标制定执行计划：

目标：{goal}

计划：
""")
    return plan


# 模式 2：少样本规划（Few-shot）

def few_shot_planner(goal):
    """给 LLM 看几个例子"""
    plan = llm.generate(f"""
以下是几个目标和对应计划的例子：

例子1：
目标：做一份番茄炒蛋
计划：
1. 准备食材（鸡蛋、番茄、调料）
2. 番茄切块，鸡蛋打散
3. 先炒鸡蛋，盛起
4. 炒番茄，加入鸡蛋
5. 调味装盘

例子2：
目标：学习 Python
计划：
1. 安装 Python 环境
2. 学习基础语法
3. 练习简单项目
4. 学习高级特性
5. 完成综合项目

现在，请为以下目标制定计划：

目标：{goal}
计划：
""")
    return plan


# 模式 3：分解式规划（Decomposition）

def decomposition_planner(goal):
    """递归分解任务"""
    # 先判断是否是原子任务
    if is_atomic_task(goal):
        return [goal]

    # 分解为子任务
    subtasks = llm.generate(f"""
将以下任务分解为 3-5 个子任务：

任务：{goal}

子任务：
1. ...
2. ...
3. ...
""")

    # 递归分解每个子任务
    full_plan = []
    for subtask in subtasks:
        full_plan.extend(decomposition_planner(subtask))

    return full_plan
```

### 1.7 Planner 完整代码实现

> [!tip] 可运行的代码
> 以下代码是完整的、可直接使用的 Planner 实现，包含三种核心策略的完整逻辑。

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


# ========== 数据结构定义 ==========

@dataclass
class PlanStep:
    """规划步骤"""
    description: str           # 步骤描述
    tool_name: Optional[str] = None  # 需要调用的工具名称
    tool_args: Optional[Dict] = None   # 工具参数
    expected_output: Optional[str] = None  # 预期输出


@dataclass
class Plan:
    """完整执行计划"""
    steps: List[PlanStep]      # 步骤列表
    reasoning: str                 # 制定计划的推理过程
    estimated_steps: int          # 预估步骤数


# ========== 基类定义 ==========

class BasePlanner(ABC):
    """规划器基类"""

    @abstractmethod
    def create_plan(self, goal: str, context: Dict) -> Plan:
        """创建执行计划"""
        pass

    @abstractmethod
    def should_replan(self, current_step: PlanStep, result: Any, plan: Plan) -> bool:
        """判断是否需要重新规划"""
        pass

    @abstractmethod
    def decide_next_action(self, goal: str) -> PlanStep:
        """决定下一步行动（ReAct 专用）"""
        pass


# ========== 策略 1：ReAct Planner（即时规划）==========

class ReActPlanner(BasePlanner):
    """即时规划器 - 每步都重新思考"""

    def __init__(self, llm, tools: Dict):
        self.llm = llm
        self.tools = tools
        self.trajectory = []  # 记录执行轨迹

    def create_plan(self, goal: str, context: Dict) -> Plan:
        """ReAct 不创建预设计划，只返回第一步"""
        return Plan(
            steps=[],  # 无预设计划
            reasoning="即时规划，每步动态决策",
            estimated_steps=0  # 未知
        )

    def should_replan(self, current_step: PlanStep, result: Any, plan: Plan) -> bool:
        """ReAct 总是需要"重新规划"（每步都思考）"""
        return True

    def decide_next_action(self, goal: str) -> PlanStep:
        """决定下一步行动（ReAct 的核心）"""

        # 构建思考 prompt
        trajectory_str = self._format_trajectory(self.trajectory)

        thought = self.llm.generate(f"""
目标：{goal}

历史轨迹：
{trajectory_str}

当前状态：{self._get_current_state()}

请思考下一步该做什么。如果已完成目标，请直接给出最终答案并标注 [FINISH]。

思考：""")

        # 检查是否完成
        if "[FINISH]" in thought:
            return PlanStep(description="完成任务")

        # 解析行动
        action = self._parse_action_from_thought(thought)

        return action

    def _format_trajectory(self, trajectory: List) -> str:
        """格式化轨迹"""
        if not trajectory:
            return "（无历史）"

        lines = []
        for i, item in enumerate(trajectory, 1):
            lines.append(f"步骤 {i}:")
            lines.append(f"  思考: {item['thought']}")
            lines.append(f"  行动: {item['action']}")
            lines.append(f"  结果: {str(item['result'])[:100]}...")

        return "\n".join(lines)

    def _get_current_state(self) -> str:
        """获取当前状态"""
        return f"已完成 {len(self.trajectory)} 步"

    def _parse_action_from_thought(self, thought: str) -> PlanStep:
        """从思考中解析行动"""
        # 方式 1：使用 LLM 结构化输出
        action = self.llm.generate(f"""
从以下思考中提取要执行的行动：

思考：{thought}

如果需要调用工具，请以 JSON 格式输出：
{{
    "tool": "工具名称",
    "args": {{"参数": "值"}},
    "description": "行动描述"
}}

如果不需要调用工具，返回：
{{"tool": null, "args": {{}}, "description": "说明原因"}}）
""")

        try:
            data = json.loads(action)
            return PlanStep(
                description=data["description"],
                tool_name=data.get("tool"),
                tool_args=data.get("args")
            )
        except:
            # 解析失败，返回默认步骤
            return PlanStep(description=thought[:100])


# ========== 策略 2：Plan-and-Execute Planner（事前规划）==========

class PlanExecutePlanner(BasePlanner):
    """事前规划器 - 先完整规划，再执行"""

    def __init__(self, llm, tools: Dict):
        self.llm = llm
        self.tools = tools

    def create_plan(self, goal: str, context: Dict) -> Plan:
        """一次性创建完整计划"""

        # 获取可用工具列表
        tools_list = "\n".join([
            f"- {name}: {tool.__doc__ or '无描述'}"
            for name, tool in self.tools.items()
        ])

        plan_prompt = f"""
目标：{goal}

可用工具：
{tools_list}

请制定一个详细的执行计划。每个步骤应该：
1. 明确要做什么
2. 指定使用的工具（如果需要）
3. 说明预期的输出

计划格式：
1. [步骤描述]
   工具：[工具名称] 或 "思考"
   输入：[参数] 或 "无"
   目的：[这一步要达到什么目的]

2. [步骤描述]
   ...

请开始制定计划：
"""

        response = self.llm.generate(plan_prompt)

        # 解析计划
        steps = self._parse_plan_text(response)

        return Plan(
            steps=steps,
            reasoning=response,
            estimated_steps=len(steps)
        )

    def should_replan(self, current_step: PlanStep, result: Any, plan: Plan) -> bool:
        """Plan-and-Execute 不重新规划"""
        return False

    def decide_next_action(self, goal: str) -> PlanStep:
        """此方法不适用于 Plan-and-Execute"""
        raise NotImplementedError("Plan-and-Execute 不使用此方法")

    def _parse_plan_text(self, plan_text: str) -> List[PlanStep]:
        """解析计划文本"""
        steps = []
        current_step = None

        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 检测步骤编号（如 "1. ", "2. "）
            if line[0].isdigit() and line[1] in ". ":
                if current_step:
                    steps.append(current_step)

                # 提取步骤描述
                desc = line.split(".", 1)[1].strip()
                current_step = PlanStep(description=desc)

            # 检测工具
            elif "工具：" in line:
                tool_name = line.split("工具：", 1)[1].strip()
                current_step.tool_name = tool_name

            elif "输入：" in line:
                input_str = line.split("输入：", 1)[1].strip()
                if input_str != "无":
                    try:
                        current_step.tool_args = json.loads(input_str)
                    except:
                        current_step.tool_args = {"query": input_str}

            elif "目的：" in line:
                current_step.expected_output = line.split("目的：", 1)[1].strip()

        if current_step:
            steps.append(current_step)

        return steps


# ========== 策略 3：Adaptive Planner（动态规划）==========

class AdaptivePlanner(BasePlanner):
    """动态规划器 - 有计划但会调整"""

    def __init__(self, llm, tools: Dict):
        self.llm = llm
        self.tools = tools
        self.execution_history = []
        self.current_plan: Optional[Plan] = None

    def create_plan(self, goal: str, context: Dict) -> Plan:
        """创建初始计划"""
        # 复用 Plan-and-Execute 的规划逻辑
        base_planner = PlanExecutePlanner(self.llm, self.tools)
        self.current_plan = base_planner.create_plan(goal, context)
        return self.current_plan

    def should_replan(self, current_step: PlanStep, result: Any, plan: Plan) -> bool:
        """评估是否需要调整计划"""

        # 记录执行历史
        self.execution_history.append({
            "step": current_step,
            "result": result
        })

        # 使用 LLM 评估
        evaluation = self.llm.generate(f"""
当前步骤：{current_step.description}

执行结果：
{str(result)[:500]}

历史执行情况：
{self._format_execution_history()}

请评估：
1. 这个步骤是否成功完成？
2. 如果失败，原因是什么？
3. 是否需要调整后续计划？

以 JSON 格式输出：
{{
    "success": true/false,
    "needs_replan": true/false,
    "reason": "原因说明",
    "adjustment": "如何调整（如果需要）"
}}
""")

        try:
            eval_data = json.loads(evaluation)
            return eval_data["needs_replan"]
        except:
            return False  # 默认不调整

    def decide_next_action(self, goal: str) -> PlanStep:
        """此方法不适用于 Adaptive"""
        raise NotImplementedError("Adaptive 使用 create_plan + should_replan")

    def adjust_plan(self, goal: str) -> Plan:
        """调整计划"""

        adjustment_prompt = f"""
原始目标：{goal}

原始计划：
{self._format_plan(self.current_plan)}

执行历史：
{self._format_execution_history()}

请根据执行情况调整剩余计划：
"""

        response = self.llm.generate(adjustment_prompt)

        # 解析调整后的计划
        steps = self._parse_plan_text(response)

        return Plan(
            steps=steps,
            reasoning=f"调整后的计划（基于 {len(self.execution_history)} 步执行历史）",
            estimated_steps=len(steps)
        )

    def _format_execution_history(self) -> str:
        """格式化执行历史"""
        if not self.execution_history:
            return "（无历史）"

        lines = []
        for i, item in enumerate(self.execution_history, 1):
            lines.append(f"步骤 {i}: {item['step'].description}")
            lines.append(f"  结果: {str(item['result'])[:100]}...")

        return "\n".join(lines)

    def _format_plan(self, plan: Plan) -> str:
        """格式化计划"""
        lines = []
        for i, step in enumerate(plan.steps, 1):
            lines.append(f"{i}. {step.description}")
            if step.tool_name:
                lines.append(f"   工具：{step.tool_name}")
        return "\n".join(lines)


# ========== Planner 工厂（工厂模式）==========

class PlannerFactory:
    """规划器工厂 - 简化创建不同类型的规划器"""

    @staticmethod
    def create(planner_type: str, llm, tools: Dict) -> BasePlanner:
        """
        创建规划器

        Args:
            planner_type: "react" | "plan_execute" | "adaptive"
            llm: LLM 实例
            tools: 工具字典 {"tool_name": tool_function}

        Returns:
            BasePlanner 实例
        """
        planners = {
            "react": ReActPlanner,
            "plan_execute": PlanExecutePlanner,
            "adaptive": AdaptivePlanner
        }

        planner_class = planners.get(planner_type)
        if not planner_class:
            raise ValueError(f"未知的规划器类型: {planner_type}")

        return planner_class(llm=llm, tools=tools)


# ========== 完整的 Agent 运行流程 ==========

class Agent:
    """完整的 Agent 实现 - 集成 Planner + Tool Use + Memory + Reflection"""

    def __init__(self, planner_type: str, llm, tools: Dict, enable_memory: bool = True):
        self.planner = PlannerFactory.create(planner_type, llm, tools)
        self.tools = tools
        self.llm = llm
        self.enable_memory = enable_memory
        self.memory = [] if enable_memory else None

    def run(self, goal: str, max_iterations: int = 20):
        """运行 Agent"""

        print(f"\n{'='*60}")
        print(f"目标：{goal}")
        print(f"规划策略：{self.planner.__class__.__name__}")
        print(f"{'='*60}\n")

        # 1. 创建初始计划
        plan = self.planner.create_plan(goal, context={})

        # 如果是 ReAct，没有预设计划
        if not plan.steps:
            return self._run_react_loop(goal, max_iterations)

        # 2. 执行计划
        results = []
        completed_steps = 0

        for i, step in enumerate(plan.steps, 1):
            print(f"\n[步骤 {i}/{len(plan.steps)}] {step.description}")

            # 执行步骤
            try:
                if step.tool_name and step.tool_name in self.tools:
                    result = self.tools[step.tool_name](**step.tool_args)
                else:
                    # 无工具，直接用 LLM 思考
                    result = self.llm.generate(f"请完成：{step.description}")

                results.append(result)
                completed_steps += 1
                print(f"  ✓ 完成")

                # 记录到记忆
                if self.enable_memory:
                    self.memory.append({"step": step, "result": result})

            except Exception as e:
                print(f"  ✗ 失败：{e}")
                results.append({"error": str(e)})

                # 记录失败到记忆
                if self.enable_memory:
                    self.memory.append({"step": step, "error": str(e)})

            # 3. 检查是否需要调整计划
            if self.planner.should_replan(step, results[-1], plan):
                print(f"\n  → 检测到需要调整计划...")

                if hasattr(self.planner, 'adjust_plan'):
                    plan = self.planner.adjust_plan(goal)
                    print(f"  → 计划已调整，剩余 {len(plan.steps)} 步")
                else:
                    print(f"  → 继续执行")

        # 4. 生成最终答案
        return self._generate_final_answer(goal, plan, results)

    def _run_react_loop(self, goal: str, max_iterations: int):
        """ReAct 循环执行"""

        for iteration in range(max_iterations):
            # 决定下一步
            next_step = self.planner.decide_next_action(goal)

            # 检查是否完成
            if not next_step.tool_name:
                return next_step.description

            # 执行行动
            try:
                result = self.tools[next_step.tool_name](**next_step.tool_args)
            except Exception as e:
                result = f"错误：{str(e)}"

            # 记录轨迹
            self.planner.trajectory.append({
                "thought": next_step.description,
                "action": next_step,
                "result": result
            })

            print(f"[迭代 {iteration + 1}] {next_step.description}")
            print(f"  → {str(result)[:100]}...")

    def _generate_final_answer(self, goal: str, plan: Plan, results: List) -> str:
        """生成最终答案"""

        results_str = "\n".join([
            f"- {str(r)[:200]}" for r in results
        ])

        return self.llm.generate(f"""
目标：{goal}

执行计划：
{self._format_plan_for_display(plan)}

执行结果：
{results_str}

请根据以上信息给出最终答案：
""")

    def _format_plan_for_display(self, plan: Plan) -> str:
        """格式化计划用于显示"""
        if not plan.steps:
            return "（无预设计划）"

        lines = []
        for i, step in enumerate(plan.steps, 1):
            lines.append(f"{i}. {step.description}")
        return "\n".join(lines)
```

---

## 组件二：Tool Use（工具使用）

> [!quote] 核心思想
> **"Actions speak louder than words."** - 行动胜于言语

### 2.1 工具使用的作用

```
Tool Use = Agent 的手脚

功能：
✅ 调用外部 API
✅ 执行代码
✅ 操作数据库
✅ 搜索信息
✅ 处理文件

核心问题：
"我该如何完成这个具体动作？"
```

### 2.2 工具定义

```python
# 标准工具定义（LangChain 风格）

from langchain.tools import tool
from typing import Literal

# 方式 1：装饰器定义
@tool
def search_web(query: str, engine: Literal["google", "bing"] = "google") -> str:
    """
    搜索互联网信息

    Args:
        query: 搜索关键词
        engine: 搜索引擎（google 或 bing）

    Returns:
        搜索结果摘要
    """
    # 实际搜索逻辑
    return f"搜索 '{query}' 的结果..."


# 方式 2：类定义
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量")

@tool(args_schema=SearchInput)
def advanced_search(query: str, num_results: int = 5) -> str:
    """高级搜索工具"""
    return f"搜索到 {num_results} 条关于 '{query}' 的结果"


# 方式 3：结构化工具
from langchain.tools import StructuredTool

def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="calculator",
    description="计算数学表达式，例如：2+2, sin(0.5), 2**10",
    args_schema=type("Input", (BaseModel,), {"__annotations__": {"expression": str}})
)
```

### 2.3 工具类型全景

```
┌─────────────────────────────────────────────────────────┐
│                      工具类型分类                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  信息获取类                                              │
│  ├─ Web Search（网页搜索）                              │
│  ├─ Wikipedia（维基百科）                               │
│  ├─ Arxiv（学术论文）                                   │
│  └─ News API（新闻资讯）                                │
│                                                         │
│  数据处理类                                              │
│  ├─ Calculator（计算器）                                 │
│  ├─ Data Analysis（数据分析）                            │
│  ├─ SQL Database（数据库查询）                           │
│  └─ CSV/Excel Processor（表格处理）                      │
│                                                         │
│  代码执行类                                              │
│  ├─ Python REPL（Python 解释器）                        │
│  ├─ Code Interpreter（代码解释器）                       │
│  └─ Shell/Bash（命令行）                                 │
│                                                         │
│  文件操作类                                              │
│  ├─ File Reader（文件读取）                             │
│  ├─ File Writer（文件写入）                             │
│  └─ Directory Browser（目录浏览）                        │
│                                                         │
│  API 调用类                                              │
│  ├─ HTTP Request（HTTP 请求）                           │
│  ├─ API Clients（专用 API 客户端）                       │
│  └─ Webhook（触发器）                                   │
│                                                         │
│  通信协作类                                              │
│  ├─ Email Sender（邮件发送）                            │
│  ├─ Slack/Discord Bot（消息通知）                        │
│  └─ Calendar（日程管理）                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.4 工具调用机制

```python
# 机制 1：Function Calling（OpenAI 风格）

def function_calling_mode():
    """使用 LLM 原生的 Function Calling"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "纽约现在的天气怎么样？"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
    )

    # LLM 决定调用工具
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # 执行工具
        result = get_weather(**arguments)

        # 继续对话
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "纽约现在的天气怎么样？"},
                response.choices[0].message,  # 工具调用消息
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            ]
        )


# 机制 2：文本解析（兼容非 Function Calling 模型）

def text_parsing_mode():
    """从文本中解析出工具调用"""

    prompt = """
    可用工具：
    - search(query): 搜索信息
    - calculator(expr): 计算表达式

    用户问题：100 的平方根是多少？

    请决定是否需要调用工具。如果需要，按以下格式输出：
    Action: calculator
    Input: sqrt(100)
    """

    response = llm.generate(prompt)

    # 解析响应
    action = parse_action(response)  # "calculator"
    input_data = parse_input(response)  # "sqrt(100)"

    # 执行
    if action == "calculator":
        result = calculator(input_data)
```

### 2.5 工具调用最佳实践

```python
# 实践 1：清晰的工具描述

@tool
def bad_tool(query: str) -> str:
    """搜索信息"""  # ❌ 描述太模糊
    return search(query)

@tool
def good_tool(query: str) -> str:
    """
    搜索互联网信息，包括最新新闻、技术文章、百科知识等。

    适用场景：
    - 需要获取实时信息
    - 需要查找具体资料
    - 需要验证事实

    不适用：
    - 纯计算问题（用 calculator）
    - 代码编写（用 code_interpreter）
    """  # ✅ 描述详细、明确边界
    return search(query)


# 实践 2：参数验证

@tool
def validated_tool(email: str, age: int) -> str:
    """
    注册用户信息

    Args:
        email: 邮箱地址（必须包含 @）
        age: 年龄（必须 18-100）

    Returns:
        注册结果
    """
    # 验证
    if "@" not in email:
        return "错误：邮箱格式不正确"
    if not (18 <= age <= 100):
        return "错误：年龄必须在 18-100 之间"

    # 执行
    return register(email, age)


# 实践 3：错误处理

@tool
def robust_tool(url: str) -> str:
    """
    获取网页内容

    包含完整的错误处理和重试机制
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text[:5000]  # 限制长度
    except requests.Timeout:
        return "错误：请求超时，请稍后重试"
    except requests.HTTPError as e:
        return f"错误：HTTP {e.response.status_code}"
    except Exception as e:
        return f"错误：{str(e)}"


# 实践 4：工具组合（Tool Composition）

def create_super_search_tool():
    """组合多个搜索工具"""

    google_search = Tool(name="google", func=google_search, ...)
    bing_search = Tool(name="bing", func=bing_search, ...)
    wiki_search = Tool(name="wikipedia", func=wiki_search, ...)

    # 创建一个超级搜索工具
    @tool
    def super_search(query: str) -> str:
        """
        综合搜索工具，自动选择最佳搜索源

        策略：
        1. 先搜 Wikipedia（如果有条目）
        2. 再搜 Google（获取最新信息）
        3. 合并结果
        """
        # 搜索 Wikipedia
        wiki_result = wiki_search.run(query)

        # 如果 Wikipedia 有结果，直接返回
        if wiki_result and "没有找到" not in wiki_result:
            return wiki_result

        # 否则搜索 Google
        google_result = google_search.run(query)
        return google_result

    return super_search
```

### 2.6 常用工具库

```python
# LangChain 内置工具
from langchain_community.tools import (
    # 搜索类
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    GoogleSerperAPIResults,

    # 数据类
    PythonREPL,
    PythonAstREPLTool,

    # 文件类
    ReadFileTool,
    WriteFileTool,
    DirectoryBrowserTool,

    # 其他
    ShellTool,
    RequestsGetTool,
    WolframAlphaQueryRun,
)

# 自定义工具集合
MY_TOOLKIT = [
    search_tool,
    calculator_tool,
    python_repl_tool,
    file_reader_tool,
    database_query_tool,
    api_call_tool,
]
```

---

## 组件三：Memory（记忆系统）

> [!quote] 核心思想
> **"Experience is the best teacher."** - 经验是最好的老师

### 3.1 记忆系统的作用

```
Memory = Agent 的大脑海马体

功能：
✅ 存储对话历史
✅ 记住关键信息
✅ 检索相关经验
✅ 支持长期学习

核心问题：
"我该如何记住和利用过去的经验？"
```

### 3.2 记忆类型全景

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 记忆系统                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  短期记忆 (Short-term Memory)                    │   │
│  │  ├─ 对话历史 (Chat History)                     │   │
│  │  ├─ 当前任务状态 (Task State)                   │   │
│  │  ├─ 中间结果 (Intermediate Results)             │   │
│  │  └─ 上下文窗口 (Context Window)                 │   │
│  │                                                  │   │
│  │  特点：快速访问、容量有限、易丢失                │   │
│  │  实现：LLM 上下文、会话缓冲                       │   │
│  └─────────────────────────────────────────────────┘   │
│                          ↓                              │
│                   精华提取                              │
│                          ↓                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  长期记忆 (Long-term Memory)                     │   │
│  │  ├─ 向量数据库 (Vector DB)                       │   │
│  │  ├─ 知识图谱 (Knowledge Graph)                   │   │
│  │  ├─ 关键事实 (Key Facts)                         │   │
│  │  └─ 用户偏好 (User Preferences)                  │   │
│  │                                                  │   │
│  │  特点：持久化、大容量、需要检索                   │   │
│  │  实现：Chroma/Qdrant + Embedding                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.3 短期记忆实现

```python
# 方式 1：简单列表存储（适合短对话）

class SimpleMemory:
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # 超出限制时移除旧消息（保留系统消息）
        if len(self.messages) > self.max_messages:
            # 保留第一条系统消息
            system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
            self.messages = [system_msg] + self.messages[-(self.max_messages-1):] if system_msg else self.messages[-(self.max_messages):]

    def get_context(self):
        return self.messages


# 方式 2：滑动窗口（保留重要信息）

class SlidingWindowMemory:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.all_messages = []  # 所有消息
        self.summary = ""       # 历史摘要

    def add_message(self, role: str, content: str):
        self.all_messages.append({"role": role, "content": content})

        # 超出窗口大小时，生成摘要
        if len(self.all_messages) > self.window_size:
            old_messages = self.all_messages[:-self.window_size//2]

            # 用 LLM 生成摘要
            self.summary = llm.generate(f"""
            以下是之前的对话历史，请生成简洁摘要：

            {format_messages(old_messages)}

            摘要：
            """)

            # 只保留最近的消息 + 摘要
            self.all_messages = self.all_messages[-(self.window_size//2):]

    def get_context(self):
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"历史对话摘要：{self.summary}"})
        context.extend(self.all_messages)
        return context


# 方式 3：Token 计数管理

class TokenManagedMemory:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # 估算 Token 数
        total_tokens = sum(count_tokens(msg["content"]) for msg in self.messages)

        # 超出限制时移除旧消息
        while total_tokens > self.max_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            total_tokens -= count_tokens(removed["content"])

    def get_context(self):
        return self.messages


# 使用示例
memory = SimpleMemory(max_messages=10)

memory.add_message("user", "你好")
memory.add_message("assistant", "你好！有什么我可以帮助你的？")
memory.add_message("user", "我叫小明")

# 获取上下文
context = memory.get_context()
# [{"role": "user", "content": "你好"}, ...]
```

### 3.4 长期记忆实现

```python
# 方式 1：向量数据库存储（语义记忆）

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreMemory
from langchain.docstore import InMemoryDocstore

class SemanticMemory:
    def __init__(self):
        # 初始化向量存储
        embedding_function = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=embedding_function,
        )

    def remember(self, content: str, metadata: dict = None):
        """记住信息"""
        # 创建文档
        from langchain.docstore.document import Document
        doc = Document(page_content=content, metadata=metadata or {})

        # 存储到向量库
        self.vectorstore.add_documents([doc])

    def recall(self, query: str, k=3) -> list:
        """回忆相关信息"""
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]


# 使用示例
memory = SemanticMemory()

# 记住信息
memory.remember(
    "用户最喜欢的编程语言是 Python",
    metadata={"type": "preference", "timestamp": "2024-01-15"}
)
memory.remember(
    "用户正在做一个机器学习项目",
    metadata={"type": "project", "timestamp": "2024-01-16"}
)

# 回忆信息
query = "用户喜欢什么技术？"
relevant_memories = memory.recall(query)
# ["用户最喜欢的编程语言是 Python", "用户正在做一个机器学习项目"]


# 方式 2：知识图谱存储（结构化记忆）

class GraphMemory:
    """使用知识图谱存储实体和关系"""

    def __init__(self):
        # 存储实体和关系
        self.entities = {}   # {name: {attributes}}
        self.relations = []  # [(source, relation, target)]

    def remember_entity(self, name: str, attributes: dict):
        """记住实体"""
        self.entities[name] = attributes

    def remember_relation(self, source: str, relation: str, target: str):
        """记住关系"""
        self.relations.append((source, relation, target))

    def recall_entity(self, name: str) -> dict:
        """回忆实体"""
        return self.entities.get(name, {})

    def recall_relations(self, entity: str) -> list:
        """回忆与实体相关的所有关系"""
        related = []
        for s, r, t in self.relations:
            if s == entity:
                related.append((r, t))
            elif t == entity:
                related.append(("reverse_" + r, s))
        return related


# 使用示例
memory = GraphMemory()

# 记住实体
memory.remember_entity("小明", {
    "name": "小明",
    "occupation": "工程师",
    "location": "北京"
})

# 记住关系
memory.remember_relation("小明", "喜欢", "Python")
memory.remember_relation("小明", "在做", "机器学习项目")
memory.remember_relation("机器学习项目", "使用", "TensorFlow")

# 查询
print(memory.recall_entity("小明"))
# {"name": "小明", "occupation": "工程师", "location": "北京"}

print(memory.recall_relations("小明"))
# [("喜欢", "Python"), ("在做", "机器学习项目")]


# 方式 3：混合记忆（综合方案）

class HybridMemory:
    """结合短期、语义、结构化记忆"""

    def __init__(self):
        self.short_term = SimpleMemory(max_messages=20)
        self.semantic = SemanticMemory()
        self.graph = GraphMemory()
        self.key_facts = []  # 关键事实列表

    def add_message(self, role: str, content: str):
        # 添加到短期记忆
        self.short_term.add_message(role, content)

        # 提取并存储关键信息
        if role == "user":
            self._extract_and_store(content)

    def _extract_and_store(self, content: str):
        """从内容中提取关键信息"""

        # 用 LLM 提取实体、偏好、关系
        extracted = llm.generate(f"""
        从以下对话中提取关键信息：

        {content}

        请提取：
        1. 实体（人名、地名、物名）
        2. 用户偏好
        3. 重要事实

        以 JSON 格式输出：
        {{
            "entities": [],
            "preferences": [],
            "facts": []
        }}
        """)

        data = json.loads(extracted)

        # 存储到各记忆系统
        for entity in data.get("entities", []):
            self.graph.remember_entity(entity["name"], entity)

        for pref in data.get("preferences", []):
            self.semantic.remember(f"用户偏好：{pref}", metadata={"type": "preference"})

        for fact in data.get("facts", []):
            self.key_facts.append(fact)

    def get_context(self, query: str = None) -> dict:
        """获取完整上下文"""
        context = {
            "chat_history": self.short_term.get_context(),
            "relevant_memories": [],
            "key_facts": self.key_facts[-5:]  # 最近 5 个关键事实
        }

        if query:
            # 从语义记忆中检索相关信息
            context["relevant_memories"] = self.semantic.recall(query)

        return context
```

### 3.5 记忆检索策略

```python
# 策略 1：相似度检索

def retrieve_by_similarity(query: str, memory, k=3):
    """根据语义相似度检索记忆"""
    return memory.vectorstore.similarity_search(query, k=k)


# 策略 2：时间衰减检索

def retrieve_with_time_decay(query: str, memory, k=3):
    """考虑时间因素的检索"""
    results = memory.vectorstore.similarity_search_with_score(query, k=k*2)

    # 加入时间衰减因子
    import time
    current_time = time.time()

    scored_results = []
    for doc, score in results:
        # 获取时间戳
        timestamp = doc.metadata.get("timestamp", current_time)
        age_days = (current_time - timestamp) / 86400

        # 时间衰减：越新的记忆权重越高
        time_factor = 1.0 / (1.0 + age_days * 0.1)
        final_score = score * 0.7 + time_factor * 0.3

        scored_results.append((doc, final_score))

    # 重新排序
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_results[:k]]


# 策略 3：重要性加权

def retrieve_by_importance(query: str, memory, k=3):
    """考虑记忆重要性"""
    results = memory.vectorstore.similarity_search_with_score(query, k=k*2)

    # 加入重要性因子（由 LLM 评估）
    scored_results = []
    for doc, score in results:
        importance = doc.metadata.get("importance", 0.5)
        final_score = score * 0.6 + importance * 0.4
        scored_results.append((doc, final_score))

    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_results[:k]]
```

---

## 组件四：Reflection（反思机制）

> [!quote] 核心思想
> **"The unexamined life is not worth living."** - 未经审视的人生不值得过

### 4.1 反思机制的作用

```
Reflection = Agent 的元认知能力

功能：
✅ 评估执行结果
✅ 发现错误和问题
✅ 改进执行策略
✅ 学习和成长

核心问题：
"我做得怎么样？如何改进？"
```

### 4.2 反思类型全景

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 反思机制                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  结果验证 (Result Validation)                    │   │
│  │  ├─ 检查输出格式                                 │   │
│  │  ├─ 验证事实正确性                               │   │
│  │  ├─ 检查任务完成度                               │   │
│  │  └─ 边界条件检查                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  错误检测 (Error Detection)                      │   │
│  │  ├─ 语法错误                                     │   │
│  │  ├─ 逻辑错误                                     │   │
│  │  ├─ 执行失败                                     │   │
│  │  └─ 异常处理                                     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  策略调整 (Strategy Adjustment)                  │   │
│  │  ├─ 更换工具                                     │   │
│  │  ├─ 调整参数                                     │   │
│  │  ├─ 修改计划                                     │   │
│  │  └─ 尝试替代方案                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  学习改进 (Learning & Improvement)               │   │
│  │  ├─ 记录成功经验                                 │   │
│  │  ├─ 分析失败原因                                 │   │
│  │  ├─ 更新知识库                                   │   │
│  │  └─ 优化行为模式                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.3 反思实现模式

```python
# 模式 1：自我评估（Self-Evaluation）

def self_reflection_agent(goal, max_iterations=3):
    """带有自我反思的 Agent"""

    history = []

    for iteration in range(max_iterations):
        # 执行任务
        result = execute_task(goal, history)

        # 自我评估
        reflection = llm.generate(f"""
        目标：{goal}

        执行结果：
        {result}

        请评估：
        1. 结果是否完全满足目标？
        2. 是否有错误或不足？
        3. 如何改进？

        评估结果（JSON）：
        {{
            "satisfied": true/false,
            "errors": [],
            "improvements": []
        }}
        """)

        eval_result = json.loads(reflection)

        # 如果满足要求，返回结果
        if eval_result["satisfied"]:
            return result

        # 否则，记录反思结果并重试
        history.append({
            "iteration": iteration,
            "result": result,
            "reflection": eval_result
        })

    # 达到最大迭代次数，返回最后一次结果
    return result


# 模式 2：Critic Agent（批评者模式）

def critic_agent_mode(goal):
    """使用一个独立的 Critic Agent 来评估"""

    # 生成器 Agent
    generator = Agent(role="generator")
    result = generator.execute(goal)

    # 批评者 Agent
    critic = Agent(role="critic")
    critique = critic.evaluate(f"""
    目标：{goal}

    结果：
    {result}

    请提供批评和改进建议。
    """)

    # 生成器根据批评改进
    if critique["needs_improvement"]:
        result = generator.improve(result, critique["suggestions"])

    return result


# 模式 3：树搜索反思（Tree of Thoughts + Reflection）

def tree_search_reflection(goal, max_branches=3):
    """探索多个可能的解决方案"""

    # 生成多个候选方案
    candidates = []
    for i in range(max_branches):
        candidate = generate_solution(goal, variation=i)
        score = evaluate_solution(goal, candidate)
        candidates.append((candidate, score))

    # 选择最佳方案
    best_candidate, best_score = max(candidates, key=lambda x: x[1])

    # 对最佳方案进行反思改进
    reflection = reflect_on_solution(goal, best_candidate)
    if reflection["can_improve"]:
        improved = improve_solution(best_candidate, reflection["suggestions"])
        return improved

    return best_candidate


# 模式 4：持续反思（Continuous Reflection）

class ReflectiveAgent:
    """持续反思和改进的 Agent"""

    def __init__(self):
        self.memory = HybridMemory()
        self.success_patterns = []   # 成功模式
        self.failure_patterns = []   # 失败模式

    def execute_with_reflection(self, goal):
        while True:
            # 执行
            result = self._execute(goal)

            # 反思
            reflection = self._reflect(goal, result)

            # 学习
            self._learn_from(reflection)

            # 判断是否需要继续
            if reflection["complete"]:
                return result

            # 调整策略
            goal = self._adjust_goal(goal, reflection)

    def _reflect(self, goal, result):
        """反思执行结果"""
        reflection = llm.generate(f"""
        目标：{goal}
        结果：{result}

        历史成功模式：{self.success_patterns}
        历史失败模式：{self.failure_patterns}

        请分析：
        1. 任务是否完成？
        2. 如果完成，使用了什么成功模式？
        3. 如果未完成，犯了什么错误？
        4. 应该如何调整？
        """)

        return json.loads(reflection)

    def _learn_from(self, reflection):
        """从反思中学习"""
        if reflection["success"]:
            self.success_patterns.append(reflection["pattern"])
        else:
            self.failure_patterns.append(reflection["error"])
```

### 4.4 代码执行反思示例

```python
# 示例：代码生成 Agent 的反思机制

def code_generation_agent(task_description):
    """带反思的代码生成 Agent"""

    # 生成代码
    code = llm.generate(f"""
    任务：{task_description}

    请生成 Python 代码：
    """)

    # 反思阶段 1：静态检查
    static_check = llm.generate(f"""
    代码：
    {code}

    请检查：
    1. 语法是否正确？
    2. 是否有明显错误？
    3. 是否符合任务要求？

    如果发现问题，请指出并提供修改建议。
    """)

    # 如果有问题，修复
    if "问题" in static_check:
        code = llm.generate(f"""
        原代码：
        {code}

        发现的问题：
        {static_check}

        请修复这些问题，输出修改后的代码：
        """)

    # 反思阶段 2：执行测试
    try:
        exec_result = execute_code(code)
        execution_success = True
    except Exception as e:
        exec_result = str(e)
        execution_success = False

    # 反思阶段 3：综合评估
    final_reflection = llm.generate(f"""
    任务：{task_description}

    代码：
    {code}

    执行结果：
    {exec_result}

    执行是否成功：{execution_success}

    请评估：
    1. 代码是否正确完成了任务？
    2. 是否有改进空间？
    3. 最终评分（0-10）？
    """)

    return {
        "code": code,
        "execution_result": exec_result,
        "reflection": final_reflection
    }


# 实际运行示例
"""
任务：写一个函数计算斐波那契数列第 n 项

[生成代码]
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

[反思 1：静态检查]
代码语法正确，但递归效率低，对于大的 n 会有性能问题。

[修复代码]
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

[反思 2：执行测试]
执行结果：fibonacci(10) = 55 ✓
执行结果：fibonacci(50) = 12586269025 ✓
执行成功！

[反思 3：综合评估]
代码正确完成任务，使用了迭代优化，性能良好。
评分：9/10
"""
```

### 4.5 反思触发条件

```python
class ReflectionTrigger:
    """决定何时触发反思"""

    def should_reflect(self, context) -> bool:
        """
        判断是否需要反思

        触发条件：
        1. 任务失败
        2. 结果质量低
        3. 存在错误
        4. 达到关键节点
        5. 用户要求
        """

        # 条件 1：执行失败
        if context.get("execution_failed"):
            return True

        # 条件 2：结果质量低
        if context.get("quality_score", 1.0) < 0.7:
            return True

        # 条件 3：检测到错误
        if context.get("errors"):
            return True

        # 条件 4：达到关键节点（如每 N 步）
        if context.get("step", 0) % 5 == 0:
            return True

        # 条件 5：用户明确要求
        if context.get("user_request_reflection"):
            return True

        return False


# 使用示例
trigger = ReflectionTrigger()

while not task_complete:
    result = agent.step()

    if trigger.should_reflect(agent.context):
        reflection = agent.reflect()
        agent.adjust(reflection)
```

---

## 四大组件协作示例

```python
# 完整的 Agent 实现（四大组件协作）

class CompleteAgent:
    """完整的 Agent 实现"""

    def __init__(self, llm, tools):
        # 四大组件
        self.planner = Planner(llm)
        self.tool_use = ToolUse(tools)
        self.memory = HybridMemory()
        self.reflection = Reflection(llm)

        self.llm = llm

    def run(self, goal: str, max_iterations=10):
        """运行 Agent"""

        # 1. 规划
        plan = self.planner.create_plan(goal)
        self.memory.add("plan", plan)

        results = []

        # 2. 执行循环
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")

            # 获取上下文
            context = self.memory.get_context(query=goal)

            # 3. 决定行动
            action = self.planner.decide_action(goal, context, results)
            print(f"Action: {action}")

            # 4. 执行行动
            result = self.tool_use.execute(action)
            print(f"Result: {result}")

            # 5. 记录结果
            results.append(result)
            self.memory.add("result", result)

            # 6. 反思
            should_continue = self.reflection.evaluate(goal, results)
            print(f"Reflection: {should_continue}")

            # 7. 判断是否完成
            if not should_continue["continue"]:
                break

            # 8. 调整计划
            if should_continue.get("adjust_plan"):
                plan = self.planner.adjust_plan(plan, should_continue["reasoning"])
                self.memory.add("plan", plan)

        # 9. 生成最终答案
        final_answer = self.llm.generate(f"""
        目标：{goal}

        执行过程：
        {format_results(results)}

        请总结最终结果：
        """)

        return final_answer


# 使用示例
agent = CompleteAgent(
    llm=OpenAI(model="gpt-4"),
    tools=[search_tool, calculator_tool, python_tool]
)

result = agent.run("分析 2024 年 AI 领域的重要进展并生成报告")
print(result)
```

---

## 七、ReAct 模式

> [!info] 核心模式
> **ReAct** 是最基础、最常用的 Agent 推理模式，结合了**推理**和**行动**

### 7.1 ReAct 原理

```
ReAct = Reasoning + Acting

循环：
Thought（思考） → Action（行动） → Observation（观察） → ...

类比人类：
遇到问题 → 思考怎么办 → 采取行动 → 观察结果 → 继续思考...
```

### 7.2 ReAct Prompt 模板

```python
REACT_PROMPT = """
回答以下问题，你可以使用以下工具：

{tools}

使用以下格式：

Question: 你要回答的问题
Thought: 你应该思考做什么
Action: 要使用的工具
Action Input: 工具的输入
Observation: 工具的输出结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}
"""


# 实际运行示例
"""
Question: 查找 Python 的发明者是谁，他现在多大年纪？

Thought: 我需要先搜索 Python 的发明者
Action: 搜索
Action Input: Python 编程语言发明者
Observation: Python 由 Guido van Rossum 在 1989 年发明

Thought: 现在我需要知道 Guido van Rossum 的出生年份来计算年龄
Action: 搜索
Action Input: Guido van Rossum 出生年份
Observation: Guido van Rossum 出生于 1956 年 1 月 31 日

Thought: 我现在可以计算他的年龄了，当前是 2024 年
Action: 计算器
Action Input: 2024 - 1956
Observation: 68

Thought: 我现在知道最终答案了
Final Answer: Python 由 Guido van Rossum 发明，他出生于 1956 年，今年约 68 岁。
"""
```

---

## 八、CoT（Chain of Thought）

### 8.1 CoT 原理

```python
# 让模型展示思考过程

def cot_prompt(question):
    return f"""
请一步步思考并回答以下问题：

问题：{question}

思考过程：
1. 首先，...
2. 然后，...
3. 接着，...
4. 最后，...

答案："""


# 示例
"""
问题：如果 3 个苹果 6 元，那么 5 个苹果多少钱？

思考过程：
1. 首先，计算 1 个苹果的价格：6 元 ÷ 3 个 = 2 元/个
2. 然后，计算 5 个苹果的价格：2 元/个 × 5 个 = 10 元
3. 检查：3 个苹果 6 元，5 个苹果应该更贵，10 元合理

答案：5 个苹果 10 元
"""
```

### 8.2 Zero-shot CoT

```python
# 简单的"让我们一步步思考"就能触发 CoT

ZERO_SHOT_COT_PROMPT = """
问题：{question}

让我们一步步思考：
"""


# 这种简单的方法在很多场景下有效！
```

---

## 九、Plan-and-Execute

### 9.1 原理

```
传统 ReAct：
思考 → 行动 → 观察 → 思考 → 行动 → ...
(边规划边执行)

Plan-and-Execute：
先完整规划 → 再逐步执行
(规划与执行分离)
```

### 9.2 实现

```python
def plan_and_execute(goal, tools):
    # 阶段 1：规划
    plan = llm.generate(f"""
    目标：{goal}

    请制定一个详细的执行计划，包含所有步骤。

    计划：
    1. ...
    2. ...
    3. ...
    """)

    print("执行计划：")
    print(plan)

    # 阶段 2：执行
    steps = parse_steps(plan)
    results = []

    for i, step in enumerate(steps, 1):
        print(f"\n执行步骤 {i}/{len(steps)}: {step}")

        # 执行每个步骤
        result = execute_step(step, tools)
        results.append(result)

        print(f"结果: {result}")

    # 阶段 3：综合
    final_answer = llm.generate(f"""
    目标：{goal}

    执行计划：
    {plan}

    执行结果：
    {format_results(results)}

    请综合以上信息，给出最终答案：
    """)

    return final_answer
```

---

## 十、其他推理模式

### 10.1 ReWOO（Reasoning WithOut Observation）

```python
# 先完整规划所有推理和行动，再执行
# 优势：减少 LLM 调用次数

def rewOO_agent(goal, tools):
    # 一次性生成完整计划
    plan = llm.generate(f"""
    目标：{goal}

    可用工具：{[t.name for t in tools]}

    请制定完整的执行计划，包括：
    1. 推理步骤
    2. 需要调用的工具及参数
    3. 最终综合

    计划：
    """)

    # 解析并执行计划
    steps = parse_rewOO_plan(plan)
    tool_results = execute_steps(steps, tools)

    # 生成最终答案
    return synthesize_answer(plan, tool_results)
```

### 10.2 Self-Consistency

```python
# 生成多个推理路径，选择最一致的答案

def self_consistency(question, n=5):
    answers = []

    for _ in range(n):
        answer = llm.generate(f"问题：{question}\n\n让我们一步步思考：")
        answers.append(answer)

    # 统计最频繁的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]

    return most_common
```

### 10.3 Tree of Thoughts（ToT）

```python
# 探索多个可能的推理路径

def tree_of_thoughts(problem, max_depth=3, breadth=3):
    """树搜索推理"""

    from queue import PriorityQueue

    # 优先队列：按启发式评分排序
    queue = PriorityQueue()
    queue.put((0, [], problem))

    best_solution = None
    best_score = float('-inf')

    while not queue.empty():
        priority, path, current_state = queue.get()

        # 评估当前状态
        score = evaluate_state(current_state)

        if score > best_score:
            best_solution = (path, current_state)
            best_score = score

        # 达到最大深度
        if len(path) >= max_depth:
            continue

        # 生成多个可能的下一步
        thoughts = generate_thoughts(current_state, n=breadth)

        for thought in thoughts:
            new_state = apply_thought(current_state, thought)
            new_priority = score + estimate_priority(new_state)
            queue.put((new_priority, path + [thought], new_state))

    return best_solution
```

---

## 十一、Agent 框架对比

### 11.1 框架选型表

| 框架 | 类型 | 特点 | 学习曲线 | 适用场景 |
|:-----|:-----|:-----|:--------|:---------|
| **LangChain Agent** | 通用框架 | 功能丰富、生态成熟 | 中等 | 通用开发 |
| **LlamaIndex Agent** | 数据导向 | RAG 集成强 | 中等 | 数据密集应用 |
| **LangGraph** | 状态机 | 可控性强、可视化 | 较陡 | 复杂工作流 |
| **CrewAI** | 多Agent | 角色扮演、协作 | 简单 | 团队协作场景 |
| **AutoGen** | 多Agent | 对话式协作 | 简单 | 研究探索 |
| **Semantic Kernel** | 企业级 | 微软生态 | 中等 | 企业应用 |
| **Claude Agent SDK** | 官方SDK | 原生MCP、自动压缩 | 简单 | Claude生态 |
| **OpenAI Agents SDK** | 官方SDK | 快速上手、生产可用 | 简单 | OpenAI生态 |
| **MCP Server** | 协议标准 | 统一连接、跨平台 | 中等 | 工具集成 |

---

## 十二、LangChain Agent

> [!info] 最流行的 Agent 框架
> LangChain 1.0+ 提供统一的 `create_agent` 接口，支持所有主流 LLM 提供商。

### 12.1 核心架构

```
LangChain 生态分层
┌─────────────────────────────────────────┐
│   应用层: Deep Agents / Projects        │
├─────────────────────────────────────────┤
│   编排层: LangGraph (状态机编排)        │
├─────────────────────────────────────────┤
│   链路层: LangChain + LCEL (链式调用)  │
├─────────────────────────────────────────┤
│   监控层: LangSmith (调试/追踪/评估)   │
└─────────────────────────────────────────┘
```

### 12.2 2026 最新：统一 Agent API

```python
# 新版统一接口 (推荐)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 定义工具
@tool
def search_db(query: str, limit: int = 10) -> str:
    """搜索客户数据库"""
    return f"找到 {limit} 条关于 '{query}' 的记录"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建 Agent (统一接口)
agent = create_agent(
    model="gpt-5",  # 或 "claude-3-5-sonnet-20241022"
    tools=[search_db, calculate],
    prompt="你是一个智能助手，擅长搜索和计算"
)

# 运行
result = agent.invoke("搜索 Python 创始人，计算他 2026 年的年龄")
```

### 12.3 静态 vs 动态模型

```python
# 静态模型 (推荐简单场景)
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5")
result = model.invoke("什么是 PyCharm?")

# 动态模型 (运行时切换)
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",      # 备选模型 1
            "claude-3-5-sonnet"  # 备选模型 2
        )
    ]
)
# 主模型失败时自动切换到备选模型
```

### 12.4 中间件系统 (Middleware)

| 中间件 | 功能 | 使用场景 |
|--------|------|----------|
| **Summarization** | 自动总结对话历史 | 接近 token 限制时压缩 |
| **Human-in-the-loop** | 暂停等待人工确认 | 敏感操作前审批 |
| **Context editing** | 管理/修剪上下文 | 控制 prompt 长度 |
| **PII detection** | 检测个人隐私信息 | 数据隐私保护 |

```python
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    SummarizationMiddleware
)

agent = create_agent(
    model="gpt-5",
    tools=[search_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            require_approval_for_tools=["delete", "update"]
        ),
        SummarizationMiddleware(
            max_tokens=150000,
            summarize_threshold=0.8
        )
    ]
)
```

### 12.5 Agent 类型对比

```python
# Type 1: Tool Calling Agent (2026 推荐)
from langchain.agents import create_tool_calling_agent
# 原生支持 Function Calling
# 适合: OpenAI, Anthropic, Google 等

# Type 2: ReAct Agent
from langchain.agents import create_react_agent
# 经典 ReAct 模式 (推理 + 行动)
# 适合: 任何 LLM，不依赖特定功能

# Type 3: JSON Agent
from langchain.agents import create_json_agent
# JSON 格式工具调用
# 适合: 需要结构化输入输出的工具

# Type 4: Self-Ask with Search
from langchain.agents import create_self_ask_with_search_agent
# 自问自答 + 搜索
# 适合: 事实性问答

# Type 5: Conversational React
from langchain.agents import create_conversational_react_agent
# 支持对话记忆
# 适合: 多轮对话场景
```

### 12.6 LangChain vs LangGraph 选择

| 特性 | **LangChain** | **LangGraph** |
|:-----|---------------|---------------|
| **核心理念** | 链 (Chain) | 图 (Graph) |
| **执行模式** | 线性: A → B → C | 任意拓扑: 循环/分支/并行 |
| **状态管理** | 无状态，数据单向流动 | 有状态机，中央状态共享 |
| **适用场景** | 简单 RAG、问答、对话 | 复杂 Agent、Multi-Agent |
| **抽象级别** | 高级 API，快速开发 | 低级控制，灵活编排 |
| **学习曲线** | 低 | 中等 |

**选择建议**:
- 简单任务、快速原型 → **LangChain**
- 复杂流程、需要循环/分支 → **LangGraph**
- Multi-Agent 协作 → **LangGraph**

---

## 十三、LangGraph

> [!info] 状态机 Agent，可控性强
> LangGraph 专为构建**有状态、多步骤**的复杂工作流设计，支持循环、分支、并行执行。

### 13.1 为什么选择 LangGraph？

```
LangChain (LCEL)          LangGraph
━━━━━━━━━━━━━━━━━━━━   ━━━━━━━━━━━━━━━━━━━━

  prompt → llm → tool    ┌─────────────┐
       (线性链式)        │   Agent     │
                          │  状态机    │
                         └─────────────┘
                            /    |    \
                          分支   循环   并行
```

**核心优势**:
- ✅ **状态管理**: 中央状态对象，所有节点共享
- ✅ **循环支持**: 可实现 "思考-行动-观察" 循环
- ✅ **条件分支**: 基于状态动态路由
- ✅ **可视化**: LangGraph Studio 可视化调试
- ✅ **检查点**: 支持暂停/恢复、人工介入

### 13.2 三大核心概念

```python
# 1. 状态 (State) - 整个图的"共享内存"
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    """所有节点共享的状态"""
    messages: list           # 对话历史
    current_step: str        # 当前步骤
    should_continue: bool    # 是否继续

# 2. 节点 (Nodes) - 执行具体任务的函数
def planning_node(state: AgentState) -> dict:
    """规划节点：接收状态，返回更新"""
    plan = planner.plan(state["messages"])
    return {"current_step": plan}  # 返回需要更新的字段

def tool_node(state: AgentState) -> dict:
    """工具节点：执行工具并更新状态"""
    result = tools.execute(state["current_step"])
    return {"messages": state["messages"] + [result]}

# 3. 边 (Edges) - 定义节点间的流转
workflow = StateGraph(AgentState)
workflow.set_entry_point("planner")           # 入口边
workflow.add_edge("planner", "tools")          # 普通边
workflow.add_conditional_edges(               # 条件边
    "tools",
    lambda x: "continue" if x["should_continue"] else "end",
    {"continue": "planner", "end": END}
)
```

### 13.3 完整示例：ReAct Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.tools import tool

# 1. 定义工具
@tool
def search(query: str) -> str:
    """搜索网络信息"""
    return f"关于 '{query}' 的搜索结果"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search, calculator]
tool_node = ToolNode(tools)

# 2. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, "对话历史"]

# 3. 定义节点
def should_continue(state: AgentState) -> str:
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: AgentState, config):
    """调用 LLM 决策"""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# 4. 构建图
model = ChatOpenAI(model="gpt-5")
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

# 5. 编译并运行
app = workflow.compile()

# 支持流式输出
for event in app.stream({"messages": [("user", "今天天气怎么样？")] }):
    print(event)
```

### 13.4 检查点与记忆

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点器
memory = MemorySaver()

# 编译时添加检查点
app = workflow.compile(checkpointer=memory)

# 使用 thread_id 保持对话记忆
config = {"configurable": {"thread_id": "user-123"}}

# 第一次对话
response = app.invoke(
    {"messages": [("user", "我叫张三")]},
    config=config
)

# 第二次对话 (会记住之前的对话)
response = app.invoke(
    {"messages": [("user", "我叫什么名字？")]},
    config=config
)
# 输出: "你叫张三"
```

### 13.5 可视化与调试

```python
# 生成 Mermaid 图表
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    print(app.get_graph().print_ascii())
```

### 13.6 高级特性

#### 并行执行
```python
from langgraph.graph import Send

def map_requests(state):
    """将任务分配给多个 Agent"""
    return [Send("process_agent", {"task": t}) for t in state["tasks"]]

workflow.add_conditional_edges("router", map_requests)
```

#### 人工介入
```python
from langgraph.checkpoint.memory import MemorySaver

# 使用 interrupt 等待人工输入
def human_review_node(state):
    human_input = input("请确认: [y/n] ")
    if human_input == "y":
        return {"approved": True}
    return {"approved": False}
```

### 13.7 LangChain 节点 vs LangGraph

```python
# LangChain 节点可以在 LangGraph 中使用
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# 创建 LangChain
prompt = ChatPromptTemplate.from_template("总结: {input}")
chain = prompt | model | StrOutputParser()

# 在 LangGraph 节点中使用
def langchain_node(state):
    result = chain.invoke({"input": state["messages"]})
    return {"summary": result}

workflow.add_node("langchain_node", langchain_node)
```

---

## 十四、CrewAI & AutoGen

### 14.1 CrewAI（角色扮演 Agent）

```python
from crewai import Agent, Task, Crew

# 定义 Agent（角色）
researcher = Agent(
    role="研究员",
    goal="研究最新的 AI 技术",
    backstory="你是一位经验丰富的 AI 研究员",
    tools=[search_tool, wikipedia_tool],
    verbose=True
)

writer = Agent(
    role="技术作家",
    goal="撰写清晰的技术文章",
    backstory="你擅长将复杂的技术概念写成易懂的文章",
    tools=[calculator_tool],
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究 2024 年最重要的 AI 进展",
    agent=researcher,
    expected_output="一份详细的研究报告"
)

write_task = Task(
    description="根据研究报告撰写一篇文章",
    agent=writer,
    expected_output="一篇通俗易懂的技术文章"
)

# 创建团队并执行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True
)

result = crew.kickoff()
```

### 14.2 AutoGen（对话式 Agent）

```python
import autogen

# 定义 Agent
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="分析数据集 data.csv 并生成可视化"
)

# Agent 会自动对话完成任务
```

---

## 2026年最新 Agent 技术发展

> [!info] 2026年更新
> 本章节补充2025-2026年Agent领域的最新技术突破，包括Claude Agent SDK、MCP协议、GraphRAG等前沿技术。

---

### 一、Claude Agent SDK（2025.9发布）

#### 1.1 核心理念

```
Claude Agent SDK = "给Claude一台电脑，让Agent像人类一样工作"

关键特性：
✅ Python + TypeScript 双版本支持
✅ 原生 MCP（Model Context Protocol）集成
✅ 自动上下文压缩（接近200k token时自动总结）
✅ 支持超30小时自主编码任务
✅ 内置工具：Read、Write、Edit、Bash、Glob、Web Search
```

#### 1.2 快速上手

```python
# Python 版本
from anthropic import Anthropic
from anthropic_agents import Agent, ToolRegistry

# 创建 Agent
agent = Agent(
    model="claude-sonnet-4-20250514",
    tools=ToolRegistry.auto_discover(),  # 自动发现 MCP 工具
    max_iterations=100,
    auto_compress=True  # 启用自动上下文压缩
)

# 执行任务
result = agent.run("重构这个项目，添加测试，并优化性能")
```

```typescript
// TypeScript 版本
import { Agent } from '@anthropic-ai/sdk';
import { MCPClient } from '@anthropic-ai/mcp';

const agent = new Agent({
  model: 'claude-sonnet-4-20250514',
  tools: await MCPClient.discover(),
  maxIterations: 100,
  autoCompress: true
});

const result = await agent.run('重构项目并添加测试');
```

#### 1.3 对比其他SDK

| 特性 | Claude Agent SDK | OpenAI Agents SDK |
|:-----|:----------------|:------------------|
| 状态管理 | 客户端控制 | 客户端控制 |
| 工具执行 | 推理链内执行 | 推理链内执行 |
| MCP 支持 | ✅ 原生集成 | ✅ 支持 |
| 核心理念 | "少抽象，多授权" | "快速上手+生产可用" |
| 语言 | Python, TypeScript | Python, JavaScript |

---

### 二、MCP（Model Context Protocol）

#### 2.1 什么是MCP

```
MCP = "AI 应用的 USB-C 接口"

本质：开源标准协议，实现 LLM 与外部数据源/工具的安全双向连接

类比：
• USB-C 统一了设备连接标准
• MCP 统一了 AI Agent 工具连接标准

价值：
✅ 降低集成成本（不再为每个工具写适配器）
✅ 跨平台兼容（一次实现，到处使用）
✅ 安全可控（标准化的权限管理）
✅ 生态丰富（MCP注册表已接近2000个服务器）
```

#### 2.2 MCP架构

```
┌─────────────────────────────────────────────────────────┐
│                    MCP 架构                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
│  │   LLM    │ ←──→ │  MCP     │ ←──→ │   Tool   │      │
│  │ Agent    │      │ Client   │      │ Server   │      │
│  └──────────┘      └──────────┘      └──────────┘      │
│         ↑                                  ↑            │
│         │                                  │            │
│    Model Context                   External Data/API   │
│                                                         │
│  MCP 传输协议：                                          │
│  • stdio（标准输入输出）                                 │
│  • SSE（Server-Sent Events）                            │
│  • HTTP（REST API）                                     │
│  • WebSocket（实时双向通信）                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 2.3 MCP Server 示例

```typescript
// 创建一个简单的文件系统 MCP Server
import { MCPServer } from '@anthropic-ai/mcp';

const server = new MCPServer({
  name: 'filesystem',
  version: '1.0.0'
});

// 注册工具
server.tool('read_file', '读取文件内容', {
  type: 'object',
  properties: {
    path: { type: 'string', description: '文件路径' }
  }
}, async ({ path }) => {
  return {
    content: [{
      type: 'text',
      text: await fs.readFile(path, 'utf-8')
    }]
  };
});

// 启动服务器
server.start();
```

#### 2.4 MCP生态

```
主流平台支持：
✅ Claude（原生支持）
✅ OpenAI（支持）
✅ 微软（Copilot）
✅ 谷歌（Gemini）
✅ 亚马逊（Bedrock）
✅ 国内：百度、阿里、腾讯

MCP 注册表分类：
• 文件操作：filesystem、git、code-editor
• 数据库：postgres、mysql、mongodb、redis
• 云服务：aws、azure、gcp
• 通讯：slack、email、calendar
• 开发工具：github、jira、linear
• 数据分析：pandas、sql、visualization
```

---

### 三、GraphRAG（2025-2026年热点）

#### 3.1 什么是GraphRAG

```
GraphRAG = Knowledge Graph + RAG

微软提出，融合知识图谱与检索增强生成

核心优势：
✅ 回答准确率提升 20-50 个百分点
✅ 更好地处理复杂关系推理
✅ 减少幻觉问题
✅ 支持全局语义理解

适用场景：
• 金融（股权关系、风险传导）
• 医疗（症状-疾病-药物关系）
• 法律（案例引用、法条关联）
• 保险（理赔欺诈检测）
```

#### 3.2 GraphRAG vs 传统RAG

| 维度 | 传统RAG | GraphRAG |
|:-----|:--------|:---------|
| 检索方式 | 向量相似度 | 图谱关系遍历 |
| 上下文 | 局部文档 | 全局知识 |
| 推理能力 | 较弱 | 强（关系推理） |
| 准确率 | 基准 | +20-50% |
| 实现复杂度 | 简单 | 较复杂 |

#### 3.3 实现示例

```python
# 使用 Microsoft GraphRAG
from graphrag import GraphRAG
from graphrag.vector_store import VectorStore
from graphrag.knowledge_graph import KnowledgeGraph

# 1. 创建知识图谱
kg = KnowledgeGraph()
kg.add_entity("公司A", type="公司")
kg.add_entity("公司B", type="公司")
kg.add_relation("公司A", "投资", "公司B", amount="1亿")

# 2. 创建 GraphRAG
rag = GraphRAG(
    knowledge_graph=kg,
    vector_store=VectorStore(),
    llm="gpt-4"
)

# 3. 查询
result = rag.query("公司A投资了哪些公司？")
```

---

### 四、AgentDevOps（2026新范式）

#### 4.1 从DevOps到AgentDevOps

```
传统 DevOps：
• 关注：系统可用性
• 指标：CPU、内存、响应时间
• 目标：服务稳定运行

AgentDevOps：
• 关注：业务结果达标
• 指标：推理链路、任务完成率、成本
• 目标：Agent正确完成任务
```

#### 4.2 AgentDevOps核心要素

```
┌─────────────────────────────────────────────────────────┐
│                  AgentDevOps 框架                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 可观察性（Observability）                            │
│     • 推理链路追踪                                       │
│     • 工具调用记录                                       │
│     • 成本分析                                           │
│                                                         │
│  2. 评估（Evaluation）                                   │
│     • 任务完成率                                         │
│     • 结果质量评分                                       │
│     • 错误分类统计                                       │
│                                                         │
│  3. 调试（Debugging）                                    │
│     • 推理步骤回放                                       │
│     • 中间状态检查                                       │
│     • A/B测试                                            │
│                                                         │
│  4. 优化（Optimization）                                 │
│     • Prompt优化                                        │
│     • 工具选择优化                                       │
│     • 上下文压缩                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 五、RaaS（结果即服务）

#### 5.1 商业模式创新

```
传统 SaaS：
• 订阅制（按月/年收费）
• 用户数计费
• 功能访问权

RaaS（Result as a Service）：
• 按结果计费
• 按对话次数
• 按问题解决量

理念转变：
从"卖工具" → "卖结果"

案例：
• 客服Agent：按解决工单数计费
• 销售Agent：按成交金额抽成
• 研发Agent：按完成需求计费
```

---

### 六、Xcode + Claude Agent（2026.2）

#### 6.1 Agentic Coding

```
苹果与 Anthropic 合作：

Xcode 26.3 原生集成 Claude Agent

核心能力：
✅ 理解整个项目结构
✅ 执行复杂目标任务（跨文件重构）
✅ 视觉闭环：看得见 Xcode Previews
✅ 调用原生 API 和文档搜索
✅ 支持异步长时任务

应用场景：
• 自动生成单元测试
• 重构遗留代码
• 修复编译错误
• 生成文档注释
```

---

### 七、行业趋势与挑战

#### 7.1 2026年四大挑战

```
1. 标准碎片化
   • MCP vs OpenAPI vs 自定义协议
   • 需要行业统一标准

2. 权限管理
   • Agent访问权限边界
   • 审计追踪
   • 合规要求

3. 成本控制
   • Token消耗巨大
   • MCP连接开销
   • 需要智能缓存

4. 可靠性
   • Agent幻觉
   • 工具调用失败
   • 需要人工确认机制
```

#### 7.2 反向观点：CLI vs MCP

```
2026年2月，有观点提出"MCP泡沫破裂"：

论点：
• 80-90%日常工作流，MCP过于繁重
• CLI + 强大推理模型可能是更好选择

理由：
✅ CLI工具（bash, git, rg, grep）更成熟可靠
✅ 前沿模型对Shell使用进行了大量训练
✅ MCP消耗大量Token
✅ 对于简单任务，CLI更快更直接

结论：
• 复杂集成用MCP
• 简单任务用CLI
• 根据场景选择
```

#### 7.3 市场数据

```
市场规模：
• 2024年：51亿美元
• 2030年预测：471亿美元
• 复合年增长率：44.8%

中国市场（2025上半年）：
• 融资总额：超80亿元人民币
• 主要投资方向：
  - 垂直行业Agent
  - 企业级平台
  - 开发者工具
```

---

### 八、A2A协议（Agent to Agent）

#### 8.1 什么是A2A

```
A2A = Agent to Agent Protocol

Google发布，专为AI Agent之间互操作性设计

目标：
• 让不同Agent相互协作
• 标准化Agent间通信
• 构建Agent生态系统

类比：
• HTTP让Web浏览器相互通信
• A2A让Agent相互通信
```

#### 8.2 A2A应用场景

```
多Agent协作：
• 旅行Agent + 预订Agent + 支付Agent
• 分析Agent + 可视化Agent + 报告Agent
• 监控Agent + 诊断Agent + 修复Agent

跨平台协作：
• Claude Agent + OpenAI Agent
• 不同公司的Agent互相调用
```

---

## 十五、单 Agent vs 多 Agent

### 15.1 对比

| 维度 | 单 Agent | 多 Agent |
|:-----|:--------|:---------|
| **复杂度** | 简单 | 复杂 |
| **协作能力** | 无 | 强 |
| **专业性** | 通用 | 专业分工 |
| **成本** | 低 | 高 |
| **可靠性** | 中 | 高 |
| **适用场景** | 简单任务 | 复杂项目 |

---

## 十六、Multi-Agent 协作模式

### 16.1 协作模式

```
┌─────────────────────────────────────────────────────────┐
│              Multi-Agent 协作模式                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 层级模式（Hierarchical）                             │
│     Manager Agent                                       │
│        ├─ Worker Agent 1                                │
│        ├─ Worker Agent 2                                │
│        └─ Worker Agent 3                                │
│                                                         │
│  2. 对话模式（Conversational）                          │
│     Agent A ↔ Agent B ↔ Agent C                         │
│     (类似 AutoGen)                                      │
│                                                         │
│  3. 竞争模式（Adversarial）                             │
│     Generator Agent ↔ Critic Agent                      │
│     (互相挑战、改进)                                     │
│                                                         │
│  4. 协作模式（Collaborative/CrewAI）                     │
│     各 Agent 扮演不同角色，共同完成目标                  │
│                                                         │
│  5. 顺序模式（Sequential）                               │
│     Agent 1 → Agent 2 → Agent 3                         │
│     (流水线式处理)                                       │
│                                                         │
│  6. 跨平台协作（A2A Protocol）                          │
│     不同平台的Agent通过A2A协议相互调用                   │
│     (Claude Agent ↔ OpenAI Agent)                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 16.2 A2A协议（Agent to Agent）

> [!info] 2026年新增
> **A2A（Agent to Agent）协议**由Google发布，专为AI Agent之间的互操作性设计，被称为"Agent界的HTTP"。

#### 核心概念

```
A2A协议 = 让Agent互相通信的标准协议

类比：
• HTTP = Web浏览器与服务器之间的协议
• A2A = Agent与Agent之间的协议

解决的问题：
✅ 不同厂商的Agent如何协作
✅ Agent发现与注册
✅ 消息格式标准化
✅ 权限与安全控制
```

#### A2A架构

```
┌─────────────────────────────────────────────────────────┐
│                    A2A 协议架构                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │ Claude Agent │ ←→  │  A2A Protocol│  ←→           │
│  └──────────────┘      │   (Layer)    │               │
│                        └──────────────┘               │
│                              ↑                         │
│                        ┌──────┴──────┐                │
│                        ↓             ↓                │
│                 ┌──────────┐  ┌──────────┐            │
│                 │OpenAI    │  │Google    │            │
│                 │Agent     │  │Agent     │            │
│                 └──────────┘  └──────────┘            │
│                                                         │
│  A2A消息格式：                                          │
│  {                                                      │
│    "from": "agent-id",                                  │
│    "to": "target-agent-id",                             │
│    "action": "task/request/response",                   │
│    "payload": { ... },                                  │
│    "context": { ... }                                   │
│  }                                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 应用场景

```
跨平台协作示例：

场景1：旅行规划
用户 → 旅行Agent
      ↓ (A2A)
      预订Agent (机票)
      ↓ (A2A)
      预订Agent (酒店)
      ↓ (A2A)
      支付Agent

场景2：软件开发
用户 → 需求分析Agent
      ↓ (A2A)
      设计Agent
      ↓ (A2A)
      编码Agent (Claude)
      ↓ (A2A)
      测试Agent (OpenAI)
      ↓ (A2A)
      部署Agent

场景3：数据分析
用户 → 数据收集Agent
      ↓ (A2A)
      清洗Agent
      ↓ (A2A)
      分析Agent
      ↓ (A2A)
      可视化Agent
```

#### A2A vs 其他集成方式

| 对比维度 | A2A协议 | MCP | 直接API调用 |
|:--------|:--------|:-----|:------------|
| 目的 | Agent间通信 | Agent-工具连接 | 点对点集成 |
| 标准化 | ✅ 行业标准 | ✅ 行业标准 | ❌ 自定义 |
| 发现机制 | ✅ Agent注册表 | ✅ MCP注册表 | ❌ 需手动配置 |
| 安全控制 | ✅ 标准化 | ✅ 标准化 | ⚠️ 各自实现 |
| 适用场景 | Agent协作 | 工具集成 | 简单场景 |

#### 实现示例

```python
# 使用A2A协议的示例
from a2a_protocol import Agent, MessageBus

# 创建Agent
travel_agent = Agent(
    id="travel-agent-001",
    capabilities=["plan_trip", "book_flight"],
    protocol="a2a-v1.0"
)

flight_agent = Agent(
    id="flight-agent-001",
    capabilities=["search_flights", "book_flights"],
    protocol="a2a-v1.0"
)

# 注册到消息总线
bus = MessageBus()
bus.register(travel_agent)
bus.register(flight_agent)

# Agent间通信
response = travel_agent.send_to(
    target="flight-agent-001",
    action="search_flights",
    payload={
        "from": "PEK",
        "to": "SHA",
        "date": "2026-03-15"
    }
)
```

### 16.3 2026年多Agent发展趋势

```
行业趋势：

1. 从单Agent到多Agent协作
   • 单Agent：适合简单任务
   • 多Agent：适合复杂项目（主流）

2. 从同构到异构协作
   • 同构：相同LLM的多个Agent
   • 异构：不同LLM（Claude + GPT）协作

3. 从集中式到分布式编排
   • 集中式：中央调度器
   • 分布式：Agent自治协作

4. 企业级多Agent平台
   • 亚马逊：Bedrock Multi-Agent
   • 微软：Autogen Studio
   • 谷歌：A2A Framework
   • 国内：百度千帆、阿里百炼
```

---

## 十七、Autonomous Agent

### 17.1 特征

```
Autonomous Agent = 高度自主的 Agent

特征：
✅ 自主设定目标
✅ 长期运行
✅ 自我改进
✅ 环境适应
✅ 无需持续监督

代表项目：
• AutoGPT
• BabyAGI
• AgentGPT
• Devin（AI 软件工程师）
```

---

## 十八、应用场景

### 18.1 应用全景

```
┌─────────────────────────────────────────────────────────┐
│                   Agent 应用场景                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  办公自动化                                              │
│  ├─ 邮件分类和回复                                       │
│  ├─ 日程安排和提醒                                       │
│  ├─ 文档生成和总结                                       │
│  └─ 数据录入和整理                                       │
│                                                         │
│  开发工具                                                │
│  ├─ 代码生成和补全（Cursor, Copilot）                    │
│  ├─ 代码审查和重构                                       │
│  ├─ Bug 修复                                             │
│  └─ 文档生成                                             │
│                                                         │
│  研究助手                                                │
│  ├─ 文献搜索和总结                                       │
│  ├─ 实验设计和分析                                       │
│  ├─ 数据分析                                             │
│  └─ 报告撰写                                             │
│                                                         │
│  客服系统                                                │
│  ├─ 问题解答                                             │
│  ├─ 工单处理                                             │
│  ├─ 情感识别                                             │
│  └─ 人工转接                                             │
│                                                         │
│  教育辅导                                                │
│  ├─ 个性化学习                                           │
│  ├─ 作业批改                                             │
│  ├─ 答疑解惑                                             │
│  └─ 学习进度跟踪                                         │
│                                                         │
│  金融分析                                                │
│  ├─ 市场研究                                             │
│  ├─ 风险评估                                             │
│  ├─ 投资建议                                             │
│  └─ 财报分析                                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 18.2 2026年新兴应用场景

> [!info] 最新应用
> 2025-2026年涌现的新一代Agent应用场景

#### 智能编程助手

```
Xcode + Claude Agent（2026.2）

能力升级：
✅ 视觉闭环：Agent"看得见"Xcode Previews界面
✅ 自动生成单元测试（覆盖率>90%）
✅ 跨文件重构（理解整个项目结构）
✅ 修复编译错误并解释原因
✅ 生成API文档注释

实际案例：
• 独立开发者：3人周 → 1人天（重构遗留代码）
• 初级开发者：效率提升200%
• 代码审查：节省80%时间
```

#### GraphRAG应用

```
金融领域：股权穿透分析

传统RAG问题：
• 无法理解复杂的股权关系
• 对"A公司的实际控制人是谁"这类问题回答不准

GraphRAG解决方案：
• 构建股权关系图谱
• 支持多层穿透查询
• 风险传导路径分析

效果：
• 准确率从60% → 95%
• 支持实时监管报送
```

```
医疗领域：诊疗辅助

应用：
• 症状-疾病-药物关系图谱
• 药物相互作用检测
• 个性化治疗方案推荐

效果：
• 诊断准确率提升30%
• 减少药物不良反应
```

#### 企业级多Agent协作

```
场景：电商订单处理

Agent协作流程：
┌──────────────┐
│ 订单接收Agent│
└──────┬───────┘
       │
       ├─→ 库存检查Agent ─→ [OK] ─┐
       │                           │
       ├─→ 风控检测Agent ────→ [OK]┤
       │                           │
       └─→ 价格计算Agent ───→ [OK]┘
                                   │
                            ┌──────┴───────┐
                            │ 订单确认Agent │
                            └──────────────┘

效果：
• 订单处理时间：从5分钟 → 10秒
• 自动化率：95%
• 人力成本：降低70%
```

#### 知识管理Agent

```
企业知识库 + MCP连接

功能：
✅ 自动分类文档
✅ 智能问答（准确率92%）
✅ 知识图谱构建
✅ 跨系统搜索（邮件、文档、IM）

技术栈：
• Claude 4.5（理解能力）
• MCP（连接企业系统）
• GraphRAG（知识图谱）

效果：
• 信息检索时间：从30分钟 → 5秒
• 知识复用率：提升3倍
```

#### Agentic DevOps

```
软件开发全流程Agent化

需求分析Agent
    ↓
设计Agent
    ↓
编码Agent（Claude + Xcode）
    ↓
测试Agent
    ↓
部署Agent
    ↓
监控Agent

效果：
• 开发周期：缩短60%
• 代码质量：Bug减少40%
• 发布频率：从每月 → 每天
```

#### 教育个性化Agent

```
应用：K12个性化辅导

功能：
• 学习路径规划
• 薄弱点诊断
• 个性化习题推荐
• 学习进度跟踪
• 家长报告生成

技术：
• Multi-Agent（学科Agent + 心理Agent）
• 长期记忆（学生学习记录）
• GraphRAG（知识点关联）

效果：
• 学习效率：提升50%
• 学生参与度：提升80%
• 教师工作量：降低40%
```

### 18.3 行业应用统计（2026）

```
Agent应用渗透率：

行业           | 2024 | 2026预测
--------------|------|----------
软件开发       | 35%  | 75%
客户服务       | 20%  | 60%
金融分析       | 15%  | 50%
内容创作       | 25%  | 70%
教育培训       | 10%  | 40%
医疗健康       | 5%   | 25%
制造业         | 8%   | 30%

最成功的应用场景（按ROI排序）：
1. 智能客服（ROI: 300%）
2. 代码助手（ROI: 250%）
3. 文档自动化（ROI: 200%）
4. 数据分析（ROI: 180%）
5. 内容生成（ROI: 150%）
```

---

## 十九、最佳实践

### 19.1 开发原则

```
Agent 开发十大原则：

1. 明确目标：清楚地定义 Agent 要解决的问题

2. 简单开始：从简单功能开始，逐步添加复杂度

3. 工具优先：提供高质量、描述清晰的工具

4. 限制循环：设置最大迭代次数，防止无限循环

5. 错误处理：优雅地处理工具调用失败

6. 可观察性：记录 Agent 的决策过程

7. 测试驱动：编写测试用例验证行为

8. 人机协作：设计人工介入的机制

9. 安全第一：限制 Agent 的权限和访问范围

10. 持续优化：根据反馈不断改进
```

### 19.2 推荐技术栈

```
生产环境推荐：

LLM:
• Claude 4.5（推理能力强）
• GPT-4（功能全面）
• DeepSeek-R1（性价比高）

框架:
• LangChain（通用场景）
• LangGraph（复杂工作流）
• CrewAI（多 Agent 协作）

工具:
• Tavily（搜索）
• Exa（AI 搜索）
• Python REPL（代码执行）

向量数据库:
• Qdrant（长期记忆）

部署:
• LangServe（服务化）
• LangSmith（监控调试）
```

---

## 二十、评估与调试

### 20.1 评估指标

| 指标 | 说明 | 目标值 |
|:-----|:-----|:------|
| **任务成功率** | 成功完成任务的百分比 | >80% |
| **平均迭代次数** | 完成任务的平均循环次数 | 越少越好 |
| **工具调用准确率** | 正确调用工具的比例 | >90% |
| **响应时间** | 完成任务的总时间 | <30秒 |
| **成本** | 每次 Agent 运行的成本 | 可接受范围 |

### 20.2 调试技巧

```python
# 1. 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. 使用 LangSmith 追踪
from langsmith import trace
@trace
def my_agent(...):
    ...

# 3. 记录决策过程
class TracedAgent:
    def __init__(self):
        self.trace = []

    def run(self, goal):
        self.trace.append({"event": "start", "goal": goal})
        # ...
```

---

## 二十一、完整知识地图

### 21.1 Agent 知识体系

```
                    Agent 全面解析 (本文件)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
      基础原理           架构模式           实战应用
         │                 │                 │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │         │      │         │      │         │
  四大组件   推理模式   单Agent   多Agent  框架实现
    │         │      │         │      │         │
  Planner   ReAct    ReAct   CrewAI  LangChain
  ToolUse   CoT      LangGraph AutoGen LangGraph
  Memory   ToT
Reflection
```

### 21.2 学习路径

```
第一步：理解基础
  → 本文件的一至六章
  → 理解 Agent 是什么、为什么、怎么做

第二步：核心模式
  → 本文件的七至十章
  → 掌握 ReAct、CoT 等推理模式

第三步：框架实战
  → LangChain Agent
  → 动手实现一个简单的 Agent

第四步：进阶架构
  → LangGraph
  → Multi-Agent 系统

第五步：实战项目
  → 构建一个完整的 Agent 应用
  → 部署和优化
```

---

## 二十二、相关笔记索引

### 22.1 相关技术

| 笔记 | 说明 |
|:-----|:-----|
| [[AI研究/AI学习/03-实战应用/RAG全面解析]] | RAG 检索增强生成 |
| [[AI研究/AI学习/02-模型原理/Transformer全面解析]] | LLM 基础架构 |

### 22.2 学习导航

| 笔记 | 说明 |
|:-----|:-----|
| [[AI研究/AI学习/00-知识库索引]] | 知识库导航中心 |
| [[AI研究/AI学习/AI模型系统性学习路径]] | 完整学习路线 |

---

## 二十三、快速参考

### 23.1 核心要点

```
Agent = LLM + 规划 + 工具 + 记忆 + 反思

四大组件：
  Planner（规划器）：分解任务、制定计划
  Tool Use（工具使用）：调用外部工具和 API
  Memory（记忆系统）：存储和检索信息
  Reflection（反思机制）：评估和改进

核心公式：
  高质量 Agent = 清晰目标 + 优质工具 + 合理架构 + 持续优化

最佳实践：
  ReAct 模式 + LangChain + 好的工具集 + 反思机制

常见错误：
  ❌ 工具描述不清楚
  ❌ 没有循环限制
  ❌ 缺少错误处理
  ❌ 没有记忆机制
```

### 23.2 一句话总结

> **Agent 让 LLM 能够自主规划和执行任务，就像雇了一个智能助手，能思考、能行动、能记忆、能改进。**

---

## 二十四、LangChain 学习资源（2026）

### 24.1 官方资源

| 资源 | 链接 | 说明 |
|:-----|:-----|:-----|
| **官方文档 (Python)** | https://python.langchain.com/ | 最新完整文档 |
| **中文文档** | https://python.langchain.cn/ | 中文翻译版本 |
| **LangGraph 文档** | https://langchain-ai.github.io/langgraph/ | 状态图框架 |
| **LangSmith** | https://docs.smith.langchain.com/ | 调试/监控/评估 |
| **GitHub 仓库** | https://github.com/langchain-ai/langchain | 源代码 |

### 24.2 中文资源

| 资源 | 链接 |
|:-----|:-----|
| **LangChain 中文网** | https://www.langchain.com.cn/ |
| **500页中文教程** | https://www.bandianxiang.com/info/uQvi66O |
| **CSDN 教程合集** | 搜索 "LangChain" |
| **Cookbook 示例** | https://cookbook.langchain.com.cn/ |

### 24.3 学习路径

```
第一阶段：基础概念（1-2周）
├── 1. 安装与环境配置
├── 2. Models I/O - LLM 调用
├── 3. Prompts - 提示词模板
└── 4. Memory - 记忆机制

第二阶段：LCEL 语法（2-3周）
├── 1. Chain 链式调用
├── 2. Runnable 接口
├── 3. 自定义 Runnable
└── 4. 路由与分支

第三阶段：Agents（3-4周）
├── 1. Tools 工具定义
├── 2. Agent 类型
├── 3. AgentExecutor
└── 4. 自定义 Agent

第四阶段：LangGraph（4-6周）
├── 1. State 状态管理
├── 2. Nodes 与 Edges
├── 3. 检查点机制
└── 4. Multi-Agent 编排

第五阶段：生产实践
├── 1. LangSmith 监控
├── 2. 性能优化
├── 3. 错误处理
└── 4. 部署方案
```

### 24.4 实战项目推荐

```python
# 项目 1: 简单问答 Agent
# 技能: LLM 调用、Prompt 模板
# 时间: 1-2 天

# 项目 2: 带 RAG 的知识库问答
# 技能: 文档加载、向量化、检索
# 时间: 3-5 天

# 项目 3: ReAct Agent (工具调用)
# 技能: Tools、Agent、工具定义
# 时间: 5-7 天

# 项目 4: Multi-Agent 协作系统
# 技能: LangGraph、状态管理
# 时间: 1-2 周

# 项目 5: 完整的客服 Agent
# 技能: 全栈、记忆、知识库
# 时间: 2-3 周
```

### 24.5 社区资源

| 资源 | 链接 |
|:-----|:-----|
| **Discord 社区** | https://discord.gg/langchain |
| **YouTube 官方频道** | https://www.youtube.com/@LangChain |
| **LangChainHub** | https://smith.langchain.com/hub |
| **Twitter/X** | @LangChainAI |

### 24.6 2026 新特性速览

```python
# 新版统一 Agent API
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5",
    tools=[search_tool],
    middleware=[SummarizationMiddleware()]
)

# 新版 LCEL 语法
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest")
chain = prompt | model | parser

# LangGraph 预构建 Agent
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools, checkpointer=memory)
```

---

#Agent #智能体 #LLM #MultiAgent #LangChain #CrewAI #AutoGen
