---
title: LangChain vs LangGraph 对比分析
date: 2026-03-01
tags:
  - LangChain
  - LangGraph
  - 框架对比
  - 技术选型
cssclass: comparison-page
status: active
---

# LangChain vs LangGraph 对比分析

> [!info] 核心问题
> **我应该使用 LangChain 还是 LangGraph？**
> 两者不是竞争关系，而是互补关系。选择取决于你的应用复杂度和需求。

> [!tip] 相关笔记
> - [[AI研究/AI学习/03-实战应用/LangChain全面解析]] - LangChain 详解
> - [[AI研究/AI学习/03-实战应用/LangGraph全面解析]] - LangGraph 详解
> - [[AI研究/AI学习/03-实战应用/Agent全面解析]] - Agent 详解

---

## 📑 目录

- [[#一、核心区别一图看懂]]
- [[#二、详细对比表]]
- [[#三、架构对比]]
- [[#四、代码对比]]
- [[#五、场景选择]]
- [[#六、混合使用]]
- [[#七、迁移指南]]
- [[#八、性能对比]]

---

## 一、核心区别一图看懂

### 1.1 可视化对比

```
LangChain (LCEL)                    LangGraph
━━━━━━━━━━━━━━━━━━               ━━━━━━━━━━━━━━━━━━

    线性链式流程                          图结构流程

┌─────┐    ┌─────┐    ┌─────┐          ┌──────────┐
│Prompt│ → │ LLM │ → │Parser│          │          │
└─────┘    └─────┘    └─────┘          │   Agent  │
                                        │  状态机   │
    A → B → C → D                       │          │
  (单向流动)                          └────┬─────┘
                                          │    │
                              ┌───────────┘    └──────────┐
                              │                              │
                         ┌────┴────┐                    ┌───┴───┐
                         │ Tool A  │                    │Tool B │
                         └─────────┘                    └───────┘
                              │                              │
                              └───────────┬──────────────────┘
                                          │
                                   ┌──────┴──────┐
                                   │  继续或结束  │
                                   └─────────────┘

     无状态，数据单向传递                  有状态，中央状态共享
```

### 1.2 核心差异

| 维度 | **LangChain (LCEL)** | **LangGraph** |
|:-----|:---------------------|:--------------|
| **核心抽象** | 链 (Chain) | 图 (Graph) |
| **执行模式** | 线性: A → B → C | 任意拓扑: 循环/分支/并行 |
| **状态管理** | 无状态 | 有状态机 (State Machine) |
| **控制流** | 隐式 (通过 \| 操作符) | 显式 (节点和边) |
| **学习曲线** | 低 | 中等 |
| **调试难度** | 中等 | 低 (可视化) |
| **适用场景** | 简单工作流 | 复杂 Agent 系统 |

---

## 二、详细对比表

### 2.1 功能对比

| 功能 | LangChain | LangGraph |
|:-----|:----------|:----------|
| **顺序执行** | ✅ 原生支持 | ✅ 支持 |
| **条件分支** | ⚠️ 需要 RunnableBranch | ✅ 原生条件边 |
| **循环** | ❌ 不支持 | ✅ 原生支持 |
| **并行执行** | ⚠️ RunnableParallel | ✅ 原生支持 |
| **状态持久化** | ❌ 需要外部实现 | ✅ 内置检查点 |
| **人工介入** | ❌ 不支持 | ✅ interrupt 机制 |
| **可视化** | ❌ 无 | ✅ LangGraph Studio |
| **时间旅行** | ❌ 不支持 | ✅ 检查点回溯 |

### 2.2 代码风格对比

```python
# ═══════════════════════════════════════════════
# LangChain (LCEL) - 简洁声明式
# ═══════════════════════════════════════════════
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("问题")


# ═══════════════════════════════════════════════
# LangGraph - 显式状态机
# ═══════════════════════════════════════════════
from langgraph.graph import StateGraph, END

class State(TypedDict):
    messages: list
    context: str

def retrieve_node(state: State):
    docs = retriever.invoke(state["messages"][-1])
    return {"context": format_docs(docs)}

def generate_node(state: State):
    response = llm.invoke([
        SystemMessage(content=system_prompt.format(context=state["context"])),
        state["messages"][-1]
    ])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
result = app.invoke({"messages": [("user", "问题")]})
```

### 2.3 复杂度对比

| 场景 | LangChain 代码量 | LangGraph 代码量 | 复杂度 |
|:-----|:----------------|:----------------|:-------|
| 简单 RAG | ~10 行 | ~40 行 | LangChain 更简单 ✅ |
| 带循环的 Agent | 不支持 | ~50 行 | LangGraph 必需 ✅ |
| 条件分支 | ~20 行 | ~30 行 | 相当 |
| Multi-Agent | ~100 行 | ~80 行 | LangGraph 更清晰 ✅ |
| 需要暂停/恢复 | 需要外部存储 | ~5 行 | LangGraph 更简单 ✅ |

---

## 三、架构对比

### 3.1 LangChain 架构

```
┌─────────────────────────────────────────────────────────┐
│                    LangChain 架构                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │            LCEL Expression                 │         │
│  │                                          │         │
│  │   Runnable  →  Runnable  →  Runnable      │         │
│  │   (Prompt)   (LLM)      (Parser)          │         │
│  └────────────────────────────────────────────┘         │
│                    ↓                                     │
│  ┌────────────────────────────────────────────┐         │
│  │              Components                    │         │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────┐ │         │
│  │  │  Models  │  │ Prompts  │  │ Chains  │ │         │
│  │  └──────────┘  └──────────┘  └─────────┘ │         │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────┐ │         │
│  │  │  Memory  │  │  Agents  │  │Retrievers│ │         │
│  │  └──────────┘  └──────────┘  └─────────┘ │         │
│  └────────────────────────────────────────────┘         │
│                    ↓                                     │
│  ┌────────────────────────────────────────────┐         │
│  │              Integrations                  │         │
│  │  OpenAI / Anthropic / Google / ...        │         │
│  └────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘

特点:
• 线性数据流
• 组件通过 | 操作符连接
• 数据单向传递，无状态
```

### 3.2 LangGraph 架构

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph 架构                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│                    ┌─────────────┐                      │
│                    │    State    │                      │
│                    │  (状态中心)  │                      │
│                    └──────┬──────┘                      │
│                           │                             │
│          ┌────────────────┼────────────────┐            │
│          │                │                │            │
│     ┌────┴────┐      ┌───┴────┐      ┌───┴────┐         │
│     │  Node 1 │      │ Node 2 │      │ Node 3 │         │
│     │  (规划)  │  →   │ (工具)  │  →   │ (反思)  │         │
│     └─────────┘      └────────┘      └────────┘         │
│          │                │                │             │
│          └────────────────┼────────────────┘             │
│                           │                             │
│                    ┌──────┴──────┐                      │
│                    │  Checkpoint │                      │
│                    │  (检查点)    │                      │
│                    └─────────────┘                      │
│                           │                             │
│                    ┌──────┴──────┐                      │
│                    │ Conditional │                      │
│                    │    Edges    │                      │
│                    │ (条件边)     │                      │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘

特点:
• 中央状态对象
• 节点间共享和修改状态
• 支持循环、分支、并行
• 可视化调试
```

---

## 四、代码对比

### 4.1 场景一：简单问答

**需求**: 用户问题 → 检索文档 → LLM 回答

```python
# ═══════════════════════════════════════════════
# LangChain - 推荐 ✅ 更简洁
# ═══════════════════════════════════════════════

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "根据上下文回答: {context}\n问题: {question}"
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("什么是 LangChain?")


# ═══════════════════════════════════════════════
# LangGraph - 过度设计 ❌
# ═══════════════════════════════════════════════

from langgraph.graph import StateGraph, END

class State(TypedDict):
    question: str
    context: str
    answer: str

def retrieve_node(state: State):
    docs = retriever.invoke(state["question"])
    return {"context": format_docs(docs)}

def generate_node(state: State):
    response = llm.invoke(f"根据上下文回答: {state['context']}\n问题: {state['question']}")
    return {"answer": response.content}

workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")

app = workflow.compile()
result = app.invoke({"question": "什么是 LangChain?"})
```

**结论**: 简单场景用 **LangChain**

---

### 4.2 场景二：需要循环的 Agent

**需求**: Agent 思考 → 行动 → 观察 → 判断是否继续

```python
# ═══════════════════════════════════════════════
# LangChain - 不支持 ❌ 无法实现循环
# ═══════════════════════════════════════════════

# AgentExecutor 内部有循环，但用户无法控制
# 无法实现:
# - 自定义循环逻辑
# - 循环内条件判断
# - 状态持久化


# ═══════════════════════════════════════════════
# LangGraph - 完美支持 ✅ 原生循环
# ═══════════════════════════════════════════════

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, "对话历史"]
    iteration_count: int

def agent_node(state: AgentState):
    """LLM 决策节点"""
    response = model.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """工具执行节点"""
    tool_calls = state["messages"][-1].tool_calls
    results = [execute_tool(tc) for tc in tool_calls]
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    """判断是否继续"""
    # 最多执行 5 次
    if state["iteration_count"] >= 5:
        return "end"

    # 检查是否有工具调用
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"

    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# 执行时会循环: agent → tools → agent → tools → ... → END
```

**结论**: 需要循环用 **LangGraph**

---

### 4.3 场景三：条件分支

**需求**: 根据用户查询类型，选择不同的处理流程

```python
# ═══════════════════════════════════════════════
# LangChain - RunnableBranch
# ═══════════════════════════════════════════════

from langchain_core.runnables import RunnableBranch

def classify_input(input_text: str) -> str:
    """分类输入"""
    if "代码" in input_text:
        return "code"
    elif "写作" in input_text:
        return "writing"
    else:
        return "general"

chain = RunnableBranch(
    (lambda x: classify_input(x) == "code", code_chain),
    (lambda x: classify_input(x) == "writing", writing_chain),
    general_chain
)

result = chain.invoke("写一段 Python 代码")


# ═══════════════════════════════════════════════
# LangGraph - 条件边
# ═══════════════════════════════════════════════

def route_function(state: State) -> str:
    input_text = state["messages"][-1].content

    if "代码" in input_text:
        return "code_agent"
    elif "写作" in input_text:
        return "writing_agent"
    else:
        return "general_agent"

workflow = StateGraph(State)
workflow.add_node("router", route_node)
workflow.add_node("code_agent", code_agent_node)
workflow.add_node("writing_agent", writing_agent_node)
workflow.add_node("general_agent", general_agent_node)

workflow.add_conditional_edges(
    "router",
    route_function,
    {
        "code_agent": "code_agent",
        "writing_agent": "writing_agent",
        "general_agent": "general_agent"
    }
)

app = workflow.compile()
```

**结论**: 两者都支持，**LangGraph** 的条件边更清晰

---

### 4.4 场景四：Multi-Agent 协作

**需求**: 多个 Agent 协作完成复杂任务

```
用户请求
    ↓
分发器 Agent
    ├─→ 研究 Agent ──→ 研究结果
    ├─→ 写作 Agent ──→ 草稿
    └─→ 审核 Agent ──→ 审核意见
         ↓
    汇总 Agent
         ↓
      最终结果
```

```python
# ═══════════════════════════════════════════════
# LangChain - 复杂且难维护 ❌
# ═══════════════════════════════════════════════

# 需要手动管理:
# - Agent 间的通信
# - 结果的合并
# - 错误处理
# - 状态同步
# 代码量: ~150+ 行


# ═══════════════════════════════════════════════
# LangGraph - 清晰且易维护 ✅
# ═══════════════════════════════════════════════

class MultiAgentState(TypedDict):
    messages: list
    research_result: str
    draft: str
    review: str
    final_output: str

def dispatcher_node(state: MultiAgentState):
    """分发任务给各个 Agent"""
    task = state["messages"][-1].content
    # 触发并行执行
    return {"task": task}

def research_agent_node(state: MultiAgentState):
    """研究 Agent"""
    result = research_agent.invoke(state["task"])
    return {"research_result": result}

def writer_agent_node(state: MultiAgentState):
    """写作 Agent"""
    result = writer_agent.invoke(
        f"基于研究结果: {state['research_result']}"
    )
    return {"draft": result}

def reviewer_agent_node(state: MultiAgentState):
    """审核 Agent"""
    result = reviewer_agent.invoke(state["draft"])
    return {"review": result}

def synthesizer_node(state: MultiAgentState):
    """汇总 Agent"""
    final = synthesizer_agent.invoke(
        f"研究: {state['research_result']}\n"
        f"草稿: {state['draft']}\n"
        f"审核: {state['review']}"
    )
    return {"final_output": final}

workflow = StateGraph(MultiAgentState)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("writer_agent", writer_agent_node)
workflow.add_node("reviewer_agent", reviewer_agent_node)
workflow.add_node("synthesizer", synthesizer_node)

# 分发后并行执行
workflow.add_conditional_edges(
    "dispatcher",
    lambda x: "parallel",
    {
        "research": "research_agent",
        "write": "writer_agent",
        "review": "reviewer_agent"
    }
)

# 所有 Agent 完成后汇总
workflow.add_edge(["research_agent", "writer_agent", "reviewer_agent"], "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()
```

**结论**: Multi-Agent 用 **LangGraph**

---

## 五、场景选择

### 5.1 决策树

```
开始
  │
  ├─ 需要循环？
  │   ├─ 是 → LangGraph ✅
  │   └─ 否 ↓
  │
  ├─ 需要 Multi-Agent？
  │   ├─ 是 → LangGraph ✅
  │   └─ 否 ↓
  │
  ├─ 需要人工介入/暂停恢复？
  │   ├─ 是 → LangGraph ✅
  │   └─ 否 ↓
  │
  ├─ 需要可视化调试？
  │   ├─ 是 → LangGraph ✅
  │   └─ 否 ↓
  │
  └─ 简单线性流程？
      ├─ 是 → LangChain ✅
      └─ 否 ↓
          LangGraph（可扩展性）
```

### 5.2 使用场景矩阵

| 场景 | 推荐框架 | 原因 |
|:-----|:---------|:-----|
| **简单问答** | LangChain | 代码简洁，学习曲线低 |
| **RAG 应用** | LangChain | LCEL 天然适配 |
| **聊天机器人** | LangChain | 记忆机制完善 |
| **单 Agent** | LangChain | AgentExecutor 够用 |
| **带循环的 Agent** | LangGraph | 原生支持循环 |
| **Multi-Agent** | LangGraph | 状态管理清晰 |
| **需要人工确认** | LangGraph | interrupt 机制 |
| **长时间任务** | LangGraph | 检查点持久化 |
| **复杂决策流程** | LangGraph | 条件边灵活 |
| **需要可视化** | LangGraph | LangGraph Studio |
| **快速原型** | LangChain | 开发速度快 |
| **生产级系统** | LangGraph | 可控性强 |

### 5.3 典型应用案例

**LangChain 最佳实践**:
```
✅ 文档问答系统
✅ 客服聊天机器人
✅ 内容摘要工具
✅ 简单的 AI 应用
✅ API 集成工具
```

**LangGraph 最佳实践**:
```
✅ 自主研究 Agent
✅ 代码编写 Agent
✅ 任务管理系统
✅ Multi-Agent 协作
✅ 工作流自动化
✅ 需要人工审核的流程
```

---

## 六、混合使用

### 6.1 两者可以协同工作

```
LangChain 节点可以在 LangGraph 中使用

┌─────────────────────────────────────┐
│         LangGraph 工作流            │
│                                     │
│  ┌─────────┐    ┌─────────┐        │
│  │ Node A  │ → │ Node B  │        │
│  │(规划)   │    │(LangChain)│     │
│  └─────────┘    └────┬────┘        │
│                      │              │
│                      │  内部使用    │
│                      │  LCEL 构建   │
│                      │  的链        │
│                      ↓              │
│              prompt│llm│parser     │
│                      │              │
│               ┌──────┴──────┐       │
│               │   Node C    │       │
│               │  (反思)     │       │
│               └─────────────┘       │
└─────────────────────────────────────┘
```

### 6.2 混合使用示例

```python
# 在 LangGraph 节点中使用 LangChain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 用 LangChain 创建一个 RAG 链
rag_prompt = ChatPromptTemplate.from_template(
    "上下文: {context}\n问题: {input}"
)
rag_chain = create_retrieval_chain(
    retriever,
    rag_prompt | llm
)

# 在 LangGraph 中使用
def rag_node(state: State):
    # 调用 LangChain 链
    result = rag_chain.invoke({"input": state["question"]})
    return {"answer": result["answer"]}

workflow = StateGraph(State)
workflow.add_node("rag", rag_node)
# ... 其他节点
```

### 6.3 最佳实践

```python
# ═══════════════════════════════════════════════
# 推荐模式: LangGraph 作为编排层
# ═══════════════════════════════════════════════

# 1. 用 LangChain 构建原子功能
retrieval_chain = retriever | format_docs
generation_chain = prompt | llm | parser
analysis_chain = another_prompt | llm

# 2. 用 LangGraph 编排整体流程
def retrieval_node(state):
    context = retrieval_chain.invoke(state["query"])
    return {"context": context}

def generation_node(state):
    response = generation_chain.invoke({
        "context": state["context"],
        "query": state["query"]
    })
    return {"response": response}

def analysis_node(state):
    analysis = analysis_chain.invoke(state["response"])
    return {"analysis": analysis}

workflow = StateGraph(State)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("generation", generation_node)
workflow.add_node("analysis", analysis_node)
# ... 定义边
```

---

## 七、迁移指南

### 7.1 从 LangChain 迁移到 LangGraph

**步骤 1: 识别迁移需求**

```python
# 需要迁移的信号:
# ❌ AgentExecutor 满足不了需求
# ❌ 需要自定义循环逻辑
# ❌ 需要人工介入
# ❌ Multi-Agent 协作复杂
```

**步骤 2: 重构为 LangGraph**

```python
# ═══════════════════════════════════════════════
# Before: LangChain AgentExecutor
# ═══════════════════════════════════════════════

from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    verbose=True
)

result = executor.invoke({"input": query})


# ═══════════════════════════════════════════════
# After: LangGraph
# ═══════════════════════════════════════════════

from langgraph.prebuilt import create_react_agent

# 一行代码迁移！
app = create_react_agent(llm, tools)

# 需要更多控制？手动构建
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list
    iteration_count: int

def agent_node(state):
    response = agent.invoke({"messages": state["messages"]})
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}

def should_continue(state):
    return state["iteration_count"] < 5

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_conditional_edges("agent", should_continue, {True: "agent", False: END})

app = workflow.compile()
```

### 7.2 迁移检查清单

```
迁移前检查:
☐ 明确当前 LangChain 链的功能
☐ 识别需要增强的部分（循环、分支、持久化）
☐ 评估迁移成本

迁移步骤:
☐ 定义 State 结构
☐ 将每个 Runnable 转换为 Node
☐ 定义 Edges（边）
☐ 添加检查点（如需持久化）
☐ 测试和验证

迁移后优化:
☐ 可视化调试
☐ 性能优化
☐ 添加监控
```

---

## 八、性能对比

### 8.1 基准测试

| 操作 | LangChain | LangGraph | 差异 |
|:-----|:----------|:----------|:-----|
| **简单链调用** | ~50ms | ~60ms | LangChain 稍快 ✅ |
| **Agent 单步** | ~200ms | ~210ms | 相当 |
| **状态保存** | N/A | ~10ms | LangGraph 支持 ✅ |
| **条件分支** | ~5ms | ~8ms | LangChain 稍快 ✅ |
| **检查点恢复** | 不支持 | ~20ms | LangGraph 独有 ✅ |

### 8.2 资源消耗

| 指标 | LangChain | LangGraph |
|:-----|:----------|:----------|
| **内存开销** | 低 | 中等（状态对象） |
| **启动时间** | 快 | 中等 |
| **调试难度** | 中等 | 低（可视化） |
| **代码维护性** | 简单场景高 | 复杂场景高 |

### 8.3 总结

| 框架 | 优势 | 劣势 |
|:-----|:-----|:-----|
| **LangChain** | • 简洁<br>• 快速开发<br>• 学习曲线低<br>• 性能稍好 | • 不支持循环<br>• 状态管理弱<br>• 复杂场景难维护 |
| **LangGraph** | • 支持循环<br>• 状态管理强<br>• 可视化调试<br>• 可控性强 | • 学习曲线陡<br>• 代码量多<br>• 简单场景过度设计 |

---

## 九、快速决策

### 9.1 一句话总结

> **简单任务用 LangChain，复杂任务用 LangGraph，两者可以混合使用。**

### 9.2 选择公式

```
IF (需要循环 OR 需要Multi-Agent OR 需要人工介入 OR 需要长时间运行):
    使用 LangGraph ✅
ELSE IF (简单问答 OR 基础 RAG OR 聊天机器人 OR 快速原型):
    使用 LangChain ✅
ELSE:
    从 LangChain 开始，需要时迁移到 LangGraph 🔄
```

### 9.3 学习建议

```
第一阶段（必学）: LangChain
├── LCEL 语法
├── 基础组件
└── 简单链构建

第二阶段（按需）: LangGraph
├── 当 LangChain 满足不了需求时
├── 学习 State、Nodes、Edges
└── 构建复杂 Agent

第三阶段（精通）: 混合使用
├── LangChain 构建组件
├── LangGraph 编排流程
└── LangSmith 监控调试
```

---

**相关笔记**:
- [[AI研究/AI学习/03-实战应用/LangChain全面解析]]
- [[AI研究/AI学习/03-实战应用/LangGraph全面解析]]
- [[AI研究/AI学习/03-实战应用/Agent全面解析]]

**返回导航**:
- [[AI研究/AI学习/00-知识库索引]]
- [[AI研究/AI学习/AI模型系统性学习路径]]
