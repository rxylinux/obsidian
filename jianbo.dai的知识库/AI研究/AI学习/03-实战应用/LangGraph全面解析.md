---
title: LangGraph 全面解析
date: 2026-03-01
tags:
  - LangGraph
  - Agent框架
  - 状态机
  - 工作流编排
cssclass: main-page
status: active
---

# LangGraph 全面解析

> [!info] 核心定位
> **LangGraph** 是 LangChain 团队开发的**状态机式 Agent 框架**，用有向图定义 Agent 行为，提供强大的可控性和可视化能力。

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **学习路线**：[[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线
> - **框架对比**：[[AI研究/AI学习/03-实战应用/LangChain vs LangGraph 对比分析]] - 框架选择指南
> - **相关笔记**：
>   - [[AI研究/AI学习/03-实战应用/LangChain全面解析]] - LangChain 框架教程
>   - [[AI研究/AI学习/03-实战应用/Agent全面解析]] - Agent 体系知识
>   - [[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 检索增强生成
> - **官方文档**：https://langchain-ai.github.io/langgraph/

---

## 📑 目录

### 核心概念
- [[#一、为什么需要 LangGraph]]
- [[#二、核心概念解析]] ← 重点
- [[#三、与 LangChain Agent 的区别]]

### 基础用法
- [[#四、第一个 LangGraph Agent]]
- [[#五、状态（State）管理]]
- [[#六、节点（Node）定义]]

### 高级特性
- [[#七、边（Edge）类型]]
- [[#八、条件路由]]
- [[#九、记忆与持久化]]
- [[#十、人机交互（Human-in-the-Loop）]]

### 实战模式
- [[#十一、常见 Agent 模式实现]]
- [[#十二、Multi-Agent 协作]]
- [[#十三、时间旅行调试]]

### 生产实践
- [[#十四、LangSmith 集成]]
- [[#十五、部署与监控]]
- [[#十六、最佳实践]]

---

## 一、为什么需要 LangGraph

### 1.1 传统 Agent 的痛点

```python
# 传统 LangChain Agent 的问题
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# 问题 1: 行为不可控
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
# ❌ Agent 会循环思考-行动-观察，无法精确控制流程

# 问题 2: 难以实现复杂逻辑
# ❌ 无法实现 "先调研，再编码，最后测试" 这种固定流程

# 问题 3: 调试困难
# ❌ 无法回溯到某个中间状态

# 问题 4: 缺乏可视化
# ❌ 看不到 Agent 的决策路径
```

### 1.2 LangGraph 的解决方案

```
┌─────────────────────────────────────────────────────────┐
│              LangGraph = 状态机 + Agent                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  传统 Agent:          LangGraph:                        │
│                                                         │
│  ┌─────────┐         ┌─────┐    ┌─────┐    ┌─────┐    │
│  │   LLM   │         │Plan │───▶│Code │───▶│Test │    │
│  │  ┌──┐   │         └─────┘    └─────┘    └─────┘    │
│  │  │Think│  │                                       │
│  │  └──┘   │           ┌─────┐                       │
│  │  ┌──┐   │           │Plan │◀──┐                  │
│  │  │Act │   │           └─────┘   │                  │
│  │  └──┘   │              │       │                  │
│  │  ┌──┐   │              ▼       │                  │
│  │  │Obs │  │           ┌─────┐   │                  │
│  │  └──┘   │           │Fail │───┘                  │
│  │   ...   │           └─────┘                       │
│  └─────────┘                                         │
│                                                         │
│  ❌ 循环不可控        ✅ 流程精确可控                   │
│  ❌ 无法分支          ✅ 支持条件分支                   │
│  ❌ 难以调试          ✅ 状态可回溯                     │
│  ❌ 无可视化          ✅ 可视化图结构                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 适用场景

```
何时使用 LangGraph？

├─ 需要**精确控制** Agent 行为
│   └─ LangGraph ✅
│
├─ 需要**固定流程**的 Agent（如：调研→分析→报告）
│   └─ LangGraph ✅
│
├─ 需要**条件分支**逻辑（如：成功→结束，失败→重试）
│   └─ LangGraph ✅
│
├─ 需要**人机交互**（如：人工审核关键步骤）
│   └─ LangGraph ✅
│
├─ 需要**可视化调试**
│   └─ LangGraph ✅
│
└─ 简单的、探索性的 Agent 任务
    └─ LangChain Agent / ReAct ✅
```

---

## 二、核心概念解析

### 2.1 核心概念图谱

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph 核心概念                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│  │    State      │  │     Node      │  │    Edge     │ │
│  │   （状态）     │  │    （节点）    │  │    （边）    │ │
│  └───────────────┘  └───────────────┘  └─────────────┘ │
│         │                  │                │           │
│         ▼                  ▼                ▼           │
│  • 存储所有数据        • 处理逻辑         • 连接节点    │
│  • 在节点间传递        • 接收状态         • 定义流程    │
│  • 逐步更新           • 返回更新         • 条件路由    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Graph（图）                          │   │
│  │          = Nodes + Edges + State                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │            CompiledGraph（编译后的图）            │   │
│  │         .compile() → 可执行的应用                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 概念详解

#### State（状态）

```python
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    """状态在所有节点间共享和传递"""

    # 消息历史
    messages: list[BaseMessage]

    # 带合并策略的字段（累加）
    steps: Annotated[list[str], add]  # 自动累加

    # 覆盖式字段
    current_plan: str | None

    # 控制字段
    should_continue: bool

# 状态的更新方式：
# 1. 覆盖：return {"current_plan": "new plan"}
# 2. 累加：return {"steps": ["step1"]}  # 自动追加
# 3. 不变：不返回该字段
```

#### Node（节点）

```python
def my_node(state: AgentState) -> dict:
    """
    节点函数：
    1. 接收当前状态
    2. 执行逻辑
    3. 返回状态更新（部分或全部）
    """
    # 读取状态
    messages = state["messages"]

    # 执行逻辑
    response = llm.invoke(messages)

    # 返回更新
    return {
        "messages": [response],  # 添加新消息
        "should_continue": False  # 更新控制标志
    }

# 节点类型：
# • 同步节点：普通函数
# • 异步节点：async def my_node(...)
```

#### Edge（边）

```python
# 1. 普通边（无条件）
graph.add_edge("node_a", "node_b")  # a 执行完 → b

# 2. 条件边（根据状态路由）
def route_function(state: AgentState) -> str:
    """返回下一个节点名称"""
    if state["should_continue"]:
        return "continue_node"
    return "end_node"

graph.add_conditional_edges(
    "node_a",
    route_function,
    {
        "continue_node": "continue_node",
        "end_node": END
    }
)

# 3. 入口点
graph.set_entry_point("first_node")
```

---

## 三、与 LangChain Agent 的区别

| 维度 | LangChain Agent | LangGraph |
|:-----|:----------------|:----------|
| **控制方式** | LLM 自主决策 | 状态机预定义 |
| **流程确定性** | 低（每次可能不同） | 高（固定流程） |
| **调试难度** | 高（黑盒） | 低（每步可见） |
| **可视化** | ❌ | ✅（Mermaid 图） |
| **状态回溯** | ❌ | ✅（checkpointer） |
| **人机交互** | ⚠️ 困难 | ✅ 原生支持 |
| **学习曲线** | 低 | 中等 |
| **适用场景** | 探索性任务 | 复杂工作流 |

---

## 四、第一个 LangGraph Agent

### 4.1 简单的 ReAct Agent

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from operator import add

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]

# 2. 定义工具
def search_tool(query: str) -> str:
    return f"搜索结果：{query}"

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "计算错误"

tools = {
    "search": search_tool,
    "calculator": calculator_tool
}

# 3. 定义节点
def call_model(state: AgentState):
    """LLM 决策节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    """工具执行节点"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    results = []
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool(**tool_call["args"])
        results.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call["id"]
        })

    return {"messages": results}

def should_continue(state: AgentState) -> str:
    """条件路由"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 4. 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 5. 编译并运行
app = workflow.compile()

result = app.invoke({
    "messages": [HumanMessage(content="计算 123 * 456")]
})

print(result["messages"][-1].content)
# 输出：55488
```

---

## 五、状态（State）管理

### 5.1 状态定义模式

```python
# 模式 1：简单 TypedDict
class SimpleState(TypedDict):
    input: str
    output: str

# 模式 2：带合并策略的状态
class MergedState(TypedDict):
    # 使用 Annotated 定义合并策略
    messages: Annotated[list, add]  # 列表累加
    tags: Annotated[set, lambda a, b: a | b]  # 集合合并

# 模式 3：使用 Pydantic（推荐）
from pydantic import BaseModel, Field

class StrictState(BaseModel):
    """Pydantic 提供类型验证"""
    query: str = Field(description="用户查询")
    results: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=10)

    class Config:
        arbitrary_types_allowed = True
```

### 5.2 状态更新规则

```python
def node_with_partial_update(state: AgentState) -> dict:
    """部分更新：只返回需要修改的字段"""
    return {
        "current_plan": "新计划",  # 更新
        # messages 保持不变
    }

def node_with_full_update(state: AgentState) -> dict:
    """完全更新：返回所有字段"""
    return {
        "messages": [...],
        "current_plan": "...",
        "should_continue": True
    }

def node_with_merge(state: MergedState) -> dict:
    """合并更新：带 Annotated 的字段会自动合并"""
    return {
        "messages": [new_msg],  # 自动追加到列表
        "tags": {"new_tag"}      # 自动合并到集合
    }
```

### 5.3 状态归约器（Reducer）

```python
from typing import Annotated
from operator import add, or_

# 内置归约器
class StateWithReducers(TypedDict):
    # 列表：追加
    items: Annotated[list, add]

    # 集合：并集
    tags: Annotated[set, or_]

    # 字典：合并
    metadata: Annotated[dict, lambda a, b: {**a, **b}]

    # 自定义：保留最新
    last_update: Annotated[str, lambda a, b: b]

# 自定义归约器
def custom_reducer(old, new):
    """保留最长的字符串"""
    return old if len(old) > len(new) else new

class CustomState(TypedDict):
    content: Annotated[str, custom_reducer]
```

---

## 六、节点（Node）定义

### 6.1 节点函数签名

```python
# 同步节点
def sync_node(state: AgentState) -> dict:
    return {"field": "value"}

# 异步节点
async def async_node(state: AgentState) -> dict:
    await some_async_operation()
    return {"field": "value"}

# 带配置的节点
from langchain_core.runnables import RunnableConfig

def configurable_node(state: AgentState, config: RunnableConfig) -> dict:
    """访问配置参数"""
    temperature = config.get("configurable", {}).get("temperature", 0.7)
    return {"temperature": temperature}

# 带元数据的节点
def node_with_metadata(state: AgentState, metadata: dict) -> dict:
    """访问节点元数据"""
    node_id = metadata.get("node_id")
    return {"node_id": node_id}
```

### 6.2 节点类型

```python
# 1. LLM 节点
def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 2. 工具节点
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

# 3. 路由节点
def router_node(state: AgentState) -> dict:
    """根据内容路由到不同分支"""
    query = state["query"].lower()

    if "搜索" in query:
        return {"next": "search"}
    elif "计算" in query:
        return {"next": "calculate"}
    else:
        return {"next": "chat"}

# 4. 聚合节点
def aggregator_node(state: AgentState):
    """聚合多个分支的结果"""
    all_results = state["branch_results"]
    aggregated = summarize(all_results)
    return {"final_result": aggregated}

# 5. 验证节点
def validator_node(state: AgentState) -> dict:
    """验证结果，决定是否重新执行"""
    result = state["result"]

    if not is_valid(result):
        return {"should_retry": True, "retry_count": state["retry_count"] + 1}

    return {"should_retry": False, "validated_result": result}
```

---

## 七、边（Edge）类型

### 7.1 边类型总览

```
┌─────────────────────────────────────────────────────────┐
│                    边（Edge）类型                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 普通边（Edge）                                       │
│     ┌─────┐                                           │
│     │  A  │────────────────────────▶ ┌─────┐          │
│     └─────┘                           │  B  │          │
│                                        └─────┘          │
│                                         无条件执行        │
│                                                         │
│  2. 条件边（Conditional Edge）                           │
│     ┌─────┐                                           │
│     │  A  │────┐                                      │
│     └─────┘    │                                      │
│            ┌───┴───┐                                   │
│            │ 条件  │                                   │
│            │ 函数  │                                   │
│            └───┬───┘                                   │
│         ┌──────┴──────┐                                │
│         ▼             ▼                                │
│     ┌─────┐       ┌─────┐                             │
│     │  B  │       │  C  │                             │
│     └─────┘       └─────┘                             │
│      条件1          条件2                               │
│                                                         │
│  3. 入口边（Entry Point）                                │
│         START                                          │
│           │                                             │
│           ▼                                             │
│         ┌─────┐                                        │
│         │  A  │                                        │
│         └─────┘                                        │
│                                                         │
│  4. 终止边（END）                                        │
│     ┌─────┐                                           │
│     │  A  │─────────────────────▶ END                  │
│     └─────┘                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 7.2 边的定义

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# 1. 设置入口点
workflow.set_entry_point("start_node")

# 2. 添加普通边
workflow.add_edge("node_a", "node_b")  # A → B

# 3. 添加条件边
def routing_function(state: AgentState) -> str:
    """返回下一个节点名称"""
    score = state.get("score", 0)

    if score > 80:
        return "high_score_node"
    elif score > 50:
        return "mid_score_node"
    else:
        return "low_score_node"

workflow.add_conditional_edges(
    "evaluate_node",           # 源节点
    routing_function,          # 路由函数
    {
        "high_score_node": "high_score_node",   # 路由映射
        "mid_score_node": "mid_score_node",
        "low_score_node": "low_score_node"
    }
)

# 4. 连接到 END
workflow.add_edge("final_node", END)

workflow.add_conditional_edges(
    "check_done",
    lambda x: "end" if x["done"] else "continue",
    {
        "end": END,
        "continue": "process_node"
    }
)
```

---

## 八、条件路由

### 8.1 路由函数模式

```python
# 模式 1：简单字符串返回
def simple_router(state: AgentState) -> str:
    if state["status"] == "success":
        return "success_node"
    return "failure_node"

# 模式 2：基于消息内容路由
def message_router(state: AgentState) -> str:
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls"):
        return "tool_node"
    return "end"

# 模式 3：多条件路由
def multi_condition_router(state: AgentState) -> str:
    confidence = state.get("confidence", 0)
    has_tools = state.get("has_tools", False)

    if confidence > 0.9 and has_tools:
        return "auto_execute"
    elif confidence > 0.7:
        return "human_review"
    else:
        return "regenerate"

# 模式 4：使用枚举
from enum import Enum

class Route(str, Enum):
    TOOL = "tool"
    END = "end"
    RETRY = "retry"

def enum_router(state: AgentState) -> Route:
    if state["error_count"] > 3:
        return Route.END
    if state["needs_tool"]:
        return Route.TOOL
    return Route.RETRY
```

### 8.2 复杂路由场景

```python
# 场景 1：带默认值的路由
def safe_router(state: AgentState) -> str:
    """确保总是返回有效的路由"""
    routes = {
        "search": "search_node",
        "calculate": "calc_node",
        "chat": "chat_node"
    }
    return routes.get(state["intent"], "chat_node")  # 默认值

# 场景 2：概率路由
import random

def probabilistic_router(state: AgentState) -> str:
    """基于概率的路由（用于 A/B 测试）"""
    if random.random() < 0.3:
        return "experimental_node"
    return "stable_node"

# 场景 3：基于 LLM 的路由
def llm_router(state: AgentState) -> str:
    """让 LLM 决定路由"""
    routing_prompt = f"""
    根据以下对话，决定下一步操作：
    {format_messages(state["messages"])}

    可选操作：
    - search: 搜索信息
    - calculate: 数学计算
    - chat: 继续对话

    只返回操作名称。
    """
    response = llm.invoke(routing_prompt)
    return response.content.strip().lower()
```

---

## 九、记忆与持久化

### 9.1 Checkpointer 基础

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. 内存检查点（开发用）
memory = MemorySaver()

# 2. SQLite 持久化（生产用）
db_saver = SqliteSaver.from_conn_string("agent_state.db")

# 3. 编译时指定 checkpointer
app = workflow.compile(
    checkpointer=db_saver,
    interrupt_before=["human_review"]  # 在节点前中断
)

# 4. 运行时传入 thread_id
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(
    {"messages": [HumanMessage("你好")]},
    config=config
)

# 5. 从检查点恢复
result = app.invoke(
    {"messages": [HumanMessage("继续")]},
    config=config  # 相同的 thread_id
)
```

### 9.2 时间旅行调试

```python
# 获取线程的所有状态
states = list(app.get_state(config))

# 恢复到某个历史状态
app.get_state(config).values  # 查看当前状态
app.get_state(config, checkpoint_id="xxx").values  # 查看历史状态

# 回退到之前的检查点
app.restart(config)  # 从头开始
app.revert(config, checkpoint_id="xxx")  # 回退到指定点

# 分支创建（从历史状态创建新分支）
new_config = app.branch(config, checkpoint_id="xxx")
```

### 9.3 长期记忆模式

```python
from langchain_core.messages import BaseMessage
from typing import Annotated
from operator import add

class MemoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    summary: str

# 消息压缩节点
def summarize_messages(state: MemoryState) -> dict:
    """当消息过多时进行压缩"""
    messages = state["messages"]

    if len(messages) > 50:  # 阈值
        summary = llm.invoke(f"总结以下对话：\n{messages}")
        return {
            "summary": summary.content,
            "messages": messages[-10:]  # 保留最近 10 条
        }

    return {}

# 添加到图中
workflow.add_node("summarize", summarize_messages)
workflow.add_conditional_edges(
    "chat",
    lambda x: "summarize" if len(x["messages"]) > 50 else "continue"
)
```

---

## 十、人机交互（Human-in-the-Loop）

### 10.1 中断与审批

```python
# 编译时设置中断点
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["critical_action"],  # 在节点前中断
    interrupt_after=["data_collection"]    # 在节点后中断
)

# 运行到中断点
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config)

# 检查是否被中断
state = app.get_state(config)
print(state.next)  # ['critical_action'] - 下一个要执行的节点

# 人工审批后继续
approval = input("批准执行吗？(y/n): ")
if approval.lower() == "y":
    result = app.invoke(None, config)  # 继续执行
else:
    # 修改状态后重新路由
    app.update_state(config, {"approved": False})
    result = app.invoke(None, config)
```

### 10.2 人工输入节点

```python
from langgraph.prebuilt import create_react_agent

# 创建带人工输入的 Agent
def human_input_node(state: AgentState):
    """等待人工输入"""
    question = state["messages"][-1].content

    # 在实际应用中，这里应该通过 API 等待前端输入
    # 这里用 input() 演示
    response = input(f"问题：{question}\n请回答：")

    return {"messages": [HumanMessage(content=response)]}

# 添加到图中
workflow.add_node("human_input", human_input_node)

workflow.add_conditional_edges(
    "agent",
    lambda x: "human_input" if x["needs_human"] else "continue",
    {
        "human_input": "human_input",
        "continue": "tools"
    }
)
```

---

## 十一、常见 Agent 模式实现

### 11.1 ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(name="搜索", func=search_func, description="搜索互联网"),
    Tool(name="计算", func=calculator, description="数学计算")
]

# 创建 ReAct Agent
agent = create_react_agent(
    llm,
    tools,
    state_modifier="你是一个有帮助的助手。"
)

# 编译
app = agent.compile()

# 运行
result = app.invoke({
    "messages": [("user", "搜索 2024 年最好的 AI 编程工具")]
})
```

### 11.2 Plan-and-Execute Agent

```python
def planner_node(state: AgentState):
    """规划节点：生成执行计划"""
    prompt = f"""
    目标：{state['goal']}

    请制定详细的执行计划，每个步骤包括：
    - 步骤编号
    - 要执行的动作
    - 预期结果

    计划：
    """
    plan = llm.invoke(prompt)
    return {"plan": plan.content}

def executor_node(state: AgentState):
    """执行节点：按计划执行"""
    plan = state["plan"]
    # 解析并执行计划
    results = execute_plan(plan)
    return {"results": results}

def reflection_node(state: AgentState):
    """反思节点：评估执行结果"""
    results = state["results"]
    assessment = llm.invoke(f"评估以下结果：{results}")
    return {"assessment": assessment.content}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("reflection", reflection_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflection")

workflow.add_conditional_edges(
    "reflection",
    lambda x: "planner" if x.get("needs_replan") else END,
    {
        "planner": "planner",
        "end": END
    }
)
```

### 11.3 多策略 Agent

```python
class StrategyState(TypedDict):
    query: str
    strategy: str
    result: str

def strategy_router(state: StrategyState) -> str:
    """根据查询选择策略"""
    query = state["query"].lower()

    if "搜索" in query or "查找" in query:
        return "search_strategy"
    elif "计算" in query or "数学" in query:
        return "calc_strategy"
    else:
        return "chat_strategy"

def search_strategy(state: StrategyState):
    """搜索策略"""
    # 实现搜索逻辑
    return {"result": "搜索结果..."}

def calc_strategy(state: StrategyState):
    """计算策略"""
    # 实现计算逻辑
    return {"result": "计算结果..."}

def chat_strategy(state: StrategyState):
    """对话策略"""
    # 实现对话逻辑
    return {"result": "对话回复..."}

workflow = StateGraph(StrategyState)
workflow.add_node("search", search_strategy)
workflow.add_node("calculate", calc_strategy)
workflow.add_node("chat", chat_strategy)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", strategy_router)
```

---

## 十二、Multi-Agent 协作

### 12.1 顺序协作

```python
# 研究员 → 写作者 → 审核者

class MultiAgentState(TypedDict):
    topic: str
    research: str
    draft: str
    final_article: str

def researcher(state: MultiAgentState):
    """研究员 Agent"""
    prompt = f"研究主题：{state['topic']}\n\n收集详细信息。"
    research = researcher_llm.invoke(prompt)
    return {"research": research.content}

def writer(state: MultiAgentState):
    """写作者 Agent"""
    prompt = f"""
    基于以下研究结果撰写文章：
    {state['research']}
    """
    draft = writer_llm.invoke(prompt)
    return {"draft": draft.content}

def editor(state: MultiAgentState):
    """审核者 Agent"""
    prompt = f"""
    审核并优化以下文章：
    {state['draft']}
    """
    final = editor_llm.invoke(prompt)
    return {"final_article": final.content}

workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("editor", editor)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", END)
```

### 12.2 并行协作

```python
from langgraph.graph import Send

# 多个专家同时处理

class ParallelState(TypedDict):
    query: str
    expert_results: Annotated[list, add]

def expert_a(state: ParallelState):
    return {"expert_results": f"专家A的分析：{state['query']}"}

def expert_b(state: ParallelState):
    return {"expert_results": f"专家B的分析：{state['query']}"}

def expert_c(state: ParallelState):
    return {"expert_results": f"专家C的分析：{state['query']}"}

def aggregator(state: ParallelState):
    """聚合所有专家结果"""
    combined = "\n\n".join(state["expert_results"])
    summary = llm.invoke(f"综合以下专家意见：\n{combined}")
    return {"final_result": summary.content}

# 构建并行图
workflow = StateGraph(ParallelState)
workflow.add_node("expert_a", expert_a)
workflow.add_node("expert_b", expert_b)
workflow.add_node("expert_c", expert_c)
workflow.add_node("aggregator", aggregator)

workflow.set_entry_point("router")

# 路由到所有专家（并行）
workflow.add_conditional_edges(
    "router",
    lambda x: ["expert_a", "expert_b", "expert_c"],
    {
        "expert_a": "expert_a",
        "expert_b": "expert_b",
        "expert_c": "expert_c"
    }
)

# 所有专家完成后聚合
workflow.add_edge(["expert_a", "expert_b", "expert_c"], "aggregator")
```

### 12.3 辩论式协作

```python
class DebateState(TypedDict):
    proposition: str
    arguments_for: list[str]
    arguments_against: list[str]
    rounds: int

def proponent(state: DebateState):
    """正方论点"""
    prompt = f"""
    支持：{state['proposition']}

    反方论点：{state['arguments_against']}

    请提出更有力的支持论点。
    """
    argument = llm.invoke(prompt)
    return {"arguments_for": [argument.content]}

def opponent(state: DebateState):
    """反方论点"""
    prompt = f"""
    反对：{state['proposition']}

    正方论点：{state['arguments_for']}

    请提出更有力的反对论点。
    """
    argument = llm.invoke(prompt)
    return {"arguments_against": [argument.content]}

def judge(state: DebateState):
    """裁判综合判断"""
    if state["rounds"] >= 3:
        prompt = f"""
        正方论点：{state['arguments_for']}
        反方论点：{state['arguments_against']}

        请做出最终判断。
        """
        verdict = llm.invoke(prompt)
        return {"verdict": verdict.content, "done": True}

    return {"rounds": state["rounds"] + 1}

workflow = StateGraph(DebateState)
workflow.add_node("proponent", proponent)
workflow.add_node("opponent", opponent)
workflow.add_node("judge", judge)

workflow.set_entry_point("proponent")
workflow.add_edge("proponent", "opponent")
workflow.add_edge("opponent", "judge")

workflow.add_conditional_edges(
    "judge",
    lambda x: "end" if x.get("done") else "proponent",
    {"end": END, "proponent": "proponent"}
)
```

---

## 十三、时间旅行调试

### 13.1 查看历史状态

```python
# 获取所有检查点
for checkpoint in app.get_state_history(config):
    print(f"Checkpoint ID: {checkpoint.id}")
    print(f"Timestamp: {checkpoint.timestamp}")
    print(f"State: {checkpoint.values}")
    print("---")

# 查看特定检查点
state = app.get_state(config, checkpoint_id="xxx")
print(state.values)
print(state.next)  # 下一步要执行的节点
```

### 13.2 状态回滚

```python
# 回滚到之前的检查点
app.revert(config, checkpoint_id="xxx")

# 修改状态后继续
app.update_state(
    config,
    {"messages": [HumanMessage("新的输入")]}
)

# 继续执行
result = app.invoke(None, config)
```

### 13.3 分支探索

```python
# 从某个检查点创建新分支
branch_config = app.branch(config, checkpoint_id="xxx")

# 在新分支中执行不同的操作
result_a = app.invoke({"input": "路径A"}, branch_config)
result_b = app.invoke({"input": "路径B"}, branch_config)

# 比较结果
compare_results(result_a, result_b)
```

---

## 十四、LangSmith 集成

### 14.1 启用追踪

```python
import os

# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-agent"

# 编译时自动启用追踪
app = workflow.compile()

# 所有执行都会被追踪到 LangSmith
result = app.invoke(initial_state, config)
```

### 14.2 自定义标签和元数据

```python
result = app.invoke(
    initial_state,
    config,
    {
        "tags": ["production", "v1.0"],  # 标签
        "metadata": {  # 元数据
            "user_id": "user-123",
            "session_id": "session-456"
        }
    }
)
```

### 14.3 在 LangSmith 中查看

```
LangSmith 提供：
✅ 完整的执行轨迹
✅ 每个节点的输入输出
✅ 执行时间分析
✅ Token 使用统计
✅ 错误追踪
✅ 可视化图结构
```

---

## 十五、部署与监控

### 15.1 部署为 API

```python
from fastapi import FastAPI
from langgraph serve

app = FastAPI()

# LangGraph 内置服务器
langgraph_app = workflow.compile()

# 添加到 FastAPI
@app.post("/agent/run")
async def run_agent(request: dict):
    result = await langgraph_app.ainvoke(
        request["state"],
        config=request.get("config")
    )
    return result

# 或使用 LangGraph CLI
# langgraph dev --port 8123
```

### 15.2 监控指标

```python
# 自定义监控
class MonitoredState(TypedDict):
    messages: Annotated[list, add]
    step_count: int
    error_count: int
    total_tokens: int

def monitoring_node(state: MonitoredState):
    """记录监控指标"""
    return {
        "step_count": state.get("step_count", 0) + 1,
        "error_count": state.get("error_count", 0)
    }

# 导出到监控系统
import prometheus_client

step_counter = prometheus_client.Counter('langgraph_steps', 'Total steps')
error_counter = prometheus_client.Counter('langgraph_errors', 'Total errors')

def monitored_node(state):
    step_counter.inc()
    try:
        result = process(state)
        return result
    except Exception as e:
        error_counter.inc()
        raise
```

---

## 十六、最佳实践

### 16.1 设计原则

```python
# ✅ 好的设计
class GoodState(TypedDict):
    """清晰的状态定义"""
    query: str          # 用户输入
    context: str        # 上下文
    response: str       # 响应
    error: str | None   # 错误信息

# ❌ 不好的设计
class BadState(TypedDict):
    """混乱的状态定义"""
    data: dict  # 太灵活，难以理解
    info: list  # 含义不明确
    stuff: Any  # 类型不明确

# ✅ 好的节点命名
def validate_input(state: State): ...
def search_database(state: State): ...
def generate_response(state: State): ...

# ❌ 不好的节点命名
def node1(state: State): ...
def process(state: State): ...
def do_stuff(state: State): ...
```

### 16.2 错误处理

```python
def safe_node(state: AgentState):
    """带错误处理的节点"""
    try:
        result = risky_operation()
        return {"result": result, "error": None}

    except Exception as e:
        logger.error(f"节点执行失败: {e}")
        return {
            "error": str(e),
            "should_retry": True,
            "retry_count": state.get("retry_count", 0) + 1
        }

def error_handler(state: AgentState):
    """错误处理节点"""
    if state.get("retry_count", 0) > 3:
        return {"error": "重试次数过多，放弃"}

    # 重试
    return {"should_retry": True}
```

### 16.3 性能优化

```python
# 1. 使用异步节点
async def async_node(state: State):
    result = await async_operation()
    return {"data": result}

# 2. 并行执行独立节点
workflow.add_edge(["node_a", "node_b"], "node_c")

# 3. 缓存 LLM 结果
from langchain.cache import InMemoryCache
langchain.cache = InMemoryCache()

# 4. 状态压缩
def compress_if_needed(state: State):
    if len(state["messages"]) > 100:
        compressed = compress_messages(state["messages"])
        return {"messages": compressed}
    return {}
```

---

## 十七、常见疑问（FAQ）

> [!question] 路由策略相关疑问
> 本节收录学习 LangGraph 路由策略时的常见疑问和深度解析。

### 17.1 路由策略是用户决定的吗？

#### 疑问
路由策略是指用户可以决定某件事的流程吗？

#### 解答

**不完全是。** 路由策略有两种决策方式：

```
┌─────────────────────────────────────────────────────────┐
│              路由决策的两种方式                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  方式 1：开发者预设规则（自动路由）                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  if state["score"] > 60:                        │   │
│  │      return "pass"                              │   │
│  │  else:                                           │   │
│  │      return "fail"                              │   │
│  │                                                  │   │
│  │  代码里写死的规则，运行时自动判断                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  方式 2：运行时用户决策（人机交互）                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  interrupt_before=["human_review"]              │   │
│  │                                                  │   │
│  │  暂停执行，等待用户输入，然后继续                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**示例对比：**

```python
# 自动路由（开发者预设）
def auto_router(state: PublishState) -> str:
    if state["quality_score"] >= 90:
        return "publish"      # 直接发布
    elif state["quality_score"] >= 60:
        return "minor_edit"   # 小修后发布
    else:
        return "major_edit"   # 大修

# 用户路由（人机交互）
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["wait_approval"]  # 暂停等待用户输入
)

# 运行时用户决策
app.update_state(config, {"user_choice": user_input})
```

---

### 17.2 路由不应该是 LLM 自动决定吗？为什么还要开发者决定？

#### 疑问
传统 Agent 是 LLM 自动决策，为什么 LangGraph 要让开发者控制路由？

#### 解答

**LangGraph 支持两种方式，你可以选择。**

```
┌─────────────────────────────────────────────────────────┐
│            传统 Agent vs LangGraph 路由对比              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  传统 Agent（LangChain）：                                │
│  ┌────────┐    ┌────────┐    ┌────────┐                │
│  │ 用户   │───▶│  LLM   │───▶│ LLM    │                │
│  │ 输入   │    │  推理  │    │ 决策   │                │
│  └────────┘    └────────┘    └────┬───┘                │
│                                   │                     │
│  LLM 100% 自主决策                                       │
│  优点：灵活、智能  缺点：不可控、可能出错                  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LangGraph 方式A（LLM 决定）：                            │
│  ┌────────┐    ┌────────┐    ┌────────┐                │
│  │ 用户   │───▶│  LLM   │───▶│ LLM    │                │
│  │ 输入   │    │  推理  │    │ 决策   │                │
│  └────────┘    └────────┘    └────┬───┘                │
│                                   │                     │
│  LLM 决定，但你可以看到它选了什么                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LangGraph 方式B（开发者决定）：                           │
│  ┌────────┐    ┌────────┐    ┌────────┐                │
│  │ 用户   │───▶│  LLM   │───▶│ 代码   │                │
│  │ 输入   │    │  推理  │    │ if判断 │                │
│  └────────┘    └────────┘    └────┬───┘                │
│                                   │                     │
│  开发者写规则，LLM 不能改变                               │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LangGraph 方式C（混合）：                                │
│  有些路由固定，有些由 LLM 决定，可混合使用！                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**为什么要开发者控制？**

1. **安全性**：银行转账不能跳过验证
2. **可控性**：LLM 可能犯错（选错工具、陷入循环）
3. **调试性**：固定流程更容易排查问题
4. **合规性**：某些流程必须按规范执行

**最佳实践：混合使用**

```python
def smart_router(state: State) -> str:
    """结合 LLM 智能和开发者控制"""

    # 1. 关键安全检查：开发者硬编码
    if state.get("safety_check_failed"):
        return "human_review"  # 强制人工审核

    # 2. 常规流程：LLM 自主决策
    if state.get("needs_decision"):
        decision = llm.invoke(f"根据 {state} 选择下一步")
        return decision.content

    # 3. 默认规则：开发者预设
    return "default_node"
```

---

### 17.3 开发者控制的路由跟规则引擎有什么差别？

#### 疑问
开发者用 if/else 写路由，跟规则引擎有什么本质区别？

#### 解答

**本质上确实相似，但有重要区别。**

| 维度 | LangGraph 路由 | 规则引擎 |
|:-----|:---------------|:---------|
| **本质** | if/else 代码 | 专门的规则系统 |
| **学习成本** | 低（会写代码就行） | 中高（需要学习规则语言） |
| **灵活性** | 需要改代码 | 规则可配置、热部署 |
| **复杂度** | 适合简单逻辑 | 适合复杂规则网络 |
| **推理能力** | 无 | 前向/后向链、冲突解决 |
| **适用角色** | 开发者 | 开发者 + 业务人员 |
| **与 LLM 结合** | ✅ 原生支持 | ⚠️ 需要自己集成 |

**什么时候用哪个？**

```
├─ 简单的 Agent 路由（<10 条规则）
│   └─ LangGraph 路由 ✅
│
├─ 复杂业务规则（>50 条规则，业务人员要改）
│   └─ 规则引擎 ✅
│
├─ 需要和 LLM 混合决策
│   └─ LangGraph ✅（原生支持 LLM）
│
├─ 金融、医疗等合规场景（规则必须可审计、可配置）
│   └─ 规则引擎 ✅
│
└─ 快速原型、敏捷迭代
    └─ LangGraph 路由 ✅
```

**实际例子对比：**

```python
# LangGraph 路由：简单直接
def order_router(state: OrderState) -> str:
    if state["amount"] > 1000:
        return "manual_review"
    elif state["risk_score"] > 80:
        return "reject"
    else:
        return "auto_approve"

# 规则引擎：可配置、可热部署
# 规则文件（业务人员可直接修改）
rule "大额订单审核"
    when
        $order: Order(amount > 1000)
    then
        $order.setStatus("REVIEW");
end

rule "VIP客户特权"
    when
        $order: Order(customer.type == "VIP")
    then
        $order.setPriority("HIGH");
end
```

---

### 17.4 那LangGraph 的价值到底在哪？

#### 疑问
如果开发者控制路由 = 规则引擎，LLM 控制路由 = 传统 Agent，那 LangGraph 的独特价值是什么？

#### 解答

**LangGraph 的核心价值：在一个框架里同时支持多种策略！**

```
┌─────────────────────────────────────────────────────────┐
│           LangGraph 的真正价值                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ❌ 不是：提供一个更好的规则引擎                           │
│  ✅ 而是：在一个框架里同时支持                             │
│                                                          │
│      • LLM 自主决策（智能）                               │
│      • 开发者规则控制（确定）                             │
│      • 人机交互干预（灵活）                               │
│      • 组合使用（混合）                                   │
│                                                         │
│  核心价值：你可以在同一个图里混合使用这些策略！              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**真实案例：客服 Agent**

```python
from langgraph.graph import StateGraph, END

class CustomerServiceState(TypedDict):
    user_query: str
    intent: str
    confidence: float
    user_verified: bool

def hybrid_router(state: CustomerServiceState) -> str:
    """混合路由策略"""

    # 1. 安全/合规：开发者硬编码规则
    if "密码" in state["user_query"] or "转账" in state["user_query"]:
        return "verification_required"  # 强制验证

    # 2. 高风险操作：人机交互
    if state.get("needs_approval"):
        return "human_agent"  # 转人工

    # 3. 常规问题：LLM 自主决策
    if state["confidence"] > 0.8:
        if state["intent"] == "refund":
            return "process_refund"
        elif state["intent"] == "complaint":
            return "handle_complaint"

    # 4. 不确定：默认路由
    return "general_chat"

# 同一个 Agent 里：
# • 有些路由是硬规则（安全）
# • 有些路由需要人审（风险）
# • 有些路由 LLM 决定（灵活）
# • 这些可以随意组合！
```

**总结：**

```
传统 Agent：只能 LLM 决定 ✅
规则引擎：   只能用规则 ✅
LangGraph：  LLM 决定 ✅  +  开发者决定 ✅  +  人机交互 ✅  +  混合 ✅
```

---

### 17.5 快速参考

| 问题 | 简短回答 |
|:-----|:---------|
| 路由是用户决定的吗？ | 不完全是，有自动路由和用户路由两种方式 |
| LLM 能自动决定路由吗？ | 可以，LangGraph 支持 |
| 为什么要开发者控制？ | 安全性、可控性、调试性 |
| 跟规则引擎区别？ | LangGraph 更简单，规则引擎更复杂强大 |
| LangGraph 价值在哪？ | 一个框架同时支持多种路由策略，可混合使用 |

---

## 总结

```
┌─────────────────────────────────────────────────────────┐
│              LangGraph 学习总结                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  核心概念：                                              │
│  • State（状态）- 在节点间传递的数据                     │
│  • Node（节点）- 处理逻辑的函数                          │
│  • Edge（边）- 连接节点的路径                            │
│  • Graph（图）- 完整的工作流                             │
│                                                         │
│  关键优势：                                              │
│  ✅ 精确控制 Agent 行为                                  │
│  ✅ 可视化调试和监控                                     │
│  ✅ 支持复杂的工作流                                     │
│  ✅ 状态持久化和时间旅行                                 │
│  ✅ 人机交互能力                                         │
│                                                         │
│  适用场景：                                              │
│  • 需要固定流程的 Agent                                  │
│  • Multi-Agent 协作                                     │
│  • 需要人工审批的工作流                                  │
│  • 复杂的条件分支逻辑                                    │
│                                                         │
│  学习资源：                                              │
│  • 官方文档：https://langchain-ai.github.io/langgraph/  │
│  • GitHub：https://github.com/langchain-ai/langgraph    │
│  • 示例库：https://langchain-ai.github.io/langgraph/gallery/ │
│                                                         │
└─────────────────────────────────────────────────────────┘

---

## 相关笔记

### 相关技术

| 笔记 | 说明 |
|:-----|:-----|
| [[AI研究/AI学习/03-实战应用/LangChain全面解析]] | LangChain 框架详解 |
| [[AI研究/AI学习/03-实战应用/Agent全面解析]] | Agent 智能体详解 |
| [[AI研究/AI学习/03-实战应用/RAG全面解析]] | RAG 检索增强生成 |

### 学习导航

| 笔记 | 说明 |
|:-----|:-----|
| [[AI研究/AI学习/00-知识库索引]] | 知识库导航中心 |
| [[AI研究/AI学习/AI模型系统性学习路径]] | 完整学习路线 |

### 基础原理

| 笔记 | 说明 |
|:-----|:-----|
| [[AI研究/AI学习/02-模型原理/Transformer全面解析]] | LLM 基础架构 |
| [[AI研究/AI学习/常见术语对照]] | AI/ML 术语词典 |

---

#LangGraph #状态机 #Agent #LangChain #工作流
```
