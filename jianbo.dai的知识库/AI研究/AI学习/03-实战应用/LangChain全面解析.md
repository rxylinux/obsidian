---
title: LangChain 全面解析
date: 2026-03-01
tags:
  - LangChain
  - LLM框架
  - LCEL
  - AI应用开发
cssclass: main-page
status: active
---

# LangChain 全面解析

> [!info] 核心概念
> **LangChain** 是构建 LLM 应用的标准化框架，被誉为"AI 应用的乐高积木"。它提供组件化工具，让开发者快速搭建智能应用。

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **学习路线**：[[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线
> - **相关笔记**：
>   - [[AI研究/AI学习/03-实战应用/LangGraph全面解析]] - LangGraph 深度解析
>   - [[AI研究/AI学习/03-实战应用/Agent全面解析]] - Agent 智能体详解
>   - [[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 检索增强生成

---

## 📑 目录

### 基础入门
- [[#一、LangChain 简介]]
- [[#二、生态系统架构]]
- [[#三、安装与环境配置]]

### 核心模块
- [[#四、Models I/O（模型输入输出）]]
- [[#五、Prompts（提示词管理）]]
- [[#六、Memory（记忆机制）]]
- [[#七、Indexes（索引与检索）]]
- [[#八、Chains（链式调用）]]
- [[#九、Agents（智能体）]]

### LCEL 语法
- [[#十、LCEL 核心概念]]
- [[#十一、LCEL 高级用法]]

### 实战应用
- [[#十二、RAG 应用]]
- [[#十三、聊天机器人]]
- [[#十四、API 集成]]

### 进阶主题
- [[#十五、LangGraph 入门]]
- [[#十六、LangSmith 监控]]
- [[#十七、性能优化]]

### 学习资源
- [[#十八、学习路径]]
- [[#十九、实战项目]]

---

## 一、LangChain 简介

### 1.1 什么是 LangChain

```
传统 LLM 开发:
用户需求 → 写代码 → 调 API → 处理响应 → 重复劳动
每次项目都从零开始，效率低下

LangChain 开发:
用户需求 → 选择组件 → 组合链条 → 完成
像搭乐高积木一样快速构建应用
```

### 1.2 核心价值

| 价值 | 说明 |
|:-----|:-----|
| **标准化** | 统一的接口，支持所有主流 LLM |
| **组件化** | 可复用的组件，减少重复代码 |
| **可组合** | 灵活组合，满足各种需求 |
| **生态丰富** | 大量集成和工具 |

### 1.3 设计理念

```python
# LangChain 的核心理念：可组合性
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

# 每个部分都是独立的组件
prompt = ChatPromptTemplate.from_template("告诉我关于{topic}的有趣事实")
llm = ChatOpenAI(model="gpt-5")
parser = StrOutputParser()

# 像搭积木一样组合
chain = prompt | llm | parser

# 执行
result = chain.invoke({"topic": "量子计算"})
```

---

## 二、生态系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  LangChain 生态系统                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  LangChain │  │  LangGraph │  │  LangSmith │       │
│  │            │  │            │  │            │       │
│  │  组件框架   │  │  编排框架   │  │  监控平台   │       │
│  │  LCEL语法  │  │  状态机    │  │  调试/评估  │       │
│  └────────────┘  └────────────┘  └────────────┘       │
│         │               │               │              │
│         └───────────────┴───────────────┘              │
│                        │                               │
│                 ┌──────┴──────┐                        │
│                 │   Providers │                        │
│                 │             │                        │
│                 │ OpenAI      │                        │
│                 │ Anthropic   │                        │
│                 │ Google      │                        │
│                 │ ...         │                        │
│                 └─────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 三层架构

| 层级 | 组件 | 职责 |
|:-----|:-----|:-----|
| **应用层** | Deep Agents / Projects | 复杂自治 Agent、多 Agent 协作 |
| **编排层** | **LangGraph** | 状态化流程控制、可视化状态流 |
| **链路层** | **LangChain / LCEL** | 模型调用、提示管理、工具集成 |
| **监控层** | **LangSmith** | 调试、观测、评估、成本追踪 |

### 2.3 LangChain vs LangGraph

> [!info] 重要区别
> 两者互补，不互斥。LangGraph 中的节点可以使用 LangChain 构建。

| 特性 | **LangChain** | **LangGraph** |
|:-----|---------------|---------------|
| **核心理念** | 链 (Chain) | 图 (Graph) |
| **执行模式** | 线性: A → B → C | 任意拓扑: 循环/分支/并行 |
| **状态管理** | 无状态，数据单向流动 | 有状态机，中央状态共享 |
| **适用场景** | 简单 RAG、问答、对话 | 复杂 Agent、Multi-Agent |
| **抽象级别** | 高级 API，快速开发 | 低级控制，灵活编排 |
| **学习曲线** | 低 | 中等 |

```python
# LangChain - 线性链式
chain = prompt | llm | output_parser

# LangGraph - 图结构
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_conditional_edges("agent", should_continue)
```

---

## 三、安装与环境配置

### 3.1 基础安装

```bash
# 创建虚拟环境
python -m venv langchain_env
source langchain_env/bin/activate  # Windows: langchain_env\Scripts\activate

# 安装核心包
pip install langchain langchain-core langchain-openai

# 可选：安装社区包
pip install langchain-community

# 可选：安装 LangGraph
pip install langgraph
```

### 3.2 环境变量配置

```bash
# 创建 .env 文件
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
```

```python
# Python 中加载环境变量
from dotenv import load_dotenv
import os

load_dotenv()

# 验证
assert os.environ.get("OPENAI_API_KEY"), "请设置 OPENAI_API_KEY"
```

### 3.3 版本推荐（2026）

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
langchain = "^1.0.7"
langchain-core = "^1.0.7"
langchain-openai = "^1.0.3"
langchain-community = "^0.3.0"
langgraph = "^1.0.3"
langsmith = "^0.4.43"
python-dotenv = "^1.0.0"
```

---

## 四、Models I/O（模型输入输出）

### 4.1 核心概念

```
Models I/O 是 LangChain 的基础，处理与 LLM 的所有交互
┌──────────────┐
│    LLMs      │  文本生成
├──────────────┤
│ Chat Models  │  对话模型 (推荐)
├──────────────┤
│  Embeddings  │  向量嵌入
└──────────────┘
```

### 4.2 统一模型初始化

```python
# 推荐方式：使用 init_chat_model
from langchain.chat_models import init_chat_model

# OpenAI
model = init_chat_model("gpt-5")

# Anthropic
model = init_chat_model("claude-3-5-sonnet-latest")

# Google
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 自动检测
model = init_chat_model("claude-3-5-sonnet-20241022")
# LangChain 会自动识别为 Anthropic
```

### 4.3 基础调用

```python
# 简单调用
result = model.invoke("Hello, how are you?")
print(result.content)

# 带参数调用
result = model.invoke(
    "What is 2+2?",
    temperature=0.7,
    max_tokens=100
)

# 批量调用
results = model.batch([
    "What is AI?",
    "What is ML?",
    "What is DL?"
])

# 流式输出
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="")
```

### 4.4 消息格式

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

# 构建消息列表
messages = [
    SystemMessage(content="你是一个有帮助的助手"),
    HumanMessage(content="你好！"),
    AIMessage(content="你好！有什么我可以帮你的吗？"),
    HumanMessage(content="介绍一下 LangChain")
]

result = model.invoke(messages)
```

### 4.5 工具调用 (Tool Calling)

```python
# 1. 定义工具
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city} 今天天气晴朗，温度 25°C"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [get_weather, calculator]

# 2. 绑定工具到模型
model_with_tools = model.bind_tools(tools)

# 3. 调用
result = model_with_tools.invoke("北京今天天气怎么样？")

# 检查是否有工具调用
if result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"调用工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
```

---

## 五、Prompts（提示词管理）

### 5.1 PromptTemplate

```python
from langchain.prompts import PromptTemplate

# 基础模板
prompt = PromptTemplate(
    input_variables=["product"],
    template="为 {product} 写一个吸引人的广告标语"
)

# 格式化提示
formatted = prompt.format(product="智能手表")
print(formatted)
# 输出: 为 智能手表 写一个吸引人的广告标语

# 在链中使用
chain = prompt | model | StrOutputParser()
result = chain.invoke({"product": "智能手表"})
```

### 5.2 ChatPromptTemplate

```python
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

# 方式1：从字符串创建
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}助手"),
    ("user", "{input}")
])

# 方式2：使用消息类
from langchain_core.messages import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),  # 对话历史
    HumanMessage(content="{input}")
])

# 使用
chain = prompt | model
result = chain.invoke({
    "role": "编程",
    "input": "解释 Python 的列表推导式",
    "history": []
})
```

### 5.3 输出解析器

```python
from langchain.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser
)
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. 字符串解析器
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})

# 2. JSON 解析器
json_parser = JsonOutputParser()

# 定义输出结构
class Product(BaseModel):
    name: str = Field(description="产品名称")
    price: float = Field(description="产品价格")
    description: str = Field(description="产品描述")

parser = PydanticOutputParser(pydantic_object=Product)

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据用户输入生成产品信息\n{format_instructions}"),
    ("user", "{input}")
])

chain = prompt | model | parser
result = chain.invoke({
    "input": "一款性价比高的智能手表",
    "format_instructions": parser.get_format_instructions()
})

# 3. 列表解析器
list_parser = CommaSeparatedListOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "列出5个{topic}相关的关键词，用逗号分隔"),
    ("user", "{input}")
])

chain = prompt | model | list_parser
result = chain.invoke({"topic": "编程语言"})
```

### 5.4 Few-Shot Prompting

```python
from langchain.prompts.few_shot import FewShotPromptTemplate

# 定义示例
examples = [
    {
        "question": "什么是 Python?",
        "answer": "Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名。"
    },
    {
        "question": "什么是 JavaScript?",
        "answer": "JavaScript 是一种用于网页开发的脚本语言，支持动态内容和交互。"
    }
]

# 创建示例提示模板
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="问题: {question}\n答案: {answer}\n"
)

# 创建 FewShot 提示
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="以下是几个问答示例：",
    suffix="问题: {input}\n答案:",
    input_variables=["input"]
)

# 使用
chain = few_shot_prompt | model | StrOutputParser()
result = chain.invoke({"input": "什么是 Java?"})
```

---

## 六、Memory（记忆机制）

### 6.1 为什么需要 Memory

```
无记忆的对话:
用户: 我叫张三
AI: 你好张三！

用户: 我叫什么名字？
AI: 对不起，我不知道  ← 遗忘了之前的对话

有记忆的对话:
用户: 我叫张三
AI: 你好张三！[记住: name="张三"]

用户: 我叫什么名字？
AI: 你叫张三  ← 记住了之前的对话
```

### 6.2 Memory 类型

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationKGMemory
)
```

#### 6.2.1 ConversationBufferMemory

```python
# 保存所有对话历史
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建对话链
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True
)

# 对话
response1 = conversation.predict(input="我叫李明")
response2 = conversation.predict(input="我叫什么名字？")

# 查看记忆
print(memory.load_memory_variables({}))
# {'chat_history': [HumanMessage(...), AIMessage(...), ...]}
```

#### 6.2.2 ConversationBufferWindowMemory

```python
# 只保存最近的 k 轮对话
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=2,  # 只保留最近 2 轮
    memory_key="chat_history",
    return_messages=True
)

conversation = ConversationChain(
    llm=model,
    memory=memory
)

# 超过 2 轮的对话会被遗忘
```

#### 6.2.3 ConversationSummaryMemory

```python
# 自动总结旧对话，节省 token
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=model,
    memory_key="chat_history"
)

conversation = ConversationChain(
    llm=model,
    memory=memory
)

# 长对话会自动总结，避免超出 token 限制
```

### 6.3 在 LCEL 中使用 Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# 创建带记忆的提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 使用 RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain = prompt | model

# 添加记忆功能
with_message_history = RunnableWithMessageHistory(
    chain,
    # 获取历史记录的函数
    lambda session_id: memory_store[session_id],
    # 输入消息的键
    input_messages_key="input",
    # 历史消息的键
    history_messages_key="history",
)

# 使用
memory_store = {}
config = {"configurable": {"session_id": "abc123"}}

response1 = with_message_history.invoke(
    {"input": "我叫王五"},
    config=config
)

response2 = with_message_history.invoke(
    {"input": "我叫什么名字？"},
    config=config
)
# 输出: 你叫王五
```

---

## 七、Indexes（索引与检索）

### 7.1 RAG 基础流程

```
文档 → 加载 → 分割 → 向量化 → 存储 → 检索 → 生成
│      │      │       │       │       │       │
│      │      │       │       │       │       └→ LLM
│      │      │       │       │       └──→ 查询向量
│      │      │       │       └──→ 向量数据库
│      │      │       └──→ Embeddings
│      │      └──→ TextSplitter
│      └──→ DocumentLoader
└──→ 你的文档
```

### 7.2 文档加载

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)

# 加载单个文本文件
loader = TextLoader("example.txt")
documents = loader.load()

# 加载 PDF
pdf_loader = PyPDFLoader("document.pdf")
pages = pdf_loader.load()

# 加载网页
web_loader = WebBaseLoader("https://example.com")
docs = web_loader.load()

# 加载整个目录
dir_loader = DirectoryLoader(
    "./data",
    glob="**/*.txt",
    loader_cls=TextLoader
)
docs = dir_loader.load()
```

### 7.3 文档分割

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# 递归字符分割（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 每块大小
    chunk_overlap=200,      # 重叠大小
    length_function=len,    # 长度函数
    separators=["\n\n", "\n", "。", " ", ""]  # 分隔符
)

splits = splitter.split_documents(documents)

# Token 分割
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    encoding_name="cl100k_base"  # OpenAI 的 token 编码
)
```

### 7.4 向量化

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 或 text-embedding-3-large
)

# 本地 Embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 嵌入文本
vector = embeddings.embed_query("这是一段文本")
print(f"向量维度: {len(vector)}")

# 批量嵌入
vectors = embeddings.embed_documents([
    "第一段文本",
    "第二段文本",
    "第三段文本"
])
```

### 7.5 向量存储

```python
from langchain_community.vectorstores import Chroma, FAISS
from langchain_chroma import Chroma

# 创建 Chroma 向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化目录
)

# 创建 FAISS 向量存储
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

# 保存 FAISS 索引
vectorstore.save_local("faiss_index")

# 加载已保存的向量存储
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 相似度搜索
results = vectorstore.similarity_search("查询内容", k=3)

# 带分数的相似度搜索
results = vectorstore.similarity_search_with_score("查询内容", k=3)
for doc, score in results:
    print(f"分数: {score:.4f}")
    print(f"内容: {doc.page_content}\n")
```

### 7.6 完整 RAG 链

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)

# 2. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回 top 3
)

# 3. 创建 RAG 提示模板
rag_prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

上下文:
{context}

问题: {question

答案:
""")

# 4. 格式化文档
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 5. 构建 RAG 链
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

# 6. 查询
result = rag_chain.invoke("LangChain 是什么？")
```

---

## 八、Chains（链式调用）

### 8.1 LCEL（LangChain Expression Language）

```python
# LCEL 核心概念：管道（Pipeline）
prompt = ChatPromptTemplate.from_template("告诉我关于{topic}的有趣事实")
llm = ChatOpenAI(model="gpt-5")
parser = StrOutputParser()

# 使用 | 操作符组合
chain = prompt | llm | parser

# 等价于：
# chain = prompt.invoke() → llm.invoke() → parser.invoke()

# 调用
result = chain.invoke({"topic": "量子计算"})
```

### 8.2 基础 Chains

```python
# LLMChain
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一个广告标语"
)

chain = LLMChain(llm=model, prompt=prompt)
result = chain.run(product="智能手表")

# SimpleSequentialChain - 顺序执行
from langchain.chains import SimpleSequentialChain

# 第一个链
synopsis_chain = LLMChain(
    llm=model,
    prompt=PromptTemplate(
        input_variables=["title"],
        template="为电影'{title}'写一个简短大纲"
    )
)

# 第二个链
review_chain = LLMChain(
    llm=model,
    prompt=PromptTemplate(
        input_variables=["synopsis"],
        template="根据以下大纲写影评:\n{synopsis}"
    )
)

# 组合
overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain],
    verbose=True
)

result = overall_chain.run("星际穿越")
```

### 8.3 路由链 (Routing)

```python
from langchain.chains import RouterChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# 定义多个子链
physics_chain = LLMChain(
    llm=model,
    prompt=PromptTemplate(
        input_variables=["input"],
        template="你是一个物理专家。回答: {input}"
    )
)

math_chain = LLMChain(
    llm=model,
    prompt=PromptTemplate(
        input_variables=["input"],
        template="你是一个数学专家。回答: {input}"
    )
)

# 定义路由逻辑
def route(inputs):
    topic = inputs["topic"]
    if "物理" in topic:
        return physics_chain
    elif "数学" in topic:
        return math_chain
    else:
        return general_chain

# 使用
from langchain.chains import TransformChain

router_chain = RouterChain(
    chains=[physics_chain, math_chain],
    default=general_chain
)
```

### 8.4 自定义 Chain

```python
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from typing import Dict, Optional

class CustomChain(Chain):
    """自定义链示例"""

    input_key: str = "input"
    output_key: str = "output"

    @property
    def input_keys(self):
        return [self.input_key]

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 自定义逻辑
        input_text = inputs[self.input_key]

        # 处理输入
        processed = f"[处理] {input_text}"

        # 调用 LLM
        response = model.invoke(processed)

        return {self.output_key: response.content}

# 使用
custom_chain = CustomChain()
result = custom_chain.run("输入文本")
```

---

## 九、Agents（智能体）

> [!info] Agent 详解
> 关于 Agent 的详细内容，请参考：[[AI研究/AI学习/03-实战应用/Agent全面解析]]

### 9.1 Agent 基础

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

# 定义工具
@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 '{query}' 的搜索结果"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search, calculator]

# 创建 Agent
from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(model, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # 最大迭代次数
)

# 运行
result = agent_executor.invoke({
    "input": "搜索 Python 的发明者，然后计算他 2026 年的年龄"
})
```

### 9.2 Agent 类型

| 类型 | 说明 | 适用场景 |
|:-----|:-----|:-----|
| **tool-calling** | 原生 Function Calling | OpenAI, Anthropic, Google |
| **ReAct** | 推理 + 行动 | 任何 LLM |
| **JSON** | JSON 格式工具调用 | 结构化输入输出 |
| **Self-Ask** | 自问自答 | 事实性问答 |
| **Conversational** | 带对话记忆 | 多轮对话 |

### 9.3 自定义工具

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 定义输入结构
class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")
    limit: int = Field(default=5, description="结果数量")

# 定义工具函数
def search_func(query: str, limit: int = 5) -> str:
    """搜索信息"""
    return f"找到 {limit} 条关于 '{query}' 的结果"

# 创建工具
search_tool = StructuredTool.from_function(
    func=search_func,
    name="search",
    description="搜索信息",
    args_schema=SearchInput
)

# 或使用装饰器
@tool
def advanced_search(
    query: str,
    limit: int = 5,
    language: str = "zh"
) -> str:
    """高级搜索

    Args:
        query: 搜索查询
        limit: 返回结果数量
        language: 搜索语言
    """
    return f"在 {language} 中搜索 '{query}'，返回 {limit} 条结果"
```

### 9.4 ToolKit 工具包

```python
# LangChain 提供预构建的工具包
from langchain_community.agent_toolkits import (
    SQLDatabaseToolkit,
    ZapierToolkit,
    GitHubToolkit
)

# SQL 工具包
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///chinook.db")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

sql_tools = toolkit.get_tools()
agent_executor = create_react_agent(
    model,
    sql_tools,
    prompt
)
```

---

## 十、LCEL 核心概念

### 10.1 Runnable 接口

```python
# 所有 LCEL 继承 Runnable 基类
from langchain_core.runnables import Runnable

# 核心方法
result = runnable.invoke(input)              # 同步调用
result = await runnable.ainvoke(input)       # 异步调用

for chunk in runnable.stream(input):         # 流式输出
    print(chunk)

result = runnable.batch([input1, input2])    # 批量调用
```

### 10.2 RunnablePassthrough

```python
from langchain_core.runnables import RunnablePassthrough

# 传递输入，不做修改
chain = RunnablePassthrough() | model

# 在字典中使用
chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()  # 传递原始输入
} | prompt | model

# RunnablePassthrough.assign() 添加新字段
chain = RunnablePassthrough.assign(
    processed=lambda x: x["input"].upper()
)
```

### 10.3 RunnableParallel

```python
from langchain_core.runnables import RunnableParallel

# 并行执行多个 Runnable
chain = RunnableParallel(
    summary=prompt1 | model,
    translation=prompt2 | model,
    analysis=prompt3 | model
)

result = chain.invoke({"input": "文本"})

# 结果是字典
{
    "summary": "...",
    "translation": "...",
    "analysis": "..."
}
```

### 10.4 RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

# 包装任意函数
def custom_function(input_text: str) -> str:
    return input_text.upper()

chain = RunnableLambda(custom_function) | model

# 或使用装饰器
@RunnableLambda
def process_input(x):
    return x["input"].upper()

chain = process_input | model
```

### 10.5 RunnableBranch

```python
from langchain_core.runnables import RunnableBranch

# 条件分支
chain = RunnableBranch(
    (lambda x: "动物" in x["topic"], animal_chain),
    (lambda x: "植物" in x["topic"], plant_chain),
    default_chain
)

result = chain.invoke({"topic": "动物", "input": "..."})
```

---

## 十一、LCEL 高级用法

### 11.1 流式输出

```python
# 流式输出 token
for chunk in chain.stream("讲一个故事"):
    print(chunk.content, end="")

# 异步流式
async for chunk in chain.astream("讲一个故事"):
    print(chunk.content, end="")
```

### 11.2 批量处理

```python
# 同步批量
results = chain.batch([
    "问题1",
    "问题2",
    "问题3"
])

# 异步批量
results = await chain.abatch([
    "问题1",
    "问题2"
])

# 返回最大并发数
results = await chain.abatch(
    inputs,
    config={"max_concurrency": 5}
)
```

### 11.3 错误处理

```python
from langchain_core.runnables import RunnableRetry

# 自动重试
chain_with_retry = chain | RunnableRetry(
    max_attempts=3,
    retry_exceptions=(Exception,)
)

# 回退策略
from langchain_core.runnables import RunnableWithFallbacks

fallback_chain = (
    chain
    | RunnableWithFallbacks.from_config(
        {
            "fallbacks": [
                ChatOpenAI(model="gpt-4o-mini"),
                ChatOpenAI(model="gpt-3.5-turbo")
            ]
        }
    )
)
```

### 11.4 动态路由

```python
from langchain_core.runnables import RouterRunnable

# 定义路由函数
def route_function(input):
    if "代码" in input:
        return "code_chain"
    elif "写作" in input:
        return "writing_chain"
    else:
        return "general_chain"

# 创建路由
router = RouterRunnable(
    {
        "code_chain": code_prompt | model,
        "writing_chain": writing_prompt | model,
        "general_chain": general_prompt | model
    },
    route_function
)

result = router.invoke("写一段 Python 代码")
```

---

## 十二、RAG 应用

> [!info] RAG 详解
> 关于 RAG 的详细内容，请参考：[[AI研究/AI学习/03-实战应用/RAG全面解析]]

### 12.1 基础 RAG

```python
# 完整的 RAG 应用
from langchain_core.runnables import RunnablePassthrough

# 1. 加载和分割文档
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 2. 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 创建提示模板
template = """
你是一个有帮助的助手。使用以下上下文片段回答问题。
如果不知道答案，就说不知道，不要编造答案。

上下文:
{context}

问题: {question}

答案:
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. 构建 RAG 链
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 5. 查询
result = rag_chain.invoke("LangChain 支持哪些 LLM？")
```

### 12.2 多查询 RAG

```python
from langchain.prompts import ChatPromptTemplate

# 多查询提示
multi_query_prompt = ChatPromptTemplate.from_template(
    """用户问题: {question}
    生成 3 个不同版本的查询，从不同角度检索相关文档。
    用换行分隔每个查询。
    """
)

# 生成查询
query_chain = multi_query_prompt | model | StrOutputParser()

# 获取所有查询的唯一文档
from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """扁平化并去重"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# 构建多查询 RAG 链
multi_query_chain = (
    {
        "question": RunnablePassthrough()
    }
    | {
        "queries": query_chain,
        "original_question": RunnablePassthrough()
    }
    | {
        "documents": lambda x: get_unique_union(
            [retriever.invoke(q) for q in x["queries"].split("\n")]
        ),
        "question": lambda x: x["original_question"]
    }
    | rag_prompt | model | StrOutputParser()
)
```

### 12.3 Hybrid Search（混合搜索）

```python
# 结合 BM25 和语义搜索
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 检索器
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 3

# 向量检索器
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 组合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]  # 权重
)

# 使用
rag_chain = (
    {
        "context": ensemble_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt | model | StrOutputParser()
)
```

---

## 十三、聊天机器人

### 13.1 基础聊天机器人

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# 链
chain = prompt | model

# 添加记忆
def get_session_history(session_id):
    # 返回该 session 的历史记录
    return history_store.get(session_id, [])

 runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用
history_store = {}
config = {"configurable": {"session_id": "user-123"}}

response1 = runnable_with_history.invoke(
    {"input": "你好，我叫小明"},
    config=config
)

response2 = runnable_with_history.invoke(
    {"input": "我叫什么名字？"},
    config=config
)
# 输出: 你叫小明
```

### 13.2 带工具的聊天机器人

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 晴天，25°C"

@tool
def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_weather, get_time]

# 绑定工具
model_with_tools = model.bind_tools(tools)

# 提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，可以使用工具获取信息"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建 Agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

agent = create_tool_calling_agent(model_with_tools, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 添加记忆
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### 13.3 流式聊天

```python
# 流式输出
async for chunk in agent_with_memory.stream(
    {"input": "今天北京天气怎么样？"},
    config=config
):
    if "content" in chunk:
        print(chunk["content"], end="")
```

---

## 十四、API 集成

### 14.1 HTTP API 部署

```python
# 使用 FastAPI 部署
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await chain_with_memory.ainvoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}}
        )
        return ChatResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 运行: uvicorn api:app --reload
```

### 14.2 LangServe 部署

```python
# LangServe 是 LangChain 官方的部署方案
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI()

# 添加链路由
add_routes(
    app,
    chain,
    path="/chain"
)

# 访问: http://localhost:8000/chain/playground
```

### 14.3 外部 API 调用

```python
import requests
from langchain_core.tools import tool

@tool
def call_external_api(query: str) -> str:
    """调用外部 API"""
    response = requests.get(
        "https://api.example.com/search",
        params={"q": query}
    )
    return response.json()

# 在 Agent 中使用
tools = [call_external_api]
```

---

## 十五、LangGraph 入门

> [!info] LangGraph 详解
> 关于 LangGraph 的详细内容，请参考：[[AI研究/AI学习/03-实战应用/LangGraph全面解析]]

### 15.1 基础示例

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 定义状态
class State(TypedDict):
    messages: list
    next_action: str

# 定义节点
def agent_node(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    # 执行工具
    result = tools.execute(state["next_action"])
    return {"messages": state["messages"] + [result]}

# 条件边
def should_continue(state: State):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# 构建图
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")

# 编译
app = workflow.compile()
```

### 15.2 带记忆的 Agent

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点
memory = MemorySaver()

# 编译时添加
app = workflow.compile(checkpointer=memory)

# 使用
config = {"configurable": {"thread_id": "user-123"}}
response = app.invoke(
    {"messages": [("user", "你好")]},
    config=config
)
```

---

## 十六、LangSmith 监控

### 16.1 LangSmith 简介

```
LangSmith 是 LangChain 的官方监控平台，提供：
- Tracing: 追踪每一次调用
- Debugging: 调试复杂链
- Evaluation: 评估输出质量
- Monitoring: 生产环境监控
```

### 16.2 配置 LangSmith

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 设置环境变量
os.environ["LANGSMITH_API_KEY"] = "lsv2_..."
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# 自动开始追踪
# 所有 chain 调用都会被记录
```

### 16.3 自定义追踪

```python
from langsmith import trace

@trace
def my_function(input_text: str) -> str:
    # 这个函数的调用会被追踪
    result = chain.invoke(input_text)
    return result
```

### 16.4 评估

```python
from langsmith.evaluation import evaluate

# 定义评估函数
def custom_evaluator(run, example):
    # 自定义评估逻辑
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    return {"score": prediction == reference}

# 运行评估
results = evaluate(
    chain.invoke,
    data="my-dataset",  # LangSmith 数据集
    evaluators=[custom_evaluator],
    num_experiments=3
)
```

---

## 十七、性能优化

### 17.1 缓存

```python
# 启用缓存
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# 第一次调用会请求 LLM
result1 = model.invoke("什么是 AI?")

# 第二次调用从缓存读取
result2 = model.invoke("什么是 AI?")

# 使用 Redis 缓存
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(host="localhost", port=6379)
set_llm_cache(RedisCache(redis_client))
```

### 17.2 异步调用

```python
# 异步链
import asyncio

async def async_chain():
    results = await chain.abatch([
        "问题1",
        "问题2",
        "问题3"
    ])
    return results

# 运行
results = asyncio.run(async_chain())
```

### 17.3 流式处理

```python
# 流式输出，减少延迟
async for chunk in chain.astream("长文本输入"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 17.4 批量处理

```python
# 批量处理提高效率
inputs = [
    {"input": "问题1"},
    {"input": "问题2"},
    {"input": "问题3"}
]

results = chain.batch(inputs)
```

---

## 十八、学习路径

### 18.1 第一阶段：基础入门（1-2周）

```
Week 1:
├── 安装与环境配置
├── Models I/O - LLM 调用
├── Prompts - 提示词模板
└── Memory - 记忆机制

Week 2:
├── LCEL 语法基础
├── Chains - 链式调用
└── 简单 RAG 应用
```

### 18.2 第二阶段：进阶应用（2-3周）

```
Week 3-4:
├── Agents - 智能体开发
├── 工具定义与集成
└── 复杂 RAG 系统

Week 5:
├── LangGraph 基础
├── 状态管理与可视化
└── Multi-Agent 编排
```

### 18.3 第三阶段：生产实践（3-4周）

```
Week 6-7:
├── LangSmith 监控与调试
├── 性能优化
└── 错误处理

Week 8-9:
├── API 部署
├── 生产环境最佳实践
└── 完整项目开发
```

---

## 十九、实战项目

### 19.1 项目一：文档问答系统

**技能**: RAG、向量存储、文档加载

```python
# 功能需求:
# - 上传 PDF/Word 文档
# - 智能问答
# - 引用来源

# 技术栈:
# - PyPDFLoader / DocxLoader
# - Chroma 向量存储
# - RAG 链
```

### 19.2 项目二：智能客服机器人

**技能**: Agents、工具调用、对话管理

```python
# 功能需求:
# - 回答常见问题
# - 查询订单状态
# - 处理退款请求

# 技术栈:
# - ReAct Agent
# - SQL Toolkit
# - API 工具
# - 对话记忆
```

### 19.3 项目三：代码助手

**技能**: Agents、代码执行、文件操作

```python
# 功能需求:
# - 代码生成
# - 代码解释
# - Bug 修复
# - 代码重构

# 技术栈:
# - Code Interpreter
# - File Tools
# - GitHub Toolkit
```

### 19.4 项目四：多 Agent 协作系统

**技能**: LangGraph、Multi-Agent

```python
# 功能需求:
# - 研究员 Agent
# - 写作 Agent
# - 审核 Agent
# - 协作完成报告

# 技术栈:
# - LangGraph
# - 状态机编排
# - Agent 间通信
```

### 19.5 项目五：AI 编程助手

**技能**: 全栈、复杂系统

```python
# 功能需求:
# - 需求分析
# - 代码生成
# - 测试生成
# - 文档生成
# - Git 操作

# 技术栈:
# - Multi-Agent
# - LangGraph
# - LangSmith
# - GitHub API
```

---

## 二十、学习资源

### 20.1 官方资源

| 资源 | 链接 |
|:-----|:-----|
| **官方文档** | https://python.langchain.com/ |
| **中文文档** | https://python.langchain.cn/ |
| **GitHub** | https://github.com/langchain-ai/langchain |
| **LangGraph** | https://langchain-ai.github.io/langgraph/ |
| **LangSmith** | https://docs.smith.langchain.com/ |

### 20.2 中文资源

| 资源 | 链接 |
|:-----|:-----|
| **中文网** | https://www.langchain.com.cn/ |
| **500页教程** | https://www.bandianxiang.com/info/uQvi66O |
| **Cookbook** | https://cookbook.langchain.com.cn/ |

### 20.3 社区

| 资源 | 链接 |
|:-----|:-----|
| **Discord** | https://discord.gg/langchain |
| **Twitter** | @LangChainAI |
| **YouTube** | https://www.youtube.com/@LangChain |

### 20.4 推荐阅读

```python
# 学习顺序推荐
1. python.langchain.com/docs/introduction/     # 官方入门
2. python.langchain.com/docs/tutorials/       # 官方教程
3. python.langchain.com/docs/concepts/        # 核心概念
4. cookbook.langchain.com/                    # 实战案例
5. langgraph.dev/                             # LangGraph
```

---

## 二十一、快速参考

### 21.1 核心代码片段

```python
# 初始化模型
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-5")

# LCEL 链
chain = prompt | model | parser

# RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)

# Agent
from langchain.agents import create_agent
agent = create_agent(model="gpt-5", tools=tools)

# LangGraph
from langgraph.graph import StateGraph
workflow = StateGraph(State)
app = workflow.compile()
```

### 21.2 常用导入

```python
# 核心模块
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 记忆
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 工具
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentExecutor

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
```

### 21.3 最佳实践

```
✅ 使用 init_chat_model() 统一模型初始化
✅ 使用 LCEL 构建链
✅ 启用 LangSmith 追踪
✅ 添加错误处理和重试
✅ 使用缓存减少 API 调用
✅ 实现流式输出提升体验
✅ 添加监控和日志
❌ 不要硬编码 API Key
❌ 不要忽略错误处理
❌ 不要过度依赖单一模型
```

### 21.4 一句话总结

> **LangChain 是构建 LLM 应用的标准框架，通过组件化设计和 LCEL 语法，让开发者像搭乐高积木一样快速构建智能应用。**

---

**相关笔记**:
- [[AI研究/AI学习/03-实战应用/LangGraph全面解析]]
- [[AI研究/AI学习/03-实战应用/Agent全面解析]]
- [[AI研究/AI学习/03-实战应用/RAG全面解析]]

**返回导航**:
- [[AI研究/AI学习/00-知识库索引]]
- [[AI研究/AI学习/AI模型系统性学习路径]]
