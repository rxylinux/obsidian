---
title: RAG 项目记录
date: 2026-02-28
tags:
  - RAG
  - LangChain
  - VectorDB
  - 项目
status: in-progress
---

# RAG 项目记录

> [!info] 项目目标
> 构建一个检索增强生成 (RAG) 系统，实现知识库问答

> [!tip] 相关笔记
> - **RAG 全面解析**：[[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 完整知识体系
> - **术语对照**：[[AI研究/AI学习/常见术语对照]] - RAG 相关术语
> - **模型原理**：[[AI研究/AI学习/02-模型原理/Transformer研读]] - LLM 基础
> - **学习路径**：[[AI研究/AI学习/AI模型系统性学习路径]] - 实战阶段规划
> - **周计划**：[[AI研究/AI学习/周学习计划]] - 项目进度跟踪

---

## 项目概述

### 什么是 RAG？

```
用户问题
    │
    ▼
┌─────────┐     ┌──────────────┐     ┌─────────┐
│ 检索器  │────▶│ 相关文档片段  │────▶│ LLM    │
│(Retriever)   │  (Context)    │     │ + 生成  │
└─────────┘     └──────────────┘     └────┬────┘
                                            │
                                            ▼
                                        带引用的回答
```

**核心组件**：
1. **文档处理**：切分、嵌入
2. **向量存储**：Embedding + 向量数据库
3. **检索策略**：相似度搜索、重排序
4. **生成合成**：LLM + Prompt

### 项目规划

> [!todo]- 开发清单
> - [ ] 阶段1：基础 RAG 搭建
> - [ ] 阶段2：优化检索质量
> - [ ] 阶段3：高级特性（多轮、混合检索）
> - [ ] 阶段4：评估与部署

### RAG 完整流程总览

> [!info] 三阶段工作流
> 理解 RAG 系统需要区分三个不同的阶段

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG 三阶段工作流                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【准备阶段】（离线 - Offline）                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  原始文档                                                │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 文档加载     │ PDF/TXT/MD                            │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 文档切分     │ Chunking                              │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌────────────────────────────────┐                    │   │
│  │  │  向量化 / 分词                  │                    │   │
│  │  │  ├─ Embedding → 向量 (Dense)    │                    │   │
│  │  │  └─ 分词 → 倒排索引 (Sparse)    │                    │   │
│  │  └──────────────┬─────────────────┘                    │   │
│  │                 │                                       │   │
│  │                 ▼                                       │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 存储到数据库  │ 向量数据库 + 搜索引擎                  │   │
│  │  └──────────────┘                                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  【查询阶段】（在线 - Online）                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  用户查询                                                │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 查询优化器    │ (可选)                                │   │
│  │  │ - 查询重写    │                                       │   │
│  │  │ - HyDE       │                                       │   │
│  │  │ - 查询扩展    │                                       │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 检索器        │ Dense/BM25/Hybrid                    │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 检索数据      │ 候选文档 (Top-K)                      │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼ (可选)                                         │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ Reranker     │ 重排序 → 精选文档 (Top-N)              │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 生成器 (LLM)  │ + Prompt + 上下文                     │   │
│  │  └──────┬───────┘                                       │   │
│  │         │                                               │   │
│  │         ▼                                               │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ 最终答案      │ + 引用来源                            │   │
│  │  └──────────────┘                                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  【训练阶段】（RAG 2.0 特有）                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  联合训练                                                │   │
│  │  ┌────────────────────────────────┐                    │   │
│  │  │   检索器 ←─────┐              │                    │   │
│  │  │        │        │ 梯度回传      │                    │   │
│  │  │        │        │ (双向流动)    │                    │   │
│  │  │        ▼        │              │                    │   │
│  │  │   生成器 ───────┘              │                    │   │
│  │  │                                  │                    │   │
│  │  │   目标：全局最优                 │                    │   │
│  │  │   - 生成器反馈检索质量            │                    │   │
│  │  │   - 检索器学习生成器偏好          │                    │   │
│  │  │   - 协同进化                     │                    │   │
│  │  └────────────────────────────────┘                    │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键理解**：

| 阶段 | 时机 | 目的 | RAG 1.0 | RAG 2.0 |
|:-----|:-----|:-----|:--------|:--------|
| **准备阶段** | 系统初始化 | 构建索引 | ✅ 有 | ✅ 有 |
| **查询阶段** | 用户请求 | 检索+生成 | ✅ 有 | ✅ 有 |
| **训练阶段** | 模型优化 | 联合训练 | ❌ 无 | ✅ 有 |

**项目实现对应**：

- **阶段1-2**: 准备阶段 + 查询阶段（基础 RAG 1.0）
- **阶段3**: 查询阶段优化（高级特性）
- **阶段4**: 评估与部署
- **进阶**: 训练阶段（RAG 2.0，需要更多资源）

### 数据与检索器对应关系

> [!warning] 关键设计原则
> **准备阶段生成的数据类型，决定了查询阶段必须使用的检索器类型！**

```
准备阶段决定数据 → 查询阶段必须对应

┌──────────────────┐      ┌──────────────────┐
│ 准备阶段数据生成  │  →   │ 查询阶段检索器    │
├──────────────────┤      ├──────────────────┤
│ Embedding 向量化  │  →   │ Dense 检索器      │
│ 分词 Tokenization │  →   │ BM25 检索器      │
│ 两者都做          │  →   │ 混合检索器        │
└──────────────────┘      └──────────────────┘
```

**类比理解**：
- 准备阶段 = 建图书馆（建书架、分类系统）
- 查询阶段 = 图书管理员找书
- 有什么工具才能干什么活！

**设计顺序**：
```
1. 先决定需求（性能、成本、场景）
2. 再决定用什么检索器
3. 最后决定准备阶段生成什么数据

自底而上：数据存储 → 检索方式 → 检索器
```

---

## 技术选型

### 框架对比

| 框架 | 优点 | 缺点 | 适用场景 |
|:-----|:-----|:-----|:---------|
| **LangChain** | 生态丰富、组件化 | 学习曲线陡 | 复杂应用 |
| **LlamaIndex** | RAG 优化、易用 | 定制性弱 | 快速原型 |
| **Haystack** | 深度定制 | 文档少 | 生产环境 |
| **原生实现** | 完全控制 | 开发成本高 | 学习研究 |

**本项目选择**：LangChain（灵活 + 社区支持）

### Embedding 模型

| 模型 | 维度 | 特点 | 价格 |
|:-----|:----:|:-----|:-----|
| OpenAI text-embedding-3-small | 1536 | 性价比高 | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | 最强性能 | $0.13/1M tokens |
| BGE-M3 | 1024 | 多语言、开源 | 免费 |
| jina-embeddings-v2 | 768 | 支持长文本 | 免费 |

**本项目选择**：BGE-M3（开源 + 中文友好）

### 向量数据库

| 数据库 | 特点 | 部署 |
|:------|:-----|:-----|
| **Chroma** | 轻量、易用 | 本地 |
| **FAISS** | 高性能、无服务器 | 本地 |
| **Pinecone** | 托管、扩展性好 | 云 |
| **Milvus** | 开源、功能全 | 自建 |

**本项目选择**：Chroma（开发阶段）→ Milvus（生产环境）

---

## 阶段1：基础 RAG

### 项目结构

```
rag_project/
├── data/              # 原始文档
│   └── documents/
├── src/
│   ├── config.py      # 配置文件
│   ├── embeddings.py  # Embedding 封装
│   ├── vectorstore.py # 向量数据库
│   ├── retriever.py   # 检索器
│   └── chain.py       # RAG 链
├── notebooks/         # Jupyter 笔记
├── tests/             # 测试
└── requirements.txt
```

### 配置文件

```python
# src/config.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class Config:
    # 模型配置
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.0

    # 向量数据库
    vector_db_path: str = "./chroma_db"
    collection_name: str = "knowledge_base"

    # 检索配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # API配置
    api_key: str = "your-api-key"

config = Config()
```

### 文档处理

```python
# src/processing.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_documents(self, path):
        """加载文档"""
        from langchain_community.document_loaders import (
            TextLoader, PDFLoader, DirectoryLoader
        )

        if path.endswith('.pdf'):
            loader = PDFLoader(path)
        else:
            loader = TextLoader(path, autodetect_encoding=True)

        return loader.load()

    def split_documents(self, documents):
        """切分文档"""
        return self.splitter.split_documents(documents)

# 使用示例
processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
docs = processor.load_documents("data/document.pdf")
chunks = processor.split_documents(docs)
print(f"切分后: {len(chunks)} 个片段")
```

### Embedding 封装

```python
# src/embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed_query(self, text):
        """嵌入查询"""
        return self.model.embed_query(text)

    def embed_documents(self, texts):
        """嵌入文档"""
        return self.model.embed_documents(texts)

# 测试
embed_model = EmbeddingModel()
query_vector = embed_model.embed_query("什么是机器学习？")
print(f"向量维度: {len(query_vector)}")
```

### 向量数据库

```python
# src/vectorstore.py
from langchain.vectorstores import Chroma
from langchain.schema import Document

class VectorStore:
    def __init__(self, embedding_model, persist_directory="./chroma_db"):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.db = None

    def create_index(self, documents, collection_name="knowledge_base"):
        """创建索引"""
        self.db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        return self.db

    def load_index(self, collection_name="knowledge_base"):
        """加载已有索引"""
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )
        return self.db

    def search(self, query, top_k=5):
        """相似度搜索"""
        results = self.db.similarity_search(query, k=top_k)
        return results

# 使用
vector_store = VectorStore(embed_model)
vector_store.create_index(chunks)
results = vector_store.search("Transformer是什么？")
```

### RAG 链

```python
# src/chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from anthropic import Anthropic

class RAGChain:
    def __init__(self, vector_store, llm_model="claude-3-5-sonnet-20241022"):
        self.vector_store = vector_store.db
        self.llm = Anthropic(api_key=config.api_key)
        self.model_name = llm_model
        self.qa_chain = None

    def create_prompt(self):
        """创建 RAG Prompt"""
        template = """你是一个有用的助手。请根据以下上下文回答问题。

上下文信息：
{context}

问题：{question}

请基于上下文信息回答，如果上下文中没有相关信息，请明确说明。

回答："""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def create_chain(self):
        """创建 RAG 链"""
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k}
        )

        prompt = self.create_prompt()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return self.qa_chain

    def query(self, question):
        """查询"""
        if self.qa_chain is None:
            self.create_chain()

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

# 使用
rag = RAGChain(vector_store, llm_model="claude-3-5-sonnet-20241022")
response = rag.query("什么是 Attention 机制？")
print(response["answer"])
print("\n来源：")
for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']}")
```

---

## 阶段2：优化检索质量

### 文档切分策略

```python
# 语义切分
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embed_model,
    breakpoint_threshold_type="percentile"
)

# 固定大小 + 语义
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
```

### 重排序 (Rerank)

```python
# 使用 Cohere Rerank 或本地模型
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

def rerank_documents(query, documents, top_k=5):
    """重排序文档"""
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)

    # 按分数排序
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

### 混合检索

```python
# 关键词 + 向量
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
vector_retriever = vector_store.db.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

---

## 阶段3：高级特性

### 多轮对话

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
```

### 查询重写

```python
def query_expansion(query, llm):
    """查询扩展"""
    prompt = f"""生成3个不同方式表达的问题，用于检索相关文档。

原问题：{query}

扩展问题："""

    expanded = llm.generate(prompt)
    return [query] + expanded.split('\n')

def query_decomposition(query, llm):
    """复杂问题分解"""
    prompt = f"""将复杂问题分解为多个简单子问题。

问题：{query}

子问题："""

    sub_queries = llm.generate(prompt)
    return sub_queries.split('\n')
```

### RAG Fusion

```python
def rag_fusion(query, retriever, llm):
    """RAG Fusion: 查询重写 + 检索 + 重排序"""
    # 1. 查询重写
    queries = query_expansion(query, llm)

    # 2. 多路检索
    all_docs = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    # 3. 去重 + 重排序
    unique_docs = list(set(all_docs))
    reranked = rerank_documents(query, unique_docs)

    return reranked
```

---

## 阶段4：评估与部署

### 评估指标

```python
# 使用 RAGAS 评估
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

result = evaluate(
    dataset=test_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]
)

print(result)
```

| 指标 | 含义 | 目标 |
|:-----|:-----|:-----|
| Faithfulness | 忠实度 | >0.8 |
| Answer Relevancy | 相关性 | >0.7 |
| Context Recall | 召回准确率 | >0.8 |
| Context Precision | 上下文精确度 | >0.8 |

### 部署方案

```python
# FastAPI 服务
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def query(request: QueryRequest):
    result = rag.query(request.question)
    return {
        "answer": result["answer"],
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result["source_documents"]
        ]
    }
```

---

## 踩坑记录

### 问题1：检索不相关

**原因**：
- 切分太小，语义不完整
- Embedding 模型不匹配领域

**解决**：
- 增大 chunk_size 到 1024
- 使用领域内 Embedding 模型

### 问题2：回答幻觉

**原因**：
- 检索到的文档质量差
- Prompt 没有约束

**解决**：
- 添加重排序
- Prompt 明确要求"不知道就说不知道"

### 问题3：响应慢

**原因**：
- 文档太多
- 每次都检索

**解决**：
- 使用 Hybrid Search
- 添加查询缓存
- 使用更快的小模型

---

## 扩展方向

- [ ] 多模态 RAG（图片、表格）
- [ ] Agent 化（自主查询优化）
- [ ] 知识图谱增强
- [ ] 本地模型（Ollama + LLaMA3）

---

## 参考资源

### 文档
- [LangChain 文档](https://python.langchain.com/)
- [LlamaIndex 文档](https://docs.llamaindex.ai/)

### 教程
- [RAG 从零到一](https://www.anthropic.com/index/retrieval-augmented-generation)
- [Advanced RAG Techniques](https://blog.langchain.dev/techniques-for-improving-retrieval-augmented-generation/)

### 论文
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

## 相关笔记

### 核心学习
- [[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 完整知识体系

### 进阶学习
- [[AI研究/AI学习/04-深入前沿/RAG 2.0 全面解析]] - RAG端到端训练
- [[AI研究/AI学习/04-深入前沿/知识蒸馏全面解析]] - 模型压缩技术

### 基础原理
- [[AI研究/AI学习/02-模型原理/Transformer全面解析]] - LLM基础架构
- [[AI研究/AI学习/02-模型原理/Transformer研读]] - Transformer深度研读
- [[AI研究/AI学习/02-模型原理/Embedding模型全面解析]] - Embedding 模型
- [[AI研究/AI学习/02-模型原理/检索方式全面对比]] - 检索方式对比

### 辅助工具
- [[AI研究/AI学习/常见术语对照]] - RAG相关术语
- [[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线
- [[AI研究/AI学习/周学习计划]] - 项目进度跟踪

---

#RAG #LangChain #项目 #向量数据库
