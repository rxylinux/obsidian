---
title: Embedding 模型全面解析
date: 2026-03-01
tags:
  - Embedding
  - 向量检索
  - 深度学习
  - RAG
  - 对比学习
status: active
---

# Embedding 模型全面解析

> [!info] 核心概念
> Embedding 是将高维离散数据（如文本、图像）映射到低维连续向量空间的技术，使得语义相似的对象在向量空间中距离更近。

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **RAG 全面解析**：[[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 完整知识体系
> - **RAG 2.0**：[[AI研究/AI学习/04-深入前沿/RAG 2.0 全面解析]] - RAG 系统进阶
> - **检索方式对比**：[[AI研究/AI学习/02-模型原理/检索方式全面对比]] - 不同检索方式详解
> - **Transformer**：[[AI研究/AI学习/02-模型原理/Transformer全面解析]] - LLM 基础

---

## 📑 目录

### 模型对比
- [[#一、2025-2026 主流模型]]
- [[#二、分场景推荐]]
- [[#三、中文场景推荐]]
- [[#四、成本对比]]

### 技术原理
- [[#五、训练范式演进]]
- [[#六、核心训练方法]]
- [[#七、模型架构]]
- [[#八、维度选择]]

### 实战应用
- [[#九、向量数据库选型]]
- [[#十、检索优化技巧]]
- [[#十一、性能优化]]
- [[#十二、质量评估]]

### 前沿趋势
- [[#十三、MRL 嵌套式嵌入]]
- [[#十四、Late Chunking]]
- [[#十五、多语言与多模态]]
- [[#十六、LLM-as-Embedding]]

---

## 一、2025-2026 主流模型

### 🏆 MTEB 榜单前列模型

| 模型 | 公司 | 维度 | MTEB得分 | 特点 |
|:-----|:-----|:-----|:---------|:-----|
| **text-embedding-3-large** | OpenAI | 3072 | ~72.5 | 商业标杆，质量稳定 |
| **bge-m3** | BAAI | 1024 | ~71.8 | 多语言、多功能、开源王者 |
| **voyage-3** | VoyageAI | 1024 | ~71.5 | 长文本优秀，商业性价比高 |
| **gte-qwen-1.5-7b-instruct** | Alibaba | 4096 | ~72.0 | 性能最强但推理慢 |
| **e5-mistral-7b-instruct** | Microsoft | 4096 | ~71.2 | 指令微调，质量高 |
| **jina-embeddings-v3** | Jina AI | 1024 | ~70.8 | 自适应，支持多任务 |
| **nomic-embed-text-v1.5** | Nomic AI | 768 | ~69.5 | 完全开源，可本地部署 |
| **mixedbread-ai/mxbai-embed-large-v1** | Mxbai | 1024 | ~70.2 | 高性能，开源友好 |

---

## 二、分场景推荐

### 生产环境（商业项目）

```
首选：OpenAI text-embedding-3-large
  ✅ 质量稳定，API 成熟
  ✅ 无需自己部署
  ❌ 需要付费，数据出境问题

备选：Voyage-3
  ✅ 性价比高（$0.06/1M tokens）
  ✅ 长文本支持好
  ✅ 中文支持不错
```

### 私有部署/数据敏感

```
首选：BGE-M3 (BAAI)
  ✅ 多语言（中英均衡）
  ✅ 功能多样（dense/sparse/colbert）
  ✅ 完全开源

备选：Nomic-embed-text-v1.5
  ✅ 完全开源（数据+代码+权重）
  ✅ 768 维度，存储友好
  ✅ 支持 Matryoshka（可变长度）
```

### 追求极致性能

```
首选：GTE-Qwen-1.5-7B-Instruct
  ✅ MTEB 榜单前列
  ✅ 指令微调，理解能力强
  ❌ 推理成本高，需要 GPU

备选：E5-Mistral-7B-Instruct
  ✅ 微软出品，质量有保证
  ❌ 同样需要 GPU 推理
```

---

## 三、中文场景推荐

| 场景 | 推荐模型 | 理由 |
|:-----|:---------|:-----|
| **通用中文** | **BGE-M3** | 中英均衡，中文效果优秀 |
| **纯中文** | **gte-qwen** | 基于 Qwen，中文原生 |
| **轻量部署** | **bge-small-zh-v1.5** | 330 维度，速度优先 |
| **长文本中文** | **BGE-M3** | 支持 8192 token |
| **开源合规** | **Nomic v1.5** | 完全开源，Apache 2.0 |

---

## 四、成本对比（2025 年价格）

| 模型 | 输入成本 | 输出成本 | 月度 $100 预算可处理 |
|:-----|:---------|:---------|:-------------------|
| OpenAI ada-002 | $0.10/1M | - | ~10 亿 tokens |
| OpenAI text-embedding-3-large | $0.13/1M | - | ~7.7 亿 tokens |
| Voyage-3 | $0.06/1M | - | ~16.7 亿 tokens |
| Cohere embed-v3 | $0.03/1M | - | ~33 亿 tokens |
| 开源模型（自部署） | GPU 成本 | - | 取决于部署规模 |

---

## 五、训练范式演进

```
传统方法 (2018前)
├─ Word2Vec (2013)
├─ GloVe (2014)
└─ FastText (2016)
    问题：词级嵌入，无法处理上下文

上下文嵌入 (2018-2021)
├─ BERT (2018)
├─ RoBERTa (2019)
├─ SimCSE (2021) ──────────────┐
└─ E5 (2022) ──────────────────┤ 对比学习时代

现代方法 (2022-2026)
├─ Contriever (2022)
├─ BGE (2023) ─────────────────┤ 数据规模+负采样优化
├─ GTE (2023)
└─ MRL (2024) ─────────────────┘ 架构创新
```

---

## 六、核心训练方法

### 6.1 对比学习 (Contrastive Learning)

```python
# 核心思想：拉近正样本，推远负样本

class ContrastiveLearning:
    """
    SimCSE 风格的训练
    """
    def __init__(self, encoder, temperature=0.05):
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, sentences):
        # 同一句两次编码（通过 dropout 噪声）
        z1 = self.encoder(sentences)  # (batch, dim)
        z2 = self.encoder(sentences)  # (batch, dim) - 再次编码

        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # 损失：对角线为正样本，其他为负样本
        batch_size = len(sentences)
        labels = torch.arange(batch_size)  # [0, 1, 2, ..., batch-1]
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

# 关键公式：
# Loss = -log(exp(sim(z_i, z_i)/τ) / Σ_j exp(sim(z_i, z_j)/τ))
```

### 6.2 难负样本挖掘 (Hard Negative Mining)

```python
# 问题：随机负样本太简单，学不到东西
# 解决：使用"难"负样本 - 与 query 相关但不是答案的文档

class HardNegativeMining:
    def __init__(self, encoder, mine_hard_negatives=True):
        self.encoder = encoder
        self.mine_hard_negatives = mine_hard_negatives

    def get_hard_negatives(self, query, candidates, k=5):
        """用当前模型找出"难"负样本"""
        query_emb = self.encoder.encode(query)
        candidate_embs = self.encoder.encode(candidates)

        # 计算所有候选的相似度
        similarities = cosine_similarity(query_emb, candidate_embs)

        # 找出最相似但不是正样本的（难负样本）
        # 它们"看起来相关"但其实不是
        hard_neg_indices = np.argsort(similarities)[-k-1:-1]

        return [candidates[i] for i in hard_neg_indices]
```

### 6.3 E5-style 训练（指令微调）

```python
# E5: 核心创新是给 query 和 document 添加不同的指令前缀

class E5StyleTraining:
    def __init__(self, encoder):
        self.encoder = encoder
        self.query_prefix = "query: "
        self.doc_prefix = "passage: "

    def encode(self, text, is_query=True):
        """根据角色添加不同前缀"""
        prefix = self.query_prefix if is_query else self.doc_prefix
        text_with_prefix = prefix + text
        return self.encoder.encode(text_with_prefix)

# 为什么指令前缀有效？
# 1. 帮助模型区分 query 和 document 的不同语义空间
# 2. Query 通常是信息需求，document 是信息供给
# 3. 同样的词，在 query 和 doc 中权重不同
```

### 6.4 In-batch Negatives（批量负样本）

```python
# 关键优化：利用同一个 batch 内的其他样本作为负样本

class InBatchNegatives:
    def forward(self, batch):
        """
        batch = [(q1, d1+), (q2, d2+), (q3, d3+), ...]
        """
        queries = [item['query'] for item in batch]
        positives = [item['positive'] for item in batch]

        # 编码
        q_embs = self.encoder.encode(queries)      # (batch, dim)
        p_embs = self.encoder.encode(positives)    # (batch, dim)

        # 计算相似度矩阵 (batch x batch)
        # sim_matrix[i, j] = similarity(q_i, p_j)
        sim_matrix = torch.matmul(q_embs, p_embs.T) / self.temperature

        # 对角线是正样本 (q1-d1+, q2-d2+, ...)
        # 非对角线是负样本 (q1-d2+, q1-d3+, ...)
        labels = torch.arange(len(batch))

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

# 优势：
# - 每个样本有 (batch_size - 1) 个免费负样本
# - batch_size=64 时，每个样本有 63 个负样本
# - 负样本质量高（来自真实数据，非随机）
```

---

## 七、模型架构

### 主流架构选择

```
Transformer Encoder 架构家族

BERT (Google, 2018)
├─ 12层, 768维, 110M 参数
├─ 双向编码，上下文感知
└─ 大 embedding 模型的基石

RoBERTa (Facebook, 2019)
├─ BERT 改进版
├─ 更长训练、更大 batch
└─ 效果稳定提升

DistilBERT (HuggingFace, 2019)
├─ BERT 蒸馏版
├─ 6层, 768维, 40% 参数
├─ 保留 97% 性能
└─ 速度提升 60%

现代 embedding 基座
├─ BGE: 基于 BERT/RoBERTa
├─ GTE: 基于 Qwen（LLM 蒸馏）
├─ E5: 基于 BERT
└─ Nomic: 基于 BERT，架构创新
```

---

## 八、维度选择

### 维度权衡表

| 维度 | 存储成本 | 检索速度 | 精度 | 适用场景 |
|:-----|:---------|:---------|:-----|:---------|
| **256** | 极低 | 极快 | 中等 | 边缘设备、实时检索 |
| **384** | 低 | 很快 | 良好 | 小规模系统 |
| **512** | 较低 | 快 | 良好 | 平衡选择 |
| **768** | 中等 | 中等 | 很好 | 大多数场景 |
| **1024** | 较高 | 较慢 | 优秀 | 追求质量 |
| **3072** | 高 | 慢 | 极佳 | 不差钱、不差空间 |
| **4096+** | 很高 | 很慢 | 最佳 | 研究级、特殊需求 |

### 存储成本估算（100 万文档）

```
256 维   ≈ 1 GB
768 维   ≈ 2.8 GB
1024 维  ≈ 3.8 GB
3072 维  ≈ 11.4 GB
4096 维  ≈ 15.3 GB
```

---

## 九、向量数据库选型

| 数据库 | 推荐场景 | 优势 | 劣势 | 许可证 |
|:-------|:---------|:-----|:-----|:-------|
| **Chroma** | 本地开发、原型 | 零配置、Python 原生 | 规模受限 | Apache 2.0 |
| **Qdrant** | 生产环境 | 性能优秀、API 友好 | 学习曲线 | Apache 2.0 |
| **Milvus** | 大规模部署 | 可扩展性强 | 复杂度高 | Apache 2.0 |
| **Pinecone** | 快速上线 | 托管服务、零运维 | 成本高、数据出境 | 专有 |
| **Weaviate** | 语义搜索 | 多模态支持 | 资源占用高 | BSD 3-Clause |

### 数据库选择决策树

```
你的需求是？

├─ 本地原型/学习
│   └─ Chroma ✅
│
├─ 生产环境 + < 1000万向量
│   └─ Qdrant ✅
│
├─ 生产环境 + > 1000万向量
│   └─ Milvus ✅
│
├─ 不差钱 + 快速上线
│   └─ Pinecone ✅
│
└─ 多模态（文本+图像）
    └─ Weaviate ✅
```

---

## 十、检索优化技巧

### 10.1 混合检索（Hybrid Search）

```python
# 结合向量检索和关键词检索（BM25）

from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """
    混合检索：向量相似度 + BM25 分数
    """
    def __init__(self, documents, embedding_model, alpha=0.5):
        self.documents = documents
        self.embedding_model = embedding_model
        self.alpha = alpha  # 向量检索权重

        # 初始化 BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # 预计算文档向量
        self.doc_embeddings = embedding_model.encode(documents)

    def retrieve(self, query, top_k=10):
        # 向量检索分数
        query_emb = self.embedding_model.encode(query)
        vector_scores = cosine_similarity(
            query_emb.reshape(1, -1),
            self.doc_embeddings
        )[0]

        # BM25 分数
        bm25_scores = self.bm25.get_scores(query.split())

        # 归一化
        vector_scores = self._normalize(vector_scores)
        bm25_scores = self._normalize(bm25_scores)

        # 加权融合
        final_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores

        # 返回 top-k
        top_indices = np.argsort(final_scores)[-top_k:][::-1]

        return [
            {'doc': self.documents[i], 'score': final_scores[i]}
            for i in top_indices
        ]
```

### 10.2 重排序（Reranking）

```python
# 两阶段检索：快速召回 → 精细排序

class TwoStageRetriever:
    def __init__(self, retriever, reranker, top_k=100, rerank_top_k=10):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query):
        # 第一阶段：快速召回大量候选
        candidates = self.retriever.retrieve(query, top_k=self.top_k)

        # 第二阶段：精细排序
        reranked = self.reranker.rerank(query, candidates)

        return reranked[:self.rerank_top_k]

# 性能对比
# 向量检索：~100 ms（召回 100 个）
# 重排序：   ~200 ms（排序 100 个）
# 总计：     ~300 ms，但质量大幅提升
```

---

## 十一、性能优化

### 批量处理加速

```python
import torch
from tqdm import tqdm

def batch_encode(model, texts, batch_size=256, show_progress=True):
    """
    批量编码，充分利用 GPU
    """
    model.eval()
    embeddings = []

    iterator = tqdm(range(0, len(texts), batch_size),
                    disable=not show_progress)

    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            batch_embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=True
            )

            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

# 性能对比（10 万文档）
# 逐个编码：~2 小时
# 批量编码（batch_size=256）：~10 分钟
# 加速比：~12x
```

### FAISS 加速

```python
import faiss
import numpy as np

class FAISSRetriever:
    def __init__(self, embeddings, index_type="IVF"):
        self.dimension = embeddings.shape[1]

        if index_type == "IVF":
            # 倒排索引：适合大规模数据
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                100  # nlist：聚类中心数量
            )
            self.index.train(embeddings.astype('float32'))
        elif index_type == "HNSW":
            # 图索引：更快但内存占用高
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)

        self.index.add(embeddings.astype('float32'))

    def search(self, query_embedding, top_k=10):
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        return indices[0]

# 性能对比（100 万向量，768 维）
# 暴力搜索：   ~500 ms
# IVF 索引：   ~50 ms
# HNSW 索引：  ~10 ms
```

---

## 十二、质量评估

```python
def evaluate_retrieval(queries, ground_truth, retriever, top_k=10):
    """
    评估检索质量

    Args:
        queries: 查询列表
        ground_truth: 每个查询的相关文档 ID 列表
        retriever: 检索器
        top_k: 检索返回数量
    """
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'mrr': [],  # Mean Reciprocal Rank
    }

    for query, relevant_ids in zip(queries, ground_truth):
        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r['id'] for r in results]

        # Precision@K
        precision = len(set(retrieved_ids) & set(relevant_ids)) / top_k
        metrics['precision@k'].append(precision)

        # Recall@K
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
        metrics['recall@k'].append(recall)

        # MRR（第一个相关文档的倒数排名）
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                mrr = 1 / (i + 1)
                metrics['mrr'].append(mrr)
                break
        else:
            metrics['mrr'].append(0)

    return {
        'precision@k': np.mean(metrics['precision@k']),
        'recall@k': np.mean(metrics['recall@k']),
        'mrr': np.mean(metrics['mrr'])
    }
```

---

## 十三、MRL 嵌套式嵌入

### Matryoshka Representations Learning (MRL)

```
俄罗斯套娃式嵌入：一个向量，多种长度

传统 Embedding 问题：
  - 1024 维模型只能用 1024 维
  - 想降维？重新训练或简单截断（效果差）

MRL 解决方案：
  - 训练时同时优化多个维度 [d/64, d/32, ..., d]
  - 推理时可自由选择维度
  - 小维度：快速检索、低存储
  - 大维度：高质量表示
```

```python
from sentence_transformers import SentenceTransformer

# 使用支持 MRL 的模型（如 Nomic v1.5）
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

text = "RAG 2.0 是端到端联合训练的检索增强生成方法"

# 灵活选择维度
embedding_64 = model.encode(text, output_dim=64)    # 快速检索
embedding_256 = model.encode(text, output_dim=256)  # 平衡
embedding_768 = model.encode(text, output_dim=768)  # 高质量

# 优势：
# - 同一个模型，支持不同应用场景
# - 粗排用 64 维（快速）
# - 精排用 768 维（精确）
# - 存储成本降低 10x（用 64 维）
```

---

## 十四、Late Chunking

```
传统分块问题：
  文章 → 先分块 → 分别编码 → 向量
  问题：分块后的片段失去上下文

Late Chunking 创新：
  文章 → 先整体编码 → 后分块 → 向量
  优势：每个 chunk 都有完整上下文
```

```python
# Late Chunking 实现
class LateChunking:
    def __init__(self, model, chunk_size=512):
        self.model = model
        self.chunk_size = chunk_size

    def encode_late_chunking(self, text):
        # 获取所有 token 的嵌入
        token_embeddings = self.model.encode_tokens(text)

        # 然后按 chunk 划分 token 嵌入
        chunk_embeddings = []
        for i in range(0, len(token_embeddings), self.chunk_size):
            chunk_tokens = token_embeddings[i:i + self.chunk_size]
            # 聚合（平均池化）
            chunk_emb = chunk_tokens.mean(dim=0)
            chunk_embeddings.append(chunk_emb)

        return chunk_embeddings

# 效果对比（实验数据）
# 传统分块：  68.5%
# Late Chunking：74.2%
# 提升：    +5.7%
```

---

## 十五、多语言与多模态

### 多语言 Embedding

| 模型 | 语言数 | 中文表现 | 特点 |
|:-----|:-------|:---------|:-----|
| **BGE-M3** | 100+ | ⭐⭐⭐⭐⭐ | 最强多语言，中英均衡 |
| **GTE-multilingual** | 100+ | ⭐⭐⭐⭐ | 质量优秀 |
| **E5-mistral** | 100+ | ⭐⭐⭐⭐ | LLM 级别能力 |
| **Cohere embed-v3** | 100+ | ⭐⭐⭐ | 商业 API |
| **LaBSE** | 109 | ⭐⭐⭐ | Google 开源 |

### 多模态 Embedding

```python
# Jina CLIP / OpenAI CLIP：文本+图像
import clip

# 加载模型
model, preprocess = clip.load("ViT-B/32", device="cuda")

# 编码文本
text = "一只猫在睡觉"
text_tokens = clip.tokenize([text]).to("cuda")
text_features = model.encode_text(text_tokens)

# 编码图像
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to("cuda")
image_features = model.encode_image(image)

# 计算相似度（文本和图像在同一空间！）
similarity = torch.cosine_similarity(text_features, image_features)
```

---

## 十六、LLM-as-Embedding

```python
# 新趋势：直接用 LLM 生成 embedding
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def llm_embedding(text):
    # 获取 LLM 的隐藏状态
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 使用最后一层的平均池化
    last_hidden = outputs.hidden_states[-1]
    embedding = last_hidden.mean(dim=1)

    return embedding.squeeze(0)

# 优势：
# - 利用 LLM 的强大理解能力
# - 无需单独训练 embedding 模型

# 劣势：
# - 推理成本高（7B 模型 vs 300M 模型）
# - 速度慢
# - 资源要求高
```

---

## 十七、实践建议总结

```
2025-2026 最佳实践：

1. 模型选择
   - 通用：BGE-M3
   - 生产：Voyage-3 / OpenAI
   - 中文：BGE-M3 / GTE-Qwen
   - 私有：Nomic v1.5

2. 优化策略
   - 混合检索（向量 + BM25）
   - 重排序（两阶段检索）
   - 查询改写（LLM 辅助）

3. 基础设施
   - 小规模：Chroma
   - 生产：Qdrant
   - 大规模：Milvus
   - 快速上线：Pinecone

4. 性能优化
   - 批量编码
   - GPU 加速
   - FAISS 索引
   - MRL 灵活维度

5. 前沿技术
   - 关注 MRL 模型
   - 尝试 Late Chunking
   - 探索多模态
```

---

## 十八、相关笔记

### 深入学习
- [[AI研究/AI学习/03-实战应用/RAG全面解析]] - RAG 完整知识体系
- [[AI研究/AI学习/04-深入前沿/RAG 2.0 全面解析]] - RAG 系统进阶
- [[AI研究/AI学习/02-模型原理/检索方式全面对比]] - 不同检索方式详解
- [[AI研究/AI学习/02-模型原理/Transformer全面解析]] - LLM 基础架构

### 实战应用
- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - RAG 1.0 项目实战

---

#Embedding #向量检索 #深度学习 #对比学习 #RAG
