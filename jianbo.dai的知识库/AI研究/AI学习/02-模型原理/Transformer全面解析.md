---
title: Transformer 全面解析
date: 2026-02-28
tags:
  - Transformer
  - 注意力机制
  - NLP
  - 深度学习
  - 架构原理
status: active
---

# Transformer 全面解析

> [!info] 说明
> 本笔记系统介绍Transformer的原理、架构变体、主流模型、性能优化及实战应用，覆盖2026年最新的模型发展（Llama 3、DeepSeek等）

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **术语速查**：[[AI研究/AI学习/常见术语对照]] - AI/ML术语词典

---

## 📑 目录

> [!tip] 使用说明
> 点击下方的任何章节链接，即可跳转到对应内容（支持 `Ctrl/Cmd + Click` 在新面板打开）

### 基础理论
- [[#一、核心原理]]
  - [[#1.1 什么是Transformer]]
  - [[#1.2 Self-Attention机制]]
  - [[#1.3 Multi-Head Attention]]
  - [[#1.4 Position Encoding]]
  - [[#1.5 Feed-Forward Network]]
  - [[#1.6 Layer Normalization]]
  - [[#1.7 完整Transformer Block]]
- [[#二、为什么Transformer成为主流]]
  - [[#2.1 与RNN/LSTM对比]]
  - [[#2.2 与CNN对比]]
  - [[#2.3 并行计算优势]]
  - [[#2.4 可扩展性]]

### 架构详解
- [[#三、主流Transformer架构详解]]
  - [[#3.1 Encoder-only架构]]
  - [[#3.2 Decoder-only架构]]
  - [[#3.3 Encoder-Decoder架构]]
  - [[#3.4 Vision Transformer (ViT)]]
  - [[#3.5 混合与特殊架构]]
- [[#四、架构对比总结]]
  - [[#4.1 架构特性对比]]
  - [[#4.2 任务适用对比]]
  - [[#4.3 性能对比]]
  - [[#4.4 选择决策树]]

### 实战应用
- [[#五、开源框架与模型]]
  - [[#5.1 Hugging Face Transformers]]
  - [[#5.2 开源预训练模型]]
  - [[#5.3 模型选择指南]]
- [[#六、性能与耗时分析]]
  - [[#6.1 计算复杂度]]
  - [[#6.2 内存占用]]
  - [[#6.3 KV Cache优化]]
  - [[#6.4 量化（Quantization）]]
  - [[#6.5 Flash Attention]]
  - [[#6.6 性能优化总结]]
- [[#七、实战案例]]
  - [[#7.1 文本分类（BERT微调）]]
  - [[#7.2 命名实体识别（NER）]]
  - [[#7.3 机器翻译（T5、NLLB）]]
  - [[#7.4 文本生成（GPT、Llama）]]
  - [[#7.5 问答系统（RAG + LLM）]]
  - [[#7.6 摘要生成（BART、T5）]]
  - [[#7.7 代码生成（CodeLlama、StarCoder）]]
  - [[#7.8 多模态（CLIP、GPT-4V）]]

### 参考指南
- [[#八、方案选择建议]]
  - [[#8.1 按任务类型选择]]
  - [[#8.2 按资源限制选择]]
  - [[#8.3 按语言选择]]
  - [[#8.4 部署方案选择]]
- [[#九、常见问题]]
  - [[#9.1 BERT vs GPT的选择]]
  - [[#9.2 上下文长度限制]]
  - [[#9.3 幻觉问题（Hallucination）]]
  - [[#9.4 训练不稳定]]
- [[#十、学习资源]]
  - [[#10.1 核心论文]]
  - [[#10.2 在线教程]]
  - [[#10.3 书籍]]
  - [[#10.4 实践平台]]
- [[#十一、相关笔记]]
  - [[#深入学习]]
  - [[#实战应用]]
  - [[#基础知识]]

---

## 一、核心原理

### 1.1 什么是Transformer

**Transformer** 是一种完全基于注意力机制的深度学习架构，于2017年由Vaswani等人在论文《Attention Is All You Need》中提出。

```
传统序列模型 vs Transformer：

RNN/LSTM:
├─ 序列化计算（t时刻依赖t-1时刻）
├─ 难以并行训练
└─ 长距离依赖信息丢失

CNN (Temporal Convolution):
├─ 可以并行计算
├─ 感受野有限（需要堆叠多层）
└─ 难以捕捉非常长距离的依赖

Transformer:
├─ 完全并行计算
├─ 全局感受野（任意位置间直接连接）
└─ 长距离依赖完美捕捉
```

### 1.2 Self-Attention机制

Self-Attention是Transformer的核心创新，允许序列中的每个位置直接关注其他所有位置。

#### 直观理解

```
句子: "The cat sat on the mat"

处理 "cat" 时:
├─ Query (Q): "cat" 想要找什么？
├─ Key (K): 每个词能提供什么？
├─ Value (V): 每个词的实际内容
└─ Attention: Q与K匹配度，加权V

结果:
├─ Attention("cat", "The")   = 低  (不相关)
├─ Attention("cat", "cat")   = 高  (自身)
├─ Attention("cat", "sat")   = 中  (主谓关系)
└─ Attention("cat", "mat")   = 中  (位置关系)
```

#### 数学表达

**Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: Query矩阵
- $K \in \mathbb{R}^{n \times d_k}$: Key矩阵
- $V \in \mathbb{R}^{n \times d_v}$: Value矩阵
- $d_k$: Key/Query维度
- $\sqrt{d_k}$: 缩放因子（防止softmax饱和）

**计算步骤**:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        mask: (batch, seq_len, seq_len) 可选
    """
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 应用mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### 1.3 Multi-Head Attention

多头注意力允许模型从不同的表示子空间和位置关注信息。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$

$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中 $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 线性投影并分头
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k)

        # 2. 转置以便并行计算
        # (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 4. 拼接多头
        # (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5. 最终线性变换
        output = self.W_o(attn_output)

        return output, attn_weights
```

**为什么多头？**

| 头 | 学习内容 | 示例 |
|:---|:--------|:-----|
| 头1 | 语法关系 | 主谓一致 |
| 头2 | 语义关联 | 近义词 |
| 头3 | 指代消解 | it → cat |
| 头4 | 长距离依赖 | 第一句...最后一句 |

### 1.4 Position Encoding

由于Self-Attention本身是置换不变的，需要显式注入位置信息。

#### Sinusoidal位置编码（原始方案）

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        # sin和cos交替
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为buffer（不是参数，但会保存到state_dict）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

#### 现代位置编码方案

| 方案 | 特点 | 使用模型 | 优势 |
|:-----|:-----|:---------|:-----|
| **Sinusoidal** | 固定函数 | 原始Transformer | 外推能力强 |
| **Learned** | 可学习参数 | GPT-2/3, BERT | 灵活，固定长度 |
| **RoPE** | 旋转变换 | LLaMA, Mistral | 相对位置，外推 |
| **ALiBi** | 线性偏置 | BLOOM, MPT | 简单，优秀外推 |
| **xPos** | 扩展RoPE | 长序列模型 | 更好的长序列 |

#### RoPE详解（当前主流）

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    当前主流方案（LLaMA, Mistral, DeepSeek等）
    """
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model

        # 计算旋转角度
        theta = 1.0 / (10000 ** torch.arange(0, d_model, 2).float() / d_model)
        seq_idx = torch.arange(max_seq_len).float()
        idx_theta = torch.outer(seq_idx, theta)

        # 计算cos和sin
        self.register_buffer('cos', torch.cos(idx_theta))
        self.register_buffer('sin', torch.sin(idx_theta))

    def rotate(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 分成两半
        x1 = x[..., ::2]  # 偶数维度
        x2 = x[..., 1::2]  # 奇数维度

        # 应用旋转
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # 拼接
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)

        return x_rot
```

### 1.5 Feed-Forward Network

Position-wise FFN对每个位置独立处理：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # 现代用GELU，原始用ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
```

**设计要点**:
- $d_{ff}$ 通常是 $d_{model}$ 的4倍
- 等价于对每个位置独立应用一个两层MLP
- 引入非线性变换

### 1.6 Layer Normalization

LayerNorm是Transformer稳定训练的关键：

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 对每个样本的所有维度归一化
        # x: (batch, seq_len, d_model)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

**Pre-Norm vs Post-Norm**:

```python
# Post-Norm (原始Transformer)
def transformer_block_post(x):
    attn_out = attention(x)
    x = x + layer_norm(attn_out)  # Norm在残差之后
    ffn_out = ffn(x)
    x = x + layer_norm(ffn_out)
    return x

# Pre-Norm (现代主流，更稳定)
def transformer_block_pre(x):
    attn_out = attention(layer_norm(x))  # Norm在残差之前
    x = x + attn_out
    ffn_out = ffn(layer_norm(x))
    x = x + ffn_out
    return x
```

### 1.7 完整Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Pre-Norm结构
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with Pre-Norm
        attn_out, _ = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_out)

        # FFN with Pre-Norm
        x = x + self.ffn(self.norm2(x))

        return x
```

---

## 二、为什么Transformer成为主流

### 2.1 与RNN/LSTM对比

| 维度 | RNN/LSTM | Transformer | 说明 |
|:-----|:---------|:------------|:-----|
| **并行计算** | ❌ 序列依赖 | ✅ 完全并行 | 训练速度差异巨大 |
| **长距离依赖** | ❌ 梯度消失/爆炸 | ✅ O(1)路径长度 | 任意位置直接连接 |
| **感受野** | 递增 | 全局 | 第一层就能看到全部序列 |
| **可扩展性** | 受限 | 优秀 | 可堆叠更多层 |
| **性能** | 中等 | 优秀 | 各项NLP任务SOTA |

```
长距离依赖示例:

句子: "The man who wore a red jacket and blue jeans
        and carried a black umbrella walked quickly
        through the rainy street because he was late."

问题: "he" 指代谁？

RNN: 信息需要传递30+词，逐渐丢失
Transformer: "he" 直接关注 "man"，距离不影响
```

### 2.2 与CNN对比

| 维度 | CNN (Temporal) | Transformer |
|:-----|:---------------|:------------|
| **感受野** | 需要堆叠多层扩大 | 天然全局 |
| **感受野计算** | 线性增长 | O(1) |
| **归纳偏置** | 局部性 | 无（灵活性强） |
| **长序列** | 需要深层网络 | 无需深层 |

**CNN的优势**: 归纳偏置适合数据量少的场景
**Transformer的优势**: 大数据下表现更好，上限更高

### 2.3 并行计算优势

```
RNN训练:
t=1: ───▶ t=2: ───▶ t=3: ───▶ t=4:
(串行计算，无法并行)

Transformer训练:
t=1 ──┐
t=2 ──┤
t=3 ──┼──▶ 全部位置同时计算
t=4 ──┘

结果: Transformer训练速度提升10-100倍
```

### 2.4 可扩展性

Transformer展现了惊人的缩放法则（Scaling Laws）：

```
性能 ∝ (参数量、数据量、计算量)^α

模型规模演进:
2017: Transformer Base (110M)
2018: BERT-Base (110M), BERT-Large (340M)
2019: GPT-2 (1.5B)
2020: GPT-3 (175B)
2023: LLaMA 2 (70B)
2024: GPT-4 (~1.8T), Llama 3 (405B)
2025: DeepSeek V3 (671B MoE)
```

---

## 三、主流Transformer架构详解

### 3.1 Encoder-only架构

**代表模型**: BERT, RoBERTa, ALBERT, DeBERTa, ELMo (早期)

#### 架构原理

```
Input Embedding
      │
      ▼
┌─────────────────────────────────────────────────┐
│           Encoder Stack (N层)                   │
│  ┌─────────────────────────────────────────┐   │
│  │ ┌─────────┐    ┌──────────────────────┐ │   │
│  │ │  Norm   │    │     Feed Forward     │ │   │
│  │ └────┬────┘    └──────────┬───────────┘ │   │
│  │      │                    │              │   │
│  │      ▼                    ▼              │   │
│  │  Multi-Head Self-Attention (双向)       │   │
│  │      │                    ▲              │   │
│  │      └────────────────────┘              │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
      │
      ▼
  [CLS] Token → 分类头
  各Token → 序列标注
```

#### 特点

| 特点 | 说明 |
|:-----|:-----|
| **双向注意力** | 每个token可以关注上下文 |
| **Masked LM** | 预训练时随机mask部分token |
| **Next Sentence** | 预测句子关系（部分模型） |
| **适合理解** | 分类、NER、QA等理解任务 |

#### 主流Encoder模型

| 模型 | 参数 | 发布时间 | 特点 |
|:-----|:-----|:---------|:-----|
| **BERT-Base** | 110M | 2018 | 双向Transformer |
| **BERT-Large** | 340M | 2018 | 24层，性能更好 |
| **RoBERTa** | 355M | 2019 | 优化训练策略 |
| **ALBERT** | 12M | 2019 | 参数共享，轻量化 |
| **DeBERTa** | 1.5B | 2020 | 解耦注意力 |
| **ELECTRA** | 14M-330M | 2020 | 替代token检测 |

#### 代码示例：文本分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 加载预训练模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 准备数据
dataset = load_dataset("glue", "sst2")  # 情感分析

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. 训练配置
training_args = TrainingArguments(
    output_dir="./bert-sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 4. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

# 5. 推理
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "Positive" if probs[0][1] > 0.5 else "Negative"

print(predict_sentiment("This movie is absolutely amazing!"))  # Positive
```

#### 适用场景

| 任务 | 说明 | 模型选择 |
|:-----|:-----|:---------|
| **文本分类** | 情感分析、主题分类 | BERT, RoBERTa |
| **命名实体识别** | 人名、地名、机构名提取 | BERT+CRF |
| **问答系统** | 抽取式问答 | BERT, RoBERTa |
| **语义相似度** | 句子匹配 | SBERT |
| **自然语言推理** | 前提-假设关系 | RoBERTa, DeBERTa |

---

### 3.2 Decoder-only架构

**代表模型**: GPT系列, LLaMA系列, Mistral, Qwen, DeepSeek

#### 架构原理

```
Input Embedding
      │
      ▼
┌─────────────────────────────────────────────────┐
│           Decoder Stack (N层)                   │
│  ┌─────────────────────────────────────────┐   │
│  │ ┌─────────┐    ┌──────────────────────┐ │   │
│  │ │  Norm   │    │     Feed Forward     │ │   │
│  │ └────┬────┘    └──────────┬───────────┘ │   │
│  │      │                    │              │   │
│  │      ▼                    ▼              │   │
│  │  Masked Self-Attention (因果)            │   │
│  │      │                    ▲              │   │
│  │      └────────────────────┘              │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
      │
      ▼
    LM Head → 概率分布 (词表大小)
```

#### 因果注意力（Causal Attention）

```python
def causal_mask(seq_len):
    """
    创建因果mask，确保每个位置只能看到自己和之前的位置
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

# 可视化
# seq_len=5时的mask:
# [[1, 0, 0, 0, 0],   位置0只能看自己
#  [1, 1, 0, 0, 0],   位置1可以看0,1
#  [1, 1, 1, 0, 0],   位置2可以看0,1,2
#  [1, 1, 1, 1, 0],   位置3可以看0,1,2,3
#  [1, 1, 1, 1, 1]]   位置4可以看全部
```

#### 主流Decoder模型对比

| 模型 | 参数 | 发布时间 | 特点 | 上下文 |
|:-----|:-----|:---------|:-----|:-------|
| **GPT-3** | 175B | 2020 | Few-shot学习 | 2048 |
| **GPT-4** | ~1.8T | 2023 | 多模态，强大推理 | 8192-32768 |
| **LLaMA** | 7B-65B | 2023 | 开源高效 | 2048 |
| **LLaMA 2** | 7B-70B | 2023 | 对话优化 | 4096 |
| **LLaMA 3** | 8B-405B | 2024 | 8K上下文，强推理 | 8192 |
| **Mistral 7B** | 7B | 2023 | GQA, Sliding Window | 8192 |
| **Mixtral 8x7B** | 47B | 2024 | MoE架构 | 32768 |
| **Qwen 2.5** | 0.5B-72B | 2024 | 多语言强 | 32768 |
| **DeepSeek V2** | 236B | 2024 | MoE + MLA | 128K |
| **DeepSeek V3** | 671B | 2025 | MoE优化 | 64K |

#### 代码示例：文本生成

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型
model_name = "meta-llama/Llama-3-8B"  # 或 "deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 文本生成
def generate_text(prompt, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
prompt = "Write a Python function to implement binary search:"
response = generate_text(prompt, max_new_tokens=512)
print(response)

# 3. 流式生成
def stream_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 256,
        "temperature": 0.7
    }

    import threading
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)
```

#### 训练策略：微调LLM

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. 加载基础模型
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 准备LoRA微调
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,              # LoRA秩
    lora_alpha=32,     # LoRA缩放
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 可训练参数占比

# 3. 准备数据
dataset = load_dataset("json", data_files="my_instructions.json")

def format_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_prompt)

# 4. 训练
training_args = TrainingArguments(
    output_dir="./llama-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

# 5. 保存和推理
model.save_pretrained("./my-finetuned-llama")
```

#### 适用场景

| 任务 | 说明 | 模型选择 |
|:-----|:-----|:---------|
| **文本生成** | 创意写作、代码生成 | GPT-4, Llama 3 |
| **对话系统** | 聊天机器人 | Llama 3, DeepSeek |
| **代码生成** | 程序补全、代码解释 | DeepSeek Coder, CodeLlama |
| **长文本生成** | 报告、文章 | GPT-4, Claude |
| **通用AI** | 多任务处理 | GPT-4, DeepSeek V3 |

---

### 3.3 Encoder-Decoder架构

**代表模型**: T5, BART, mBART, M2M100, NLLB

#### 架构原理

```
     Encoder (双向)                    Decoder (因果)
┌───────────────────┐           ┌───────────────────┐
│                   │           │                   │
│  Encoder Stack × N│──────────▶│  Decoder Stack × N│
│                   │ Cross-Attn│                   │
└───────────────────┘           └───────────────────┘
        │                              │
        ▼                              ▼
    Input                        Output Probs
```

#### 交叉注意力（Cross Attention）

```python
class CrossAttention(nn.Module):
    """
    Decoder中的交叉注意力
    Q来自Decoder，K和V来自Encoder输出
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):
        """
        x: Decoder当前层输出
        encoder_output: Encoder最终输出
        """
        attn_out, _ = self.attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=mask
        )
        return self.norm(x + attn_out)
```

#### 主流Encoder-Decoder模型

| 模型 | 参数 | 发布时间 | 特点 | 任务 |
|:-----|:-----|:---------|:-----|:-----|
| **原始Transformer** | 213M | 2017 | 机器翻译基线 | 翻译 |
| **T5-Base** | 220M | 2019 | Text-to-Text框架 | 通用NLP |
| **T5-11B** | 11B | 2019 | 大规模T5 | 通用NLP |
| **BART-Large** | 400M | 2019 | 去噪自编码 | 摘要、翻译 |
| **mBART-50** | 680M | 2020 | 多语言翻译 | 多语言翻译 |
| **NLLB-200** | 54.5B | 2022 | 200语言翻译 | 低资源语言 |
| **M2M100** | 12B | 2020 | 多对多翻译 | 多语言翻译 |

#### T5: Text-to-Text框架

T5将所有NLP任务转化为文本到文本格式：

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5
model_name = "t5-base"  # 或 "t5-large", "t5-11b"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 不同任务的prompt格式
def t5_format(task_type, text):
    prefixes = {
        "summarize": "summarize: ",
        "translate": "translate English to German: ",
        "classify": "sst2 sentence: ",
        "qa": "question: ",
        "generate": "generate: "
    }
    return prefixes.get(task_type, "") + text

# 1. 文本摘要
text = "The tower is 324 metres (1,063 ft) tall..."
input_text = t5_format("summarize", text)
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(**inputs, max_length=150, min_length=40, length_penalty=2.0)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. 翻译
input_text = t5_format("translate", "The house is wonderful.")
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # "Das Haus ist wunderbar."

# 3. 情感分类
input_text = t5_format("classify", "This movie is great!")
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)  # "positive"
```

#### 代码示例：机器翻译

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# 1. 加载翻译模型
model_name = "facebook/nllb-200-distilled-600M"  # 支持200语言
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 设置源语言和目标语言
tokenizer.src_lang = "eng_Latn"
tgt_lang_code = "zho_Hans"  # 简体中文

# 2. 翻译
def translate(text, src_lang="eng_Latn", tgt_lang="zho_Hans"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=256
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
result = translate("Hello, how are you today?")
print(result)  # "你好，你好吗？"

# 3. 微调到特定领域
dataset = load_dataset("opus100", "en-zh")

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["zh"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-en-zh",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
```

#### 适用场景

| 任务 | 说明 | 推荐模型 |
|:-----|:-----|:---------|
| **机器翻译** | 语言互译 | NLLB, M2M100 |
| **文本摘要** | 生成摘要 | BART, T5 |
| **文档改写** | 句子重写 | T5, PEGASUS |
| **代码翻译** | 跨语言代码翻译 | CodeT5, TransCoder |

---

### 3.4 Vision Transformer (ViT)

**代表模型**: ViT, Swin Transformer, MAE, CLIP, DINO

#### 架构原理

将图像分割为patches，将每个patch视为一个"token"：

```
图像 (224×224×3)
      │
      ▼
┌─────────────────────────────────────┐
│  分割成patches (16×16, 共196个patch) │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  展平并投影到 embedding 维度         │
│  (196 patches × 768-dim)            │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  添加位置编码 + [CLS] token          │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│     Transformer Encoder (12层)       │
└─────────────────────────────────────┘
      │
      ▼
  [CLS] token → MLP Head → 分类
```

#### 代码示例：ViT图像分类

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# 1. 加载预训练ViT
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 2. 图像分类
def classify_image(image_path):
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    # 获取Top-5预测
    top5_probs = torch.nn.functional.softmax(logits, dim=-1)[0].topk(5)

    results = []
    for score, idx in zip(top5_probs.values, top5_probs.indices):
        label = model.config.id2label[idx.item()]
        results.append(f"{label}: {score.item():.4f}")

    return predicted_class, results

# 示例
label, top5 = classify_image("cat.jpg")
print(f"Predicted: {label}")
print("\nTop-5:")
for result in top5:
    print(f"  {result}")
```

#### Swin Transformer

Swin引入了层次结构和移位窗口注意力：

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Swin使用分层特征，类似CNN
model_name = "microsoft/swin-base-patch4-window7-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Swin的优势:
# 1. 层次化特征表示（适合检测、分割）
# 2. 窗口注意力降低复杂度（线性复杂度）
# 3. 移位窗口实现跨窗口连接
```

#### 多模态：CLIP

```python
from transformers import CLIPProcessor, CLIPModel

# CLIP: 连接图像和文本
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# 图像-文本匹配
def clip_match(image_path, texts):
    image = Image.open(image_path)

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # 图像到文本相似度
    probs = logits_per_image.softmax(dim=-1)

    return probs[0].tolist()

# 示例
texts = ["a cat", "a dog", "a bird", "a car"]
probs = clip_match("cat.jpg", texts)
for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.4f}")
```

#### Vision模型对比

| 模型 | 参数 | 发布时间 | 特点 | 应用 |
|:-----|:-----|:---------|:-----|:-----|
| **ViT-Base** | 86M | 2020 | 纯Transformer视觉 | 图像分类 |
| **Swin-Base** | 88M | 2021 | 层次化，窗口注意力 | 检测、分割 |
| **MAE** | 340M | 2022 | 掩码自编码预训练 | 自监督学习 |
| **CLIP** | 400M | 2021 | 图文对比学习 | 多模态 |
| **DINOv2** | 300M | 2023 | 自监督，强表示 | 通用视觉 |
| **SAM** | 636M | 2023 | 任意分割 | 图像分割 |

---

### 3.5 混合与特殊架构

#### Mixture of Experts (MoE)

```python
# Mixtral 8x7B: 8个专家，每次激活2个
from transformers import MixtralForCausalLM, AutoTokenizer

model_name = "mistralai/Mixtral-8x7B-v0.1"
model = MixtralForCausalLM.from_pretrained(model_name)

# MoE结构:
# - 47B总参数
# - 每个token只使用13B参数（激活2/8专家）
# - 推理效率接近13B密集模型
```

#### 长上下文模型

| 模型 | 上下文长度 | 技术 |
|:-----|:----------|:-----|
| **Claude 2/3** | 100K-200K | 特殊注意力模式 |
| **GPT-4 Turbo** | 128K | 稀疏注意力 |
| **Gemini 1.5 Pro** | 1M | 多种注意力混合 |
| **Moon Dream** | 32K | 旋转编码优化 |

#### 检索增强生成 (RAG)

```python
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import DPRQuestionEncoder, DPRContextEncoder

# RAG结合检索和生成
# 1. 检索器：DPR (Dense Passage Retrieval)
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# 2. 生成器：BART
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index_name="custom",
    passages=your_passages
)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# 3. 推理
question = "What is the capital of France?"
input_dict = tokenizer(question, return_tensors="pt")
generated = model.generate(**input_dict)
answer = tokenizer.decode(generated[0], skip_special_tokens=True)
```

---

## 四、架构对比总结

### 4.1 架构特性对比

| 特性 | Encoder-only | Decoder-only | Encoder-Decoder |
|:-----|:-------------|:-------------|:---------------|
| **注意力类型** | 双向 | 因果（单向） | Enc:双向, Dec:单向+交叉 |
| **训练方式** | Masked LM | Causal LM | Span Corruption |
| **并行训练** | 完全并行 | 完全并行 | 完全并行 |
| **并行推理** | ✅ | ❌ (序列) | ❌ (序列) |
| **双向理解** | ✅ | ❌ | ✅ (Encoder部分) |
| **生成能力** | 弱 | 强 | 强 |
| **理解能力** | 强 | 中等 | 强 |
| **参数效率** | 高 | 高 | 中等 |
| **训练数据** | 文本对 | 纯文本 | 文本对 |

### 4.2 任务适用对比

| 任务类型 | 最佳架构 | 推荐模型 | 理由 |
|:---------|:---------|:---------|:-----|
| **文本分类** | Encoder-only | BERT, RoBERTa | 双向理解更好 |
| **命名实体识别** | Encoder-only | BERT+CRF | 需要双向上下文 |
| **机器翻译** | Encoder-Decoder | NLLB, T5 | 需要理解+生成 |
| **文本摘要** | Encoder-Decoder | BART, T5 | 需要理解+生成 |
| **对话系统** | Decoder-only | Llama 3, GPT-4 | 灵活生成 |
| **代码生成** | Decoder-only | DeepSeek Coder | 灵活生成 |
| **文本生成** | Decoder-only | GPT-4, Claude | 强生成能力 |
| **问答系统** | Encoder-only | BERT (抽取式) | 精确匹配 |
| **问答系统** | Decoder-only | GPT-4 (生成式) | 灵活回答 |
| **语义相似度** | Encoder-only | SBERT | 句子嵌入 |

### 4.3 性能对比

| 模型 | 参数 | GLUE | SQuAD | MMLU | HumanEval |
|:-----|:-----|:-----|:------|:-----|:----------|
| **BERT-Large** | 340M | 82.1 | 93.2 | - | - |
| **RoBERTa-Large** | 355M | 88.5 | 94.6 | - | - |
| **GPT-3 (175B)** | 175B | - | - | 43.9 | - |
| **LLaMA 3 70B** | 70B | - | - | 82.0 | 81.7 |
| **DeepSeek V3** | 671B | - | - | 88.5 | 92.0 |
| **GPT-4** | ~1.8T | - | - | 86.4 | 67.0 |

### 4.4 选择决策树

```
                     你的任务是什么？
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    理解任务            生成任务            翻译/摘要
        │                  │                  │
        ▼                  ▼                  ▼
  Encoder-only      Decoder-only      Encoder-Decoder
        │                  │                  │
    ┌───┴────┐        ┌────┴────┐        ┌───┴───┐
    │        │        │         │        │       │
  分类    NER     对话     代码生成   翻译   摘要
    │        │        │         │        │       │
  BERT    BERT    Llama   DeepSeek  NLLB   BART
 RoBERTa  ALBERT  Mistral  CodeLlama T5     T5
```

---

## 五、开源框架与模型

### 5.1 Hugging Face Transformers

**核心组件**:

```python
from transformers import (
    AutoConfig,           # 模型配置
    AutoTokenizer,        # 分词器
    AutoModel,            # 模型
    AutoModelForCausalLM, # 因果语言模型
    Trainer,              # 训练器
    TrainingArguments,    # 训练参数
    pipeline,             # 高级API
)

# 1. 使用pipeline快速开始
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# 2. 支持的任务
tasks = [
    "sentiment-analysis",  # 情感分析
    "text-classification", # 文本分类
    "question-answering",  # 问答
    "fill-mask",           # 填空
    "text-generation",     # 文本生成
    "translation",         # 翻译
    "summarization",       # 摘要
    "feature-extraction",  # 特征提取
    "token-classification", # 标记分类
    "zero-shot-classification", # 零样本分类
]

# 3. 自定义模型
config = AutoConfig.from_pretrained("bert-base-uncased")
config.hidden_size = 768
config.num_hidden_layers = 12

model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 5.2 开源预训练模型

#### Encoder-only模型

| 模型 | 链接 | 特点 |
|:-----|:-----|:-----|
| **BERT** | `bert-base-uncased` | 基础模型 |
| **RoBERTa** | `roberta-base` | 优化训练 |
| **DeBERTa** | `microsoft/deberta-base` | 解耦注意力 |
| **ALBERT** | `albert-base-v2` | 参数共享 |
| **DistilBERT** | `distilbert-base-uncased` | 蒸馏BERT |

#### Decoder-only模型

| 模型 | 链接 | 特点 |
|:-----|:-----|:-----|
| **LLaMA 3** | `meta-llama/Llama-3-8B` | 开源SOTA |
| **Mistral** | `mistralai/Mistral-7B-v0.1` | 高效 |
| **Qwen 2.5** | `Qwen/Qwen2.5-7B` | 多语言 |
| **Phi-3** | `microsoft/Phi-3-mini-4k-instruct` | 轻量级 |
| **Gemma** | `google/gemma-7b` | Google开源 |

#### Encoder-Decoder模型

| 模型 | 链接 | 特点 |
|:-----|:-----|:-----|
| **T5** | `t5-base` | Text-to-Text |
| **BART** | `facebook/bart-base` | 去噪自编码 |
| **NLLB** | `facebook/nllb-200-distilled-600M` | 多语言 |
| **mBART** | `facebook/mbart-large-50` | 多语言 |

#### Vision Transformer模型

| 模型 | 链接 | 特点 |
|:-----|:-----|:-----|
| **ViT** | `google/vit-base-patch16-224` | 基础ViT |
| **Swin** | `microsoft/swin-base-patch4-window7-224` | 层次化 |
| **CLIP** | `openai/clip-vit-base-patch32` | 多模态 |
| **SAM** | `facebook/sam-vit-base` | 图像分割 |

### 5.3 模型选择指南

#### 按任务选择

```
文本分类 → BERT/RoBERTa
NER → BERT+CRF
抽取式QA → BERT/RoBERTa
生成式QA → GPT-4/Llama 3
摘要 → BART/T5
翻译 → NLLB/M2M100
对话 → Llama 3/Mistral
代码 → DeepSeek Coder
多模态 → CLIP/LLaVA
```

#### 按资源选择

| GPU内存 | 推荐模型 | 量化 |
|:--------|:---------|:-----|
| < 8GB | DistilBERT, Phi-3, TinyLlama | INT4/INT8 |
| 8-16GB | BERT-Large, Llama-3-8B, Mistral-7B | INT8 |
| 16-32GB | Llama-3-70B (4bit), Mixtral-8x7B | 4bit |
| > 32GB | Llama-3-70B (full), GPT-Q models | BF16/FP16 |

#### 按语言选择

| 语言 | 推荐模型 |
|:-----|:---------|
| **中文** | Qwen 2.5, Yi, DeepSeek |
| **英文** | Llama 3, Mistral, GPT |
| **多语言** | Qwen 2.5, NLLB, Aya |
| **代码** | DeepSeek Coder, StarCoder, CodeLlama |

---

## 六、性能与耗时分析

### 6.1 计算复杂度

| 组件 | 复杂度 | 说明 |
|:-----|:-------|:-----|
| **Self-Attention** | O(n²·d) | n=序列长度, d=模型维度 |
| **FFN** | O(n·d²) | 每个位置独立 |
| **整体** | O(n²·d) | 注意力是瓶颈 |

```
序列长度 vs 计算量（假设d=512）:

n=512:   512² × 512 = 134M operations
n=1024:  1024² × 512 = 537M operations  (4x)
n=2048:  2048² × 512 = 2.1B operations  (16x)
n=4096:  4096² × 512 = 8.6B operations  (64x)

结论: 序列长度翻倍，计算量4倍
```

### 6.2 内存占用

```python
def estimate_memory(params, seq_len, batch_size=1, precision=16):
    """
    估算模型内存占用

    params: 模型参数量（百万）
    seq_len: 序列长度
    batch_size: 批大小
    precision: 精度 (16=FP16/BF16, 32=FP32)
    """
    # 模型参数内存
    model_mem = params * 1e6 * precision / 8 / (1024**3)  # GB

    # 激活内存（简化估算）
    # 激活 ~ O(seq_len² × d_model × layers)
    hidden_dim = int(params ** 0.5 * 512)  # 粗略估算
    layers = int(params / 7)  # 假设每层7M参数
    activation_mem = (seq_len ** 2 * hidden_dim * layers * batch_size *
                      precision / 8 / (1024**3))

    # KV Cache内存（仅Decoder）
    kv_mem = (2 * layers * hidden_dim * seq_len * batch_size *
              precision / 8 / (1024**3))

    return {
        "model": model_mem,
        "activation": activation_mem,
        "kv_cache": kv_mem,
        "total": model_mem + activation_mem + kv_mem
    }

# 示例: Llama-3-8B
llama_mem = estimate_memory(params=8000, seq_len=2048, batch_size=1, precision=16)
print(f"Llama-3-8B 内存占用:")
print(f"  模型参数: {llama_mem['model']:.2f} GB")
print(f"  激活: {llama_mem['activation']:.2f} GB")
print(f"  KV Cache: {llama_mem['kv_cache']:.2f} GB")
print(f"  总计: {llama_mem['total']:.2f} GB")
```

### 6.3 KV Cache优化

KV Cache是Decoder推理的关键优化：

```python
class KVCache:
    """
    KV Cache实现
    缓存之前的Key和Value，避免重复计算
    """
    def __init__(self, n_layers, n_heads, d_head, max_len, dtype=torch.bfloat16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_len = max_len

        # 预分配缓存
        self.keys = torch.zeros(n_layers, max_len, n_heads, d_head, dtype=dtype)
        self.values = torch.zeros(n_layers, max_len, n_heads, d_head, dtype=dtype)
        self.seq_len = 0

    def update(self, layer_idx, new_k, new_v):
        """
        更新指定层的KV缓存
        new_k: (batch, n_new, n_heads, d_head)
        """
        n_new = new_k.size(1)
        self.keys[layer_idx, self.seq_len:self.seq_len+n_new] = new_k[0]
        self.values[layer_idx, self.seq_len:self.seq_len+n_new] = new_v[0]
        self.seq_len += n_new

    def get(self, layer_idx):
        """获取指定层的KV（截至当前位置）"""
        return (
            self.keys[layer_idx, :self.seq_len].unsqueeze(0),  # (1, seq_len, n_heads, d_head)
            self.values[layer_idx, :self.seq_len].unsqueeze(0)
        )

# 使用KV Cache的生成
def generate_with_kv_cache(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 首次前向传播（完整序列）
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values
    current_token = outputs.logits[:, -1:].argmax(dim=-1)

    generated = [current_token]

    # 后续生成（每次只处理一个token）
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True
            )

        past_key_values = outputs.past_key_values
        current_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated.append(current_token)

        if current_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(torch.cat(generated))
```

### 6.4 量化（Quantization）

量化可以显著减少模型大小和加速推理：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. INT4量化（推荐，最小精度损失）
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 二次量化
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config_4bit,
    device_map="auto"
)
# 内存: 16GB → ~5GB

# 2. INT8量化
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config_8bit,
    device_map="auto"
)
# 内存: 16GB → ~8GB

# 3. GPTQ量化（离线量化）
from transformers import GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    tokenizer=tokenizer
)

model_gptq = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-3-8B-GPTQ",
    device_map="auto"
)
```

**量化效果对比**:

| 量化 | 精度 | 内存 | 速度 | 精度损失 |
|:-----|:-----|:-----|:-----|:---------|
| FP16 | 16bit | 16GB | 基准 | - |
| INT8 | 8bit | 8GB | ~2x | <1% |
| INT4 | 4bit | 5GB | ~3x | 1-3% |
| GPTQ-4 | 4bit | 5GB | ~3x | <1% |

### 6.5 Flash Attention

Flash Attention是注意力机制的IO精确算法优化：

```python
# Flash Attention 2（自动启用）
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # 启用Flash Attention 2
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Flash Attention优势:
# 1. 减少内存: O(n) → O(sqrt(n)) for attention matrix
# 2. 加速: 2-4x faster attention
# 3. 支持更长序列: 可处理更长上下文
```

### 6.6 性能优化总结

| 优化技术 | 加速比 | 内存节省 | 代价 |
|:---------|:-------|:---------|:-----|
| **KV Cache** | 10x+ | - | 序列越长越有效 |
| **INT8量化** | 2x | 50% | <1%精度损失 |
| **INT4量化** | 3x | 70% | 1-3%精度损失 |
| **Flash Attention** | 2-4x | 50%+ | 需要特定GPU |
| **Fused Kernels** | 1.5-2x | - | 需要优化库 |
| **Speculative Decoding** | 2-3x | - | 需要草稿模型 |
| **Continuous Batching** | 2-10x | - | 服务端优化 |

---

## 七、实战案例

### 7.1 文本分类（BERT微调）

#### 场景描述

对电影评论进行情感分类（正面/负面）。

#### 代码实现

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# 1. 加载数据集
dataset = load_dataset("rotten_tomatoes")  # 电影评论情感分类

# 2. 加载模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False  # DataCollator会处理padding
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 4. 定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary")
    }

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./bert-sentiment",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,  # 混合精度训练
    logging_steps=100,
)

# 6. 创建模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 7. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

# 8. 评估
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1 Score: {results['eval_f1']:.4f}")

# 9. 推理
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    sentiment = "Positive" if probs[0][1] > 0.5 else "Negative"
    confidence = probs[0].max().item()

    return sentiment, confidence

# 示例
texts = [
    "This movie was absolutely fantastic! Best film of the year.",
    "Terrible movie, complete waste of time and money.",
    "It was okay, nothing special but watchable."
]

for text in texts:
    sentiment, conf = predict_sentiment(text)
    print(f"Text: {text[:60]}...")
    print(f"Sentiment: {sentiment} (confidence: {conf:.2%})\n")
```

#### 训练技巧

| 技巧 | 说明 | 实现 |
|:-----|:-----|:-----|
| **学习率预热** | 前10%步数线性增加 | `warmup_steps=500` |
| **早停** | 验证集不再提升时停止 | `load_best_model_at_end=True` |
| **梯度累积** | 模拟大批次 | `gradient_accumulation_steps=4` |
| **混合精度** | FP16加速，减少内存 | `fp16=True` |
| **类别平衡** | 处理不平衡数据 | 加权Loss或过采样 |

---

### 7.2 命名实体识别（NER）

#### 场景描述

从文本中识别人名、地名、机构名等实体。

#### 代码实现

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
import torch

# 1. 加载CoNLL-2003数据集
dataset = load_dataset("conll2003")

# 标签映射
label_list = dataset["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# 2. 加载BERT NER模型
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 3. NER预测函数
def extract_entities(text, model, tokenizer, label_list):
    """从文本中提取命名实体"""

    # 分词
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测标签
    predictions = outputs.logits.argmax(dim=-1)[0]
    predicted_labels = [label_list[p.item()] for p in predictions]

    # 对齐token和单词
    encoding = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # 提取实体
    entities = []
    current_entity = None

    for token, label, (start, end) in zip(tokens, predicted_labels, encoding["offset_mapping"]):
        if label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": label[2:],
                "text": token.replace('##', ''),
                "start": start,
                "end": end
            }
        elif label.startswith('I-') and current_entity:
            # 继续当前实体
            if token.startswith('##'):
                current_entity["text"] += token[2:]
            else:
                current_entity["text"] += " " + token
            current_entity["end"] = end

    if current_entity:
        entities.append(current_entity)

    return entities

# 4. 示例
text = """Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California.
Elon Musk is the CEO of Tesla, Inc. based in Austin, Texas."""

entities = extract_entities(text, model, tokenizer, label_list)

print("命名实体识别结果:")
print(f"文本: {text}\n")
print("实体:")
for entity in entities:
    print(f"  [{entity['type']}] {entity['text']} ({entity['start']}-{entity['end']})")
```

#### 从头训练NER模型

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np

# 1. 准备数据
dataset = load_dataset("conll2003")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    """对齐标签和token"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 特殊token
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # 子词token
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 2. 创建模型
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list)
)

# 3. 评估函数
metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# 4. 训练
training_args = TrainingArguments(
    output_dir="./bert-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 5. 保存模型
model.save_pretrained("./my-ner-model")
tokenizer.save_pretrained("./my-ner-model")
```

---

### 7.3 机器翻译（T5、NLLB）

#### 场景描述

将英语翻译为中文（或任意语言对）。

#### NLLB多语言翻译

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 1. 加载NLLB模型（支持200种语言）
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. 语言代码映射
LANGUAGE_CODES = {
    "英语": "eng_Latn",
    "中文": "zho_Hans",  # 简体
    "繁体中文": "zho_Hant",
    "日语": "jpn_Jpan",
    "韩语": "kor_Hang",
    "法语": "fra_Latn",
    "德语": "deu_Latn",
    "西班牙语": "spa_Latn",
    "俄语": "rus_Cyrl",
    "阿拉伯语": "arb_Arab",
}

# 3. 翻译函数
def translate(text, source_lang="英语", target_lang="中文", max_length=256):
    """翻译文本"""

    # 设置源语言
    tokenizer.src_lang = LANGUAGE_CODES[source_lang]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 设置目标语言
    forced_bos_token_id = tokenizer.lang_code_to_id[LANGUAGE_CODES[target_lang]]

    # 生成翻译
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=5,
            length_penalty=1.0
        )

    # 解码
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# 4. 批量翻译
def batch_translate(texts, source_lang="英语", target_lang="中文"):
    """批量翻译"""
    tokenizer.src_lang = LANGUAGE_CODES[source_lang]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    forced_bos_token_id = tokenizer.lang_code_to_id[LANGUAGE_CODES[target_lang]]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations

# 5. 示例
texts = [
    "Hello, how are you today?",
    "I love learning new languages.",
    "The weather is beautiful today.",
    "Machine translation has improved significantly."
]

print("英译中示例:")
for text in texts:
    translation = translate(text, "英语", "中文")
    print(f"EN: {text}")
    print(f"CN: {translation}\n")

# 6. 多语言互译
print("多语言翻译:")
en_text = "The quick brown fox jumps over the lazy dog."
print(f"英语 → 中文: {translate(en_text, '英语', '中文')}")
print(f"英语 → 日语: {translate(en_text, '英语', '日语')}")
print(f"英语 → 韩语: {translate(en_text, '英语', '韩语')}")
```

#### T5 Text-to-Text翻译

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载T5
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def t5_translate(text, task="translate English to German"):
    """使用T5进行翻译或Text-to-Text任务"""

    # 构造输入
    input_text = f"{task}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # 生成
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)

    # 解码
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 示例
print("T5 Text-to-Text示例:")
print(translate("The house is wonderful.", "translate English to German"))
# "Das Haus ist wunderbar."

print(translate("My name is John.", "translate English to French"))
# "Je m'appelle Jean."
```

#### 微调翻译模型

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset

# 1. 加载OPUS数据集（平行语料）
dataset = load_dataset("opus100", "en-zh")

# 2. 预处理
source_lang = "en"
target_lang = "zh"

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # 准备标签
    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3. 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# 4. 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./translation-model",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False
)

# 5. 训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# 6. 保存
model.save_pretrained("./my-en-zh-translation")
tokenizer.save_pretrained("./my-en-zh-translation")
```

---

### 7.4 文本生成（GPT、Llama）

#### 场景描述

使用Llama 3生成创意文本、故事、代码等。

#### 基础文本生成

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型
model_name = "meta-llama/Llama-3-8B"  # 或 "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 设置pad_token
tokenizer.pad_token = tokenizer.eos_token

# 2. 生成参数配置
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,      # 控制随机性（0-1）
    "top_p": 0.9,           # nucleus sampling
    "top_k": 50,            # top-k sampling
    "do_sample": True,      # 启用采样
    "repetition_penalty": 1.1,  # 惩罚重复
}

# 3. 生成函数
def generate_text(prompt, **kwargs):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 合并配置
    config = {**generation_config, **kwargs}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 只返回新生成的部分
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text

# 4. 示例：创意写作
story_prompt = """Write a short story about a robot who discovers emotions.

Title: The First Tear
"""

story = generate_text(story_prompt, max_new_tokens=512, temperature=0.9)
print("生成的故事:")
print(story)

# 5. 示例：代码生成
code_prompt = """Write a Python function to implement a binary search tree with insertion and search operations.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
"""

code = generate_text(code_prompt, max_new_tokens=512, temperature=0.3)
print("\n生成的代码:")
print(code)
```

#### 对话生成

```python
def chat(messages, max_new_tokens=256, temperature=0.7):
    """
    对话生成

    messages: [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
        {"role": "user", "content": "请介绍一下Python"}
    ]
    """
    # Llama 3对话模板
    chat_template = "{% for message in messages %}"
    chat_template += "{% if message['role'] == 'user' %}"
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    chat_template += "{% elif message['role'] == 'assistant' %}"
    chat_template += "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    chat_template += "{% endif %}"
    chat_template += "{% endfor %}"
    chat_template += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# 示例对话
messages = [
    {"role": "user", "content": "你好！"}
]
response = chat(messages)
print(response)

# 继续对话
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "请解释一下什么是Transformer"})
response = chat(messages)
print(response)
```

#### 流式生成

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_generate(prompt, max_new_tokens=256):
    """流式生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 创建流式输出器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # 生成配置
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id
    }

    # 在后台线程中生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 实时打印
    print("生成中: ", end="", flush=True)
    for text in streamer:
        print(text, end="", flush=True)
    print("\n")

    thread.join()

# 示例
stream_generate("写一首关于春天的诗")
```

---

### 7.5 问答系统（RAG + LLM）

#### 场景描述

构建一个基于检索增强生成的问答系统，能够回答基于文档的问题。

#### 完整实现

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 文档索引器
class DocumentIndexer:
    """使用Sentence-BERT和FAISS构建文档索引"""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents):
        """添加文档到索引"""
        self.documents.extend(documents)

        # 编码文档
        embeddings = self.encoder.encode(documents, convert_to_numpy=True)

        # 创建或更新FAISS索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings.astype('float32'))

    def search(self, query, k=3):
        """搜索相关文档"""
        if self.index is None:
            return []

        # 编码查询
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)

        # 搜索
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        # 返回结果
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": 1 / (1 + dist)  # 转换为相似度分数
                })

        return results

# 2. RAG问答系统
class RAGQuestionAnswering:
    """检索增强生成问答系统"""

    def __init__(self, llm_name="meta-llama/Llama-3-8B"):
        # 加载LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载索引器
        self.indexer = DocumentIndexer()

    def index_documents(self, documents):
        """索引文档"""
        self.indexer.add_documents(documents)
        print(f"索引了 {len(documents)} 个文档")

    def answer(self, question, max_new_tokens=256, temperature=0.5):
        """回答问题"""

        # 1. 检索相关文档
        retrieved_docs = self.indexer.search(question, k=3)

        if not retrieved_docs:
            return "抱歉，我没有找到相关文档来回答这个问题。"

        # 2. 构造提示
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])

        prompt = f"""Based on the following context, please answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

        # 3. 生成回答
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return answer.strip()

# 3. 使用示例
def create_sample_documents():
    """创建示例文档"""
    return [
        """
Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991. Python supports multiple programming
paradigms including procedural, object-oriented, and functional programming.
        """,
        """
Machine learning is a subset of artificial intelligence that focuses on building systems that
can learn from data. Common algorithms include linear regression, decision trees, neural networks,
and support vector machines. Deep learning is a subset of machine learning using neural networks
with multiple layers.
        """,
        """
Transformers are a type of neural network architecture introduced in 2017's "Attention Is All You Need" paper.
They use self-attention mechanisms to process sequential data. Transformers have become the foundation
for modern large language models like GPT, BERT, and LLaMA.
        """,
        """
NumPy is a fundamental package for scientific computing in Python. It provides support for large,
multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate
on these arrays. NumPy is widely used in data science and machine learning.
        """,
        """
Pandas is a data manipulation library for Python. It provides data structures like DataFrame and
Series for handling structured data. Pandas is built on top of NumPy and is commonly used for data
cleaning, transformation, and analysis in data science workflows.
        """
    ]

# 初始化RAG系统
rag = RAGQuestionAnswering()
rag.index_documents(create_sample_documents())

# 问答示例
questions = [
    "Who created Python and when?",
    "What is the Transformer architecture?",
    "What is the difference between NumPy and Pandas?",
    "How does deep learning relate to machine learning?"
]

for question in questions:
    print(f"问题: {question}")
    answer = rag.answer(question, max_new_tokens=150)
    print(f"回答: {answer}\n")
    print("-" * 50 + "\n")
```

#### 高级RAG：混合检索

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class HybridRetriever:
    """混合检索（稀疏+密集）"""

    def __init__(self):
        self.dense_retriever = DocumentIndexer()
        self.sparse_vectorizer = TfidfVectorizer()
        self.sparse_index = None
        self.documents = []

    def index_documents(self, documents):
        """索引文档"""
        self.documents = documents

        # 密集检索
        self.dense_retriever.add_documents(documents)

        # 稀疏检索
        self.sparse_index = self.sparse_vectorizer.fit_transform(documents)

    def search(self, query, k=3, alpha=0.5):
        """混合搜索
        alpha: 密集检索权重（0-1）
        """
        # 密集检索结果
        dense_results = self.dense_retriever.search(query, k=k*2)

        # 稀疏检索结果
        query_vec = self.sparse_vectorizer.transform([query])
        sparse_scores = (self.sparse_index * query_vec.T).toarray().flatten()

        # 组合分数
        combined_scores = {}

        # 添加密集检索分数
        for result in dense_results:
            idx = self.documents.index(result["document"])
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * result["score"]

        # 添加稀疏检索分数
        for idx, score in enumerate(sparse_scores):
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * score

        # 排序并返回top-k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {"document": self.documents[idx], "score": score}
            for idx, score in sorted_indices
        ]
```

---

### 7.6 摘要生成（BART、T5）

#### 场景描述

生成长文档的摘要。

#### BART摘要

```python
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# 1. 加载BART摘要模型
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 2. 摘要函数
def summarize_text(text, max_length=150, min_length=50):
    """生成摘要"""

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=1.0,  # >1鼓励更短摘要
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 3. 示例
long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural
intelligence displayed by humans or animals. Leading AI textbooks define the field as the study
of "intelligent agents": any system that perceives its environment and takes actions that maximize
its chance of achieving its goals. Some popular accounts use the term "artificial intelligence"
to describe machines that mimic "cognitive" functions that humans associate with the human mind,
such as "learning" and "problem solving".

AI applications include advanced web search engines, recommendation systems (used by YouTube,
Amazon and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars
(e.g. Tesla), and competing at the highest level in strategic game systems (such as chess and Go).

As machines become increasingly capable, tasks considered to require "intelligence" are often
removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical
character recognition is frequently excluded from things considered to be AI, having become a
routine technology.
"""

summary = summarize_text(long_text)
print("摘要:")
print(summary)

# 4. 批量摘要
def batch_summarize(texts, batch_size=4):
    """批量生成摘要"""
    summaries = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=150,
                min_length=50,
                num_beams=4,
                early_stopping=True
            )

        batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        summaries.extend(batch_summaries)

    return summaries
```

#### T5摘要

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5摘要模型
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def t5_summarize(text, max_length=150):
    """使用T5生成摘要"""

    # T5需要任务前缀
    task_prefix = "summarize: "
    input_text = task_prefix + text

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

#### 多种摘要风格

```python
def abstractive_summarization(text):
    """抽象式摘要（生成新句子）"""
    return summarize_text(text, max_length=150)

def extractive_summarization(text, num_sentences=3):
    """提取式摘要（选择重要句子）"""
    import nltk
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sentences = nltk.sent_tokenize(text)

    # 计算句子相似度
    embeddings = [sentence_embedding(s) for s in sentences]
    similarity_matrix = cosine_similarity(embeddings)

    # TextRank
    import networkx as nx
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # 选择top句子
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])

    return summary

def bullet_point_summary(text):
    """要点式摘要"""
    prompt = f"""Summarize the following text into 3-5 bullet points:

{text}

Bullet points:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
print("抽象式摘要:")
print(abstractive_summarization(long_text))

print("\n要点式摘要:")
print(bullet_point_summary(long_text))
```

---

### 7.7 代码生成（CodeLlama、StarCoder）

#### 场景描述

生成、补全、解释代码。

#### CodeLlama代码生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, CodeGenerationConfig

# 1. 加载CodeLlama
model_name = "codellama/CodeLlama-7b-Python-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 代码生成
def generate_code(prompt, max_new_tokens=256, language="python"):
    """生成代码"""

    # 添加语言提示
    if language == "python":
        prompt = f"'\"\"\"\n{prompt}\n\"\"\"\n\n# Here's the implementation:\n\n```python\n"
    elif language == "javascript":
        prompt = f"// {prompt}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,  # 代码生成需要较低温度
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_code = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generated_code

# 3. 示例
print("示例1: 二叉树遍历")
code = generate_code("Implement a binary tree with inorder, preorder, and postorder traversal methods", max_new_tokens=300)
print(code)

print("\n示例2: 快速排序")
code = generate_code("Write a function to implement quicksort algorithm in Python", max_new_tokens=200)
print(code)

# 4. 代码补全
def complete_code(code_prefix, max_new_tokens=128):
    """补全代码"""

    inputs = tokenizer(code_prefix, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return code_prefix + completion

# 示例
prefix = """def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    else:
"""

completion = complete_code(prefix, max_new_tokens=100)
print("\n代码补全:")
print(completion)
```

#### 代码解释

```python
def explain_code(code):
    """解释代码功能"""

    prompt = f"""Please explain the following code in detail, including:
1. What the code does
2. How it works
3. Time and space complexity

Code:
```python
{code}
```

Explanation:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True
        )

    explanation = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return explanation

# 示例
code_to_explain = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""

print("代码解释:")
print(explain_code(code_to_explain))
```

#### DeepSeek Coder（2025推荐）

```python
# DeepSeek Coder是2025年最强大的开源代码模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# DeepSeek使用对话格式
messages = [
    {"role": "user", "content": "Write a function to implement a hash table in Python with get, set, and delete operations."}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True
    )

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
```

---

### 7.8 多模态（CLIP、GPT-4V）

#### CLIP图文匹配

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 1. 加载CLIP
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

# 2. 图文相似度
def clip_image_text_similarity(image_path, texts):
    """计算图像与多个文本的相似度"""

    image = Image.open(image_path)

    inputs = processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 获取logits
    logits_per_image = outputs.logits_per_image  # 图像到文本
    probs = logits_per_image.softmax(dim=-1)

    return probs[0]

# 示例
texts = [
    "a cat sitting on a couch",
    "a dog playing in the park",
    "a bird flying in the sky",
    "a landscape with mountains"
]

probs = clip_image_text_similarity("cat.jpg", texts)

for text, prob in zip(texts, probs):
    print(f"{text}: {prob:.4f}")

# 3. 零样本图像分类
def zero_shot_classify(image_path, class_names):
    """零样本图像分类"""

    # 构造提示
    prompts = [f"a photo of a {cls}" for cls in class_names]

    probs = clip_image_text_similarity(image_path, prompts)

    # 排序
    sorted_indices = probs.argsort(descending=True)

    results = [(class_names[i], probs[i].item()) for i in sorted_indices]
    return results

# 示例
classes = ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear"]
results = zero_shot_classify("animal.jpg", classes)

print("零样本分类结果:")
for cls, score in results:
    print(f"  {cls}: {score:.4f}")
```

#### 图像描述生成

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. 加载BLIP模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. 生成图像描述
def generate_caption(image_path):
    """为图像生成描述"""

    image = Image.open(image_path)

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# 3. 条件描述生成
def generate_conditional_caption(image_path, context_text):
    """基于上下文生成描述"""

    image = Image.open(image_path)

    inputs = processor(image, text=context_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# 示例
print("无条件描述:")
print(generate_caption("scene.jpg"))

print("\n条件描述 (以'This shows'开头):")
print(generate_conditional_caption("scene.jpg", "This shows"))
```

#### LLaVA多模态对话

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

# 1. 加载LLaVA
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 多模态对话
def multimodal_chat(image_path, question):
    """多模态问答"""

    image = Image.open(image_path)

    # 构造对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# 示例
answer = multimodal_chat(
    "kitchen.jpg",
    "What objects can you see in this image? Is there anything that could be dangerous for children?"
)
print(answer)

# 3. 图像推理
answer = multimodal_chat(
    "chart.jpg",
    "Analyze this chart and describe the main trends you see. What conclusions can be drawn?"
)
print(answer)
```

---

## 八、方案选择建议

### 8.1 按任务类型选择

```
文本分类/情感分析
├─ 小数据集 (<10K): BERT, DistilBERT
├─ 中等数据集 (10K-100K): RoBERTa, DeBERTa
└─ 大数据集 (>100K): RoBERTa-Large

命名实体识别
├─ 通用领域: BERT+CRF
├─ 生物医学: BioBERT
└─ 中文: BERT-Chinese, MacBERT

问答系统
├─ 抽取式QA: BERT, RoBERTa
├─ 生成式QA: Llama 3, DeepSeek V3
└─ 领域QA: RAG + 专用LLM

机器翻译
├─ 英译中: NLLB, Qwen
├─ 多语言: NLLB-200, M2M100
└─ 低资源语言: NLLB, Aya

文本摘要
├─ 新闻摘要: BART, PEGASUS
├─ 长文档摘要: Llama 3 (长上下文)
└─ 对话摘要: T5, BART

对话系统
├─ 通用对话: Llama 3, GPT-4
├─ 中文对话: Qwen, DeepSeek
├─ 轻量级: Mistral-7B, Phi-3
└─ 领域对话: 微调通用模型

代码生成
├─ Python: DeepSeek Coder, CodeLlama
├─ 多语言: StarCoder, CodeGemma
└─ 代码解释: GPT-4, Claude
```

### 8.2 按资源限制选择

| GPU内存 | 推荐配置 | 模型 | 量化 |
|:--------|:---------|:-----|:-----|
| <8GB | CPU/Edge | DistilBERT, Phi-3 | INT4 |
| 8GB | 单卡消费级 | BERT-Base, Llama-3-8B | INT8 |
| 16GB | 高端消费级 | BERT-Large, Llama-3-8B | BF16 |
| 24GB | 3090/4090 | Llama-3-70B-4bit, Mixtral | 4bit |
| 40GB+ | A100/H100 | Llama-3-70B, DeepSeek | BF16 |

### 8.3 按语言选择

| 语言 | Encoder | Decoder | 推荐 |
|:-----|:--------|:--------|:-----|
| **中文** | ERNIE, MacBERT | Qwen, Yi, DeepSeek | Qwen 2.5 |
| **英文** | BERT, RoBERTa | Llama 3, Mistral | Llama 3 |
| **多语言** | mBERT, XLM-R | Qwen, Aya | Qwen 2.5 |
| **代码** | CodeBERT | DeepSeek Coder, StarCoder | DeepSeek Coder |

### 8.4 部署方案选择

```python
# 方案1: 本地部署（小型模型）
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
result = classifier("This is great!")

# 方案2: vLLM部署（大型模型，高吞吐）
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

# 方案3: Ollama部署（本地推理）
# ollama run llama3:8b

# 方案4: 云API（最快上手）
import openai
# 使用OpenAI、Anthropic、DeepSeek等API
```

---

## 九、常见问题

### 9.1 BERT vs GPT的选择

**选择BERT (Encoder-only)的情况**:
- 需要双向理解
- 分类、NER、抽取式QA
- 有标注数据且量适中
- 需要快速推理

**选择GPT (Decoder-only)的情况**:
- 生成任务
- 灵活的对话
- 少样本/零样本
- 复杂推理

### 9.2 上下文长度限制

| 模型 | 上下文长度 | 扩展方案 |
|:-----|:----------|:---------|
| **BERT** | 512 | 滑动窗口、长序列变体 |
| **GPT-3** | 2048-4096 | - |
| **Llama 2** | 4096 | - |
| **Llama 3** | 8192 | - |
| **Claude 3** | 200K | 内置 |
| **Gemini 1.5** | 1M | 内置 |

**处理长文本的策略**:
```python
# 1. 滑动窗口
def process_long_text(text, window_size=4000, overlap=200):
    """用滑动窗口处理长文本"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# 2. 分层摘要
def hierarchical_summary(text):
    """分层摘要：先摘要各部分，再整体摘要"""
    sections = split_into_sections(text)
    section_summaries = [summarize(s) for s in sections]
    final_summary = summarize(" ".join(section_summaries))
    return final_summary

# 3. RAG（检索相关部分）
def rag_qa_long_document(question, document):
    """从长文档中检索相关部分并回答"""
    chunks = chunk_document(document)
    relevant_chunks = retrieve_relevant_chunks(question, chunks)
    answer = generate_answer(question, relevant_chunks)
    return answer
```

### 9.3 幻觉问题（Hallucination）

**幻觉原因**:
1. 训练数据中的错误信息
2. 模型过度自信
3. 缺乏外部知识验证

**缓解方法**:

```python
# 1. 检索增强生成 (RAG)
# 提供真实文档作为上下文

# 2. 温度控制
# 使用较低温度（如0.1）减少随机性

# 3. Prompt工程
prompt = """Please answer the following question. If you are not certain about the answer,
please say "I'm not sure" rather than making up information.

Question: {question}

If you know the answer, please provide it with your confidence level."""

# 4. 自我验证
def self_verify(question, answer):
    """让模型自我验证"""
    verify_prompt = f"""Original Question: {question}
Answer: {answer}

Please verify if this answer is correct. If not, explain why."""

    verification = generate(verify_prompt)
    return verification

# 5. 多次采样投票
def ensemble_vote(question, n=5):
    """生成多个答案并投票"""
    answers = [generate(question, temperature=0.7) for _ in range(n)]
    # 选择最一致的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

### 9.4 训练不稳定

**常见问题**:
- 梯度爆炸/消失
- 损失不下降
- 过拟合

**解决方案**:

```python
# 1. 学习率调度
from transformers import get_scheduler

num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# 2. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 层次学习率（微调时）
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "layer11" in n or "layer10" in n],
        "lr": 1e-5  # 顶层用较低学习率
    },
    {
        "params": [p for n, p in model.named_parameters() if "layer11" not in n and "layer10" not in n],
        "lr": 1e-4  # 底层用较高学习率
    }
]

# 4. 正则化
training_args = TrainingArguments(
    weight_decay=0.01,    # 权重衰减
    max_grad_norm=1.0,    # 梯度裁剪
    label_smoothing_factor=0.1,  # 标签平滑
)
```

---

## 十、学习资源

### 10.1 核心论文

| 论文 | 年份 | 贡献 | 链接 |
|:-----|:-----|:-----|:-----|
| Attention Is All You Need | 2017 | Transformer原始论文 | [arXiv](https://arxiv.org/abs/1706.03762) |
| BERT: Pre-training of Deep Bidirectional Transformers | 2018 | Encoder-only | [arXiv](https://arxiv.org/abs/1810.04805) |
| GPT-2: Language Models are Unsupervised Multitask Learners | 2019 | Decoder-only | [arXiv](https://arxiv.org/abs/1905.11719) |
| GPT-3: Language Models are Few-Shot Learners | 2020 | 大规模LLM | [arXiv](https://arxiv.org/abs/2005.14165) |
| T5: Exploring the Limits of Transfer Learning | 2019 | Text-to-Text | [arXiv](https://arxiv.org/abs/1910.10683) |
| BART: Denoising Sequence-to-Sequence Pre-training | 2019 | Seq2Seq | [arXiv](https://arxiv.org/abs/1910.13461) |
| LLaMA: Open and Efficient Foundation Language Models | 2023 | 开源LLM | [arXiv](https://arxiv.org/abs/2302.13971) |
| Llama 3 | 2024 | 最强开源LLM | [arXiv](https://arxiv.org/abs/2407.21783) |
| DeepSeek V3 | 2025 | MoE LLM | [arXiv](https://arxiv.org/abs/2412.19437) |
| An Image is Worth 16x16 Words (ViT) | 2020 | Vision Transformer | [arXiv](https://arxiv.org/abs/2010.11929) |
| CLIP: Learning Transferable Visual Models | 2021 | 多模态 | [arXiv](https://arxiv.org/abs/2103.00020) |

### 10.2 在线教程

- **The Illustrated Transformer**: [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/)
- **Attention Is All You Need - Explained**: [.youtube.com](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **Stanford CS224N**: [NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- **Hugging Face Course**: [huggingface.co/course](https://huggingface.co/course)

### 10.3 书籍

- 《Natural Language Processing with Transformers》
- 《Speech and Language Processing (3rd ed.)》
- 《Deep Learning (Ian Goodfellow)》

### 10.4 实践平台

- **Hugging Face**: [huggingface.co](https://huggingface.co)
- **Papers with Code**: [paperswithcode.com](https://paperswithcode.com)
- **Kaggle**: [kaggle.com](https://kaggle.com)

---

## 十一、相关笔记

### 深入学习

- [[AI研究/AI学习/02-模型原理/Transformer研读]] - Transformer架构详细研读
- [[AI研究/AI学习/02-模型原理/GNN全面解析]] - 图神经网络解析
- [[AI研究/AI学习/神经网络类型全景总结]] - 所有神经网络架构概览
- [[AI研究/AI学习/常见术语对照]] - AI/ML术语中英文对照
- [[AI研究/AI学习/AI模型系统性学习路径]] - 系统学习路径

### 实战应用

- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - 检索增强生成项目
- [[AI研究/AI学习/车道路径筛选系统设计]] - GNN实际应用
- [[AI研究/AI学习/使用LLM进行车道路径推理]] - LLM推理应用
- [[AI研究/AI学习/04-深入前沿/论文阅读模板]] - 论文阅读笔记模板

### 基础知识

- [[AI研究/AI学习/01-基础夯实/数学基础笔记]] - 深度学习数学基础
- [[AI研究/AI学习/01-基础夯实/NumPy学习笔记]] - 数值计算基础

---

#Transformer #注意力机制 #NLP #深度学习 #架构原理
