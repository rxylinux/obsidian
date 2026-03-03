---
title: Transformer 架构研读
date: 2026-02-28
tags:
  - Transformer
  - Attention
  - 论文
  - 架构
status: in-progress
---

# Transformer 架构研读

> [!quote] "Attention Is All You Need"
> Vaswani et al., 2017
>
> Transformer 抛弃了循环和卷积，完全依赖注意力机制。

> [!info] 相关资源
> - **术语对照**：[[AI研究/AI学习/常见术语对照]] - Transformer 相关术语速查
> - **学习路径**：[[AI研究/AI学习/AI模型系统性学习路径]]

---

## 论文核心贡献

### 三大创新

1. **Self-Attention**：捕捉序列内部的长距离依赖
2. **Multi-Head Attention**：从多个子空间学习表示
3. **Positional Encoding**：注入序列位置信息

### 为什么重要？

| 架构   | 问题         | Transformer 解决 |
| :--- | :--------- | :------------- |
| [[AI研究/AI学习/常见术语对照#RNN\RNN]] | 序列化计算，难以并行 | 完全并行           |
| [[AI研究/AI学习/常见术语对照#CNN\CNN]] | 感受野有限      | 全局视野           |
| [[AI研究/AI学习/常见术语对照#LSTM\LSTM]] | 长距离遗忘      | 直接连接           |

---

## Transformer 架构

### 整体结构

```
        Encoder                    Decoder
    ┌─────────────┐          ┌─────────────┐
    │  Encoder    │          │   Decoder   │
    │   Stack × N │ ────────▶│   Stack × N │
    │             │          │             │
    └─────────────┘          └──────┬──────┘
                                    │
                                    ▼
                              Linear + Softmax
```

### Encoder Block

```
Input ──▶ [Add & Norm] ──▶ Feed Forward ──▶ [Add & Norm] ──▶ Output
            ▲                                      │
            │                                      │
         Attention ◀───────────────────────────────┘
```

**核心组件**：
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Networks
3. Residual Connection + Layer Norm

---

## Self-Attention 详解

### 核心思想

> [!tip] 直观理解
> - Query：我想找什么？
> - Key：你能提供什么？
> - Value：实际内容是什么？

### 计算步骤

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    # 1. 计算注意力分数
    scores = Q @ K.T / sqrt(d_k)  # 缩放防梯度消失

    # 2. Softmax 归一化
    attention_weights = softmax(scores, dim=-1)

    # 3. 加权求和
    output = attention_weights @ V

    return output, attention_weights
```

### 直观示例

```
句子: "The cat sat on the mat"

处理 "cat" 时:
- Query: "cat" 的查询向量
- Key: 所有词的键向量
- Value: 所有词的值向量

Attention("cat", "The")   = 低  (不相关)
Attention("cat", "cat")   = 高  (自身)
Attention("cat", "sat")   = 中  (主谓关系)
Attention("cat", "mat")   = 中  (位置关系)
```

### Multi-Head Attention

```python
def multi_head_attention(x, num_heads=8):
    """
    x: (batch, seq_len, d_model)
    d_model = 512, d_k = d_v = 64
    """
    d_model = x.shape[-1]
    d_k = d_model // num_heads

    # 1. 线性投影到多个头
    Q = linear_q(x).view(batch, seq_len, num_heads, d_k)
    K = linear_k(x).view(batch, seq_len, num_heads, d_k)
    V = linear_v(x).view(batch, seq_len, num_heads, d_v)

    # 2. 转置以便并行计算
    Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    # 3. Scaled Dot-Product Attention
    attn_output, _ = scaled_dot_product_attention(Q, K, V)

    # 4. 拼接多个头
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch, seq_len, d_model)

    # 5. 最终线性变换
    output = linear_o(attn_output)

    return output
```

**为什么多头？**
- 不同头学习不同的关注模式
- 头1：语法关系
- 头2：语义关联
- 头3：指代消解
- ...

---

## Positional Encoding

### 为什么需要？

Transformer 本身是**位置无关**的（置换等变）

```python
# 这个问题：
Input: "我 爱 AI" 和 "AI 爱 我"
# 在 Transformer 看来是一样的集合，需要位置编码
```

### Sinusoidal 位置编码

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

```python
def positional_encoding(max_len, d_model):
    """
    max_len: 最大序列长度
    d_model: 模型维度 (512)
    """
    pe = torch.zeros(max_len, d_model)

    # 位置索引
    position = torch.arange(0, max_len).unsqueeze(1)

    # 除数项
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))

    # sin 和 cos 交替
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # (1, max_len, d_model)
```

**特点**：
- 每个维度有不同频率
- 相对位置可以表示（PE(pos+k) - PE(pos)）
- 外推能力

### 现代改进

| 方案 | 特点 | 使用 |
|:-----|:-----|:-----|
| RoPE | 旋转位置编码 | LLaMA, Mistral |
| ALiBi | 线性偏置 | BLOOM, MPT |
| 学习式 | 可学习参数 | 原始 GPT-3 |

---

## Feed-Forward Network

```python
def feed_forward(x, d_model=512, d_ff=2048):
    """
    两层全连接 + ReLU
    """
    # (batch, seq_len, d_model) → (batch, seq_len, d_ff)
    hidden = relu(linear1(x))

    # (batch, seq_len, d_ff) → (batch, seq_len, d_model)
    output = linear2(hidden)

    return output
```

**作用**：
- 注意力后"消化"信息
- 引入非线性
- 等价于每个位置独立的前馈网络

---

## 残差连接与层归一化

### Add & Norm

```python
def add_and_norm(x, sublayer_output):
    """
    sublayer: Attention 或 FFN
    """
    # 残差连接
    x = x + sublayer_output

    # 层归一化
    x = layer_norm(x)

    return x
```

**Layer Normalization**：
```python
def layer_norm(x, eps=1e-6):
    """
    对每个样本的所有维度归一化
    x: (batch, seq_len, d_model)
    """
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)

    return (x - mean) / (std + eps)
```

**为什么有效？**
- 残差：缓解梯度消失
- 归一化：稳定训练

---

## 训练细节

### 超参数

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| $d_{model}$ | 512 | 模型维度 |
| $d_{ff}$ | 2048 | FFN 隐藏层维度 |
| Heads | 8 | 注意力头数 |
| Layers | 6 | Encoder/Decoder 层数 |
| Dropout | 0.1 | 正则化 |

### 优化技巧

**Warmup**：学习率预热
```python
def lr_schedule(step, d_model=512, warmup_steps=4000):
    """
    先增后稳的学习率
    """
    arg1 = step ** -0.5
    arg2 = step * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(arg1, arg2)
```

**Label Smoothing**：防止过度自信
```python
# 将 one-hot 变为平滑分布
# [0, 1, 0] → [0.05, 0.9, 0.05]
smoothed_labels = labels * (1 - epsilon) + epsilon / num_classes
```

---

## 三种变体

### Encoder-only

**代表**：BERT, RoBERTa

```
Input ──▶ Encoder ──▶ Output
```

**特点**：
- 双向注意力
- 适合理解任务
- 应用：分类、NER、QA

### Decoder-only

**代表**：GPT系列, LLaMA

```
Input ──▶ Decoder ──▶ Output
       (Masked)
```

**特点**：
- 因果注意力（只看过去）
- 适合生成任务
- 应用：文本生成、对话

### Encoder-Decoder

**代表**：T5, BART, 原始 Transformer

```
Encoder ──▶ Cross Attention ──▶ Decoder
  │                              │
Input ──────────────────────────▶Output
```

**特点**：
- 编码器理解输入
- 解码器生成输出
- 应用：翻译、摘要

---

## 实现示例

### 简化的 Transformer Encoder

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * math.sqrt(d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return x
```

---

## 延伸阅读

### 后续论文

- [ ] BERT: Pre-training of Deep Bidirectional Transformers (2018)
- [ ] GPT-2: Language Models are Unsupervised Multitask Learners (2019)
- [ ] GPT-3: Language Models are Few-Shot Learners (2020)
- [ ] LLaMA: Open and Efficient Foundation Language Models (2023)
- [ ] Mistral: Mixtral of Experts (2024)

### 改进方向

- **效率**：Flash Attention, PagedAttention
- **长度**：长上下文、稀疏注意力
- **规模**：MoE (Mixture of Experts)
- **对齐**：RLHF, DPO

---

## 学习资源

### 可视化

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need - 可视化](https://poloclub.github.io/transformer-explainer/)

### 视频

- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown - Attention](https://www.3blue1brown.com/)

### 代码

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 学习进度

> [!todo]- 学习清单
> - [ ] 通读原论文
> - [ ] 理解 Self-Attention 计算
> - [ ] 理解 Multi-Head 机制
> - [ ] 理解 Positional Encoding
> - [ ] 动手实现简化版
> - [ ] 阅读 BERT/GPT 论文

---

## 相关笔记

### 深入学习

- [[AI研究/AI学习/02-模型原理/Transformer全面解析]] - Transformer系统讲解
- [[AI研究/AI学习/神经网络类型全景总结]] - 所有神经网络架构概览
- [[AI研究/AI学习/常见术语对照]] - Transformer相关术语

### 实战应用

- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - RAG系统开发
- [[AI研究/AI学习/04-深入前沿/RAG 2.0 全面解析]] - RAG端到端训练
- [[AI研究/AI学习/04-深入前沿/知识蒸馏全面解析]] - 模型压缩技术

### 导航

- [[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
- [[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线

---

#Transformer #Attention #论文 #架构
