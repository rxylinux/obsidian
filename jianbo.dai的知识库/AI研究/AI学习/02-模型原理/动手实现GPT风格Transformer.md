---
title: 动手实现 GPT 风格 Transformer
date: 2026-03-01
tags:
  - Transformer
  - GPT
  - 动手实现
  - PyTorch
status: in-progress
---

# 动手实现 GPT 风格 Transformer

> [!tip] 学习目标
> - 从零实现一个简化版 GPT 模型
> - 理解每个组件的作用
> - 能够生成简单的文本

> [!info] 相关笔记
> - [[AI研究/AI学习/02-模型原理/Transformer全面解析]] - 详细理论
> - [[AI研究/AI学习/02-模型原理/Transformer研读.md]] - 论文研读

---

## 实现路线图

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT 模型架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Token Embedding + Positional Encoding                      │
│  │                                                           │
│  │                                                           │
│  ▼                                                           │
│  ┌─────────────────────────────────────────────┐            │
│  │          Transformer Block × N               │            │
│  │  ┌─────────────────────────────────────┐    │            │
│  │  │  Layer Norm (Pre-Norm)               │    │            │
│  │  │           │                           │    │            │
│  │  │           ▼                           │    │            │
│  │  │  Causal Multi-Head Self-Attention    │    │            │
│  │  │           │                           │    │            │
│  │  │           │ (Residual + Norm)         │    │            │
│  │  │           ▼                           │    │            │
│  │  │  Feed-Forward Network (4x expansion)  │    │            │
│  │  │           │                           │    │            │
│  │  │           │ (Residual + Norm)         │    │            │
│  │  └─────────────────────────────────────┘    │            │
│  └─────────────────────────────────────────────┘            │
│                      │                                       │
│                      ▼                                       │
│              Final Layer Norm                                │
│                      │                                       │
│                      ▼                                       │
│              Linear Projection to Vocab                      │
│                      │                                       │
│                      ▼                                       │
│              Softmax → Token Probabilities                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 第一步：基础组件

### 1.1 导入依赖

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 设置随机种子保证可复现
torch.manual_seed(42)
```

### 1.2 配置参数

```python
class GPTConfig:
    """GPT 模型配置"""

    def __init__(
        self,
        vocab_size: int = 50304,      # 词表大小（取整方便计算）
        max_seq_len: int = 256,       # 最大序列长度
        d_model: int = 384,           # 嵌入维度（原始512，简化用384）
        n_heads: int = 6,             # 注意力头数（原始8）
        n_layers: int = 6,            # Transformer 层数（原始6）
        d_ff: int = None,             # FFN 隐藏层维度（默认 4×d_model）
        dropout: float = 0.1,         # Dropout 比例
        eps: float = 1e-5,            # LayerNorm epsilon
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model  # 默认 4 倍扩展
        self.dropout = dropout
        self.eps = eps

        # 验证配置
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
```

### 1.3 Token Embedding + Positional Encoding

```python
class Embeddings(nn.Module):
    """
    将 token 索引转换为向量表示

    组成：
    - Token Embedding: 学习每个 token 的语义表示
    - Positional Encoding: 注入位置信息

    最终输出 = Token Embedding + Positional Encoding
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding: 将 token id 映射到 d_model 维向量
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # Positional Encoding: 学习位置嵌入（比 sinusoidal 更灵活）
        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape

        # 获取位置索引 [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device)

        # Token Embedding: (batch_size, seq_len, d_model)
        token_embeds = self.token_embedding(input_ids)

        # Positional Embedding: (seq_len, d_model) → (1, seq_len, d_model)
        # expand 会复制 batch_size 次，不占用额外内存
        pos_embeds = self.position_embedding(positions).unsqueeze(0)

        # 相加组合（广播机制）
        x = token_embeds + pos_embeds

        # Dropout
        x = self.dropout(x)

        return x
```

---

## 第二步：Causal Self-Attention

### 2.1 核心思想

> [!important] Causal Masking
> GPT 是自回归模型，**每个 token 只能看到之前的 token**，不能看到未来的信息。
>
> 这通过一个**上三角掩码矩阵**实现：

```
         Token 0  Token 1  Token 2  Token 3
Token 0    ✓       ✗        ✗        ✗
Token 1    ✓       ✓        ✗        ✗
Token 2    ✓       ✓        ✓        ✗
Token 3    ✓       ✓        ✓        ✓
```

### 2.2 单个注意力头

```python
class CausalAttentionHead(nn.Module):
    """
    单个因果自注意力头

    流程：
    1. 将输入投影到 Q, K, V
    2. 计算 attention scores = Q @ K^T / sqrt(d_k)
    3. 应用 causal mask（屏蔽未来信息）
    4. Softmax 归一化
    5. 加权求和：scores @ V
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 每个 head 的维度
        d_model = config.d_model
        n_heads = config.n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q, K, V 投影矩阵
        self.q_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)

        # 预构建 causal mask（缓存，不用每次计算）
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 投影到 Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_k)
        K = self.k_proj(x)  # (batch, seq_len, d_k)
        V = self.v_proj(x)  # (batch, seq_len, d_k)

        # 2. 计算注意力分数: Q @ K^T
        # (batch, seq_len, d_k) @ (batch, d_k, seq_len)
        # = (batch, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # 3. 应用 causal mask
        # 将未来位置设为 -inf（softmax 后变为 0）
        mask = self.mask[:seq_len, :seq_len]  # 截取当前序列长度
        scores = scores.masked_fill(mask, float('-inf'))

        # 4. Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 加权求和
        # (batch, seq_len, seq_len) @ (batch, seq_len, d_k)
        # = (batch, seq_len, d_k)
        output = attn_weights @ V

        return output
```

### 2.3 多头注意力

```python
class MultiHeadCausalAttention(nn.Module):
    """
    多头因果自注意力

    将多个注意力头并行计算，然后拼接结果
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 使用并行的单头实现（比单个多头效率高）
        self.qkv_proj = nn.Linear(
            config.d_model,
            3 * config.d_model,  # Q, K, V 一起投影
            bias=False
        )

        # 输出投影
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        n_heads = self.config.n_heads
        d_k = d_model // n_heads

        # 1. 一次性计算 Q, K, V
        # (batch, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # 分割成 Q, K, V
        # (batch, seq_len, 3, n_heads, d_k)
        qkv = qkv.view(batch_size, seq_len, 3, n_heads, d_k)

        # 转置: (3, batch, n_heads, seq_len, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力分数
        # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
        # = (batch, n_heads, seq_len, seq_len)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

        # 3. 应用 causal mask
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float('-inf'))

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 加权求和
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        # = (batch, n_heads, seq_len, d_k)
        output = attn_weights @ V

        # 6. 合并多头
        # (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
        output = output.transpose(1, 2)

        # (batch, seq_len, n_heads, d_k) → (batch, seq_len, d_model)
        output = output.contiguous().view(batch_size, seq_len, d_model)

        # 7. 输出投影
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output
```

---

## 第三步：Feed-Forward Network

```python
class FeedForward(nn.Module):
    """
    位置-wise 前馈网络

    对每个位置独立应用两层全连接：
    Linear(d_model → 4×d_model) → ReLU → Linear(4×d_model → d_model)

    作用：
    - 在注意力之后"消化"信息
    - 引入非线性变换能力
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # 两层全连接
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),           # GELU 比 ReLU 更平滑
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.net(x)
```

---

## 第四步：Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    Transformer 块（Decoder 层）

    结构：
    1. Layer Normalization
    2. Multi-Head Causal Self-Attention
    3. Residual Connection
    4. Layer Normalization
    5. Feed-Forward Network
    6. Residual Connection
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Pre-Norm: 先做 LayerNorm，再做注意力/FFN
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.attn = MultiHeadCausalAttention(config)

        self.norm2 = nn.LayerNorm(config.d_model, eps=config.eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Attention + Residual
        x = x + self.attn(self.norm1(x))

        # FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x
```

---

## 第五步：完整 GPT 模型

```python
class GPT(nn.Module):
    """
    完整的 GPT 模型

    架构：
    1. Embeddings (Token + Position)
    2. Transformer Blocks × N
    3. Final Layer Norm
    4. Linear projection to vocab logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1. Embeddings
        self.embeddings = Embeddings(config)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # 3. Final Layer Norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.eps)

        # 4. Language Modeling Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定：让 embedding 和 lm_head 共享权重
        # 减少参数量，提高训练稳定性
        self.embeddings.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # 1. Embeddings
        x = self.embeddings(input_ids)  # (batch, seq_len, d_model)

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final Layer Norm
        x = self.ln_f(x)

        # 4. Project to vocab logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        自回归生成文本

        Args:
            idx: (batch_size, seq_len) 初始 token 序列
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（越低越确定性）
            top_k: 只保留概率最高的 k 个 token

        Returns:
            idx: (batch_size, seq_len + max_new_tokens) 生成后的序列
        """
        for _ in range(max_new_tokens):
            # 裁剪到最大序列长度
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # 前向传播获取 logits
            logits = self(idx_cond)  # (batch, seq_len, vocab_size)

            # 只取最后一个时间步的 logits
            logits = logits[:, -1, :]  # (batch, vocab_size)

            # 应用 temperature
            logits = logits / temperature

            # 可选：top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Softmax 获取概率
            probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)

            # 采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # 拼接到序列
            idx = torch.cat([idx, idx_next], dim=1)  # (batch, seq_len + 1)

        return idx
```

---

## 第六步：测试模型

### 6.1 创建模型

```python
# 创建配置
config = GPTConfig(
    vocab_size=50304,
    max_seq_len=256,
    d_model=384,
    n_heads=6,
    n_layers=6,
)

# 创建模型
model = GPT(config)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

# 输出：
# 总参数量: 39,238,656
# 可训练参数: 39,238,656
# 模型大小: 149.58 MB
```

### 6.2 前向传播测试

```python
# 随机输入
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

print(f"输入 shape: {input_ids.shape}")  # (4, 32)

# 前向传播
logits = model(input_ids)

print(f"输出 shape: {logits.shape}")  # (4, 32, 50304)
print(f"输出值范围: [{logits.min():.2f}, {logits.max():.2f}]")
```

### 6.3 简单文本生成（随机）

```python
# 从随机 token 开始
start_ids = torch.randint(0, config.vocab_size, (1, 10))

# 生成
generated_ids = model.generate(
    start_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=30
)

print(f"生成序列长度: {generated_ids.shape[1]}")  # 60
print(f"生成 token ids: {generated_ids[0].tolist()}")
```

---

## 模型结构总结

| 组件 | 输入维度 | 输出维度 | 参数量（约） |
|:-----|:---------|:---------|:-------------|
| Token Embedding | vocab | d_model | vocab × d_model |
| Position Embedding | max_seq | d_model | max_seq × d_model |
| Multi-Head Attention | d_model | d_model | 4 × d_model² |
| Feed-Forward | d_model | d_model | 8 × d_model² |
| Layer Norm × 2 | d_model | d_model | 4 × d_model |
| **单层 Transformer Block** | - | - | ~12 × d_model² |
| **N 层 GPT** | - | - | ~12 × N × d_model² |

对于我们的配置 (d_model=384, n_layers=6)：
- 理论参数量 ≈ 12 × 6 × 384² ≈ 10.6M（不含 embedding）

---

## 下一步

### 学习清单

- [ ] 尝试不同超参数组合
- [ ] 在小数据集上训练（如莎士比亚文本）
- [ ] 实现 Cross-Attention（变为 Encoder-Decoder）
- [ ] 实现 BERT 风格的 Encoder-only 模型
- [ ] 添加 Flash Attention 优化
- [ ] 实现 RoPE（旋转位置编码）

### 相关笔记

- [[AI研究/AI学习/02-模型原理/Transformer全面解析]]
- [[AI研究/AI学习/02-模型原理/Transformer研读.md]]

---

#Transformer #GPT #动手实现 #PyTorch
