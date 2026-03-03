---
title: RNN/LSTM/GRU 全面解析
date: 2026-02-28
tags:
  - RNN
  - LSTM
  - GRU
  - 序列建模
  - 深度学习
  - 架构原理
status: active
---

# RNN/LSTM/GRU 全面解析

> [!info] 说明
> 本笔记系统介绍循环神经网络（RNN）及其变体LSTM、GRU的原理、架构对比、开源框架及实战应用，涵盖文本生成、时间序列预测、语音识别等多个场景

---

## 📑 目录

> [!tip] 使用说明
> 点击下方的任何章节链接，即可跳转到对应内容（支持 `Ctrl/Cmd + Click` 在新面板打开）

### 基础理论
- [[#一、核心原理]]
  - [[#1.1 什么是RNN（循环神经网络）]]
  - [[#1.2 序列建模基本概念]]
  - [[#1.3 时间展开]]
  - [[#1.4 数学表达]]
  - [[#1.5 梯度消失/爆炸问题]]
- [[#二、为什么需要RNN]]
  - [[#2.1 序列数据的特点]]
  - [[#2.2 与MLP的对比]]
  - [[#2.3 与CNN的对比]]
  - [[#2.4 时序依赖建模]]

### 架构详解
- [[#三、RNN变体详解]]
  - [[#3.1 Vanilla RNN（简单RNN）]]
  - [[#3.2 LSTM（Long Short-Term Memory）]]
  - [[#3.3 GRU（Gated Recurrent Unit）]]
  - [[#3.4 Bidirectional RNN]]
  - [[#3.5 Deep RNN（多层RNN）]]
- [[#四、架构对比总结]]
  - [[#4.1 RNN vs LSTM vs GRU]]
  - [[#4.2 参数量详细对比]]
  - [[#4.3 效果对比]]
  - [[#4.4 与Transformer对比]]

### 实战应用
- [[#五、开源框架与实现]]
  - [[#5.1 PyTorch RNN模块]]
  - [[#5.2 常见Pitfalls]]
  - [[#5.3 PACKED_SEQUENCE使用]]
- [[#六、性能与耗时分析]]
  - [[#6.1 序列长度 vs 性能]]
  - [[#6.2 GPU加速效果]]
  - [[#6.3 与Transformer效率对比]]
  - [[#6.4 优化建议]]
- [[#七、实战案例]]
  - [[#7.1 文本生成]]
  - [[#7.2 时间序列预测]]
  - [[#7.3 语音识别]]
  - [[#7.4 机器翻译]]
  - [[#7.5 情感分析]]
  - [[#7.6 音乐生成]]
  - [[#7.7 异常检测（时序数据）]]

### 参考指南
- [[#八、方案选择建议]]
- [[#九、常见问题]]
  - [[#Q1: 梯度消失解决方案]]
  - [[#Q2: 长序列处理]]
  - [[#Q3: 训练不稳定]]
- [[#十、学习资源]]
- [[#十一、相关笔记]]

---

## 一、核心原理

### 1.1 什么是RNN（循环神经网络）

**RNN（Recurrent Neural Network）** 是专门处理序列数据的神经网络。

```
传统神经网络：处理独立样本
├─ MLP：独立样本，无记忆
├─ CNN：空间局部连接（图像）
└─ 输入输出维度固定

RNN：处理序列数据
├─ 时序依赖建模
├─ 可变长度输入输出
└─ 隐状态传递信息
```

### 1.2 序列建模基本概念

**序列数据特点**：

```
序列 = 有序的数据点集合
├─ 时间序列：股票价格、天气、传感器数据
├─ 文本序列：字符、词、句子
├─ 语音序列：声波信号、MFCC特征
└─ 视频序列：帧序列

关键特性：
├─ 顺序重要：词序改变语义（"狗咬人" vs "人咬狗"）
├─ 长度可变：句子长度不同
└─ 上下文依赖：当前词依赖前面的词
```

### 1.3 时间展开

RNN的核心思想是**参数共享**：同一组权重在不同时间步重复使用。

```
时间展开图：

输入序列: x_1 → x_2 → x_3 → ... → x_T
            ↓     ↓     ↓           ↓
            h_1 → h_2 → h_3 → ... → h_T
            ↓     ↓     ↓           ↓
输出序列:   y_1   y_2   y_3   ...   y_T

其中：
├─ x_t: t时刻的输入
├─ h_t: t时刻的隐状态（记忆）
├─ y_t: t时刻的输出
└─ 所有时间步共享相同的W, U, V参数
```

### 1.4 数学表达

对于时间步 **t**：

```python
# Vanilla RNN前向传播
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)  # 隐状态更新
y_t = W_hy · h_t + b_y                          # 输出计算

其中：
├─ h_t ∈ R^H: 隐状态（H为隐状态维度）
├─ x_t ∈ R^D: 输入（D为输入维度）
├─ y_t ∈ R^O: 输出（O为输出维度）
├─ W_hh ∈ R^(H×H): 隐状态到隐状态的权重
├─ W_xh ∈ R^(H×D): 输入到隐状态的权重
├─ W_hy ∈ R^(O×H): 隐状态到输出的权重
└─ tanh: 激活函数
```

### 1.5 梯度消失/爆炸问题

RNN训练时通过时间反向传播（BPTT），存在严重的梯度问题：

```
梯度随时间步的变化：

∂L/∂h_t = ∏_{k=t+1}^{T} ∂h_k/∂h_{k-1} · ∂L/∂h_T

其中 ∂h_k/∂h_{k-1} ≈ W_hh · diag(φ'(·))

当 |λ_max(W_hh)| < 1: 梯度指数衰减 → 梯度消失
当 |λ_max(W_hh)| > 1: 梯度指数增长 → 梯度爆炸

直观理解：
├─ 长序列 = 多层乘法
├─ 小于1的数连乘 → 0（梯度消失）
└─ 大于1的数连乘 → ∞（梯度爆炸）
```

**影响**：

| 问题 | 现象 | 原因 |
|:-----|:-----|:-----|
| **梯度消失** | 无法学习长期依赖 | 远端信息梯度衰减到0 |
| **梯度爆炸** | 训练不稳定、NaN | 梯度值指数增长 |

**解决方案**：

```
梯度爆炸：
├─ 梯度裁剪（Gradient Clipping）
└─ ||g|| ← max(1, c/||g||) · g  当 ||g|| > c

梯度消失：
├─ LSTM/GRU门控机制
├─ 更好的初始化
└─ 残差连接
```

---

## 二、为什么需要RNN

### 2.1 序列数据的特点

```
序列数据 vs 独立数据：

独立数据（MLP适用）：
├─ 每个样本独立
├─ 样本间无顺序关系
└─ 例如：图像分类（每张图独立）

序列数据（RNN适用）：
├─ 样本间有时序关系
├─ 当前样本依赖历史信息
└─ 例如：翻译（当前词依赖前面的词）
```

### 2.2 与MLP的对比

| 特性 | MLP | RNN |
|:-----|:-----|:-----|
| **输入** | 固定维度向量 | 可变长度序列 |
| **输出** | 单一输出 | 序列输出/单一输出 |
| **记忆** | 无 | 有（隐状态） |
| **参数共享** | 无（每层独立） | 有（时间步共享） |
| **适用任务** | 分类、回归 | 序列建模 |

```python
# MLP无法处理变长序列
MLP:
x_1, x_2, ..., x_T  → 需要拼接或padding → 固定长度 → MLP → 输出

# RNN自然处理变长序列
RNN:
x_1 → h_1 ┐
x_2 → h_2 ├→ 最终隐状态 h_T → 输出
...       │
x_T → h_T ┘
```

### 2.3 与CNN的对比

| 特性 | CNN | RNN |
|:-----|:-----|:-----|
| **感受野** | 空间局部 | 时间全局（理论上） |
| **连接方式** | 局部连接 | 全连接（时间维度） |
| **参数共享** | 空间共享 | 时间共享 |
| **归纳偏置** | 局部性、平移不变性 | 时序依赖 |

```python
# 1D-CNN也可以处理序列
1D-CNN:
x = [x_1, x_2, x_3, x_4, x_5]
卷积核大小=3:
h_1 = conv(x_1, x_2, x_3)
h_2 = conv(x_2, x_3, x_4)
h_3 = conv(x_3, x_4, x_5)

# 1D-CNN vs RNN
CNN: 感受野有限（卷积核大小）
RNN: 理论上可以记住任意长的历史（实际受限于梯度消失）
```

### 2.4 时序依赖建模

```
RNN如何建模时序依赖：

短程依赖（1-3步）：
"I ate a _____"  → 容易预测（apple/banana）

中程依赖（10-20步）：
"After growing up in France, she spoke fluent _____"  → 需要记住France

长程依赖（100+步）：
长文档开头提到"The person is John"，结尾问"Who is the person?"
     ↓ 需要跨越很长距离传递信息

RNN能力：
├─ Vanilla RNN: 只能学习短程依赖
├─ LSTM: 可以学习中程依赖
└─ GRU: 类似LSTM，但更高效
```

---

## 三、RNN变体详解

### 3.1 Vanilla RNN（简单RNN）

#### 原理

最基础的RNN，使用tanh激活函数：

```python
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

#### 特点

- ✅ 结构简单，计算快速
- ✅ 参数少，易于理解
- ❌ 严重梯度消失问题
- ❌ 无法学习长期依赖（>10步）

#### 代码示例

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 参数定义
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            output: [batch_size, seq_len, output_size]
            hidden: [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐状态
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            # RNN核心计算
            h = torch.tanh(self.W_hh(h) + self.W_xh(x[:, t, :]) + self.b_h)
            outputs.append(self.W_hy(h))

        output = torch.stack(outputs, dim=1)
        return output, h

# 使用示例
rnn = VanillaRNN(input_size=128, hidden_size=256, output_size=10)
x = torch.randn(32, 50, 128)  # batch=32, seq_len=50, input_dim=128
output, hidden = rnn(x)
print(output.shape)  # [32, 50, 10]
```

---

### 3.2 LSTM（Long Short-Term Memory）

#### 原理

LSTM通过**门控机制**和**细胞状态**解决梯度消失问题：

```
LSTM结构：

输入 x_t ────────┬───────────────────────┬────────────────→ h_t
                 │                       │
                 ▼                       ▼
            ┌─────────┐           ┌─────────┐
            │  遗忘门  │           │  输入门  │
            │  f_gate  │           │  i_gate  │
            └────┬────┘           └────┬────┘
                 │                      │
                 ▼                      ▼
    ┌─────────────────────┐   ┌─────────────────┐
    │      细胞状态        │   │  候选细胞状态    │
    │   C_{t-1} ⊙ f_gate   │   |   C̃_t           │
    └─────────────────────┘   └────────┬────────┘
                 │                      │
                 ▼                      │
    ┌──────────────────────────────────┘
    │      C_t = C_{t-1} ⊙ f + i ⊙ C̃    │
    └────────────────┬───────────────────┘
                     │
                     ▼
              ┌──────────┐
              │  输出门   │
              │  o_gate   │
              └────┬─────┘
                   │
                   ▼
         h_t = o_gate ⊙ tanh(C_t)
```

#### 门控机制详解

```python
# LSTM完整公式

# 1. 遗忘门：决定丢弃多少旧信息
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
# f_t ∈ [0,1]^H，0=完全遗忘，1=完全保留

# 2. 输入门：决定写入多少新信息
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
# i_t ∈ [0,1]^H

# 3. 候选细胞状态：要写入的新信息
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

# 4. 更新细胞状态
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
# ⊙ 表示逐元素乘法

# 5. 输出门：决定输出多少信息
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

# 6. 计算隐状态
h_t = o_t ⊙ tanh(C_t)

其中：
├─ σ: sigmoid函数，输出[0,1]
├─ tanh: 输出[-1,1]
└─ [h, x]: 拼接操作
```

#### 细胞状态的意义

```
细胞状态 C_t vs 隐状态 h_t：

细胞状态 C_t：
├─ "长期记忆"通道
├─ 梯度可以顺畅传播（恒等映射路径）
└─ 解决梯度消失问题

隐状态 h_t：
├─ "短期记忆"/输出状态
├─ 用于当前时间步的输出
└─ 传递给下一时间步

关键洞察：
C_t = f_t ⊙ C_{t-1} + ...
当 f_t ≈ 1 时，C_t ≈ C_{t-1}
梯度可以无衰减地传播！
```

#### 特点

- ✅ 解决梯度消失问题
- ✅ 可以学习长期依赖（100+步）
- ✅ 门控机制自动学习记忆/遗忘
- ❌ 参数量大（4倍于Vanilla RNN）
- ❌ 计算复杂度高

#### 代码示例

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """手动实现LSTM单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 所有门和候选状态的权重可以合并计算
        self.combined = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h=None, c=None):
        """
        Args:
            x: [batch_size, input_size]
            h: [batch_size, hidden_size] 或 None
            c: [batch_size, hidden_size] 或 None
        Returns:
            h_next: [batch_size, hidden_size]
            c_next: [batch_size, hidden_size]
        """
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c is None:
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # 拼接输入和隐状态
        combined = torch.cat([x, h], dim=1)

        # 计算所有门和候选状态（一次线性变换）
        gates = self.combined(combined)

        # 分割成四个部分
        i, f, g, o = gates.chunk(4, dim=1)

        # 应用激活函数
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)     # 候选细胞状态
        o = torch.sigmoid(o)  # 输出门

        # 更新细胞状态和隐状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class CustomLSTM(nn.Module):
    """自定义LSTM层"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            outputs: [batch_size, seq_len, hidden_size]
            (h_n, c_n): 最终隐状态和细胞状态
        """
        batch_size, seq_len, _ = x.shape

        # 初始化
        h = [None] * self.num_layers
        c = [None] * self.num_layers

        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]  # 下一层的输入
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)

        # 收集最终状态
        h_n = torch.stack([h[i] for i in range(self.num_layers)], dim=0)
        c_n = torch.stack([c[i] for i in range(self.num_layers)], dim=0)

        return outputs, (h_n, c_n)


# 使用PyTorch内置LSTM
class PyTorchLSTMExample(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 词索引
        """
        # 嵌入
        x = self.embedding(x)  # [batch, seq_len, embed_size]

        # LSTM
        output, (h_n, c_n) = self.lstm(x)
        # output: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        # c_n: [num_layers, batch, hidden_size]

        # 使用最后时间步的输出
        last_output = output[:, -1, :]
        logits = self.fc(last_output)

        return logits
```

---

### 3.3 GRU（Gated Recurrent Unit）

#### 原理

GRU是LSTM的简化版本，合并了输入门和遗忘门：

```
GRU结构：

输入 x_t ────────┬────────────────────────────────→ h_t
                 │
                 ▼
            ┌─────────┐
            │ 更新门  │
            │  z_gate  │
            └────┬────┘
                 │
         ┌───────┴────────┐
         ▼                ▼
    ┌────────┐      ┌─────────┐
    │ 重置门  │      │ 候选状态 │
    │ r_gate  │      │  h̃_t    │
    └────┬───┘      └────┬────┘
         │               │
         └───────┬───────┘
                 ▼
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

#### 数学表达

```python
# GRU完整公式

# 1. 更新门：决定保留多少旧状态
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

# 2. 重置门：决定遗忘多少旧状态
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

# 3. 候选隐状态
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

# 4. 更新隐状态
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

对比LSTM：
├─ 没有独立的细胞状态
├─ 更新门 = 1 - 遗忘门（约束关系）
└─ 参数量更少（3组 vs 4组权重）
```

#### 特点

- ✅ 比LSTM更简单高效
- ✅ 参数量少25%
- ✅ 训练速度更快
- ✅ 性能与LSTM相当
- ❌ 表达能力略弱于LSTM（理论）

#### LSTM vs GRU

| 特性 | LSTM | GRU |
|:-----|:-----|:-----|
| **门数量** | 3个（输入、遗忘、输出） | 2个（更新、重置） |
| **细胞状态** | 有（C_t和h_t分离） | 无（只有h_t） |
| **参数量** | 4 × (H² + HD + H) | 3 × (H² + HD + H) |
| **训练速度** | 较慢 | 较快 |
| **长期依赖** | 更强 | 强 |
| **适用场景** | 长序列、复杂任务 | 中等序列、资源受限 |

#### 代码示例

```python
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """手动实现GRU单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # 所有门和候选状态的权重
        self.combined = nn.Linear(input_size + hidden_size, 3 * hidden_size)

    def forward(self, x, h=None):
        """
        Args:
            x: [batch_size, input_size]
            h: [batch_size, hidden_size] 或 None
        Returns:
            h_next: [batch_size, hidden_size]
        """
        batch_size = x.size(0)

        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # 拼接
        combined = torch.cat([x, h], dim=1)

        # 计算所有门和候选状态
        gates = self.combined(combined)

        # 分割
        r, z, n = gates.chunk(3, dim=1)

        # 激活
        r = torch.sigmoid(r)  # 重置门
        z = torch.sigmoid(z)  # 更新门
        n = torch.tanh(n)     # 候选状态

        # 更新隐状态
        h_next = (1 - z) * h + z * n

        return h_next


class CustomGRU(nn.Module):
    """自定义GRU层"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            outputs: [batch_size, seq_len, hidden_size]
            h_n: [num_layers, batch_size, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        h = [None] * self.num_layers
        outputs = []

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = self.cells[layer](inp, h[layer])
                inp = h[layer]
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)

        return outputs, h_n


# PyTorch内置GRU使用示例
class SentimentClassifier(nn.Module):
    """基于GRU的情感分类模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        # 双向GRU，输出维度乘以2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 词索引
        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.embedding(x)

        # GRU
        output, h_n = self.gru(x)
        # output: [batch, seq_len, hidden*2]
        # h_n: [num_layers*2, batch, hidden]

        # 拼接双向的最终隐状态
        # h_n[-2] 是前向的最后一层
        # h_n[-1] 是后向的最后一层
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits
```

---

### 3.4 Bidirectional RNN

#### 原理

双向RNN同时从前向和后向处理序列：

```
双向RNN结构：

输入序列: x_1 → x_2 → x_3 → ... → x_T

前向RNN:  → h_1 → h_2 → h_3 → ... → h_T
后向RNN:  ← g_1 ← g_2 ← g_3 ← ... ← g_T

合并:     [h_1⊕g_1] [h_2⊕g_2] [h_3⊕g_3] ... [h_T⊕g_T]
```

#### 特点

- ✅ 利用上下文信息（过去+未来）
- ✅ 适合序列标注任务（POS、NER）
- ❌ 无法用于实时生成（需要看到未来）
- ❌ 参数量翻倍

#### 代码示例

```python
import torch
import torch.nn as nn

class BiRNNTagger(nn.Module):
    """双向RNN序列标注模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # 双向，输出维度乘以2
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            tags: [batch_size, seq_len, num_tags]
        """
        x = self.embedding(x)

        output, _ = self.lstm(x)
        # output: [batch, seq_len, hidden*2]

        # 每个时间步都预测
        tags = self.fc(output)

        return tags

# 使用示例
model = BiRNNTagger(vocab_size=10000, embed_size=300, hidden_size=256, num_tags=10)
x = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
tags = model(x)
print(tags.shape)  # [32, 50, 10]
```

---

### 3.5 Deep RNN（多层RNN）

#### 原理

堆叠多个RNN层，形成深度网络：

```
多层RNN结构：

输入: x_1 → x_2 → x_3 → ... → x_T
          ↓     ↓     ↓
第1层: h_1^(1) → h_2^(1) → h_3^(1) → ... → h_T^(1)
          ↓     ↓     ↓
第2层: h_1^(2) → h_2^(2) → h_3^(2) → ... → h_T^(2)
          ↓     ↓     ↓
第3层: h_1^(3) → h_2^(3) → h_3^(3) → ... → h_T^(3)
          ↓     ↓     ↓
输出:   y_1   y_2   y_3   ...   y_T

每层学习不同层次的特征：
├─ 底层：局部模式、字符级/词级
├─ 中层：短语、句法
└─ 顶层：语义、全局信息
```

#### 特点

- ✅ 学习多层次抽象
- ✅ 更强的表达能力
- ❌ 训练难度增加
- ❌ 梯度问题更严重
- ❌ 层数不宜过深（通常2-4层）

#### 代码示例

```python
import torch
import torch.nn as nn

class DeepRNN(nn.Module):
    """多层RNN语言模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 多层LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch_size, seq_len]
            hidden: (h_0, c_0) 或 None
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: (h_n, c_n)
        """
        x = self.embedding(x)

        if hidden is None:
            output, hidden = self.lstm(x)
        else:
            output, hidden = self.lstm(x, hidden)

        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size, device):
        """初始化隐状态"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


# 不同深度的影响
model_1layer = DeepRNN(vocab_size=10000, embed_size=300, hidden_size=512, num_layers=1)
model_2layer = DeepRNN(vocab_size=10000, embed_size=300, hidden_size=512, num_layers=2)
model_4layer = DeepRNN(vocab_size=10000, embed_size=300, hidden_size=512, num_layers=4)

# 参数量对比
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"1层: {count_parameters(model_1layer):,}")  # ~10M
print(f"2层: {count_parameters(model_2layer):,}")  # ~15M
print(f"4层: {count_parameters(model_4layer):,}")  # ~25M
```

---

## 四、架构对比总结

### 4.1 RNN vs LSTM vs GRU

| 特性 | Vanilla RNN | LSTM | GRU |
|:-----|:-----------:|:----:|:---:|
| **门控机制** | 无 | 3门 | 2门 |
| **细胞状态** | 无 | 有 | 无 |
| **参数量（每层）** | H²+HD+HO | 4(H²+HD)+H | 3(H²+HD) |
| **计算复杂度** | 低 | 高 | 中 |
| **训练速度** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **长期依赖** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **内存占用** | 低 | 高 | 中 |
| **适用序列长度** | 短（<20） | 长（>100） | 中（50-100） |

> H = 隐状态维度, D = 输入维度, O = 输出维度

### 4.2 参数量详细对比

假设输入维度D=256，隐状态维度H=512：

```python
# Vanilla RNN
params_rnn = H*H + H*D + H*O  # 权重
            = 512*512 + 512*256 + 512*O
            = 262,144 + 131,072 + ...
            ≈ 393K + O*512

# LSTM (4组权重)
params_lstm = 4 * (H*H + H*D) + 4*H  # 权重 + 偏置
             = 4 * (512*512 + 512*256) + 2048
             = 4 * 393,216 + 2048
             ≈ 1,574K

# GRU (3组权重)
params_gru = 3 * (H*H + H*D) + 3*H
            = 3 * 393,216 + 1536
            ≈ 1,181K

# 比例
RNN : GRU : LSTM ≈ 1 : 3 : 4
```

### 4.3 效果对比

| 任务类型 | 推荐模型 | 理由 |
|:---------|:---------|:-----|
| **短序列分类** | Vanilla RNN / GRU | 简单快速 |
| **长序列建模** | LSTM | 长期依赖能力强 |
| **实时应用** | GRU | 速度快 |
| **资源受限** | GRU | 参数少 |
| **需要可解释性** | LSTM | 细胞状态可视化 |
| **序列标注** | Bi-LSTM | 上下文信息 |

### 4.4 与Transformer对比

| 特性 | RNN家族 | Transformer |
|:-----|:--------|:-----------|
| **计算方式** | 串行（O(n)） | 并行（O(1)） |
| **长期依赖** | 渐弱（LSTM可改善） | 强（注意力机制） |
| **训练速度** | 慢（串行） | 快（并行） |
| **推理速度** | 快 | 慢（O(n²)注意力） |
| **位置编码** | 隐式（顺序处理） | 显式（位置编码） |
| **数据需求** | 中等 | 大 |
| **适用场景** | 小数据、时序任务 | 大数据、复杂任务 |

---

## 五、开源框架与实现

### 5.1 PyTorch RNN模块

#### nn.RNN

```python
import torch
import torch.nn as nn

# 创建RNN层
rnn = nn.RNN(
    input_size=128,      # 输入维度
    hidden_size=256,     # 隐状态维度
    num_layers=2,        # 层数
    nonlinearity='tanh', # 'tanh' 或 'relu'
    batch_first=True,    # 输入格式 [batch, seq, feature]
    dropout=0.2,         #dropout概率（多层时）
    bidirectional=False  # 是否双向
)

# 前向传播
x = torch.randn(32, 50, 128)  # [batch, seq_len, input_size]
output, h_n = rnn(x)

print(output.shape)  # [32, 50, 256] 所有时间步的输出
print(h_n.shape)     # [2, 32, 256]  最终隐状态 [num_layers, batch, hidden]
```

#### nn.LSTM

```python
# 创建LSTM层
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=True  # 双向LSTM
)

# 前向传播
x = torch.randn(32, 50, 128)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # [32, 50, 512] 双向，hidden*2
print(h_n.shape)     # [4, 32, 256]  双向*num_layers
print(c_n.shape)     # [4, 32, 256]  细胞状态
```

#### nn.GRU

```python
# 创建GRU层
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2,
    bidirectional=False
)

# 前向传播
x = torch.randn(32, 50, 128)
output, h_n = gru(x)

print(output.shape)  # [32, 50, 256]
print(h_n.shape)     # [2, 32, 256]
```

### 5.2 常见Pitfalls

#### 1. 忘记设置batch_first

```python
# 错误：默认batch_first=False
rnn = nn.RNN(input_size=128, hidden_size=256)
x = torch.randn(32, 50, 128)  # [batch, seq, feature]
output, h_n = rnn(x)  # 错误！期望 [seq, batch, feature]

# 正确：设置batch_first=True
rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True)
output, h_n = rnn(x)  # 正确
```

#### 2. 忘重置隐状态

```python
# 错误：隐状态累积
for batch in dataloader:
    output, h_n = lstm(batch)  # h_n会保留上次的状态

# 正确：每次重新初始化或detach
for batch in dataloader:
    h_0 = torch.zeros(num_layers, batch_size, hidden_size)
    c_0 = torch.zeros(num_layers, batch_size, hidden_size)
    output, (h_n, c_n) = lstm(batch, (h_0, c_0))

    # 或者如果需要状态传递（如生成）
    if i == 0:
        h_n, c_n = None, None
    else:
        h_n = h_n.detach()  # 阻止梯度传播
        c_n = c_n.detach()
    output, (h_n, c_n) = lstm(batch, (h_n, c_n))
```

#### 3. 单向/双向输出处理错误

```python
# 双向LSTM的输出处理
lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True, batch_first=True)
output, (h_n, c_n) = lstm(x)

# h_n的形状: [4, batch, 256]  # 2层 * 2方向
# 如何获取最终的隐状态？

# 方法1: 取最后时间步
final_output = output[:, -1, :]  # [batch, 512]

# 方法2: 拼接双向的最后一层
# h_n[-2] 是前向最后一层
# h_n[-1] 是后向最后一层
final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, 512]
```

### 5.3 PACKED_SEQUENCE使用

处理变长序列时，padding会浪费计算：

```python
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 变长序列
seqs = [
    torch.tensor([1, 2, 3]),      # 长度3
    torch.tensor([4, 5, 6, 7]),   # 长度4
    torch.tensor([8, 9])          # 长度2
]

# 1. Padding
lengths = torch.tensor([len(seq) for seq in seqs])
padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
# padded: [[1,2,3,0], [4,5,6,7], [8,9,0,0]]

# 2. Pack（压缩掉padding）
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

# 3. 通过RNN
lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
packed_output, (h_n, c_n) = lstm(packed)

# 4. Unpack（恢复padding）
output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

# 优势：
# - 只计算真实数据，不计算padding
# - 训练更快
# - 避免padding影响学习
```

完整使用示例：

```python
class PaddedLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        """
        Args:
            x: [batch, seq_len] padded后的序列
            lengths: [batch] 每个序列的真实长度
        """
        # 嵌入
        x = self.embedding(x)

        # Pack
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM
        packed_output, (h_n, c_n) = self.lstm(packed)

        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 取每个序列最后有效时间步的输出
        last_outputs = []
        for i, length in enumerate(lengths):
            last_outputs.append(output[i, length-1, :])
        last_outputs = torch.stack(last_outputs)

        logits = self.fc(last_outputs)
        return logits

# 使用
model = PaddedLSTM(vocab_size=10000, embed_size=300, hidden_size=256, num_classes=10)
x = torch.tensor([[1,2,3,0], [4,5,0,0], [6,7,8,9]])  # padded
lengths = torch.tensor([3, 2, 4])
logits = model(x, lengths)
```

---

## 六、性能与耗时分析

### 6.1 序列长度 vs 性能

```
时间复杂度分析：

Vanilla RNN:
├─ 前向: O(T × H² + T × H × D)
├─ 反向: O(T × H² + T × H × D)
└─ 总计: O(T × (H² + HD))

LSTM:
├─ 前向: O(4 × T × (H² + HD))
├─ 反向: O(4 × T × (H² + HD))
└─ 总计: O(T × H²) （主导项）

实际耗时 (H=512, D=256, GPU):
├─ T=10:   ~1ms
├─ T=50:   ~5ms
├─ T=100:  ~10ms
├─ T=500:  ~50ms
└─ T=1000: ~100ms

耗时与序列长度基本线性关系！
```

### 6.2 GPU加速效果

```
CPU vs GPU 单层LSTM推理耗时 (H=512, T=100):

设备    | 耗时    | 加速比
--------|---------|--------
CPU     | ~500ms  | 1x
GPU     | ~10ms   | 50x

批量处理效果:
batch_size=1:  ~10ms
batch_size=16: ~15ms
batch_size=32: ~20ms
batch_size=64: ~30ms

GPU并行效果显著，但大batch受显存限制
```

### 6.3 与Transformer效率对比

```
计算复杂度对比:

模型       | 训练复杂度      | 推理复杂度
----------|---------------|------------
RNN/LSTM  | O(T × H²)     | O(T × H²)
Transformer| O(T² × H)    | O(T² × H)

序列长度对训练时间的影响 (H=512):

T    | RNN耗时 | Transformer耗时
-----|---------|----------------
64   | 1x      | 1x
128  | 2x      | 4x
256  | 4x      | 16x
512  | 8x      | 64x
1024 | 16x     | 256x

结论:
├─ 短序列 (<128): 差距不大
├─ 中序列 (128-512): RNN更快
└─ 长序列 (>512): Transformer训练非常慢
```

### 6.4 优化建议

| 优化方法 | 适用场景 | 效果 |
|:---------|:---------|:-----|
| **PACKED_SEQUENCE** | 变长序列 | 10-30%加速 |
| **减少层数** | 过深网络 | 线性加速 |
| **降低隐状态维度** | 资源受限 | 平方加速 |
| **混合精度训练** | GPU支持 | ~2x加速 |
| **torch.compile** | PyTorch 2.0+ | 20-50%加速 |

```python
# 优化示例
import torch

# 1. 混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for x, y in dataloader:
    with autocast():
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. PyTorch 2.0 编译
model = torch.compile(model, mode='reduce-overhead')

# 3. 降低精度
model = model.half()  # FP16
```

---

## 七、实战案例

### 7.1 文本生成

#### 场景描述

基于字符或词生成文本，如：

- 字符级：生成莎士比亚风格文本
- 词级：生成新闻标题、故事
- 代码生成：生成Python代码

#### 字符级语言模型

```python
import torch
import torch.nn as nn
import numpy as np

class CharRNN(nn.Module):
    """字符级RNN语言模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 字符嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: (h_n, c_n)
        """
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, prime_seq, length, temperature=1.0, device='cpu'):
        """
        生成文本
        Args:
            prime_seq: 起始字符索引列表
            length: 生成字符数
            temperature: 采样温度（<1更保守，>1更多样）
        """
        self.eval()
        with torch.no_grad():
            # 初始化
            input_seq = torch.tensor([prime_seq], dtype=torch.long, device=device)
            hidden = None

            # 处理起始序列
            output, hidden = self.forward(input_seq, hidden)

            # 生成
            generated = prime_seq.copy()
            for _ in range(length):
                # 取最后字符的输出
                logits = output[0, -1, :] / temperature

                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_char = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_char)

                # 下一步输入
                input_seq = torch.tensor([[next_char]], dtype=torch.long, device=device)
                output, hidden = self.forward(input_seq, hidden)

            return generated


# 数据准备
def prepare_text(text, seq_length=100):
    """准备文本数据"""
    # 构建字符表
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    # 创建训练样本
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length):
        sequences.append([char_to_idx[c] for c in text[i:i+seq_length]])
        targets.append(char_to_idx[text[i+seq_length]])

    return char_to_idx, idx_to_char, sequences, targets


# 训练
def train_char_rnn(model, sequences, targets, epochs=50, batch_size=64, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        permutation = torch.randperm(sequences.size(0))

        for i in range(0, sequences.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = sequences[indices]
            batch_y = targets[indices]

            optimizer.zero_grad()
            output, _ = model(batch_x)
            loss = criterion(output[:, -1, :], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (len(sequences)//batch_size):.4f}')


# 使用示例
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune..."""

char_to_idx, idx_to_char, sequences, targets = prepare_text(text, seq_length=50)

model = CharRNN(
    vocab_size=len(char_to_idx),
    embed_size=64,
    hidden_size=256,
    num_layers=2
)

# 训练
train_char_rnn(model, sequences, targets, epochs=100)

# 生成
prime = [char_to_idx[c] for c in "To be"]
generated = model.generate(prime, length=200, temperature=0.8)
text = ''.join([idx_to_char[i] for i in generated])
print(text)
```

#### 词级语言模型

```python
import torch
import torch.nn as nn
from collections import Counter
import re

class WordLSTM(nn.Module):
    """词级LSTM语言模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden


def tokenize(text):
    """简单分词"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def build_vocab(texts, min_freq=2):
    """构建词表"""
    counter = Counter()
    for text in texts:
        words = tokenize(text)
        counter.update(words)

    # 过滤低频词
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return word_to_idx, vocab


# 训练策略
def train_word_lm(model, dataloader, epochs, lr=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output, _ = model(batch_x)
            # output: [batch, seq_len, vocab_size]
            # batch_y: [batch, seq_len]

            loss = criterion(output.reshape(-1, output.size(-1)), batch_y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')


# Top-k采样
def sample_with_top_k(logits, k=5):
    """从top-k中采样"""
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=-1)
    next_idx = top_k_indices[torch.multinomial(probs, 1)].item()
    return next_idx
```

#### 训练策略总结

| 策略 | 说明 |
|:-----|:-----|
| **Teacher Forcing** | 训练时使用真实值而非预测值作为输入 |
| **梯度裁剪** | 防止梯度爆炸，max_norm=1.0 |
| **温度采样** | 控制生成多样性，0.5-1.0 |
| **Top-k采样** | 从top-k中采样，避免低概率词 |
| **Nucleus采样** | 从累积概率达到p的词中采样 |

---

### 7.2 时间序列预测

#### 场景描述

预测时间序列的未来值：

- 股票价格预测
- 天气预报
- 能源消耗预测
- 销量预测

#### 股票价格预测

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockPredictor(nn.Module):
    """基于LSTM的股票价格预测"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # 注意力权重
        self.attention = nn.Linear(hidden_size, 1)

        # 预测层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            prediction: [batch, output_size]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size]

        # 注意力机制（可选）
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context: [batch, hidden_size]

        # 预测
        prediction = self.fc(context)

        return prediction


def prepare_stock_data(prices, seq_length=60, pred_length=1):
    """
    准备股票数据
    Args:
        prices: 价格序列 [num_days]
        seq_length: 输入序列长度（天数）
        pred_length: 预测长度
    """
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # 创建样本
    X, y = [], []
    for i in range(len(prices_scaled) - seq_length - pred_length + 1):
        X.append(prices_scaled[i:i+seq_length])
        y.append(prices_scaled[i+seq_length:i+seq_length+pred_length])

    return np.array(X), np.array(y), scaler


def train_stock_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # 批量训练
        for i in range(0, len(X_train), batch_size):
            batch_x = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
            batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32)
            val_y = torch.tensor(y_val, dtype=torch.float32)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y.squeeze()).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_stock_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')


# 多变量时间序列预测
class MultivariateTimeSeriesLSTM(nn.Module):
    """多变量LSTM预测模型"""
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, feature_size]
            特征: [开盘价, 收盘价, 最高价, 最低价, 成交量, ...]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后时间步
        last_hidden = h_n[-1]  # [batch, hidden_size]

        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        output = self.fc2(x)

        return output


# 使用示例
import pandas as pd

# 假设有股票数据
# df = pd.read_csv('stock_prices.csv')
# features = ['Open', 'High', 'Low', 'Close', 'Volume']
# data = df[features].values

# model = MultivariateTimeSeriesLSTM(
#     feature_size=len(features),
#     hidden_size=128,
#     num_layers=2,
#     output_size=1  # 预测明天收盘价
# )
```

#### 天气预测

```python
class WeatherPredictor(nn.Module):
    """基于GRU的天气预测"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
            输入特征: [温度, 湿度, 气压, 风速, 降水量]
        Returns:
            output: [batch, output_size]
        """
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]  # [batch, hidden_size]

        # 批归一化
        x = self.bn(last_hidden)
        output = self.fc(x)

        return output


# Seq2Seq预测（预测未来多个时间步）
class Seq2SeqPredictor(nn.Module):
    """编码器-解码器结构，用于多步预测"""
    def __init__(self, input_size, hidden_size, output_size, pred_steps):
        super().__init__()
        self.pred_steps = pred_steps

        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 解码器
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            predictions: [batch, pred_steps, output_size]
        """
        batch_size = x.size(0)

        # 编码
        _, (h_enc, c_enc) = self.encoder(x)

        # 解码初始输入（使用编码器最后输出）
        dec_input = torch.zeros(batch_size, 1, self.fc.out_features, device=x.device)
        h_dec, c_dec = h_enc, c_enc

        predictions = []
        for _ in range(self.pred_steps):
            # 单步解码
            dec_output, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))

            # 预测
            prediction = self.fc(dec_output)
            predictions.append(prediction)

            # 下一步输入（使用预测值）
            dec_input = prediction

        predictions = torch.cat(predictions, dim=1)
        return predictions


# 预测未来7天气温
model = Seq2SeqPredictor(
    input_size=5,    # 5个特征
    hidden_size=128,
    output_size=1,   # 预测温度
    pred_steps=7     # 预测7天
)
```

---

### 7.3 语音识别

#### 场景描述

将语音信号转换为文本：

- 输入：声波特征（MFCC、Mel频谱）
- 输出：文字序列
- 特点：输入输出长度不对齐

#### CTC Loss实现

```python
import torch
import torch.nn as nn

class ASRModel(nn.Module):
    """基于CTC的语音识别模型"""
    def __init__(self, input_size, hidden_size, num_layers, vocab_size):
        super().__init__()
        # 特征提取（可选，通常前置CNN）
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 分类层
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size] MFCC特征
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # CNN特征提取
        x = x.transpose(1, 2)  # [batch, input_size, seq_len]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # [batch, seq_len, 64]

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 分类
        logits = self.fc(lstm_out)

        # Log_softmax for CTC
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs


# CTC Loss训练
def train_asr(model, dataloader, epochs):
    # CTC Loss
    # blank_idx通常设为vocab_size-1或0
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            # batch: (features, targets, input_lengths, target_lengths)
            features, targets, input_lengths, target_lengths = batch

            optimizer.zero_grad()

            # 前向传播
            log_probs = model(features)

            # CTC需要 [seq_len, batch, vocab_size]
            log_probs = log_probs.transpose(0, 1)

            # 计算CTC Loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}')


# 解码（贪心解码）
def ctc_greedy_decode(log_probs, blank_idx=0):
    """
    简单的贪心解码
    Args:
        log_probs: [seq_len, vocab_size]
    Returns:
        decoded: 解码后的索引序列
    """
    # 取argmax
    preds = torch.argmax(log_probs, dim=-1)

    # 移除blank和重复
    decoded = []
    prev = None
    for p in preds:
        if p != blank_idx and p != prev:
            decoded.append(p.item())
        prev = p

    return decoded


# Beam Search解码（更准确）
import heapq

class BeamSearchCTC:
    """CTC Beam Search解码"""
    def __init__(self, beam_width=10, blank_idx=0):
        self.beam_width = beam_width
        self.blank_idx = blank_idx

    def decode(self, log_probs):
        """
        Args:
            log_probs: [seq_len, vocab_size]
        Returns:
            best_hyp: 最佳假设序列
        """
        seq_len, vocab_size = log_probs.shape

        # 初始状态: (负对数概率, 序列)
        beams = [(0.0, [])]

        for t in range(seq_len):
            new_beams = []

            for score, hyp in beams:
                # 扩展blank
                new_score = score - log_probs[t, self.blank_idx].item()
                new_beams.append((new_score, hyp))  # blank不加入序列

                # 扩展所有token
                for v in range(vocab_size):
                    if v == self.blank_idx:
                        continue

                    new_score = score - log_probs[t, v].item()

                    # CTC合并重复规则
                    if len(hyp) > 0 and hyp[-1] == v:
                        # 重复token，跳过
                        continue

                    new_hyp = hyp + [v]
                    new_beams.append((new_score, new_hyp))

            # 保留top-k
            beams = heapq.nsmallest(self.beam_width, new_beams)

        # 返回最佳假设
        return beams[0][1]
```

#### 深度语音识别（Deep Speech 2风格）

```python
class DeepSpeech2(nn.Module):
    """Deep Speech 2 风格的语音识别模型"""
    def __init__(self, input_size, hidden_size, num_layers, vocab_size):
        super().__init__()

        # 1. CNN特征提取（降采样）
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 计算CNN后的维度
        self.conv_output_size = 32 * ((input_size // 2) // 2)

        # 2. 多层双向RNN
        self.rnn_layers = nn.ModuleList([
            nn.LSTM(
                self.conv_output_size if i == 0 else hidden_size,
                hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            )
            for i in range(num_layers)
        ])

        # 3. 层归一化和dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])

        # 4. 输出层
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch, 1, freq, time] 声学特征
        Returns:
            log_probs: [batch, time, vocab_size]
        """
        batch_size = x.size(0)

        # CNN
        x = self.conv(x)
        # x: [batch, 32, freq', time']

        # 重组为RNN输入
        x = x.transpose(1, 2)  # [batch, time', 32, freq']
        x = x.contiguous().view(batch_size, x.size(1), -1)

        # RNN层
        for i, (rnn, ln) in enumerate(zip(self.rnn_layers, self.layer_norms)):
            x, _ = rnn(x)
            x = ln(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # 输出
        logits = self.fc(x)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs
```

---

### 7.4 机器翻译

#### 场景描述

将源语言序列翻译成目标语言序列：

- 输入：源语言句子
- 输出：目标语言句子
- 特点：输入输出长度都可能变化

#### Seq2Seq + Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Seq2Seq编码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, src):
        """
        Args:
            src: [batch, src_len]
        Returns:
            outputs: [batch, src_len, hidden*2]
            hidden: [num_layers*2, batch, hidden]
            cell: [num_layers*2, batch, hidden]
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        # 双向合并: 将前向和后向的隐状态拼接
        # hidden: [2*num_layers, batch, hidden]
        # 需要重组为 [num_layers, batch, hidden*2]
        hidden = self._merge_directions(hidden)
        cell = self._merge_directions(cell)

        return outputs, hidden, cell

    def _merge_directions(self, state):
        """合并双向隐状态"""
        # state: [2*num_layers, batch, hidden]
        # 分离前向和后向
        batch = state.size(1)
        state = state.view(self.num_layers, 2, batch, self.hidden_size)
        # 拼接
        state = torch.cat([state[:, 0], state[:, 1]], dim=2)
        return state


class Attention(nn.Module):
    """Bahdanau注意力机制"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: [batch, hidden*2] 解码器当前隐状态
            encoder_outputs: [batch, src_len, hidden*2] 编码器输出
            mask: [batch, src_len] padding mask
        Returns:
            attention_weights: [batch, src_len]
            context: [batch, hidden*2]
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # 重复解码器隐状态
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 计算注意力能量
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, src_len]

        # 应用mask（padding位置设为-inf）
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention_weights = F.softmax(attention, dim=1)

        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)

        return attention_weights, context


class Decoder(nn.Module):
    """Seq2Seq解码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size * 2,  # embedding + context
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size * 2 + embed_size + hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask):
        """
        Args:
            input: [batch] 当前词索引
            hidden: [num_layers, batch, hidden]
            cell: [num_layers, batch, hidden]
            encoder_outputs: [batch, src_len, hidden*2]
            mask: [batch, src_len]
        Returns:
            prediction: [batch, vocab_size]
            hidden: [num_layers, batch, hidden]
            cell: [num_layers, batch, hidden]
            attention: [batch, src_len]
        """
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))

        # 注意力
        attention_weights, context = self.attention(hidden[-1], encoder_outputs, mask)

        # 拼接嵌入和上下文
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # 预测
        output = output.squeeze(1)  # [batch, hidden]
        embedded = embedded.squeeze(1)  # [batch, embed_size]
        context = context.squeeze(1)  # [batch, hidden*2]

        prediction = self.fc_out(torch.cat([output, context, embedded], dim=1))

        return prediction, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    """完整的Seq2Seq + Attention模型"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: [batch, src_len]
            trg: [batch, trg_len]
            teacher_forcing_ratio: teacher forcing概率
        Returns:
            outputs: [batch, trg_len, vocab_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        # 编码
        encoder_outputs, hidden, cell = self.encoder(src)

        # 创建mask
        mask = (src != 0).to(torch.float32)  # [batch, src_len]

        # 第一个输入是<sos> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # 解码
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # 获取预测的词
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs

    def translate(self, src, max_length=100, sos_idx=1, eos_idx=2):
        """翻译（推理模式）"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)

            # 编码
            encoder_outputs, hidden, cell = self.encoder(src)

            # 初始输入
            input = torch.tensor([sos_idx] * batch_size, device=self.device)

            # 生成
            outputs = []
            attention_weights = []

            mask = (src != 0).to(torch.float32)

            for _ in range(max_length):
                output, hidden, cell, attention = self.decoder(
                    input, hidden, cell, encoder_outputs, mask
                )
                attention_weights.append(attention)

                pred_token = output.argmax(1)
                outputs.append(pred_token)

                if (pred_token == eos_idx).all():
                    break

                input = pred_token

            outputs = torch.stack(outputs, dim=1)
            return outputs, torch.stack(attention_weights, dim=1)


# 训练函数
def train_seq2seq(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(model.device), trg.to(model.device)

        optimizer.zero_grad()

        # 前向传播
        output = model(src, trg)

        # 计算损失（忽略<sos>）
        output_dim = output.size(-1)
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# 使用示例
encoder = Encoder(
    vocab_size=10000, embed_size=256, hidden_size=512,
    num_layers=2, dropout=0.3
)
decoder = Decoder(
    vocab_size=8000, embed_size=256, hidden_size=512,
    num_layers=2, dropout=0.3
)
model = Seq2Seq(encoder, decoder, device='cuda')
model = model.to('cuda')

criterion = nn.CrossEntropyLoss(ignore_index=0)  # padding_idx=0
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

#### Transformer vs Seq2Seq

| 特性 | Seq2Seq+Attention | Transformer |
|:-----|:-----------------:|:-----------:|
| **编码器** | 双向LSTM | 多头自注意力 |
| **解码器** | 单向LSTM+注意力 | 掩码自注意力 |
| **并行度** | 低（RNN串行） | 高（注意力并行） |
| **长期依赖** | 中等 | 强 |
| **训练速度** | 慢 | 快 |
| **推理速度** | 快 | 慢 |
| **数据需求** | 中等 | 大 |

---

### 7.5 情感分析

#### 场景描述

分类文本的情感倾向：

- 正面/负面/中性
- 1-5星评分
- 情感强度分析

#### 基于LSTM的情感分类

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """基于LSTM的情感分析模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes,
                 num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # 多层全连接
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        embedded = self.embedding(x)

        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)

        # 双向LSTM: 拼接最后一层的前向和后向隐状态
        # h_n: [num_layers*2, batch, hidden_size]
        if self.lstm.bidirectional:
            # h_n[-2] 是前向最后一层
            # h_n[-1] 是后向最后一层
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        # 分类
        logits = self.fc(last_hidden)

        return logits


# 注意力情感分析
class AttentionSentiment(nn.Module):
    """带注意力机制的情感分析"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)

        # 注意力
        self.attention = nn.Linear(hidden_size * 2, 1)

        # 分类
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, num_classes]
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # 注意力权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # 分类
        logits = self.fc(context)

        return logits, attention_weights


# 分层注意力网络（Hierarchy Attention Network）
class HierarchicalAttention(nn.Module):
    """分层注意力网络：词级→句级→文档级"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 词级GRU
        self.word_gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.word_attention = nn.Linear(hidden_size * 2, hidden_size * 2)

        # 句级GRU
        self.sentence_gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.sentence_attention = nn.Linear(hidden_size * 2, hidden_size * 2)

        # 分类
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, documents):
        """
        Args:
            documents: [batch, num_sentences, num_words]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size, num_sentences, num_words = documents.shape

        # 词级处理
        word_outputs = []
        for s in range(num_sentences):
            words = documents[:, s, :]  # [batch, num_words]
            embedded = self.embedding(words)

            # GRU
            word_out, _ = self.word_gru(embedded)

            # 注意力
            word_att = torch.tanh(self.word_attention(word_out))
            word_weights = torch.softmax(word_att.sum(-1, keepdim=True), dim=1)
            sentence_vec = torch.sum(word_weights * word_out, dim=1)

            word_outputs.append(sentence_vec)

        # 句级处理
        sentence_matrix = torch.stack(word_outputs, dim=1)  # [batch, num_sentences, hidden*2]
        sentence_out, _ = self.sentence_gru(sentence_matrix)

        # 句级注意力
        sent_att = torch.tanh(self.sentence_attention(sentence_out))
        sent_weights = torch.softmax(sent_att.sum(-1, keepdim=True), dim=1)
        doc_vec = torch.sum(sent_weights * sentence_out, dim=1)

        # 分类
        logits = self.fc(doc_vec)

        return logits


# 训练示例
def train_sentiment(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0

        for texts, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(texts)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_sentiment_model.pth')
```

---

### 7.6 音乐生成

#### 场景描述

生成音乐序列（音符、和弦）：

- 输入：已有的音乐序列
- 输出：续写音乐
- 特点：需要考虑节奏、和声、旋律

#### 基于LSTM的音乐生成

```python
import torch
import torch.nn as nn
import numpy as np

class MusicLSTM(nn.Module):
    """基于LSTM的音乐生成模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: (h_n, c_n)
        """
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, prime_seq, length, temperature=1.0, top_k=None):
        """
        生成音乐
        Args:
            prime_seq: 起始序列
            length: 生成长度
            temperature: 采样温度
            top_k: top-k采样
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor([prime_seq], dtype=torch.long)
            generated = prime_seq.copy()

            # 处理起始序列
            output, hidden = self.forward(x)

            for _ in range(length):
                # 获取最后一个时间步的输出
                logits = output[0, -1, :] / temperature

                # Top-k过滤
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits

                # 采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_token)

                # 下一步输入
                x = torch.tensor([[next_token]], dtype=torch.long)
                output, hidden = self.forward(x, hidden)

            return generated


# 音乐表示
class MusicTokenizer:
    """音乐符号化"""
    def __init__(self):
        # 音符 + 休止符
        self.note_to_idx = {}
        self.chord_to_idx = {}

    def encode_melody(self, midi_events):
        """
        编码旋律
        Args:
            midi_events: MIDI事件列表 [(note, duration, velocity), ...]
        Returns:
            tokens: 索引序列
        """
        tokens = []
        for note, duration, velocity in midi_events:
            # 组合音符和时值
            token = (note, duration)
            if token not in self.note_to_idx:
                self.note_to_idx[token] = len(self.note_to_idx)
            tokens.append(self.note_to_idx[token])
        return tokens

    def encode_with_chords(self, melody_events, chord_events):
        """
        编码旋律+和弦
        Args:
            melody_events: 旋律事件
            chord_events: 和弦事件
        Returns:
            tokens: 包含旋律和和弦的序列
        """
        tokens = []
        # 简化：交替编码旋律和和弦
        for m, c in zip(melody_events, chord_events):
            if m not in self.note_to_idx:
                self.note_to_idx[m] = len(self.note_to_idx) + 1000  # 旋律偏移
            if c not in self.chord_to_idx:
                self.chord_to_idx[c] = len(self.chord_to_idx)
            tokens.append(self.note_to_idx[m])
            tokens.append(self.chord_to_idx[c])
        return tokens


# 双层LSTM（旋律+伴奏）
class MelodyAccompanimentModel(nn.Module):
    """生成旋律和伴奏的模型"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        # 旋律LSTM
        self.melody_embedding = nn.Embedding(vocab_size, embed_size)
        self.melody_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # 伴奏LSTM（条件生成）
        self.accomp_embedding = nn.Embedding(vocab_size, embed_size)
        self.accomp_lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)

        # 输出
        self.melody_fc = nn.Linear(hidden_size, vocab_size)
        self.accomp_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, melody_seq, accomp_seq):
        """
        Args:
            melody_seq: [batch, seq_len] 旋律序列
            accomp_seq: [batch, seq_len] 伴奏序列
        Returns:
            melody_logits: [batch, seq_len, vocab_size]
            accomp_logits: [batch, seq_len, vocab_size]
        """
        # 旋律编码
        melody_emb = self.melody_embedding(melody_seq)
        melody_out, melody_hidden = self.melody_lstm(melody_emb)

        # 伴奏生成（条件）
        accomp_emb = self.accomp_embedding(accomp_seq)
        accomp_input = torch.cat([accomp_emb, melody_out], dim=-1)
        accomp_out, _ = self.accomp_lstm(accomp_input)

        # 输出
        melody_logits = self.melody_fc(melody_out)
        accomp_logits = self.accomp_fc(accomp_out)

        return melody_logits, accomp_logits


# Groove LSTM（节奏建模）
class GrooveLSTM(nn.Module):
    """建模节奏和律动"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)

        # 双层LSTM：一层建模短程节奏，一层建模长程结构
        self.lstm_short = nn.LSTM(64, hidden_size, batch_first=True)
        self.lstm_long = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] 鼓点序列
        """
        x = self.embedding(x)

        # 短程节奏
        short_out, _ = self.lstm_short(x)

        # 长程结构
        long_out, _ = self.lstm_long(short_out)

        logits = self.fc(long_out)
        return logits
```

---

### 7.7 异常检测（时序数据）

#### 场景描述

检测时序数据中的异常：

- 网络流量异常
- 设备故障预测
- 心律失常检测
- 诈骗检测

#### 基于LSTM的异常检测

```python
import torch
import torch.nn as nn
import numpy as np

class LSTM_Autoencoder(nn.Module):
    """LSTM自编码器用于异常检测"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        # 编码器
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True
        )

        # 解码器
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            reconstructed: [batch, seq_len, input_size]
        """
        # 编码
        encoded, (h_n, c_n) = self.encoder(x)

        # 解码（使用编码器的最后隐状态作为初始状态）
        decoded, _ = self.decoder(encoded, (h_n, c_n))

        # 重构
        reconstructed = self.fc(decoded)

        return reconstructed


class LSTM_VAE(nn.Module):
    """LSTM变分自编码器"""
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super().__init__()

        # 编码器LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 变分层
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # 解码器
        self.decoder_latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        """编码为潜在分布"""
        _, (h_n, c_n) = self.encoder(x)
        # 使用最后时间步的隐状态
        h_last = h_n[-1]  # [batch, hidden_size]

        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """解码"""
        batch_size = z.size(0)

        # 初始隐状态
        h = z.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c = torch.zeros_like(h)

        # 解码器输入（从z开始）
        decoder_input = self.decoder_latent_to_hidden(z).unsqueeze(1)

        # 扩展到序列长度
        outputs = []
        for _ in range(seq_len):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            out = self.fc_output(out)
            outputs.append(out)

        reconstructed = torch.cat(outputs, dim=1)
        return reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x.size(1))
        return reconstructed, mu, logvar


class AnomalyDetector:
    """异常检测器"""
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    def compute_reconstruction_error(self, x):
        """计算重构误差"""
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(x)
            # MSE误差
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error.cpu().numpy()

    def fit_threshold(self, normal_data):
        """在正常数据上拟合阈值"""
        errors = []
        self.model.eval()
        with torch.no_grad():
            for batch in normal_data:
                error = self.compute_reconstruction_error(batch)
                errors.extend(error)

        self.threshold = np.percentile(errors, self.threshold_percentile)
        return self.threshold

    def detect(self, x):
        """检测异常"""
        error = self.compute_reconstruction_error(x)
        is_anomaly = error > self.threshold
        return is_anomaly, error


class PredictionBasedAnomaly(nn.Module):
    """基于预测的异常检测"""
    def __init__(self, input_size, hidden_size, num_layers, pred_steps=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_steps * input_size)
        self.pred_steps = pred_steps
        self.input_size = input_size

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            predictions: [batch, pred_steps, input_size]
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]

        predictions = self.fc(last_hidden)
        predictions = predictions.view(-1, self.pred_steps, self.input_size)

        return predictions


# 训练函数
def train_anomaly_detector(model, normal_loader, val_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in normal_loader:
            optimizer.zero_grad()

            # 自编码器重构
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证（使用正常数据）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()

        train_loss /= len(normal_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_anomaly_model.pth')

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')


# 使用示例
# model = LSTM_Autoencoder(input_size=10, hidden_size=64, num_layers=2)
# detector = AnomalyDetector(model, threshold_percentile=95)
#
# # 在正常数据上训练
# train_anomaly_detector(model, normal_train_loader, normal_val_loader)
#
# # 设置阈值
# detector.fit_threshold(normal_test_loader)
#
# # 检测新数据
# is_anomaly, error = detector.detect(new_data)
```

#### 多模态异常检测

```python
class MultimodalAnomalyDetector(nn.Module):
    """多模态时序异常检测（如图像+传感器）"""
    def __init__(self, image_feature_dim, sensor_dim, hidden_size):
        super().__init__()
        # 图像特征编码（CNN）
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, image_feature_dim)
        )

        # 传感器时序编码
        self.sensor_encoder = nn.LSTM(
            sensor_dim, hidden_size, batch_first=True
        )

        # 融合
        self.fusion = nn.Linear(image_feature_dim + hidden_size, hidden_size)

        # 检测头
        self.detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, images, sensors):
        """
        Args:
            images: [batch, seq_len, C, H, W]
            sensors: [batch, seq_len, sensor_dim]
        """
        batch, seq_len = images.shape[:2]

        # 编码每帧图像
        image_features = []
        for t in range(seq_len):
            feat = self.image_encoder(images[:, t])
            image_features.append(feat)
        image_features = torch.stack(image_features, dim=1)

        # 编码传感器序列
        sensor_out, _ = self.sensor_encoder(sensors)
        sensor_features = sensor_out[:, -1, :]

        # 融合
        combined = torch.cat([image_features.mean(1), sensor_features], dim=1)
        fused = self.fusion(combined)

        # 检测
        anomaly_score = self.detector(fused)

        return anomaly_score
```

---

## 八、方案选择建议

### 8.1 RNN vs Transformer 选择

```
决策流程图:

┌─────────────────┐
│  你的任务是什么？  │
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌──────┐  ┌──────┐
│序列标注│  │序列生成│
└───┬──┘  └───┬──┘
    │         │
    ↓         ↓
 ┌─┴───────────┴─┐
 │ 数据量多大？    │
 └─┬───────────┬─┘
   ↓           ↓
┌────┐      ┌────┐
│小  │      │大  │
└─┬──┘      └─┬──┘
  │           │
  ↓           ↓
RNN/LSTM   Transformer
(或GRU)    (BERT/GPT)
```

| 场景 | 推荐模型 | 原因 |
|:-----|:---------|:-----|
| **小数据集（<10K）** | LSTM/GRU | 防止过拟合 |
| **中等数据（10K-1M）** | LSTM/GRU | 性价比高 |
| **大数据（>1M）** | Transformer | 充分利用数据 |
| **实时推理** | GRU | 推理快 |
| **最长序列<100** | LSTM | 无需Transformer |
| **最长序列>500** | Transformer | 长依赖强 |
| **GPU受限** | GRU | 内存友好 |
| **需要可解释性** | LSTM（注意力可视化） | 可解释 |
| **预训练+微调** | Transformer | 有预训练模型 |

### 8.2 LSTM vs GRU 选择

```
LSTM vs GRU 选择指南:

┌─────────────────────┐
│     选择 LSTM        │
├─────────────────────┤
│ • 需要学习长期依赖   │
│ • 序列长度>100       │
│ • 需要显式控制记忆   │
│ • 数据充足           │
│ • 计算资源充足       │
└─────────────────────┘

┌─────────────────────┐
│     选择 GRU         │
├─────────────────────┤
│ • 中等序列长度       │
│ • 需要快速训练       │
│ • 内存受限           │
│ • 追求简洁           │
│ • 大多数实际任务     │
└─────────────────────┘
```

### 8.3 实施路径

```
阶段1: 快速原型（1-2周）
├─ 使用单层GRU
├─ 小隐状态维度（128）
├─ 验证任务可行性
└─ 建立baseline

阶段2: 优化性能（2-4周）
├─ 尝试LSTM
├─ 增加层数（2-3层）
├─ 调整隐状态维度
├─ 添加注意力机制
└─ 数据增强

阶段3: 部署优化（可选）
├─ 模型量化
├─ 剪枝
├─ 知识蒸馏
└─ TensorRT优化
```

### 8.4 超参数推荐

| 超参数 | 推荐范围 | 说明 |
|:-------|:---------|:-----|
| **隐状态维度** | 128-512 | 根据任务复杂度调整 |
| **层数** | 1-3 | 过深易过拟合 |
| **Dropout** | 0.2-0.5 | 防止过拟合 |
| **学习率** | 0.0001-0.001 | Adam优化器 |
| **Batch Size** | 16-128 | 根据显存调整 |
| **梯度裁剪** | 1.0-5.0 | 防止梯度爆炸 |

---

## 九、常见问题

### Q1: 梯度消失解决方案

```python
# 1. 使用LSTM/GRU
model = nn.LSTM(input_size, hidden_size, num_layers)

# 2. 梯度裁剪（解决梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 合理初始化
for name, param in model.named_parameters():
    if 'weight_ih' in name:
        nn.init.xavier_uniform_(param)
    elif 'weight_hh' in name:
        nn.init.orthogonal_(param)

# 4. 使用LayerNorm
class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        output = self.layer_norm(output)
        return output, (h, c)

# 5. 残差连接
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out + self.proj(x)
```

### Q2: 长序列处理

```python
# 1. 梯度截断（Truncated BPTT）
# 只回传固定步数
def truncated_bptt(model, x, y, chunk_size=50):
    """分段反向传播"""
    chunks = x.split(chunk_size, dim=1)
    y_chunks = y.split(chunk_size, dim=1)

    hidden = None
    total_loss = 0

    for chunk_x, chunk_y in zip(chunks, y_chunks):
        output, hidden = model(chunk_x, hidden)
        hidden = (hidden[0].detach(), hidden[1].detach())  # 阻止梯度回传

        loss = criterion(output, chunk_y)
        loss.backward()
        total_loss += loss

    return total_loss

# 2. 层次化处理（分层RNN）
class HierarchicalRNN(nn.Module):
    """分层处理长序列"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 底层RNN处理局部
        self.lower_rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        # 高层RNN处理全局
        self.upper_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, segment_size=100):
        batch, seq_len, _ = x.shape

        # 分段处理
        segments = x.split(segment_size, dim=1)
        segment_features = []

        for seg in segments:
            out, _ = self.lower_rnn(seg)
            segment_features.append(out[:, -1, :])  # 每段的最后状态

        # 高层处理
        segments_tensor = torch.stack(segment_features, dim=1)
        final_out, _ = self.upper_rnn(segments_tensor)

        return final_out

# 3. 注意力机制（快速访问长距离信息）
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)

        # 注意力加权，直接访问任意位置
        weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(weights * output, dim=1)

        return context
```

### Q3: 训练不稳定

```python
# 1. 学习率调度
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 2. Warmup
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 3. 使用更稳定的优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 4. 梯度监测
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10:
                print(f"Warning: Large gradient in {name}: {grad_norm}")
            elif grad_norm < 1e-7:
                print(f"Warning: Small gradient in {name}: {grad_norm}")
```

---

## 十、学习资源

### 经典论文

| 论文 | 年份 | 贡献 |
|:-----|:-----|:-----|
| [Finding Structure in Time](https://doi.org/10.1016/0893-6080(90)90021-E) | 1990 | 早期RNN (Elman Network) |
| [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) | 1997 | LSTM (Hochreiter & Schmidhuber) |
| [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) | 2014 | Seq2Seq |
| [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | 2014 | Seq2Seq + Attention |
| [Empirical Evaluation of Gated Recurrent Neural Networks](https://arxiv.org/abs/1412.3555) | 2014 | GRU |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 2014 | Attention机制 |

### 教程推荐

- **Andrej Karpathy's blog**: [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- **Christopher Olah**: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Distill.pub**: [Visualizing MNIST](https://distill.pub/2016/momentum/)
- **PyTorch Docs**: [RNN and LSTMs](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)

### 书籍

- 《Deep Learning》(Goodfellow et al.) - 第10章
- 《Speech and Language Processing》(Jurafsky & Martin) - 第9章
- 《Hands-On Machine Learning》(Geron) - 第15章
- 《Natural Language Processing with PyTorch》(Delip Rao & Brian McMahan)

---

## 十一、相关笔记

### 深入学习

- [[AI研究/AI学习/02-模型原理/GNN全面解析]] - 图神经网络原理
- [[AI研究/AI学习/02-模型原理/Transformer研读]] - Transformer 架构详细研读
- [[AI研究/AI学习/神经网络类型全景总结]] - 所有神经网络架构概览

### 实战应用

- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - 检索增强生成项目
- [[AI研究/AI学习/车道路径筛选系统设计]] - GNN实际应用

### 基础知识

- [[AI研究/AI学习/常见术语对照]] - AI/ML术语中英文对照
- [[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线

---

#RNN #LSTM #GRU #序列建模 #深度学习 #架构原理
