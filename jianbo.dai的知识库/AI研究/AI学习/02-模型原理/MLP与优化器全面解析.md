---
title: MLP与优化器全面解析
date: 2026-02-28
tags:
  - MLP
  - 优化器
  - 训练技巧
  - 深度学习
  - 基础原理
status: active
---

# MLP与优化器全面解析

> [!info] 说明
> 本笔记系统介绍多层感知机（MLP）的核心原理、各类优化算法的详细机制、训练技巧与正则化方法，以及丰富的实战案例

---

## 📑 目录

> [!tip] 使用说明
> 点击下方的任何章节链接，即可跳转到对应内容（支持 `Ctrl/Cmd + Click` 在新面板打开）

### 基础理论
- [[#一、MLP（多层感知机）核心原理]]
  - [[#1.1 什么是MLP]]
  - [[#1.2 全连接层（Dense Layer）]]
  - [[#1.3 激活函数]]
  - [[#1.4 前向传播]]
  - [[#1.5 反向传播]]
  - [[#1.6 数学表达]]

### 优化算法
- [[#二、优化算法详解]]
  - [[#2.1 经典优化器]]
    - [[#2.1.1 SGD（Stochastic Gradient Descent）]]
    - [[#2.1.2 Momentum（动量）]]
    - [[#2.1.3 Nesterov Accelerated Gradient（NAG）]]
  - [[#2.2 自适应优化器]]
    - [[#2.2.1 Adagrad]]
    - [[#2.2.2 RMSprop]]
    - [[#2.2.3 Adam（Adaptive Moment Estimation）]]
    - [[#2.2.4 AdamW]]
    - [[#2.2.5 AdaBelief]]
    - [[#2.2.6 Lion（Google新优化器）]]
  - [[#2.3 二阶优化]]
    - [[#2.3.1 Newton法]]
    - [[#2.3.2 L-BFGS（Limited-memory BFGS）]]

### 对比总结
- [[#三、优化器对比总结]]
  - [[#3.1 综合对比表]]
  - [[#3.2 收敛特性对比]]
  - [[#3.3 适用场景]]
  - [[#3.4 超参数敏感性]]

### 训练技巧
- [[#四、训练技巧]]
  - [[#4.1 学习率调度]]
    - [[#4.1.1 Step LR]]
    - [[#4.1.2 MultiStep LR]]
    - [[#4.1.3 Exponential LR]]
    - [[#4.1.4 Cosine Annealing]]
    - [[#4.1.5 Warmup]]
    - [[#4.1.6 OneCycleLR]]
  - [[#4.2 Batch Normalization]]
  - [[#4.3 Layer Normalization]]
  - [[#4.4 Dropout]]
  - [[#4.5 数据预处理]]
  - [[#4.6 权重初始化]]

### 正则化技术
- [[#五、正则化技术]]
  - [[#5.1 L1/L2正则化]]
  - [[#5.2 Early Stopping]]
  - [[#5.3 Data Augmentation]]
  - [[#5.4 Label Smoothing]]
  - [[#5.5 MixUp]]
  - [[#5.6 CutMix]]

### 问题解决
- [[#六、常见问题与解决]]
  - [[#6.1 梯度消失]]
  - [[#6.2 梯度爆炸]]
  - [[#6.3 过拟合]]
  - [[#6.4 欠拟合]]
  - [[#6.5 训练不稳定]]
  - [[#6.6 死神经元（Dead ReLU）]]

### 实战案例
- [[#七、实战案例]]
  - [[#7.1 案例1：简单分类任务（MLP实现MNIST分类）]]
  - [[#7.2 案例2：回归任务（房价预测）]]
  - [[#7.3 案例3：优化器对比实验]]
  - [[#7.4 案例4：学习率调度实战]]
  - [[#7.5 案例5：Learning Rate Finder]]

### 参考指南
- [[#八、方案选择建议]]
  - [[#8.1 优化器选择指南]]
  - [[#8.2 学习率设置建议]]
  - [[#8.3 Batch Size选择]]
  - [[#8.4 综合推荐配置]]
- [[#九、学习资源]]
  - [[#9.1 优化器论文]]
  - [[#9.2 PyTorch优化器文档]]
  - [[#9.3 调参技巧]]
- [[#十、相关笔记]]

---

## 一、MLP（多层感知机）核心原理

### 1.1 什么是MLP

**MLP（Multi-Layer Perceptron）**，又称多层感知机或全连接神经网络，是最基础的深度学习架构。

```
神经网络层级结构：

输入层 (Input Layer)
    ↓
隐藏层1 (Hidden Layer 1)
    ↓
隐藏层2 (Hidden Layer 2)
    ↓
    ...
    ↓
隐藏层N (Hidden Layer N)
    ↓
输出层 (Output Layer)
```

```
深度学习家族：

┌─────────────────────────────────────────────────────────────┐
│                        深度学习                              │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   MLP        │    CNN       │    RNN       │  Transformer   │
│ (全连接网络)  │  (卷积网络)   │  (循环网络)   │  (注意力机制)   │
│              │              │              │                │
│ · 向量数据   │ · 图像数据   │ · 序列数据   │ · 序列/图像    │
│ · 表格数据   │ · 网格结构   │ · 时序依赖   │ · 长距离依赖   │
│ · 特征变换   │ · 空间局部性 │ · 状态传递   │ · 并行计算     │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### 1.2 全连接层（Dense Layer）

全连接层是MLP的核心组件，每个神经元与上一层的所有神经元相连。

```python
# 数学表达
y = Wx + b

其中：
├─ x: 输入向量 [input_dim]
├─ W: 权重矩阵 [output_dim, input_dim]
├─ b: 偏置向量 [output_dim]
└─ y: 输出向量 [output_dim]

# 维度变换示例
输入: x ∈ R^(batch_size, 784)  # MNIST图像展平
权重: W ∈ R^(256, 784)         # 256个隐藏单元
偏置: b ∈ R^(256)
输出: y ∈ R^(batch_size, 256)
```

**PyTorch实现**：
```python
import torch.nn as nn

# 全连接层
fc = nn.Linear(in_features=784, out_features=256)

# 等价于
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias
```

### 1.3 激活函数

激活函数引入非线性，使网络能够学习复杂模式。

#### 1.3.1 Sigmoid

```python
sigmoid(x) = 1 / (1 + e^(-x))

# 导数
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

# 特点
├─ 输出范围：(0, 1)
├─ 优点：平滑、可解释（概率）
└─ 缺点：梯度消失严重、输出非零中心
```

```python
import torch.nn as nn
activation = nn.Sigmoid()
```

#### 1.3.2 Tanh

```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

# 导数
tanh'(x) = 1 - tanh²(x)

# 特点
├─ 输出范围：(-1, 1)
├─ 优点：零中心（输出均值为0）
└─ 缺点：仍有梯度消失问题
```

```python
activation = nn.Tanh()
```

#### 1.3.3 ReLU（Rectified Linear Unit）

```python
relu(x) = max(0, x)

# 导数
relu'(x) = 1 if x > 0 else 0

# 特点
├─ 输出范围：[0, +∞)
├─ 优点：计算简单、缓解梯度消失、稀疏激活
└─ 缺点：Dead ReLU问题（神经元死亡）
```

```python
activation = nn.ReLU()
```

**ReLU变体**：

| 变体 | 公式 | 特点 |
|:-----|:-----|:-----|
| **LeakyReLU** | max(αx, x), α∈(0,1) | 解决Dead ReLU |
| **PReLU** | max(αx, x), α可学习 | 自适应泄露参数 |
| **RReLU** | max(αx, x), α随机 | 训练时随机，测试时固定 |
| **ELU** | x if x>0 else α(e^x-1) | 负区有平滑输出 |
| **SELU** | λ·ELU(x) | 自归一化特性 |

#### 1.3.4 GELU（Gaussian Error Linear Unit）

```python
gelu(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))

其中 Φ(x) 是标准正态分布的累积分布函数

# 特点
├─ Transformer标配（BERT、GPT）
├─ 平滑、性能优于ReLU
└─ 计算稍复杂
```

```python
activation = nn.GELU()
```

#### 1.3.5 Swish

```python
swish(x) = x * sigmoid(βx)  # β=1时称SiLU

# 特点
├─ 平滑非单调
├─ 在深层网络表现优于ReLU
└─ 自门控机制
```

```python
activation = nn.SiLU()  # PyTorch中的Swish
```

**激活函数对比**：

| 激活函数 | 范围 | 梯度消失 | 计算复杂度 | 推荐场景 |
|:---------|:-----|:---------|:----------|:---------|
| **Sigmoid** | (0,1) | 严重 | 低 | 二分类输出层 |
| **Tanh** | (-1,1) | 中等 | 低 | RNN门控 |
| **ReLU** | [0,∞) | 无 | 极低 | 隐藏层首选 |
| **GELU** | (-∞,∞) | 无 | 中 | Transformer |
| **Swish** | (-∞,∞) | 无 | 中 | 深层网络 |

### 1.4 前向传播

```python
# MLP前向传播示例
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.network(x)

# 使用
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
output = model(x)  # [batch_size, 10]
```

### 1.5 反向传播

反向传播是训练神经网络的核心算法，通过链式法则计算梯度。

```python
# 链式法则示例
"""
损失函数 L
    ↓
输出层 h₃
    ↓
隐藏层 h₂
    ↓
隐藏层 h₁
    ↓
输入 x

梯度计算：
∂L/∂W₃ = ∂L/∂h₃ · ∂h₃/∂W₃
∂L/∂W₂ = ∂L/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂W₂
∂L/∂W₁ = ∂L/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁
"""

# PyTorch自动求导
import torch

x = torch.randn(2, 3, requires_grad=True)
W = torch.randn(3, 5, requires_grad=True)

# 前向传播
y = torch.matmul(x, W)
loss = y.sum()

# 反向传播
loss.backward()

print(W.grad)  # W的梯度
```

### 1.6 数学表达

完整MLP的数学表达：

```python
# 单个样本的前向传播
h⁰ = x  # 输入

for l in 1 to L:
    hˡ = σₗ(Wˡ h^(l-1) + bˡ)  # l层前向传播

# 输出层（无激活）
ŷ = W^(L+1) h^L + b^(L+1)

其中：
├─ L: 隐藏层数量
├─ Wˡ ∈ R^(dˡ × d^(l-1)): l层权重矩阵
├─ bˡ ∈ R^(dˡ): l层偏置向量
├─ σₗ(): l层激活函数
└─ hˡ ∈ R^(dˡ): l层激活值（特征表示）
```

---

## 二、优化算法详解

### 2.1 经典优化器

#### 2.1.1 SGD（Stochastic Gradient Descent）

随机梯度下降是最基础的优化算法。

```python
# SGD更新公式
θ_{t+1} = θ_t - η * ∇_θ L(θ_t)

其中：
├─ θ_t: t时刻的参数
├─ η: 学习率（Learning Rate）
└─ ∇_θ L(θ_t): 损失函数关于θ的梯度
```

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 参数更新
        optimizer.step()
```

**SGD变体**：

```python
# 1. 批量梯度下降（Batch GD）：使用全部数据
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. 随机梯度下降：每次使用单个样本
# 手动实现
for x, y in dataset:
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 3. 小批量SGD：折中方案，最常用
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

**特点**：
- 优点：简单、稳定、泛化能力强
- 缺点：收敛慢、容易陷入局部最优、对超参数敏感

#### 2.1.2 Momentum（动量）

```python
# Momentum更新公式
v_t = β * v_{t-1} + (1 - β) * ∇_θ L(θ_t)
θ_{t+1} = θ_t - η * v_t

其中：
├─ v_t: 速度向量（累积梯度）
├─ β: 动量系数（通常取0.9）
├─ η: 学习率
└─ ∇_θ L(θ_t): 当前梯度

# 另一种常见形式（PyTorch默认）
v_t = β * v_{t-1} + ∇_θ L(θ_t)
θ_{t+1} = θ_t - η * v_t
```

**物理直觉**：
- 想象一个球滚下山坡
- v_t 是速度，累积了之前的运动方向
- 即使当前梯度为0，球也会因惯性继续运动

```python
# PyTorch实现
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # 动量系数
)
```

**特点**：
- 优点：加速收敛、减少震荡、帮助逃离局部最优
- 缺点：可能在最小值附近震荡

#### 2.1.3 Nesterov Accelerated Gradient（NAG）

```python
# NAG更新公式
v_t = β * v_{t-1} + ∇_θ L(θ_t - β * η * v_{t-1})
θ_{t+1} = θ_t - η * v_t

区别：NAG在计算梯度时，使用"预测位置"的梯度
 Momentum使用当前位置的梯度
```

**直观理解**：
- Momentum：看到斜坡就加速
- NAG：向前看一眼，如果前面要上坡，提前减速

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # 启用Nesterov
)
```

**特点**：
- 优点：比Momentum更稳定、收敛更快
- 缺点：计算量稍大

---

### 2.2 自适应优化器

#### 2.2.1 Adagrad

```python
# Adagrad更新公式
G_t = G_{t-1} + (∇_θ L(θ_t))²  # 累积梯度平方
θ_{t+1} = θ_t - (η / √(G_t + ε)) * ∇_θ L(θ_t)

其中：
├─ G_t: 累积的梯度平方和（对角矩阵）
├─ ε: 平滑项（防止除零，通常1e-8）
└─ /√G_t: 自适应学习率（梯度大→学习率小）
```

```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

**特点**：
- 优点：自动调整学习率、适合稀疏数据
- 缺点：学习率单调递减、后期训练停滞

#### 2.2.2 RMSprop

```python
# RMSprop更新公式
E[g²]_t = β * E[g²]_{t-1} + (1 - β) * (∇_θ L(θ_t))²
θ_{t+1} = θ_t - (η / √(E[g²]_t + ε)) * ∇_θ L(θ_t)

其中：
├─ E[g²]_t: 梯度平方的指数移动平均
└─ β: 衰减率（通常0.9）
```

**与Adagrad区别**：
- Adagrad：累积全部历史梯度
- RMSprop：使用指数移动平均（给近期梯度更高权重）

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,  # 衰减率
    eps=1e-8
)
```

**特点**：
- 优点：解决Adagrad学习率衰减问题、适合非平稳目标
- 缺点：超参数敏感

#### 2.2.3 Adam（Adaptive Moment Estimation）

Adam是最流行的优化器，结合了Momentum和RMSprop的优点。

```python
# Adam更新公式
# 1. 一阶矩估计（动量）
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇_θ L(θ_t)

# 2. 二阶矩估计（类似RMSprop）
v_t = β₂ * v_{t-1} + (1 - β₂) * (∇_θ L(θ_t))²

# 3. 偏差修正
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# 4. 参数更新
θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)

默认超参数：
├─ β₁ = 0.9  （一阶矩衰减率）
├─ β₂ = 0.999（二阶矩衰减率）
├─ η = 0.001 （学习率）
└─ ε = 1e-8  （平滑项）
```

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,  # L2正则化（推荐用AdamW代替）
    amsgrad=False   # 是否使用AMSGrad变体
)
```

**Adam变体**：

| 变体 | 说明 | 代码 |
|:-----|:-----|:-----|
| **AdamW** | 解耦权重衰减 | `optim.AdamW()` |
| **AMSGrad** | 保证二阶矩非递减 | `Adam(amsgrad=True)` |
| **AdamP** | 投影梯度 | 第三方实现 |
| **RAdam** | Rectified Adam | 第三方实现 |

#### 2.2.4 AdamW

AdamW是对Adam的改进，将权重衰减与梯度更新解耦。

```python
# Adam vs AdamW

# Adam（权重衰减被加入梯度）
θ_{t+1} = θ_t - η * (m̂_t / (√v̂_t + ε) + λθ_t)

# AdamW（权重衰减独立应用）
θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε) - ηλθ_t

区别虽然细微，但AdamW在实践中表现更好
```

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # 权重衰减系数
)
```

**特点**：
- 优点：解耦权重衰减、训练更稳定、泛化能力更好
- 推荐：现代训练首选

#### 2.2.5 AdaBelief

```python
# AdaBelief更新公式
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇_θ L(θ_t)
s_t = β₂ * s_{t-1} + (1 - β₂) * (∇_θ L(θ_t) - m_t)²  # 不同点
θ_{t+1} = θ_t - η * m_t / (√s_t + ε)

区别：AdaBelief使用"信念方差"而非梯度平方
直观：根据预测的"可信度"调整步长
```

```python
# 需要安装：pip install adabelief-pytorch
from adabelief_pytorch import AdaBelief

optimizer = AdaBelief(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-12,  # 比Adam更小的eps
    weight_decay=0,
    weight_decouple=True,  # 类似AdamW
    rectify=False  # 是否使用RAdaBelief
)
```

**特点**：
- 优点：训练更稳定、泛化更好
- 适用：Transformer、图像分类

#### 2.2.6 Lion（Google新优化器）

```python
# Lion更新公式
c_t = β₁ * m_{t-1} + (1 - β₁) * ∇_θ L(θ_t)  # 动量
θ_{t+1} = θ_t - η * sign(c_t)  # 只用梯度方向，步长固定

特点：
├─ 只用梯度的符号（sign），不用梯度大小
├─ 内存占用少（不需要二阶矩）
└─ 更新幅度是固定的（由学习率决定）
```

```python
# 需要安装：pip install lion-pytorch
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-4,  # Lion推荐更小的学习率
    betas=(0.9, 0.99),  # β₂更大
    weight_decay=0.01
)
```

**特点**：
- 优点：内存占用少、泛化能力强、对超参数鲁棒
- 缺点：收敛可能较慢
- 适用：大模型训练、内存受限场景

---

### 2.3 二阶优化

#### 2.3.1 Newton法

```python
# Newton更新公式
θ_{t+1} = θ_t - [H^(-1)] * ∇_θ L(θ_t)

其中：
├─ H: Hessian矩阵（二阶导数矩阵）
└─ H^(-1): Hessian的逆矩阵

特点：
├─ 利用曲率信息，收敛极快（二次收敛）
└─ 计算Hessian及其逆非常昂贵
```

#### 2.3.2 L-BFGS（Limited-memory BFGS）

```python
# L-BFGS是Newton法的近似
# 不直接计算Hessian，而是用最近m步的梯度近似

特点：
├─ 二阶优化器，收敛快
├─ 适合小批量、全数据训练
└─ 不适合大规模随机训练
```

```python
optimizer = optim.LBFGS(
    model.parameters(),
    lr=1,
    max_iter=20,  # 每次优化的最大迭代次数
    history_size=100,  # 保留的历史步数
    line_search_fn='strong_wolfe'  # 线搜索策略
)

# L-BFGS需要特殊的训练循环
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

**特点**：
- 优点：收敛快、适合凸优化
- 缺点：内存占用大、不适合SGD训练
- 适用：小规模问题、微调

---

## 三、优化器对比总结

### 3.1 综合对比表

| 优化器 | 收敛速度 | 泛化能力 | 内存占用 | 超参数敏感 | 推荐场景 |
|:-------|:---------|:---------|:---------|:----------|:---------|
| **SGD** | 慢 | 强 | 低 | 高 | 大规模分类、需要最佳泛化 |
| **SGD+Momentum** | 中 | 强 | 低 | 中 | 通用场景 |
| **NAG** | 中快 | 强 | 低 | 中 | 需要稳定收敛 |
| **Adagrad** | 快（初期） | 中 | 低 | 低 | 稀疏数据、NLP |
| **RMSprop** | 快 | 中 | 低 | 中 | RNN、非平稳目标 |
| **Adam** | 快 | 中 | 中 | 低 | 快速实验、Transformer |
| **AdamW** | 快 | 强 | 中 | 低 | 现代训练首选 |
| **AdaBelief** | 快 | 强 | 中 | 低 | 不稳定任务 |
| **Lion** | 中 | 强 | 低 | 低 | 大模型、内存受限 |
| **L-BFGS** | 极快 | 强 | 高 | 中 | 小批量、微调 |

### 3.2 收敛特性对比

```
Loss曲线对比：

Adam:     ╲＿＿___（快速下降，可能过早收敛）
SGD:      ╲╲╲╲╲___（慢速下降，最终可能更低）
AdamW:    ╲＿＿＿＿（快速下降，持续优化）
Lion:     ╲╲╲_____（介于Adam和SGD之间）
```

### 3.3 适用场景

```
优化器选择决策树：

问题规模
├─ 小规模（<1万参数）
│  └─ L-BFGS 或 Adam
│
├─ 中等规模
│  ├─ 需要快速实验 → Adam/AdamW
│  └─ 追求最佳效果 → SGD + Momentum
│
└─ 大规模（>1M参数）
   ├─ Transformer → AdamW
   ├─ CNN → SGD + Momentum / AdamW
   └─ 内存受限 → Lion

数据特性
├─ 稀疏数据（推荐、NLP）
│  └─ Adagrad / Adam
│
├─ 稠密数据（图像）
│  └─ SGD / AdamW
│
└─ 非平稳目标（RL）
   └─ RMSprop / Adam

训练阶段
├─ 初期训练 → Adam（快速收敛）
├─ 精调 → SGD（更好泛化）
└─ 大模型预训练 → AdamW / Lion
```

### 3.4 超参数敏感性

| 优化器 | 学习率敏感 | 动量参数敏感 | 其他参数 |
|:-------|:----------|:------------|:---------|
| **SGD** | 高 | 中 | - |
| **Adam** | 低 | 低 | β₁, β₂（通常不用调） |
| **AdamW** | 低 | 低 | weight_decay |
| **Lion** | 中 | 低 | weight_decay |

**推荐学习率范围**：

```python
SGD:      lr = 0.1 ~ 0.01
SGD+Momentum: lr = 0.05 ~ 0.01
Adam:     lr = 0.001 ~ 0.0001
AdamW:    lr = 0.001 ~ 0.0001
Lion:     lr = 0.0001 ~ 0.00001（更小）
```

---

## 四、训练技巧

### 4.1 学习率调度

#### 4.1.1 Step LR

```python
# 固定间隔衰减
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,   # 每30个epoch衰减
    gamma=0.1       # 衰减系数
)

# 使用
for epoch in range(100):
    train_one_epoch(model, dataloader, optimizer)
    scheduler.step()

# 学习率变化：0.01 → 0.001 → 0.0001 → ...
```

#### 4.1.2 MultiStep LR

```python
# 在指定epoch衰减
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 80],  # 在这些epoch衰减
    gamma=0.1
)

# 学习率变化：0.01 →(epoch30)→ 0.001 →(epoch60)→ 0.0001
```

#### 4.1.3 Exponential LR

```python
# 指数衰减
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95  # 每个epoch乘以0.95
)

# 学习率：lr₀ → lr₀*0.95 → lr₀*0.95² → ...
```

#### 4.1.4 Cosine Annealing

```python
# 余弦退火（周期性）
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,      # 周期长度
    eta_min=0       # 最小学习率
)

# 学习率曲线：平滑下降后上升
```

**Cosine Annealing with Warm Restart**：

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,         # 首次周期长度
    T_mult=2,       # 周期倍增系数
    eta_min=1e-6
)

# 学习率周期性重置
```

#### 4.1.5 Warmup

```python
# 预热：初期用小学习率，逐渐增大
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

scheduler = LambdaLR(
    optimizer,
    lr_lambda=warmup_lambda
)

# 学习率：0 → lr（线性增长）
```

**带Warmup的Cosine Annealing**：

```python
# 组合调度器
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, ConstantLR

warmup = ConstantLR(optimizer, factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[5]
)
```

#### 4.1.6 OneCycleLR

```python
# 单周期学习率策略
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,        # 最大学习率
    total_steps=1000,   # 总步数
    pct_start=0.3,      # 上升阶段比例
    anneal_strategy='cos',  # 余弦退火
    final_div_factor=1e4  # 最终学习率 = max_lr/1e4
)

# 使用方式（按步更新）
for batch in dataloader:
    optimizer.step()
    scheduler.step()

# 学习率曲线：上升 → 下降 → 降至极低
```

**学习率策略对比**：

| 策略 | 特点 | 适用场景 |
|:-----|:-----|:---------|
| **Step** | 简单、不连续 | 传统训练 |
| **Cosine** | 平滑、周期性 | 现代训练 |
| **Warmup** | 稳定初期 | 大模型、微调 |
| **OneCycle** | 快速训练 | 快速实验 |

### 4.2 Batch Normalization

```python
# BatchNorm：对每个batch的每个特征归一化
"""
公式：
μ_B = (1/m) * Σ x_i  # batch均值
σ²_B = (1/m) * Σ (x_i - μ_B)²  # batch方差
x̂_i = (x_i - μ_B) / √(σ²_B + ε)  # 归一化
y_i = γ * x̂_i + β  # 缩放和偏移（可学习）
"""

import torch.nn as nn

class MLPWithBN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 1D用于全连接
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 评估模式很重要！
model.train()  # 使用batch统计量
model.eval()   # 使用running统计量
```

**BatchNorm优点**：
- 加速收敛
- 允许更大学习率
- 减少对初始化的依赖
- 轻微正则化效果

**BatchNorm缺点**：
- 小batch性能差
- 推理时额外计算
- 与dropout冲突

### 4.3 Layer Normalization

```python
# LayerNorm：对每个样本的所有特征归一化
"""
公式：
μ_i = (1/H) * Σ x_ij  # 样本均值
σ²_i = (1/H) * Σ (x_ij - μ_i)²  # 样本方差
x̂_ij = (x_ij - μ_i) / √(σ²_i + ε)  # 归一化
y_ij = γ * x̂_ij + β  # 缩放和偏移
"""

class MLPWithLN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 每个样本独立归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
```

**BN vs LN**：

| 特性 | BatchNorm | LayerNorm |
|:-----|:----------|:----------|
| 归一化维度 | batch方向 | 特征方向 |
| batch大小敏感 | 是 | 否 |
| RNN | 不适合 | 适合 |
| Transformer | 少用 | 标配 |
| CNN | 标配 | 少用 |

### 4.4 Dropout

```python
# Dropout：训练时随机丢弃部分神经元
"""
公式：
训练时：y = f(x) * mask / (1-p)  # mask是随机丢弃掩码
推理时：y = f(x)  # 不丢弃

其中 p 是丢弃概率
"""

import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)  # Dropout层
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

# 常见dropout率
├─ 0.2-0.3: 中等规模网络
├─ 0.5: 大规模网络（原论文推荐）
└─ 0.1-0.2: 谨慎正则化
```

**Dropout技巧**：
```python
# 1. 只在隐藏层使用，不用在输出层
# 2. 评估时务必model.eval()
# 3. 大模型用小dropout率

# 特殊变体
nn.Dropout(p=0.3)       # 随机丢弃
nn.Dropout2d(p=0.3)     # 丢弃整个通道（CNN）
nn.AlphaDropout(p=0.3)  # 自归一化dropout（配合SELU）
```

### 4.5 数据预处理

#### 4.5.1 归一化（Normalization）

```python
# Min-Max归一化：将数据缩放到[0,1]或[-1,1]
x_norm = (x - x_min) / (x_max - x_min)

# PyTorch实现
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])  # 缩放到[-1,1]
    # x_norm = (x - 0.5) / 0.5 = 2x - 1
])
```

#### 4.5.2 标准化（Standardization）

```python
# Z-score标准化：均值0，方差1
x_std = (x - μ) / σ

# PyTorch实现
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    )
])

# 计算数据集的均值和标准差
def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std
```

**预处理对比**：

| 方法 | 公式 | 输出范围 | 适用场景 |
|:-----|:-----|:---------|:---------|
| **Min-Max** | (x-min)/(max-min) | [0,1] | 图像像素 |
| **Standardization** | (x-μ)/σ | 无固定 | 一般特征 |
| **Robust Scaling** | (x-median)/IQR | 无固定 | 有异常值 |

### 4.6 权重初始化

#### 4.6.1 Xavier初始化（Glorot）

```python
# 适用于Sigmoid、Tanh激活
"""
线性层：fan_in = 输入维度, fan_out = 输出维度

Xavier Uniform:
W ~ U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))

Xavier Normal:
W ~ N(0, √(2/(fan_in+fan_out)))
"""

# PyTorch默认（Linear层）
nn.init.xavier_uniform_(layer.weight)
# 或
nn.init.xavier_normal_(layer.weight)
```

#### 4.6.2 He初始化（Kaiming）

```python
# 适用于ReLU及其变体
"""
He Uniform:
W ~ U(-√(6/fan_in), √(6/fan_in))

He Normal:
W ~ N(0, √(2/fan_in))
"""

# ReLU网络推荐
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
# 或
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# LeakyReLU
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
```

**初始化对比**：

| 初始化 | 适用激活 | 方差 | |
|:-------|:---------|:-----|:---|
| **Xavier** | Sigmoid, Tanh | 2/(fan_in+fan_out) | 保持方差 |
| **He** | ReLU | 2/fan_in | 考虑ReLU截断 |

#### 4.6.3 完整初始化示例

```python
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.ReLU) or isinstance(m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# 使用
model = MLP(784, [256, 128], 10)
initialize_weights(model)
```

---

## 五、正则化技术

### 5.1 L1/L2正则化

```python
# L1正则化：促使权重稀疏
L1_loss = Σ |w|

# L2正则化（权重衰减）：限制权重大小
L2_loss = Σ w²

# 总损失
total_loss = data_loss + λ * regularization_loss
```

**PyTorch实现**：

```python
# 方法1：通过优化器的weight_decay参数（L2）
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
# weight_decay = 0.01 即 λ=0.01的L2正则

# 方法2：手动添加L1正则
def l1_regularization(model, lambda_l1=0.01):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

# 训练循环
loss = criterion(output, target) + l1_regularization(model, 0.001)
```

**L1 vs L2**：

| 特性 | L1 | L2 |
|:-----|:---|:---|
| 稀疏性 | 强（产生稀疏权重） | 弱 |
| 特征选择 | 是 | 否 |
| 梯度 | 常数 | 与权重成正比 |
| 稳定性 | 可能不稳定 | 更稳定 |

### 5.2 Early Stopping

```python
# 早停：验证性能不再提升时停止训练
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Args:
            patience: 容忍多少个epoch不提升
            min_delta: 视为提升的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

# 使用
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(1000):
    train_one_epoch(model, train_loader)
    val_score = validate(model, val_loader)
    early_stopping(val_score)

    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 5.3 Data Augmentation

```python
# 图像数据增强
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),    # 随机水平翻转
    T.RandomVerticalFlip(p=0.5),      # 随机垂直翻转
    T.RandomRotation(15),             # 随机旋转±15度
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    T.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MLP（表格数据）增强技巧
"""
1. MixUp：混合样本和标签
2. CutMix：剪切和粘贴
3. SMOTE：合成少数类样本（针对不平衡数据）
4. 添加噪声：向特征添加高斯噪声
"""
```

### 5.4 Label Smoothing

```python
# 标签平滑：软化硬标签
"""
原始标签：[0, 0, 1, 0]  # one-hot
平滑后：  [0.025, 0.025, 0.925, 0.025]  # ε=0.1

公式：
y_smooth = (1-ε) * y_one_hot + ε / K

其中：
├─ ε: 平滑系数（通常0.1）
└─ K: 类别数
"""

# PyTorch实现
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, num_classes] 模型预测（logits）
            target: [batch_size] 目标类别
        """
        log_probs = F.log_softmax(pred, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (self.num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = -torch.sum(smooth_target * log_probs, dim=-1).mean()
        return loss

# 使用
criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
loss = criterion(predictions, targets)
```

### 5.5 MixUp

```python
# MixUp：混合两个样本和标签
"""
公式：
x̃ = λ * x_i + (1-λ) * x_j
ỹ = λ * y_i + (1-λ) * y_j

其中 λ ~ Beta(α, α)，通常 α=0.2
"""

def mixup_data(x, y, alpha=0.2):
    """
    Args:
        x: [batch_size, ...] 输入数据
        y: [batch_size] 标签
        alpha: Beta分布参数
    Returns:
        mixed_x: 混合后的输入
        y_a, y_b: 两个原始标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 训练循环
for x, y in train_loader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    output = model(x)
    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
```

### 5.6 CutMix

```python
# CutMix：剪切一块区域粘贴到另一个图像
"""
1. 从图像B中剪切一个box
2. 将这个box粘贴到图像A上
3. 标签按面积比例混合

ỹ = λ * y_a + (1-λ) * y_b
其中 λ = 1 - (box面积 / 图像面积)
"""

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size).to(x.device)

    # 计算剪切box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 混合图像
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

**正则化技术对比**：

| 技术 | 原理 | 效果 | 计算开销 |
|:-----|:-----|:-----|:---------|
| **L1/L2** | 约束权重 | 防止过拟合、L1稀疏化 | 低 |
| **Dropout** | 随机丢弃 | 防止共适应 | 低 |
| **Early Stopping** | 提前终止 | 防止过拟合 | 低 |
| **Data Augmentation** | 扩充数据 | 增加数据多样性 | 中 |
| **Label Smoothing** | 软化标签 | 防止过度自信 | 无 |
| **MixUp/CutMix** | 样本混合 | 增强泛化、校准 | 低 |

---

## 六、常见问题与解决

### 6.1 梯度消失

```python
"""
问题：深层网络中，梯度逐层衰减，导致浅层参数几乎不更新

原因：
├─ Sigmoid/Tanh：导数 < 1，连乘后指数衰减
├─ 深层网络：链式法则连乘
└─ 初始化不当：权重太小

症状：
├─ 浅层权重几乎不变
├─ 损失几乎不下降
└─ 深层学习，浅层不学习
"""

# 解决方案

# 1. 使用ReLU激活
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),  # ReLU在正区导数为1
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. 使用残差连接（ResNet）
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.fc(x)  # 残差连接

# 3. 使用批归一化
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # 稳定梯度
    nn.ReLU(),
    # ...
)

# 4. 梯度裁剪（防止梯度爆炸的同时帮助稳定性）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6.2 梯度爆炸

```python
"""
问题：梯度变得非常大，导致参数更新过大

原因：
├─ 深层网络：链式法则连乘
├─ 初始化不当：权重太大
└─ 学习率过大

症状：
├─ 损失变成NaN
├─ 参数变成Inf
└─ 训练突然崩溃
"""

# 解决方案

# 1. 梯度裁剪
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # L2裁剪
# 或
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  # 按值裁剪
optimizer.step()

# 2. 降低学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 更小的lr

# 3. 使用批归一化
# 4. 调整初始化（更小的方差）
```

### 6.3 过拟合

```python
"""
问题：训练集表现好，测试集表现差

症状：
├─ Train loss持续下降，Val loss开始上升
├─ Train accuracy >> Val accuracy
└─ 模型复杂度 > 数据复杂度

诊断曲线：
Train loss:  ╲_____
Val loss:    ╲╱___

检测方法：
└─ 绘制学习曲线（train/val loss随epoch变化）
"""

# 解决方案

# 1. 增加正则化
# L2正则
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Dropout
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # 增加dropout率
    nn.Linear(256, 10)
)

# 2. 数据增强
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor()
])

# 3. 早停
early_stopping = EarlyStopping(patience=10)

# 4. 减少模型复杂度
model = nn.Sequential(
    nn.Linear(784, 128),  # 减少隐藏层大小
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 5. Label Smoothing
criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
```

### 6.4 欠拟合

```python
"""
问题：训练集和测试集表现都差

症状：
├─ Train loss和Val loss都很高
├─ 模型无法拟合训练数据
└─ 模型容量不足或训练不足

诊断曲线：
Train loss:  ╲_______（没有降下去）
Val loss:    ╲_______
"""

# 解决方案

# 1. 增加模型复杂度
model = nn.Sequential(
    nn.Linear(784, 512),  # 增加宽度
    nn.ReLU(),
    nn.Linear(512, 256),  # 增加深度
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. 减少正则化
nn.Dropout(0.1)  # 降低dropout率
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 3. 训练更久
num_epochs = 200  # 增加训练轮数

# 4. 特征工程
# 添加更多有意义的特征
```

### 6.5 训练不稳定

```python
"""
问题：损失震荡、不收敛

症状：
├─ Loss曲线剧烈震荡
├─ 准确率忽高忽低
└─ 同样的数据不同次训练结果差异大
"""

# 解决方案

# 1. 调整学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

# 2. 使用更稳定的优化器
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 比SGD更稳定

# 3. 调整Batch Size
# 小batch带来噪声，可能不稳定
# 大batch更稳定但泛化可能变差
train_loader = DataLoader(dataset, batch_size=256)  # 增大batch

# 4. 批归一化
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU()
)

# 5. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 6.6 死神经元（Dead ReLU）

```python
"""
问题：ReLU神经元永远输出0，不再更新

原因：
├─ 初始化不当
├─ 学习率过大导致权重更新到负值区
└─ 输入的梯度总是负的

症状：
├─ 某些ReLU输出总是0
├─ 梯度分析显示某些神经元从未激活
└─ 模型容量实际上变小了
"""

# 解决方案

# 1. 使用LeakyReLU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(negative_slope=0.01),  # 负区有小梯度
    nn.Linear(256, 10)
)

# 2. 调整学习率
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 更小

# 3. 使用He初始化
nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')

# 4. 检测死神经元
def count_dead_neurons(model, x):
    """统计输出为0的神经元数量"""
    for layer in model:
        if isinstance(layer, nn.ReLU):
            out = layer(x)
            dead_ratio = (out == 0).float().mean()
            print(f"Dead neuron ratio: {dead_ratio:.2%}")
```

---

## 七、实战案例

### 7.1 案例1：简单分类任务（MLP实现MNIST分类）

```python
"""
任务：使用MLP对MNIST手写数字进行分类
输入：28×28图像（展平为784维向量）
输出：10个类别（数字0-9）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义模型
class MNIST_MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平: [batch, 1, 28, 28] -> [batch, 784]
        return self.network(x)

model = MNIST_MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10, dropout=0.3)

# 3. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 4. 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 5. 测试函数
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# 6. 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 7. 绘制学习曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(test_losses, label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Loss Curve')

axes[1].plot(train_accs, label='Train Accuracy')
axes[1].plot(test_accs, label='Test Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].set_title('Accuracy Curve')

plt.tight_layout()
plt.savefig('mnist_training_curve.png')
plt.show()

# 预期结果: Test Accuracy > 98%
```

### 7.2 案例2：回归任务（房价预测）

```python
"""
任务：使用MLP预测房价
输入：房屋特征（面积、房间数、房龄等）
输出：房价（连续值）
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 转换为PyTorch张量
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)

# 3. 定义回归模型
class RegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

model = RegressionMLP(
    input_dim=X_train.shape[1],
    hidden_dims=[64, 32, 16],
    output_dim=1,
    dropout=0.2
)

# 4. 训练配置
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 5. 训练
num_epochs = 200
batch_size = 32
best_loss = float('inf')
patience_counter = 0
early_patience = 20

for epoch in range(num_epochs):
    model.train()

    # 小批量训练
    indices = torch.randperm(X_train_t.size(0))
    epoch_loss = 0

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_x = X_train_t[batch_idx]
        batch_y = y_train_t[batch_idx]

        output = model(batch_x)
        loss = criterion(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_idx)

    epoch_loss /= len(X_train_t)

    # 验证
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)
        test_loss = criterion(test_output, y_test_t).item()

    scheduler.step(test_loss)

    # 早停
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
        best_model = model.state_dict().copy()
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')

    if patience_counter >= early_patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# 加载最佳模型
model.load_state_dict(best_model)

# 6. 评估
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()

# 反归一化
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f'\nRegression Results:')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')
```

### 7.3 案例3：优化器对比实验

```python
"""
任务：对比不同优化器在相同任务上的表现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 优化器配置
optimizers_config = {
    'SGD': {'class': optim.SGD, 'params': {'lr': 0.01, 'momentum': 0.9}},
    'Adam': {'class': optim.Adam, 'params': {'lr': 0.001}},
    'AdamW': {'class': optim.AdamW, 'params': {'lr': 0.001, 'weight_decay': 0.01}},
    'RMSprop': {'class': optim.RMSprop, 'params': {'lr': 0.001}},
}

# 训练和记录
results = {}
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for opt_name, config in optimizers_config.items():
    print(f'\n{"="*50}')
    print(f'Training with {opt_name}')
    print(f'{"="*50}')

    # 初始化模型
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = config['class'](model.parameters(), **config['params'])

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f}s')

    results[opt_name] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'epoch_times': epoch_times,
        'final_test_acc': test_accs[-1],
        'total_time': sum(epoch_times)
    }

# 结果对比
print(f'\n{"="*50}')
print('Optimizers Comparison')
print(f'{"="*50}')
print(f'{"Optimizer":<10} {"Final Test Acc":<15} {"Total Time":<10}')
print(f'{"-"*40}')
for opt_name, res in results.items():
    print(f'{opt_name:<10} {res["final_test_acc"]:<15.4f} {res["total_time"]:<10.2f}')

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Test Accuracy
for opt_name, res in results.items():
    axes[0].plot(res['test_accs'], label=opt_name, marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Test Accuracy vs Epoch')
axes[0].legend()
axes[0].grid(True)

# Loss
for opt_name, res in results.items():
    axes[1].plot(res['test_losses'], label=opt_name)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Loss')
axes[1].set_title('Test Loss vs Epoch')
axes[1].legend()
axes[1].grid(True)

# Final Accuracy Bar Chart
opt_names = list(results.keys())
final_accs = [results[opt]['final_test_acc'] for opt in opt_names]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
axes[2].bar(opt_names, final_accs, color=colors)
axes[2].set_ylabel('Final Test Accuracy')
axes[2].set_title('Final Test Accuracy Comparison')
axes[2].set_ylim(min(final_accs) - 0.01, max(final_accs) + 0.01)

for i, v in enumerate(final_accs):
    axes[2].text(i, v + 0.002, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()
```

### 7.4 案例4：学习率调度实战

```python
"""
任务：比较不同学习率调度策略的效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据准备（简化版）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# 调度器配置
schedulers_config = {
    'Constant': None,  # 无调度
    'StepLR': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5),
    'Exponential': lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.9),
    'Cosine': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10),
    'OneCycle': None,  # 特殊处理
}

results = {}
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for name, sched_fn in schedulers_config.items():
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # OneCycle特殊处理
    if name == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=len(train_loader)*num_epochs)
        use_batch_scheduler = True
    else:
        scheduler = sched_fn_config[name](optimizer) if sched_fn else None
        use_batch_scheduler = False

    lr_history = []
    loss_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if use_batch_scheduler:
                scheduler.step()
                lr_history.append(optimizer.param_groups[0]['lr'])

        if not use_batch_scheduler and scheduler:
            scheduler.step()

        if not use_batch_scheduler and scheduler:
            lr_history.append(optimizer.param_groups[0]['lr'])

        loss_history.append(epoch_loss / len(train_loader))

    results[name] = {'lr': lr_history, 'loss': loss_history}

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Learning Rate Curve
for name, res in results.items():
    x = range(len(res['lr'])) if res['lr'] else range(num_epochs)
    axes[0].plot(x, res['lr'] if res['lr'] else [0.001]*len(x), label=name, marker='o' if not res['lr'] else None)
axes[0].set_xlabel('Iteration/Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].set_title('Learning Rate Schedule')
axes[0].legend()
axes[0].grid(True)
axes[0].set_yscale('log')

# Loss Curve
for name, res in results.items():
    axes[1].plot(res['loss'], label=name)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('lr_scheduler_comparison.png')
plt.show()
```

### 7.5 案例5：Learning Rate Finder

```python
"""
任务：使用LR Range Test找到最佳学习率
原理：从很小的学习率开始，指数增加，记录loss
最佳学习率在loss下降最快处附近
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 数据准备
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# LR Finder
class LRFinder:
    def __init__(self, model, optimizer, criterion, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'lr': [], 'loss': []}

    def range_test(self, train_loader, init_lr=1e-6, final_lr=1.0, num_iter=100):
        """
        Args:
            init_lr: 初始学习率
            final_lr: 最终学习率
            num_iter: 测试迭代次数
        """
        model = self.model.to(self.device)
        optimizer = self.optimizer

        # 计算学习率增长因子
        lr_mult = (final_lr / init_lr) ** (1 / num_iter)

        # 保存初始状态
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        init_opt_state = optimizer.state_dict()

        # 设置初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr

        model.train()
        iter_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if iter_count >= num_iter:
                break

            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)

            # 平滑loss
            if iter_count == 0:
                avg_loss = loss.item()
            else:
                avg_loss = 0.9 * avg_loss + 0.1 * loss.item()

            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['loss'].append(avg_loss)

            loss.backward()
            optimizer.step()

            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_mult

            iter_count += 1

        # 恢复初始状态
        model.load_state_dict(init_state)
        optimizer.load_state_dict(init_opt_state)

        return self.history

    def plot(self, skip_start=10, skip_end=5):
        """绘制LR vs Loss曲线"""
        lrs = self.history['lr'][skip_start:-skip_end]
        losses = self.history['loss'][skip_start:-skip_end]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.grid(True)

        # 标记最小loss附近的学习率
        min_loss_idx = np.argmin(losses)
        min_lr = lrs[min_loss_idx]
        ax.scatter([min_lr], [losses[min_loss_idx]], color='red', s=100, zorder=5)
        ax.annotate(f'Min Loss LR: {min_lr:.2e}', xy=(min_lr, losses[min_loss_idx]),
                    xytext=(min_lr*10, losses[min_loss_idx]),
                    arrowprops=dict(arrowstyle='->', color='red'))

        plt.tight_layout()
        plt.savefig('lr_finder.png')
        plt.show()

        return min_lr

# 使用
model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(model, optimizer, criterion)
history = lr_finder.range_test(train_loader, init_lr=1e-6, final_lr=1, num_iter=200)

# 绘图并获取推荐学习率
recommended_lr = lr_finder.plot()
print(f'Recommended Learning Rate: {recommended_lr:.2e}')

# 使用推荐学习率训练
optimizer = optim.Adam(model.parameters(), lr=recommended_lr)
# 继续正常训练...
```

---

## 八、方案选择建议

### 8.1 优化器选择指南

```
优化器选择决策树：

问题类型
├─ 稀疏数据（推荐系统、NLP）
│  ├─ 需要快速收敛 → Adam / Adagrad
│  └─ 数据极稀疏 → Adagrad
│
├─ 大规模预训练（Transformer、大语言模型）
│  ├─ 标准 → AdamW
│  ├─ 内存受限 → Lion
│  └─ 追求数值稳定 → AdaBelief
│
├─ 计算机视觉（CNN）
│  ├─ 快速实验 → Adam / AdamW
│  ├─ 追求最佳效果 → SGD + Momentum + Cosine Annealing
│  └─ 迁移学习微调 → AdamW
│
├─ 小规模问题（<10K参数）
│  └─ L-BFGS
│
├─ 强化学习
│  └─ RMSprop / Adam
│
└─ 表格数据
   ├─ 简单任务 → AdamW
   └─ 复杂任务 → SGD + Momentum
```

### 8.2 学习率设置建议

| 优化器 | 推荐学习率 | 调整建议 |
|:-------|:----------|:---------|
| **SGD** | 0.1 - 0.01 | 大batch需要更大lr |
| **SGD+Momentum** | 0.05 - 0.01 | 比纯SGD稍小 |
| **Adam** | 0.001 - 0.0001 | 0.001是经典起点 |
| **AdamW** | 0.001 - 0.0001 | 与Adam相同 |
| **Lion** | 0.0001 - 0.00001 | 比Adam小10倍 |
| **RMSprop** | 0.001 | 默认值通常不错 |

**学习率与Batch Size关系**：

```python
# 线性缩放规则
base_lr = 0.1    # batch_size=256时的lr
base_batch = 256

# 新batch size的新学习率
new_batch = 512
new_lr = base_lr * (new_batch / base_batch)  # = 0.2

# 注意：超过一定阈值后不再线性增长
```

### 8.3 Batch Size选择

```
Batch Size选择指南：

极小（1-16）
├─ 优点：泛化好、内存占用少
├─ 缺点：训练慢、不稳定
└─ 适用：内存受限、在线学习

小（32-128）【推荐】
├─ 优点：泛化好、训练稳定
├─ 缺点：无法充分利用GPU
└─ 适用：大多数任务

中（256-512）
├─ 优点：训练快、稳定性好
├─ 缺点：泛化可能略差
└─ 适用：大规模训练

大（1024+）
├─ 优点：训练最快
├─ 缺点：泛化差、需要调整lr
└─ 适用：分布式训练

经验法则：
├─ 起始：32或64
├─ GPU利用率低 → 增大batch
├─ 泛化差 → 减小batch
└─ 大batch需要配合大学习率
```

### 8.4 综合推荐配置

```python
# 配置1：快速实验（AdamW）
config_fast = {
    'optimizer': 'AdamW',
    'lr': 0.001,
    'weight_decay': 0.01,
    'batch_size': 64,
    'scheduler': 'CosineAnnealingLR',
}

# 配置2：最佳泛化（SGD）
config_best = {
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'batch_size': 128,
    'scheduler': 'CosineAnnealingWarmRestarts',
}

# 配置3：大模型训练（AdamW/Lion）
config_large = {
    'optimizer': 'AdamW',
    'lr': 0.0001,
    'weight_decay': 0.01,
    'batch_size': 256,
    'scheduler': 'OneCycleLR',
}

# 配置4：内存受限
config_memory = {
    'optimizer': 'Lion',
    'lr': 0.00001,
    'weight_decay': 0.01,
    'batch_size': 32,
    'scheduler': 'StepLR',
}
```

---

## 九、学习资源

### 9.1 优化器论文

| 论文 | 年份 | 优化器 | arXiv/会议 |
|:-----|:-----|:-------|:-----------|
| [Adaptive Subgradient Methods](https://jmlr.org/papers/v12/duchi11a.html) | 2011 | Adagrad | JMLR |
| [On the importance of initialization and momentum](https://dl.acm.org/doi/10.5555/3044805.3044823) | 2013 | Momentum | ICML |
| [RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) | 2014 | RMSprop | Coursera |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | Adam | ICLR 2015 |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2017 | AdamW | ICLR 2018 |
| [AdaBelief Optimizer](https://arxiv.org/abs/2010.07468) | 2020 | AdaBelief | NeurIPS 2020 |
| [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) | 2023 | Lion | ICML 2023 |

### 9.2 PyTorch优化器文档

```python
# PyTorch内置优化器
torch.optim.SGD
torch.optim.Adam
torch.optim.AdamW
torch.optim.RMSprop
torch.optim.Adagrad
torch.optim.Adadelta
torch.optim.Adamax
torch.optim.NAdam
torch.optim.RAdam
torch.optim.SparseAdam  # 稀疏梯度
torch.optim.LBFGS

# 学习率调度器
torch.optim.lr_scheduler.LambdaLR
torch.optim.lr_scheduler.MultiplicativeLR
torch.optim.lr_scheduler.StepLR
torch.optim.lr_scheduler.MultiStepLR
torch.optim.lr_scheduler.ExponentialLR
torch.optim.lr_scheduler.CosineAnnealingLR
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
torch.optim.lr_scheduler.OneCycleLR
torch.optim.lr_scheduler.ReduceLROnPlateau
torch.optim.lr_scheduler.CyclicLR
torch.optim.lr_scheduler.SequentialLR
torch.optim.lr_scheduler.ChainedScheduler
```

官方文档：https://pytorch.org/docs/stable/optim.html

### 9.3 调参技巧

**通用流程**：

```
1. 数据探索和预处理
   ├─ 分析数据分布
   ├─ 标准化/归一化
   └─ 处理缺失值和异常值

2. 建立基线
   ├─ 简单模型
   ├─ 默认超参数
   └─ Adam优化器

3. 网络架构调优
   ├─ 调整层数和宽度
   ├─ 添加/移除正则化
   └─ 选择合适的激活函数

4. 优化器调优
   ├─ 尝试不同优化器
   ├─ 调整学习率
   └─ 添加学习率调度

5. 正则化调优
   ├─ 调整dropout率
   ├─ 调整weight decay
   └─ 添加数据增强

6. 精细调优
   ├─ 学习率范围测试
   ├─ 网格搜索/贝叶斯优化
   └─ Ensemble方法
```

**超参数搜索策略**：

```python
# 1. 网格搜索（Grid Search）
param_grid = {
    'lr': [0.001, 0.0001],
    'weight_decay': [0.01, 0.001, 0.0001],
    'hidden_dims': [[128, 64], [256, 128]]
}

# 2. 随机搜索（Random Search）- 更高效
# 随机采样参数组合

# 3. 贝叶斯优化
from optuna import create_study

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    # 训练并返回验证指标
    return validation_score

study = create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 4. 学习率范围测试（见案例5）
```

---

## 十、相关笔记

### 深入学习

- [[AI研究/AI学习/02-模型原理/GNN全面解析]] - 图神经网络原理与应用
- [[AI研究/AI学习/02-模型原理/Transformer研读]] - Transformer 架构详细研读
- [[AI研究/AI学习/神经网络类型全景总结]] - 所有神经网络架构概览
- [[AI研究/AI学习/常见术语对照]] - AI/ML 术语中英文对照

### 实战应用

- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - 检索增强生成项目
- [[AI研究/AI学习/04-深入前沿/RAG 2.0 全面解析]] - RAG端到端训练
- [[AI研究/AI学习/04-深入前沿/知识蒸馏全面解析]] - 模型压缩技术

### 基础知识

- [[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线
- [[AI研究/AI学习/01-基础夯实/数学基础笔记]] - 深度学习数学基础

---

#MLP #优化器 #训练技巧 #深度学习 #基础原理
