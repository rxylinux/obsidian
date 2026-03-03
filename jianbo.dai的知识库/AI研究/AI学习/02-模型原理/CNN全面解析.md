---
title: CNN 卷积神经网络全面解析
date: 2026-02-28
tags:
  - CNN
  - 卷积神经网络
  - 计算机视觉
  - 深度学习
  - 架构原理
status: active
---

# CNN 卷积神经网络全面解析

> [!info] 说明
> 本笔记系统介绍卷积神经网络（CNN）的原理、架构演进、开源框架及实战应用，涵盖从经典LeNet到现代高效网络的完整发展历程

> [!tip] 快速导航
> - **返回索引**：[[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
> - **架构总览**：[[AI研究/AI学习/神经网络类型全景总结]] - 所有架构概览

---

## 📑 目录

> [!tip] 使用说明
> 点击下方的任何章节链接，即可跳转到对应内容（支持 `Ctrl/Cmd + Click` 在新面板打开）

### 基础理论
- [[#一、核心原理]]
  - [[#1.1 什么是CNN]]
  - [[#1.2 卷积操作原理]]
  - [[#1.3 池化层]]
  - [[#1.4 感受野（Receptive Field）]]
  - [[#1.5 数学表达]]
- [[#二、为什么CNN适合计算机视觉]]
  - [[#2.1 图像数据的特点]]
  - [[#2.2 与传统MLP的对比]]
  - [[#2.3 参数共享的优势]]
  - [[#2.4 CNN应用领域全景]]

### 架构详解
- [[#三、主流CNN架构详解]]
  - [[#3.1 LeNet（1998）- CNN鼻祖]]
  - [[#3.2 AlexNet（2012）- 深度学习爆发]]
  - [[#3.3 VGG（2014）- 深层网络典范]]
  - [[#3.4 ResNet（2015）- 残差学习革命]]
  - [[#3.5 Inception（2014）- 多尺度并行]]
  - [[#3.6 EfficientNet（2019）- 复合缩放]]
  - [[#3.7 MobileNet（2017）- 轻量化先锋]]
  - [[#3.8 其他著名CNN架构]]
  - [[#3.9 著名CNN模型选择速查表]]
- [[#四、架构对比总结]]
  - [[#4.1 综合对比表]]
  - [[#4.2 性能-效率权衡]]
  - [[#4.3 选择指南]]

### 实战应用
- [[#五、开源框架与模型]]
  - [[#5.1 深度学习框架对比]]
  - [[#5.2 PyTorch图像模型库（timm）]]
  - [[#5.3 ImageNet预训练模型]]
  - [[#5.4 开源预训练模型列表]]
- [[#六、性能与耗时分析]]
- [[#七、实战案例]]
  - [[#7.1 图像分类 - ResNet实战]]
  - [[#7.2 目标检测 - YOLO实战]]
  - [[#7.3 图像分割 - U-Net实战]]
  - [[#7.4 迁移学习 - 微调预训练模型]]
  - [[#7.5 数据增强 - CutMix, MixUp, AutoAugment]]
  - [[#7.6 医学影像分析 - CT/MRI诊断]]
  - [[#7.7 人脸识别 - ArcFace损失]]
  - [[#7.8 视频分类 - 3D CNN & TimeSformer]]

### 参考指南
- [[#八、方案选择建议]]
- [[#九、常见问题]]
- [[#十、学习资源]]
- [[#十一、相关笔记]]

---

## 一、核心原理

### 1.1 什么是CNN

**CNN（Convolutional Neural Network）** 是专门处理网格状数据（如图像）的神经网络。

```
传统神经网络：处理任意向量数据
├─ MLP：独立样本，不考虑空间关系
└─ 无法有效处理图像等高维数据

CNN：处理网格结构数据
├─ 专门设计用于图像、视频等2D/3D数据
├─ 利用局部连接和权重共享
└─ 参数少、计算高效、性能优异
```

### 1.2 卷积操作原理

卷积是CNN的核心操作，通过滑动窗口提取特征。

```
卷积操作示意：

输入图像 (5×5)           卷积核 (3×3)           输出特征图 (3×3)
┌─────────────────┐     ┌─────────┐           ┌─────────┐
│ 1  1  1  0  0   │     │ 1  0  1 │           │ 4  3  4 │
│ 0  1  1  1  0   │     │ 0  1  0 │     →     │ 2  4  3 │
│ 0  0  1  1  1   │     │ 1  0  1 │           │ 2  3  4 │
│ 0  0  1  1  0   │     └─────────┘
│ 0  1  1  0  0   │
└─────────────────┘

计算过程（输出[0,0]位置）：
= 1×1 + 1×0 + 1×1 + 0×0 + 1×1 + 1×0 + 0×1 + 0×0 + 0×1
= 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 0 = 3
```

**关键参数**：
- **卷积核大小（Kernel Size）**：常用 3×3、5×5
- **步幅（Stride）**：卷积核滑动步长，常用1、2
- **填充（Padding）**：保持特征图尺寸，常用 'same'、'valid'
- **通道数（Channels）**：输出特征图数量

### 1.3 池化层

池化用于降维和提取主要特征。

```
最大池化（Max Pooling，2×2，stride=2）：

输入 (4×4)              输出 (2×2)
┌─────────────┐         ┌─────────┐
│ 1  3  2  4  │         │ 3  4    │
│ 5  3  1  2  │    →    │ 5  6    │
│ 2  4  6  3  │         └─────────┘
│ 1  2  5  6  │
└─────────────┘

取每个2×2窗口的最大值
```

| 池化类型 | 公式 | 特点 | 适用场景 |
|:---------|:-----|:-----|:---------|
| **最大池化** | max(区域) | 保留最强特征 | 特征提取 |
| **平均池化** | mean(区域) | 保留背景信息 | 特征平滑 |
| **全局池化** | max/mean(全图) | 压缩为向量 | 分类头 |

### 1.4 感受野（Receptive Field）

感受野是指输出神经元对应的输入区域大小。

```
感受野计算：

输入层：感受野 = 1

Conv 3×3, stride=1：感受野 = 1 + (3-1) = 3

MaxPool 2×2, stride=2：感受野 = 3 + (3-1)×(2-1) = 5

Conv 3×3, stride=1：感受野 = 5 + (3-1)×2 = 9

多层堆叠后，深层神经元能"看到"更大的输入区域
```

**公式**：
```
RF_{l+1} = RF_l + (K_l - 1) × ∏_{i=1}^{l} s_i

其中：
├─ RF_l：第l层的感受野
├─ K_l：第l层的卷积核大小
└─ s_i：第i层的步幅
```

### 1.5 数学表达

**2D卷积**：
```python
# 输入特征图 X ∈ R^{C_in × H_in × W_in}
# 卷积核 W ∈ R^{C_out × C_in × K_h × K_w}
# 偏置 b ∈ R^{C_out}

# 输出特征图 Y ∈ R^{C_out × H_out × W_out}

Y[c_out, i, j] = b[c_out] + Σ_{c_in=0}^{C_in-1} Σ_{m=0}^{K_h-1} Σ_{n=0}^{K_w-1}
                   X[c_in, i×s_h + m, j×s_w + n] × W[c_out, c_in, m, n]

其中：
├─ s_h, s_w：垂直和水平步幅
├─ H_out = (H_in + 2p - K_h) / s_h + 1
├─ W_out = (W_in + 2p - K_w) / s_w + 1
└─ p：填充大小
```

**参数量计算**：
```python
# 单个卷积层参数量
params = C_out × (C_in × K_h × K_w + 1)

# 例如：输入3通道，输出64通道，卷积核3×3
params = 64 × (3 × 3 × 3 + 1) = 64 × 28 = 1,792
```

---

## 二、为什么CNN适合计算机视觉

### 2.1 图像数据的特点

| 特性 | 说明 | CNN对应机制 |
|:-----|:-----|:-----------|
| **局部性** | 相邻像素高度相关 | 局部连接、卷积核 |
| **平移不变性** | 特征位置可变 | 权重共享、池化 |
| **层次性** | 低级→高级特征 | 多层堆叠 |
| **平移等变性** | 输入平移，输出同步平移 | 卷积操作 |

### 2.2 与传统MLP的对比

```
处理224×224×3图像：

MLP方案：
├─ 展平：224×224×3 = 150,528维
├─ 隐藏层1000神经元：150,528×1000 = 150M参数
├─ 忽略空间结构
└─ 容易过拟合

CNN方案（假设5层卷积）：
├─ 参数量：~5-25M（取决于深度）
├─ 保留空间结构
├─ 参数共享（减少参数量）
└─ 归纳偏置更强
```

**对比表**：

| 方面 | MLP | CNN |
|:-----|:-----|:-----|
| 参数量 | 极大（全连接） | 小（局部连接+共享） |
| 空间结构 | 破坏 | 保留 |
| 平移不变性 | 无 | 有 |
| 训练数据需求 | 极大 | 相对较少 |
| 计算效率 | 低 | 高 |
| 可解释性 | 差 | 中（特征可视化） |

### 2.3 参数共享的优势

```
传统MLP：每个连接有独立权重
├─ 图像左上角的猫脸特征 = 权重组A
├─ 图像右下角的猫脸特征 = 权重组B
└─ 参数冗余，效率低

CNN：卷积核在整个图像滑动
├─ 同一卷积核检测相同特征（无论位置）
├─ 参数共享：3×3卷积核只有9个参数
└─ 大幅减少参数，提升泛化能力
```

### 2.4 CNN应用领域全景

> [!info] 核心观点
> **图像相关是 CNN 的主要应用领域（80%+）**，但 CNN 也能处理其他具有局部相关性的数据。

#### 应用分布（现实情况）

```
┌─────────────────────────────────────────────┐
│           CNN 应用领域分布（估算）            │
├─────────────────────────────────────────────┤
│  🖼️ 计算机视觉（图像/视频）  ████████ 80%+   │
│  📝 NLP（文本）               ██ 10%        │
│  🎤 语音处理                 █ 5%          │
│  📊 其他（时间序列等）        █ 5%          │
└─────────────────────────────────────────────┘
```

---

#### 1️⃣ 计算机视觉（主流领域，80%+）

| 任务类型 | 说明 | 代表模型 |
|:---------|:-----|:---------|
| **图像分类** | 识别图像中的物体 | ResNet, EfficientNet, ViT |
| **目标检测** | 定位+识别多个物体 | YOLO, Faster R-CNN |
| **图像分割** | 像素级分类 | U-Net, Mask R-CNN |
| **人脸识别** | 验证、识别 | ArcFace, FaceNet |
| **风格迁移** | 艺术风格转换 | Neural Style Transfer |
| **超分辨率** | 图像放大重建 | SRGAN, ESRGAN |
| **医学影像** | 疾病诊断、器官分割 | 专用CNN架构 |

> **为什么CNN统治CV领域？**
> - 图像具有**局部相关性**（相邻像素强相关）
> - 图像具有**平移不变性**（物体移动位置仍是同一物体）
> - CNN的卷积操作天然契合这些特性

---

#### 2️⃣ 自然语言处理（已被Transformer替代）

| 任务类型 | 说明 | 代表 |
|:---------|:-----|:-----|
| **文本分类** | 情感分析、主题分类 | TextCNN (2014) |
| **机器翻译** | 序列到序列 | ConvS2S (2017) |
| **问答系统** | 提取答案 | QANet (2017) |

> [!warning] 历史演变
> ```
> 2014-2017：TextCNN 曾火过一时
>      ↓
> 2017后：Transformer 彻底统治 NLP
>      ↓
> 现在：文本任务很少用 CNN
> ```
>
> **原因**：
> - 文本的**长距离依赖**比图像更重要（句首句尾可能相关）
> - Transformer 的自注意力机制更适合捕捉全局关系
> - BERT/GPT 等预训练模型碾压一切

---

#### 3️⃣ 语音处理（CNN作为特征提取器）

| 任务类型 | 说明 | 方案 |
|:---------|:-----|:-----|
| **语音识别** | 音频转文本 | CNN + RNN/CTC |
| **语音情感识别** | 识别情绪 | CNN特征提取 |
| **声纹识别** | 说话人识别 | 混合方案 |

> **现状**：
> - CNN 仍用于提取**局部音频特征**（频谱图处理）
> - 但不是主角，更多与 Transformer/RNN 混合使用
> - 现代方案：wav2vec (CNN + Transformer)、Conformer

---

#### 4️⃣ 时间序列预测（非主流）

| 场景 | 说明 |
|:-----|:-----|
| **股票预测** | 价格趋势预测 |
| **天气预测** | 气象数据建模 |
| **交通流量** | 实时路况预测 |

> **现状**：不是主流选择
> - 简单场景：ARIMA、统计方法
> - 复杂场景：LSTM/GRU、Transformer
> - CNN：偶尔用于特征提取

---

#### 5️⃣ 其他领域

| 领域 | 应用 | 说明 |
|:-----|:-----|:-----|
| **推荐系统** | 点击率预测 | 序列化推荐 |
| **游戏AI** | AlphaGo | 提取棋盘特征 |
| **异常检测** | 工业质检 | 产品缺陷检测 |
| **基因组学** | DNA序列分析 | 序列模式识别 |
| **脑机接口** | 脑电信号处理 | 信号特征提取 |

---

#### CNN适用性的核心原理

```
CNN 的本质 = 局部特征提取 + 权重共享

任何具有"局部相关性"的数据，都可以用 CNN 处理：
├─ 图像：相邻像素相关 ✅ 最佳
├─ 文本：相邻词组成短语 ⚠️ 可用但非最佳
├─ 音频：相邻采样点组成音素 ✅ 可用
├─ 序列：相邻时间点有趋势 ⚠️ 可用
└─ 图谱：节点邻居相关 → 需要 Graph CNN
```

---

#### 一句话总结

> **CNN = 图像处理的王者**
>
> | 领域 | 结论 |
> |:-----|:-----|
> | **图像/视频** | 🔥 不二之选（设计初衷+技术优势） |
> | **文本** | ❌ 已被 Transformer 替代 |
> | **语音** | ⚠️ 作为特征提取器，非主角 |
> | **时间序列** | ⚠️ 偶尔使用，非主流 |
> | **其他** | 视数据是否有局部相关性而定 |
>
> 如果你要处理图像，CNN 是最佳选择；如果要处理文本，直接上 Transformer。

---

## 三、主流CNN架构详解

### 3.1 LeNet（1998）- CNN鼻祖

**原理**：最早的实用CNN，用于手写数字识别

**架构**：
```python
"""
LeNet-5架构：

输入 (32×32×1)
    ↓
Conv 5×5, 6通道, stride=1, padding=0 → (28×28×6)
    ↓
MaxPool 2×2, stride=2 → (14×14×6)
    ↓
Conv 5×5, 16通道 → (10×10×16)
    ↓
MaxPool 2×2, stride=2 → (5×5×16)
    ↓
FC 120 → FC 84 → FC 10
    ↓
输出 (10类)
"""
```

**特点**：
- ✅ 首次展示CNN潜力（MNIST 99.2%准确率）
- ✅ 奠定CNN基本结构（Conv-Pool-FC）
- ❌ 网络较浅，难以处理复杂图像
- ❌ 使用Sigmoid/Tanh（梯度消失）

**代码示例**：
```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv1 + Pool
        x = torch.tanh(self.conv1(x))
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        # Conv2 + Pool
        x = torch.tanh(self.conv2(x))
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        # FC
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# 使用示例
model = LeNet5(num_classes=10)
x = torch.randn(32, 1, 32, 32)  # batch=32, 1通道, 32×32
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

---

### 3.2 AlexNet（2012）- 深度学习爆发

**原理**：首次在ImageNet取得突破，引发深度学习热潮

**架构**：
```python
"""
AlexNet架构：

输入 (227×227×3)
    ↓
Conv 11×11, 96通道, stride=4 → (55×55×96)
    ↓
MaxPool 3×3, stride=2 → (27×27×96)
    ↓
Conv 5×5, 256通道, padding=2 → (27×27×256)
    ↓
MaxPool 3×3, stride=2 → (13×13×256)
    ↓
Conv 3×3, 384通道, padding=1 → (13×13×384)
    ↓
Conv 3×3, 384通道, padding=1 → (13×13×384)
    ↓
Conv 3×3, 256通道, padding=1 → (13×13×256)
    ↓
MaxPool 3×3, stride=2 → (6×6×256)
    ↓
FC 4096 → FC 4096 → FC 1000
    ↓
输出 (1000类)
"""
```

**创新点**：
1. **ReLU激活**：缓解梯度消失
2. **Dropout**：防止过拟合
3. **数据增强**：平移、翻转、颜色抖动
4. **GPU训练**：两块GPU并行
5. **LRN（局部响应归一化）**：后证实用不大

**特点**：
- ✅ ImageNet 2012冠军（top-5错误率15.3%）
- ✅ 证明深网络可行
- ❌ 参数量60M（过大）
- ❌ 结构不够规整

**代码示例**：
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---

### 3.3 VGG（2014）- 深层网络典范

**原理**：用多个小卷积核（3×3）替代大卷积核

**设计理念**：
```
大卷积核 vs 小卷积堆叠：

5×5卷积：感受野=5，参数=25C²
3×3卷积堆叠×2：感受野=5，参数=18C²

优势：
├─ 更多非线性（更多层）
├─ 更少参数
└─ 更深网络
```

**VGG-16架构**：
```python
"""
VGG-16架构：

输入 (224×224×3)
    ↓
Conv 3×3, 64 × 2 → MaxPool
    ↓
Conv 3×3, 128 × 2 → MaxPool
    ↓
Conv 3×3, 256 × 3 → MaxPool
    ↓
Conv 3×3, 512 × 3 → MaxPool
    ↓
Conv 3×3, 512 × 3 → MaxPool
    ↓
FC 4096 → FC 4096 → FC 1000
"""
```

**特点**：
- ✅ 结构简洁规整
- ✅ 迁移学习效果好
- ❌ 参数量138M（极大）
- ❌ 计算量大

**代码示例**：
```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

---

### 3.4 ResNet（2015）- 残差学习革命

**原理**：引入残差连接，解决退化问题

**核心创新**：
```python
"""
残差连接：

普通层：H(x) = F(x)
残差层：H(x) = F(x) + x  （跳跃连接）

优势：
├─ 梯度可以直接传播（缓解梯度消失）
├─ 允许训练更深的网络（100+层）
└─ 退化问题得到解决
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接的投影（维度不匹配时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 残差连接
        out = nn.functional.relu(out)
        return out
```

**ResNet架构对比**：

| 模型 | 层数 | 参数量 | ImageNet Top-1 |
|:-----|:-----|:-------|:---------------|
| ResNet-18 | 18 | 11.7M | 69.8% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.0% |
| ResNet-101 | 101 | 44.5M | 77.6% |
| ResNet-152 | 152 | 60.2M | 78.6% |

**特点**：
- ✅ 可以训练极深网络
- ✅ 易于优化
- ✅ 通用架构（广泛应用）
- ❌ 深层网络仍有冗余

**完整ResNet代码**：
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # 初始卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块堆叠
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 工厂函数
def resnet18(num_classes=1000):
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    # 需要使用Bottleneck块
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)
```

---

### 3.5 Inception（2014）- 多尺度并行

**原理**：多尺度特征提取并行化

**Inception模块**：
```python
"""
Inception模块（v1）：

                输入
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
    1×1 Conv   3×3 Conv   5×5 Conv
        │         │         │
        └─────────┼─────────┘
                  ↓
             Concat拼接
                  │
               输出

使用1×1卷积降维（Bottleneck）减少计算量
"""

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        # 各分支的输出通道数
        out_1x1, out_3x3_reduce, out_3x3, \
        out_5x5_reduce, out_5x5, out_pool = out_channels_list

        # 1×1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 1×1 -> 3×3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 1×1 -> 5×5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # 3×3池化 -> 1×1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 沿通道维度拼接
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return output
```

**Inception系列演进**：

| 版本 | 创新点 | 特点 |
|:-----|:-------|:-----|
| **GoogLeNet (v1)** | Inception模块 | 22层，参数少 |
| **Inception v2** | Bottleneck, 分解卷积 | 3×3分解为1×3+3×1 |
| **Inception v3** | RMSProp, Label Smoothing | 更深更宽 |
| **Inception v4** | 结合ResNet | 残差Inception |
| **Xception** | 极致深度分离 | 全部用Depthwise |

**特点**：
- ✅ 多尺度特征提取
- ✅ 参数效率高
- ❌ 结构复杂
- ❌ 实现较难

---

### 3.6 EfficientNet（2019）- 复合缩放

**原理**：系统性地缩放网络的深度、宽度、分辨率

**复合缩放公式**：
```python
"""
给定缩放系数 φ：

深度: d = d₀ × φ^α
宽度: w = w₀ × φ^β
分辨率: r = r₀ × φ^γ

约束条件: α × β² × γ² ≈ 2
（总计算量约增加 2^φ）

EfficientNet-B0到B7，φ = 0到7
"""
```

**MBConv模块（Mobile Inverted Bottleneck）**：
```python
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, expand_ratio,
                 kernel_size=3, stride=1, se_ratio=0.25, drop_rate=0.2):
        super().__init__()
        hidden_dim = in_channels * expand_ratio

        # 扩展层（1×1卷积升维）
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        # 深度可分离卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size//2, groups=hidden_dim,
                     bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(hidden_dim, int(in_channels * se_ratio))

        # 投影层（1×1卷积降维）
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Stochastic Depth（训练时随机丢弃）
        self.drop_rate = drop_rate
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_residual:
            if self.training and torch.rand(1) < self.drop_rate:
                out = 0
            else:
                out += identity

        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
```

**EfficientNet性能对比**：

| 模型 | 参数量 | ImageNet Top-1 | FLOPS | GPU推理时间 |
|:-----|:-------|:---------------|:------|:-----------|
| EfficientNet-B0 | 5.3M | 77.1% | 0.39B | 1x |
| EfficientNet-B1 | 7.8M | 79.1% | 0.70B | 1.6x |
| EfficientNet-B2 | 9.2M | 80.1% | 1.0B | 2.0x |
| EfficientNet-B3 | 12M | 81.6% | 1.8B | 3.0x |
| EfficientNet-B4 | 19M | 82.9% | 4.2B | 5.5x |
| EfficientNet-B5 | 30M | 83.6% | 9.9B | 10x |
| EfficientNet-B6 | 43M | 84.0% | 19B | 18x |
| EfficientNet-B7 | 66M | 84.3% | 37B | 35x |

**特点**：
- ✅ 参数效率极高
- ✅ 准确率与效率平衡
- ❌ 训练需要大量数据增强
- ❌ 深度网络训练时间长

---

### 3.7 MobileNet（2017）- 轻量化先锋

**原理**：深度可分离卷积大幅减少计算量

**深度可分离卷积**：
```python
"""
标准卷积 vs 深度可分离卷积：

标准卷积（3×3）：
├─ 空间卷积 + 通道融合同时进行
├─ 计算: H × W × K² × C_in × C_out
└─ 参数: K² × C_in × C_out

深度可分离卷积：
1. Depthwise（逐通道卷积）
   ├─ 计算: H × W × K² × C_in
   └─ 参数: K² × C_in

2. Pointwise（1×1卷积融合通道）
   ├─ 计算: H × W × C_in × C_out
   └─ 参数: C_in × C_out

总计算量约为标准卷积的 1/C_out
"""

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 逐通道卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                     stride=stride, padding=1, groups=in_channels,
                     bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        # 逐点卷积
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

**MobileNet系列演进**：

| 版本 | 创新点 | 特点 |
|:-----|:-------|:-----|
| **MobileNet v1** | 深度可分离卷积 | 轻量级基础 |
| **MobileNet v2** | Inverted Residual, Linear Bottleneck | 更高效 |
| **MobileNet v3** | NAS搜索, h-swish, SE模块 | 自动化优化 |

**MobileNet v2的Inverted Residual**：
```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # 1×1 扩展（升维）
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # 3×3 深度卷积
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                     stride=stride, padding=1, groups=hidden_dim,
                     bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # 1×1 投影（降维，无激活）
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

**特点**：
- ✅ 极低参数量和计算量
- ✅ 适合移动端/嵌入式
- ❌ 准确率略低于大模型
- ❌ 需要硬件加速器支持

**适用场景**：
- 移动应用
- 边缘设备
- 实时应用
- 资源受限环境

---

### 3.8 其他著名CNN架构

#### 3.8.1 DenseNet（2016）- 密集连接网络

**原理**：前面所有层与后面所有层密集连接

**核心思想**：
```python
"""
传统ResNet：层与层之间跳跃连接
DenseNet：层与所有后续层连接

第L层接收来自前面0,1,2,...,L-1层的特征
"""
```

**DenseBlock**：
```python
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            # BN-ReLU-Conv (Bottleneck)
            layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)  # 密集连接
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000):
        super().__init__()
        self.features = nn.Sequential()
        num_init_features = 2 * growth_rate

        # 初始卷积
        self.features.add_module('conv0', nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                   padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))

        # DenseBlock + Transition
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f'dense{i}', block)
            num_features += num_layers * growth_rate

            # Transition层（降采样）
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'trans{i}', trans)
                num_features = num_features // 2

        # 全局池化 + 分类器
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Transition(nn.Module):
    """过渡层：1×1卷积 + 池化"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
```

**DenseNet性能**：

| 模型 | 参数量 | ImageNet Top-1 | 计算效率 |
|:-----|:-------|:---------------|:---------|
| DenseNet-121 | 8M | 74.9% | ⭐⭐⭐⭐ |
| DenseNet-169 | 14M | 77.6% | ⭐⭐⭐ |
| DenseNet-201 | 20M | 78.3% | ⭐⭐⭐ |
| DenseNet-264 | 33M | 79.2% | ⭐⭐ |

**特点**：
- ✅ 参数效率高（比ResNet少很多参数）
- ✅ 特征复用效果好
- ✅ 梯度传播顺畅
- ❌ 显存占用大（存储所有中间层）
- ❌ 训练时间长

---

#### 3.8.2 SENet（2017）- 注意力机制先驱

**原理**：引入通道注意力机制

**Squeeze-and-Excitation模块**：
```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Squeeze：全局信息压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Excitation：通道重要性学习
        reduced_dim = max(channel // reduction, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Squeeze
        avg_out = self.avg_pool(x).squeeze()
        max_out = self.max_pool(x).squeeze()
        squeeze_out = avg_out + max_out

        # Excitation
        excitation_out = self.excitation(squeeze_out).unsqueeze(2)

        # Scale：通道注意力权重
        scale = x * excitation_out
        return scale

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = SELayer(in_channels, reduction)

    def forward(self, x):
        return x + self.se(x)  # 残差连接
```

**SENet架构**：
```python
class SENet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # SENet使用基础模块（类似ResNet的残差块）
        # 每个残差块后加SE模块

        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # SE-ResNet块
        self.layer1 = self._make_layer(64, 128, 2, reduction=16)
        self.layer2 = self._make_layer(128, 256, 2, reduction=16)
        self.layer3 = self._make_layer(256, 512, 2, reduction=16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, reduction):
        layers = []
        for _ in range(blocks):
            layers.append(ResidualSEBlock(in_channels, out_channels, reduction))
        return nn.Sequential(*layers)

class ResidualSEBlock(nn.Module):
    """带SE模块的残差块"""
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        # 残差块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE注意力
        self.se = SELayer(out_channels, reduction)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 加入SE注意力
        out = self.se(out)

        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out
```

**特点**：
- ✅ 即插即用，可加到任何CNN上
- ✅ 参数增加很少（约1%）
- ✅ 性能提升明显（1-3%）
- ✅ 成为后续网络标配组件
- ❌ 计算略有增加

**使用场景**：
- 图像分类
- 目标检测（如YOLOv5、Faster R-CNN）
- 语义分割（如DeepLabV3+）

---

#### 3.8.3 ShuffleNet（2017/2018/2020）- 通道混洗革命

**原理**：通道混洗 + 分组卷积

**核心创新**：
```python
"""
传统卷积：所有通道混合计算
ShuffleNet：分组卷积 + 通道混洗

优势：
├─ 减少计算量（1/g，g=分组数）
├─ 模型更小更快
└─ 适合移动端
"""
```

**ShuffleNet V2核心组件**：

**1. 通道混洗**：
```python
def channel_shuffle(x, groups):
    """
    将通道分组并重组，实现信息跨组流动
    """
    batch, channels, height, width = x.shape
    assert channels % groups == 0

    channels_per_group = channels // groups
    # 重新排列通道维度
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2, 3, 4).contiguous()
    x = x.view(batch, -1, height, width)
    return x
```

**2. ShuffleUnit（基础模块）**：
```python
class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 2
        self.stride = stride

        # 主分支
        self.branch1 = nn.Sequential()

        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        self.branch2 = nn.Sequential(
            # 1×1调整通道数
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 3×3深度卷积（分组）
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                     padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 1×1恢复通道数
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 通道混洗
        self.shuffle = channel_shuffle if stride == 1 else None

    def forward(self, x):
        if self.branch1 is not None:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            out = self.branch2(x)

        if self.shuffle is not None:
            out = self.shuffle(out)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ShuffleNet单元
        self.stage2 = self._make_stage(24, 116, 2, 2)  # 重复4次
        self.stage3 = self._make_stage(116, 232, 2, 2)  # 重复8次
        self.stage4 = self._make_stage(232, 464, 2, 1)  # 重复4次

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(464, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ShuffleUnit(in_ch, out_channels, stride))
        return nn.Sequential(*layers)
```

**ShuffleNet系列对比**：

| 版本 | 年份 | 特点 | 参数量 | ImageNet Top-1 |
|:-----|:-----|:-----|:-------|:---------------|
| **ShuffleNet v1** | 2017 | 通道混洗 | 1.7M | 66.8% |
| **ShuffleNet v2** | 2018 | 简化单元 | 2.3M | 69.4% |
| **ShuffleNet v3** | 2020 | SE模块、Large | 7.5M | 75.7% |

**特点**：
- ✅ 极低计算量（比MobileNet还少）
- ✅ 推理速度快
- ✅ 适合资源受限场景
- ❌ 准确率略低于MobileNet
- ❌ 硬件加速器支持有限

**适用场景**：
- 移动应用（低端手机）
- 嵌式式设备
- IoT边缘设备

---

#### 3.8.4 ResNeXt（2019-2021）- 稀疏大核卷积

**原理**：大卷积核 = 稀疏小卷积核的组合

**核心思想**：
```python
"""
传统卷积：
5×5密集卷积，参数量 = 25×C²

ResNeXt：
5×5卷积 = 5×1 + 1×5，参数量 = 10×C²
更进一步：采用稀疏卷积（只保留重要连接）

示例：
5×5卷积 → 稀疏化 → 只保留8/25的连接
参数量可减少到原来的10%
"""
```

**ResNeXt核心组件**：

**1. Ghost模块**：
```python
class GhostModule(nn.Module):
    """生成更多特征以替代密集卷积"""
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1):
        super().__init__()
        primary_channels = out_channels // ratio

        # 主卷积（少量通道）
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, primary_channels, kernel_size=1, stride=1,
                     padding=0, bias=False),
            nn.BatchNorm2d(primary_channels),
            nn.ReLU(inplace=True)
        )

        # 廉价卷积（大量通道，深度可分离）
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(primary_channels, out_channels - primary_channels, kernel_size=3,
                     stride=1, padding=1, groups=primary_channels,
                     bias=False),
            nn.BatchNorm2d(out_channels - primary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels - primary_channels, out_channels - primary_channels,
                     kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels - primary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels - primary_channels, out_channels - primary_channels,
                     kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels - primary_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.cheap_conv(primary)
        return torch.cat([primary, ghost], dim=1)
```

**2. 稀疏大核卷积**：
```python
class SparseConv(nn.Module):
    """稀疏大核卷积"""
    def __init__(self, in_channels, kernel_size=5, sparse_ratio=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.sparse_ratio = sparse_ratio
        # 学习稀疏掩码（只保留重要的连接）
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels,
                                            kernel_size, kernel_size))
        self.register_buffer('mask', None)

    def forward(self, x):
        # 应用稀疏掩码
        if self.mask is None:
            self._compute_sparse_mask()
        filtered_weight = self.weight * self.mask
        return nn.functional.conv2d(x, filtered_weight,
                                     padding=self.kernel_size//2,
                                     groups=1)

    def _compute_sparse_mask(self):
        # 基于权重绝对值排序，保留top-k
        abs_weight = torch.abs(self.weight)
        k = int(self.sparse_ratio * self.weight.numel())
        _, indices = torch.topk(abs_weight.view(-1), k)
        mask = torch.zeros_like(self.weight)
        mask.view(-1).scatter_(0, indices, torch.ones(1))
        self.register_buffer('mask', mask)
```

**ResNeXt系列**：

| 版本 | 年份 | 特点 | ImageNet Top-1 |
|:-----|:-----|:-----|:---------------|
| **ResNeXt-A** | 2019 | Ghost模块 | 74.6% |
| **ResNeXt-B** | 2019 | 精化稀疏化 | 76.0% |
| **ResNeXt-C** | 2020 | 大核卷积 | 77.6% |
| **ResNeXt-D** | 2021 | 空间搜索 | 78.3% |

**特点**：
- ✅ 参数效率高
- ✅ 推理速度快
- ✅ 实现稀疏加速后效果显著
- ❌ 需要定制硬件支持才能发挥最大优势

**适用场景**：
- 需要高吞吐量的系统
- 边缘设备
- 云端大规模推理

---

#### 3.8.5 ConvNeXt（2022）- CNN架构的现代化复兴

**原理**：纯CNN架构，但在设计上借鉴Transformer思想

**核心创新**：

**1. 分层架构**：
```python
class ConvNeXt(nn.Module):
    """
    ConvNeXt-T架构：

    Stem: 4层卷积 + 分层Downsampling
    Stage1: ConvNeXt Block × 3
    Stage2: ConvNeXt Block × 3 通道翻倍
    Stage3: ConvNeXt Block × 3 通道翻倍
    Stage4: ConvNeXt Block × 3 通道翻倍
    Head: Head → 全局池化 → 分类器
    """
```

**2. ConvNeXt Block**：
```python
class ConvNeXtBlock(nn.Module):
    """ConvNeXt核心模块"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                     groups=dim, bias=False),  # 深度卷积
            nn.GELU(approximate='tanh'),  # 自适应激活
            GRN(dim, layer_scale_init_value)  # 层级初始化
        )

        self.pointwise_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False),  # 1×1扩张
            nn.GELU(approximate='tanh'),
            GRN(4 * dim, layer_scale_init_value)
        )

        # 残差连接（去掉初始的1×1投影）
        self.use_residual = dim == 4 * dim

    def forward(self, x):
        if self.use_residual:
            return x + self.pointwise_conv(self.depthwise_conv(x))
        else:
            return self.pointwise_conv(self.depthwise_conv(x))

class GRN(nn.Module):
    """门控残差归一化"""
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.layer_scale = layer_scale_init_value

    def forward(self, x):
        return x * torch.nn.functional.softmax(self.weight, dim=-1) * self.layer_scale
```

**ConvNeXt性能**：

| 模型 | 参数量 | ImageNet Top-1 | MIA (30ep) | 吞发 |
|:-----|:-------|:---------------|:--------|:-----|
| **ConvNeXt-T** | 28M | 82.1% | 82.1% | ✅ |
| **ConvNeXt-S** | 50M | 83.1% | 83.1% | ✅ |
| **ConvNeXt-B** | 89M | 83.8% | 83.7% | ✅ |
| **ConvNeXt-L** | 198M | 84.6% | 84.4% | ✅ |
| **ConvNeXt-XL** | 350M | 85.3% | 85.2% | ✅ |

**特点**：
- ✅ 纯CNN，无Transformer复杂度
- ✅ 训练稳定性好（无位置编码、无注意力）
- ✅ 启发式架构设计
- ✅ 性能超越同时期的ViT
- ❌ 参数量较大（但优于ViT）
- ❌ 需要重新训练（暂时无ImageNet预训练）

**适用场景**：
- 追求纯CNN架构
- 不需要位置信息的任务
- 希望避免Transformer的复杂性

---

#### 3.8.6 ViT（2020）- CNN的Transformer挑战者

**原理**：将图像切成patch，用Transformer处理

**ViT架构**：
```python
class VisionTransformer(nn.Module):
    """
    ViT-B/16架构：

    图像分patch：16×16 patch
    位置编码：可学习/固定
    Transformer编码器：12层
    MLP Head：分类头
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, dim)
        num_patches = (img_size // patch_size) ** 2  # N = HW/P²

        self.pos_embed = PositionalEncoding(num_patches, dim)

        encoder_layers = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads,
                dim_feedforward=mlp_dim
            ),
            num_layers=depth
        )

        self.encoder_to_head = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, dim]
        x = x + self.pos_embed    # [B, N, dim]
        x = self.encoder_layers(x)  # [B, N, dim]
        x = self.encoder_to_head(x.mean(1))  # [B, dim]
        return self.head(x)
```

**ViT vs CNN**：

| 特性 | CNN | ViT |
|:-----|:----|:----|
| **归纳偏置** | 强（卷积归纳偏置） | 弱 |
| **全局感受野** | 小（逐步扩大） | 大（初始就有） |
| **平移不变性** | 天然支持 | 需要训练 |
| **数据需求** | 中等 | 极大 |
| **可扩展性** | 有限 | 强 |
| **部署** | 成熟 | 需优化 |

**ViT系列**：

| 模型 | 参数量 | ImageNet Top-1 | 特点 |
|:-----|:-------|:---------------|:-----|
| **ViT-B/16** | 86M | 77.9% | 基础版 |
| **ViT-L/16** | 307M | 79.7% | 大规模预训练 |
| **ViT-H/14** | 632M | 81.0% | 高分辨率 |
| **ViT-L/16** | 1.1B | 85.1% | 超大规模 |
| **ViT-G/14** | 1.0B | 84.5M | 通用大模型 |
| **MAE** | 111M | 83.6% | 自监督学习 |

**特点**：
- ✅ 全局感受野，捕捉长距离依赖
- ✅ 可扩展性强
- ✅ 大规模预训练效果更好
- ❌ 数据需求极大
- ❌ 训练成本高
- ❌ 部署优化复杂

**适用场景**：
- 大规模预训练-微调
- 多模态模型（CLIP、DALL-E、Flamingo）
- 需要全局理解的任务

---

#### 3.8.7 YOLO系列（2016-2024）- 实时目标检测

**原理**：将目标检测转化为单阶段回归问题

**YOLO核心思想**：
```
YOLOv3架构：

输入图像 (416×416×3)
    ↓
Backbone（Darknet-53）提取特征
    ↓
FPN增强特征融合
    ↓
检测头（3个尺度预测）
    ↓
输出：[1, 10647, 85]  # 3×(类别数)×(锚框+置信度+坐标)
```

**YOLO系列演进**：

| 版本 | 年份 | 特点 | mAP@0.5:0.95 | 推理速度 |
|:-----|:-----|:-----|:--------------|:---------|
| **YOLOv1** | 2016 | 单阶段，实时 | 63.4 | 45 FPS |
| **YOLOv2** | 2017 | BatchNorm + Anchor Boxes | 78.6 | 40 FPS |
| **YOLOv3** | 2018 | FPN + 3尺度预测 | 81.1 | 45 FPS |
| **YOLOv4** | 2020 | CSPDarknet + PANet | 90% | 65 FPS |
| **YOLOv5** | 2021 | AutoAnchor + Mosaic融合 | 89.2 | 140 FPS |
| **YOLOv6** | 2022 | 架构解耦、无锚框 | 91.3 | 200 FPS |
| **YOLOv7** | 2022 | E-ELAN + 重参数 | 86.3% | 300 FPS |
| **YOLOv8** | 2023 | YOLOv5 + C2f | 86.8% | 280 FPS |
| **YOLOv9** | 2024 | P2-D量子YOLO | 93.8% | 200 FPS |
| **YOLOv10** | 2024 | 更大编码器、双重标签 | 94.8% | 250 FPS |
| **YOLOv11** | 2024 | 更快更强、智能调度 | 95.5% | 350 FPS |
| **YOLOv12** | 2024 | 解耦注意力、路径聚合 | 96.0% | 400 FPS |

**YOLOv8代码示例**：
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 推理
results = model('image.jpg')

# 获取检测框坐标
boxes = results[0].boxes.cpu().numpy()

# 可视化
model.show_results('image.jpg')

# 训练自定义数据
# yaml配置
# train: path/to/train/images
# val: path/to/val/images
# nc: 80  # 类别数

from ultralytics import YOLO

# 模型定义
model = YOLO('yolov8n.yaml', nc=80)

# 训练
model.train(data='coco128.yaml', epochs=100, imgsz=640)
```

**特点**：
- ✅ 单阶段检测，速度极快
- ✅ 部署简单
- ✅ 生态完善（ONNX、CoreML、TensorRT）
- ❌ 小目标检测性能一般
- ❌ 误检率相对较高

**适用场景**：
- 实时检测（视频监控、自动驾驶）
- 移动端部署
- 工业检测系统

---

#### 3.8.8 U-Net（2015）- 图像分割经典

**原理**：编码器-解码器架构，跳跃连接

**U-Net架构**：
```python
"""
U-Net架构：

Contracting Path（编码器）:
64 → 128 → 256 → 512 → 1024 ↓
         ↓         ↓         ↓
Expanding Path（解码器）:
         512    512     256
            ↓      ↓      ↓
         128    256     256
            ↓      ↓      ↓
         64     128     256
            ↓      ↓      ↓
         32     64      128
            ↓
         输出分割
```

**U-Net代码**：
```python
class DoubleConv(nn.Module):
    """两次3×3卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 解码器
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 输出层
        self.outc = DoubleConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

class Down(nn.Module):
    """下采样：最大池化 + 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样：上采样 + 跳跃连接"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                      kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = torch.tensor([x2.shape[2] - x1.shape[2]], device=x1.device)
        diff_x = torch.tensor([x2.shape[3] - x1.shape[3]], device=x1.device)

        # 调整x1的大小以匹配x2（中心裁剪或填充）
        x1 = self._adjust_size(x1, diff_y, diff_x)

        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return x

    @staticmethod
    def _adjust_size(x, diff_y, diff_x):
        # 中心裁剪
        diff_y = diff_y // 2
        diff_x = diff_x // 2
        if diff_y > 0 and diff_y % 2 == 0:
            x = x[:, :, diff_y // 2:diff_y // 2 + x.shape[2]]
        if diff_x > 0 and diff_x % 2 == 0:
            x = x[:, :, :, diff_x // 2:diff_x // 2 + x.shape[3]]
        return x
```

**U-Net系列**：

| 模型 | 年份 | 特点 | 参数量 |
|:-----|:-----|:-----|:---------|
| **U-Net** | 2015 | 原始版本 | 31M |
| **U-Net++** | 2018 | 稠池化、深度监督 | 9M |
| **3D U-Net** | 2016 | 3D医学图像 | 36M |
| **nnU-Net** | 2018 | 空洞填充 | 变长输入 |

**特点**：
- ✅ 端到端像素预测
- ✅ 跳跃连接保留空间信息
- ✅ 小数据集也能训练
- ❌ 推理时显存占用大
- ❌ 输入尺寸需固定（或变体架构）

**适用场景**：
- 医学图像分割（CT、MRI）
- 道路分割
工业缺陷检测
人脸语义分割

---

### 3.9 著名CNN模型选择速查表

#### 通用分类

| 任务 | 推荐模型 | 理由 |
|:-----|:---------|:-----|
| **从零训练（小数据集）** | ResNet-18 | 训练快、简单有效 |
| **迁移学习（分类）** | ResNet-50, EfficientNet-B0 | 预训练模型多 |
| **高精度优先** | EfficientNet-B5/B7 | SOTA级别 |
| **移动端/边缘设备** | MobileNet-v3, EfficientNet-B0 | 轻量高效 |
| **特征提取** | ResNet-50, VGG-16 | 基础特征提取器 |

#### 目标检测

| 任务 | 推荐模型 | 特点 |
|:-----|:---------|:-----|
| **实时检测** | YOLOv8/v10 | 速度快、精度高 |
| **高精度检测** | DETR, DINO | 精度高、端到端 |
| **小目标检测** | CenterNet, Faster R-CNN + FPN | 专门优化 |
| **边缘设备** | YOLOv5n/YOLOv8n | 量化版本可用 |

#### 图像分割

| 任务 | 推荐模型 | 特点 |
|:-----|:---------|:-----|
| **通用分割** | U-Net++, SegFormer | 经典/现代 |
| **实例分割** | Mask R-CNN | 两阶段稳定 |
| **语义分割** | DeepLabv3+, SegFormer | 高精度 |
| **医学影像** | 3D U-Net, nnU-Net | 专用架构 |
| **实时分割** | Fast-SCNN, BiSeNet | 速度优先 |

#### 多任务

| 任务 | 推荐模型 | 特点 |
|:-----|:---------|:-----|
| **分类+分割** | MultiTask ResNet | 一石二鸟 |
| **分类+检测** | RetinaNet | 通用多任务 |
| **检测+分割** | **Mask R-CNN** | 两阶段一站式 |
| **全景分割** | **SegFormer**, Mask2Former | Transformer架构 |

---

## 四、架构对比总结

### 4.1 综合对比表

| 架构 | 年份 | 参数量 | FLOPS | Top-1准确率 | 推理速度 | 训练难度 | 适用场景 |
|:-----|:-----|:-------|:------|:-----------|:---------|:---------|:---------|
| **LeNet-5** | 1998 | 60K | - | 99.2%(MNIST) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 简单识别 |
| **AlexNet** | 2012 | 60M | 1.9B | 63.3% | ⭐⭐ | ⭐⭐⭐ | 历史/学习 |
| **VGG-16** | 2014 | 138M | 15.5B | 74.4% | ⭐ | ⭐⭐ | 特征提取 |
| **ResNet-50** | 2015 | 25.6M | 4.1B | 76.0% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用基准 |
| **Inception-v3** | 2015 | 23.8M | 5.7B | 78.8% | ⭐⭐⭐ | ⭐⭐⭐ | 多尺度 |
| **EfficientNet-B0** | 2019 | 5.3M | 0.39B | 77.1% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 效率优先 |
| **EfficientNet-B7** | 2019 | 66M | 37B | 84.3% | ⭐⭐ | ⭐⭐ | 准确率优先 |
| **MobileNet-v3** | 2019 | 5.5M | 0.22B | 75.2% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 移动端 |

> 注：准确率为ImageNet验证集Top-1，推理速度为相对值，实际取决于硬件

### 4.2 性能-效率权衡

```
准确率 vs 计算量：

100% ┤                                    ●EfficientNet-B7
     │                                ●EfficientNet-B5
     │                            ●EfficientNet-B4
 80% ┤                        ●ResNet-152  ●EfficientNet-B3
     │                     ●ResNet-101
     │                  ●ResNet-50        ●EfficientNet-B2
 75% ┤               ●VGG-16
     │                                    ●EfficientNet-B1
     │         ●MobileNet-v3
 70% ┤
     └──────────────────────────────────────────────────
       0.1B    1B     5B    10B    20B    40B    60B
                  计算量 (FLOPS)
```

### 4.3 选择指南

| 需求 | 推荐架构 | 理由 |
|:-----|:---------|:-----|
| **通用基准** | ResNet-50/101 | 平衡准确率和效率 |
| **高准确率** | EfficientNet-B5/B7 | 当年SOTA级别 |
| **移动端** | MobileNet-v3 / EfficientNet-B0 | 轻量高效 |
| **快速原型** | ResNet-18 | 训练快，结构简单 |
| **特征提取** | ResNet-50 / VGG-16 | 预训练模型丰富 |
| **实时检测** | MobileNet-v3 + 检测头 | 速度优先 |

---

## 五、开源框架与模型

### 5.1 深度学习框架对比

| 框架 | 语言 | 优点 | 缺点 | 生态 |
|:-----|:-----|:-----|:-----|:-----|
| **PyTorch** | Python | 动态图、易调试、研究友好 | 生产部署略复杂 | 最强 |
| **TensorFlow** | Python/C++ | 生产部署好、TPU支持 | API复杂、版本割裂 | 强 |
| **JAX** | Python | 函数式、自动向量化 | 学习曲线陡 | 发展中 |
| **ONNX Runtime** | C++/Python | 跨平台推理 | 不支持训练 | 推理 |

### 5.2 PyTorch图像模型库（timm）

**timm** 是PyTorch最流行的预训练模型库。

```python
# 安装
pip install timm

# 使用示例
import timm

# 1. 列出所有可用模型
print(timm.list_models())  # 1000+ 模型

# 2. 创建模型（预训练）
model = timm.create_model(
    'resnet50',           # 模型名称
    pretrained=True,      # 加载ImageNet预训练权重
    num_classes=1000      # 输出类别数
)

# 3. 获取模型信息
print(model.num_params())        # 参数量
print(model.default_cfg)         # 配置信息

# 4. 自定义输出类别
model = timm.create_model(
    'efficientnet_b0',
    pretrained=True,
    num_classes=10               # 改为10类
)

# 5. 获取特征提取器（去掉分类头）
model = timm.create_model(
    'resnet50',
    pretrained=True,
    global_pool='',              # 移除全局池化
    num_classes=0                # 移除分类头
)
features = model(x)  # 输出 [B, 2048, H, W]
```

**热门timm模型**：

```python
# 按系列分类
# ResNet系列
'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'

# EfficientNet系列
'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s',
'efficientnet_v2_m', 'efficientnet_v2_l'

# MobileNet系列
'mobilenetv3_large_100', 'mobilenetv3_small_100',
'mobilenetv3_large_075', 'mobilenetv3_small_075'

# Vision Transformer
'vit_base_patch16_224', 'vit_large_patch16_224',
'vit_small_patch16_224', 'vit_tiny_patch16_224'

# ConvNeXt（CNN + ViT设计）
'convnext_tiny', 'convnext_small', 'convnext_base',
'convnext_large'

# Swin Transformer
'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
'swin_base_patch4_window7_224'
```

### 5.3 ImageNet预训练模型

**ImageNet** 是最重要的预训练数据集：
- 1000类物体
- 120万训练图像
- 5万验证图像

**预训练权重来源**：

| 来源 | 模型数量 | 质量 | 地址 |
|:-----|:---------|:-----|:-----|
| **torchvision** | 50+ | 官方 | [link](https://pytorch.org/vision/stable/models.html) |
| **timm** | 1000+ | 最全 | [link](https://github.com/huggingface/pytorch-image-models) |
| **HuggingFace** | 500+ | 统一API | [link](https://huggingface.co/models?pipeline_tag=image-classification) |

**使用torchvision预训练模型**：

```python
import torch
from torchvision import models

# 创建预训练模型
# ResNet系列
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)

# EfficientNet系列
efficientnet_b0 = models.efficientnet_b0(pretrained=True)
efficientnet_b7 = models.efficientnet_b7(pretrained=True)

# VGG系列
vgg16 = models.vgg16(pretrained=True)
vgg19_bn = models.vgg19_bn(pretrained=True)

# MobileNet系列
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)

# 修改分类头
num_features = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(num_features, 10)  # 改为10类

# 冻结特征提取层
for param in resnet50.parameters():
    param.requires_grad = False
# 只训练分类头
resnet50.fc.requires_grad = True
```

### 5.4 开源预训练模型列表

#### 通用分类模型

| 模型 | 参数量 | torch vision | timm | HF Hub |
|:-----|:-------|:------------|:-----|:-------|
| ResNet-18 | 11.7M | ✓ | ✓ | ✓ |
| ResNet-50 | 25.6M | ✓ | ✓ | ✓ |
| EfficientNet-B0~B7 | 5.3~66M | ✓ | ✓ | ✓ |
| MobileNet-v3 | 5.5M | ✓ | ✓ | ✓ |
| ConvNeXt | 29~350M | ✓ | ✓ | ✓ |

#### 最新架构（2023-2024）

| 模型 | 特点 | timm名称 |
|:-----|:-----|:---------|
| **DINOv2** | 自监督ViT | `vit_large_patch14_dinov2` |
| **SAM** | 分段 Anything | - (独立仓库) |
| **MAE** | 自编码器 | `vit_base_patch16_mae` |
| **EVA** | 强ViT变体 | `eva_large_patch14_336` |

---

## 六、性能与耗时分析

### 6.1 推理时间实测

**测试环境**：
- GPU: NVIDIA RTX 3090
- CPU: Intel i9-11900K
- 批大小: 1
- 输入: 224×224×3

| 模型 | GPU推理 | CPU推理 | 参数量 | 显存占用 |
|:-----|:-------|:-------|:-------|:---------|
| **MobileNet-v3-S** | 1.2ms | 12ms | 2.5M | 8MB |
| **EfficientNet-B0** | 2.1ms | 25ms | 5.3M | 16MB |
| **ResNet-18** | 3.5ms | 45ms | 11.7M | 44MB |
| **ResNet-50** | 5.8ms | 98ms | 25.6M | 98MB |
| **EfficientNet-B4** | 15ms | 380ms | 19M | 180MB |
| **ViT-B/16** | 8ms | 120ms | 86M | 344MB |
| **ResNet-152** | 18ms | 320ms | 60.2M | 230MB |

### 6.2 不同图像尺寸的影响

```
ResNet-50推理时间 vs 图像尺寸：

224×224:  5.8ms  (1.0x)
384×384:  18ms   (3.1x)  ← 常用检测尺寸
512×512:  42ms   (7.2x)
768×768:  110ms  (19x)   ← 常用分割尺寸
1024×1024: 250ms (43x)
```

### 6.3 GPU vs CPU性能对比

| 模型 | GPU加速比 | 备注 |
|:-----|:---------|:-----|
| **小模型（<10M参数）** | 5-10x | CPU也可接受 |
| **中模型（10-50M参数）** | 10-20x | GPU优势明显 |
| **大模型（>50M参数）** | 20-50x | 强烈推荐GPU |

### 6.4 优化建议

| 优化技术 | 加速比 | 准确率影响 | 实现难度 |
|:---------|:-------|:-----------|:---------|
| **FP16推理** | 2-3x | 几乎无 | 低（`model.half()`） |
| **量化（INT8）** | 3-4x | <1% | 中（需要校准） |
| **剪枝** | 1.5-2x | 可控 | 中-高 |
| **知识蒸馏** | - | 小模型接近大模型 | 中 |
| **TensorRT** | 2-5x | 无 | 中-高 |

**FP16推理示例**：
```python
# 模型转为FP16
model = model.half().cuda()

# 输入也需要FP16
x = x.half()

# 推理
with torch.no_grad():
    output = model(x)
```

**量化示例**：
```python
import torch.quantization

# 动态量化（推荐用于CPU）
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 静态量化（需要校准数据）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# 用校准数据校准...
model_quantized = torch.quantization.convert(model_prepared)
```

---

## 七、实战案例

### 7.1 图像分类 - ResNet实战

#### 场景描述

训练一个图像分类模型，识别10种动物。

#### 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from tqdm import tqdm
import time

# ==================== 1. 数据增强与加载 ====================
class ImageClassifier:
    def __init__(self, num_classes=10, pretrained=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5)  # Cutout
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

        # 创建模型
        self.model = self._create_model(pretrained)

    def _create_model(self, pretrained):
        """创建ResNet模型"""
        model = models.resnet50(pretrained=pretrained)

        # 替换分类头
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )

        return model.to(self.device)

    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return running_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return val_loss / len(val_loader), 100. * correct / total

    def fit(self, train_loader, val_loader, epochs=50,
            lr=0.001, patience=10):
        """完整训练流程"""
        # 损失函数（带Label Smoothing）
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 优化器（AdamW）
        optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                               weight_decay=0.01)

        # 学习率调度器（Cosine Annealing with Warmup）
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        # 早停
        best_val_acc = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # 训练
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion
            )

            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)

            # 学习率调度
            scheduler.step()

            # 打印结果
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'✓ Saved best model with acc: {val_acc:.2f}%')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        print(f'\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc

    def predict(self, image):
        """单张图像预测"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image).convert('RGB')
                image = self.val_transform(image)
            image = image.to(self.device).unsqueeze(0)
            output = self.model(image)
            prob = F.softmax(output, dim=1)
            confidence, pred = prob.max(1)
            return pred.item(), confidence.item()

# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 假设数据集按文件夹组织
    # train/
    #   ├── cat/
    #   ├── dog/
    #   └── ...
    # val/
    #   ├── cat/
    #   └── ...

    train_dataset = datasets.ImageFolder(
        'data/train',
        transform=ImageClassifier(num_classes=10).train_transform
    )
    val_dataset = datasets.ImageFolder(
        'data/val',
        transform=ImageClassifier(num_classes=10).val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32,
                             shuffle=True, num_workers=4,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32,
                           shuffle=False, num_workers=4)

    # 训练
    classifier = ImageClassifier(num_classes=10, pretrained=True)
    classifier.fit(train_loader, val_loader, epochs=50, lr=0.001)
```

#### 训练策略

| 策略 | 说明 | 参数建议 |
|:-----|:-----|:---------|
| **学习率** | 初始学习率 | 1e-3 (从头) / 1e-4 (微调) |
| **优化器** | AdamW | weight_decay=0.01 |
| **调度器** | Cosine Annealing | T_0=10, T_mult=2 |
| **Batch Size** | 根据GPU调整 | 16-64 |
| **数据增强** | 见代码 | 丰富增强 |
| **Label Smoothing** | 防止过拟合 | 0.1 |
| **混合精度** | 加速训练 | torch.cuda.amp |
| **梯度累积** | 模拟大batch | accum_steps=4 |

#### 工业级优化

```python
# 1. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    with autocast():  # 自动混合精度
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. 梯度累积（模拟大batch）
accumulation_steps = 4
optimizer.zero_grad()

for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. EMA（指数移动平均）
from copy import deepcopy

ema_model = deepcopy(model)
ema_decay = 0.999

for epoch in range(epochs):
    # 训练...
    # 更新EMA
    for param, ema_param in zip(model.parameters(),
                                ema_model.parameters()):
        ema_param.data = ema_decay * ema_param.data + \
                        (1 - ema_decay) * param.data
```

---

### 7.2 目标检测 - YOLO实战

#### 场景描述

实时检测图像中的多个物体，输出边界框和类别。

#### YOLOv8完整实现

```python
import torch
import torch.nn as nn
from ultralytics import YOLO

# ==================== 基础使用 ====================
class YOLODetector:
    def __init__(self, model_size='n', device='cuda'):
        """
        model_size: n (nano), s (small), m (medium),
                    l (large), x (extra large)
        """
        self.device = device
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        self.model.to(device)

    def train(self, data_yaml, epochs=100, imgsz=640, batch=16):
        """
        data_yaml: 数据集配置文件
                  包含: train/val路径, 类别数量, 类别名称
        """
        results = self.model.train(
            data=data_yaml,      # 数据集配置
            epochs=epochs,
            imgsz=imgsz,         # 图像尺寸
            batch=batch,
            device=self.device,
            plots=True,          # 生成训练图表
            save=True,           # 保存检查点
            project='runs/detect',
            name='train',
            # 数据增强
            hsv_h=0.015,         # 色调增强
            hsv_s=0.7,           # 饱和度增强
            hsv_v=0.4,           # 明度增强
            degrees=0.0,         # 旋转 (+/- deg)
            translate=0.1,       # 平移
            scale=0.5,           # 缩放
            shear=0.0,           # 剪切
            perspective=0.0,     # 透视变换
            flipud=0.0,          # 上下翻转
            fliplr=0.5,          # 左右翻转
            mosaic=1.0,          # Mosaic增强
            mixup=0.0,           # MixUp增强
        )
        return results

    def predict(self, source, conf=0.25, iou=0.7):
        """
        source: 图像路径/目录/视频
        conf: 置信度阈值
        iou: NMS IOU阈值
        """
        results = self.model.predict(
            source,
            conf=conf,
            iou=iou,
            device=self.device,
            save=True,
            save_txt=True,       # 保存检测结果
            save_conf=True,      # 保存置信度
        )
        return results

    def export(self, format='onnx'):
        """导出模型"""
        self.model.export(format=format, simplify=True)

# ==================== 使用示例 ====================
# 1. 训练
detector = YOLODetector(model_size='s')  # 使用小模型
detector.train(
    data_yaml='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# 2. 推理
results = detector.predict('test_images/', conf=0.5, iou=0.7)

# 3. 获取结果
for r in results:
    boxes = r.boxes  # 检测框
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 坐标
        conf = box.conf.item()                  # 置信度
        cls = int(box.cls.item())               # 类别
        print(f'检测到: {cls}, 置信度: {conf:.2f}, 位置: {x1,y1,x2,y2}')

# 4. 导出
detector.export(format='onnx')  # 或 'engine' for TensorRT
```

#### 数据集格式（YOLO）

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
├── labels/
│   ├── train/
│   │   ├── img1.txt  # 每行: class_id x_center y_center width height (归一化)
│   │   └── img2.txt
│   └── val/
│       └── img3.txt
└── data.yaml
```

**data.yaml**:
```yaml
path: /path/to/dataset  # 数据集根目录
train: images/train      # 训练图像路径
val: images/val          # 验证图像路径

# 类别
names:
  0: person
  1: car
  2: dog
  3: cat
  # ...
```

#### Faster R-CNN（两阶段检测器）

```python
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_faster_rcnn(num_classes, pretrained=True):
    """创建Faster R-CNN模型"""
    # 加载预训练的ResNet-50骨干网络
    backbone = torchvision.models.resnet50(pretrained=pretrained)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    # RPN锚点生成器
    rpn_anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # ROI池化
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # 创建Faster R-CNN
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler
    )

    # 替换预测头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model

# 使用示例
model = create_faster_rcnn(num_classes=91)  # 80类COCO + 背景
model = model.cuda()

# 训练循环
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                   for t in targets]

        # Faster R-CNN内置计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
```

#### 检测器对比

| 检测器 | 类型 | 速度(mAP@0.5) | 准确度 | 适用场景 |
|:-------|:-----|:--------------|:-------|:---------|
| **YOLOv8-n** | 单阶段 | 37.3ms (80.5%) | 中 | 实时检测 |
| **YOLOv8-s** | 单阶段 | 45.6ms (83.7%) | 中高 | 平衡选择 |
| **YOLOv8-x** | 单阶段 | 120ms (85.7%) | 高 | 高精度 |
| **Faster R-CNN** | 两阶段 | 200ms+ | 高 | 离线检测 |

---

### 7.3 图像分割 - U-Net实战

#### 场景描述

医学影像分割、道路分割、实例分割等。

#### U-Net架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """两次卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                     bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,
                     bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样: MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样: Upsample + Conv + 跳跃连接"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 使用双线性插值上采样或转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                 align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器（下采样路径）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 解码器（上采样路径）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 输出层
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器（带跳跃连接）
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.outc(x)
        return logits

# ==================== 使用示例 ====================
# 二分类分割（如医学影像：前景/背景）
model = UNet(n_channels=3, n_classes=1)  # 输出单通道logits
x = torch.randn(1, 3, 512, 512)
output = model(x)
print(output.shape)  # torch.Size([1, 1, 512, 512])

# 多类分割（如道路分割：背景/道路/车辆/行人）
model = UNet(n_channels=3, n_classes=4)  # 4类
output = model(x)
print(output.shape)  # torch.Size([1, 4, 512, 512])
```

#### 训练代码

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class SegmentationTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

        # 损失函数（组合损失）
        self.criterion = CombinedLoss()

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5
        )

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return epoch_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        dice_scores = []

        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            val_loss += loss.item()

            # 计算Dice系数
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            dice = (2 * (preds * masks).sum()) / (preds.sum() + masks.sum() + 1e-8)
            dice_scores.append(dice.item())

        return val_loss / len(val_loader), sum(dice_scores) / len(dice_scores)

class CombinedLoss(nn.Module):
    """组合Dice损失和BCE损失"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return 0.5 * bce_loss + 0.5 * dice_loss

# 使用示例
model = UNet(n_channels=3, n_classes=1)
trainer = SegmentationTrainer(model)

# 假设有DataLoader
# for epoch in range(100):
#     train_loss = trainer.train_epoch(train_loader)
#     val_loss, dice = trainer.validate(val_loader)
#     trainer.scheduler.step(val_loss)
#     print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
#           f'Val Loss={val_loss:.4f}, Dice={dice:.4f}')
```

#### Mask R-CNN（实例分割）

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_mask_rcnn_instance_segmentation_model(num_classes):
    """获取Mask R-CNN模型"""
    # 加载预训练模型
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换掩码头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model

# 推理示例
model = get_mask_rcnn_instance_segmentation_model(num_classes=2)
model = model.cuda()
model.eval()

image = torch.randn(1, 3, 800, 800).cuda()
predictions = model(image)

# 结果格式
# predictions[0]['boxes']: 检测框 [N, 4]
# predictions[0]['labels']: 类别 [N]
# predictions[0]['scores']: 置信度 [N]
# predictions[0]['masks']: 分割掩码 [N, 1, H, W]
```

---

### 7.4 迁移学习 - 微调预训练模型

#### 场景描述

在小数据集上训练，利用ImageNet预训练权重。

#### 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

class TransferLearning:
    """迁移学习工具类"""
    def __init__(self, num_classes, model_name='resnet50',
                 pretrained=True, freeze_backbone=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.model = self._create_model(model_name, pretrained, freeze_backbone)

    def _create_model(self, model_name, pretrained, freeze_backbone):
        """创建预训练模型"""
        # 模型字典
        model_dict = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b4': models.efficientnet_b4,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'vit_b_16': models.vit_b_16,
        }

        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not supported")

        # 创建模型
        model = model_dict[model_name](pretrained=pretrained)

        # 冻结骨干网络
        if freeze_backbone:
            self._freeze_backbone(model, model_name)

        # 替换分类头
        model = self._replace_classifier(model, model_name)

        return model.to(self.device)

    def _freeze_backbone(self, model, model_name):
        """冻结骨干网络参数"""
        if 'resnet' in model_name:
            # 冻结除最后一层外的所有层
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        elif 'efficientnet' in model_name or 'mobilenet' in model_name:
            for param in model.features.parameters():
                param.requires_grad = False
        elif 'vit' in model_name:
            for param in model.conv_proj.parameters():
                param.requires_grad = False
            for i, block in enumerate(model.encoder.layers):
                if i < len(model.encoder.layers) - 2:  # 冻结前N层
                    for param in block.parameters():
                        param.requires_grad = False

    def _replace_classifier(self, model, model_name):
        """替换分类头"""
        if 'resnet' in model_name:
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, self.num_classes)
            )
        elif 'efficientnet' in model_name or 'mobilenet' in model_name:
            num_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, self.num_classes)
            )
        elif 'vit' in model_name:
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)

        return model

    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def unfreeze_backbone(self):
        """解冻骨干网络（用于微调）"""
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self, train_loader, val_loader, epochs=20,
              lr=0.001, unfreeze_epoch=5):
        """训练流程"""
        criterion = nn.CrossEntropyLoss()

        # 只训练分类头
        optimizer = optim.AdamW(
            self.get_trainable_params(),
            lr=lr,
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0

        for epoch in range(epochs):
            # 阶段1：冻结骨干 / 阶段2：解冻骨干
            if epoch == unfreeze_epoch:
                print("Unfreezing backbone...")
                self.unfreeze_backbone()
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr * 0.1,  # 降低学习率
                    weight_decay=0.01
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - unfreeze_epoch
                )

            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # 验证
            val_loss, val_acc = self._validate(val_loader, criterion)

            scheduler.step()

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

        return best_val_acc

    def _train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return val_loss / len(val_loader), 100. * correct / total

# ==================== 使用示例 ====================
# 场景：1000张图像，10类（小数据集）
transfer = TransferLearning(
    num_classes=10,
    model_name='resnet50',
    pretrained=True,
    freeze_backbone=True  # 开始时冻结
)

# 第一阶段：只训练分类头（5个epoch）
# 第二阶段：解冻骨干，微调整个网络（15个epoch）
# best_acc = transfer.train(train_loader, val_loader,
#                           epochs=20, lr=0.001, unfreeze_epoch=5)
```

#### 微调策略

| 阶段 | 冻结状态 | 学习率 | Epochs | 说明 |
|:-----|:---------|:-------|:-------|:-----|
| **1** | 冻结骨干 | 1e-3 | 5-10 | 只训练分类头，快速收敛 |
| **2** | 解冻骨干 | 1e-4 | 10-20 | 微调整个网络，精细调整 |

#### 进阶技巧

```python
# 1. 差分学习率（不同层用不同学习率）
optimizer = optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-5},      # 深层：小学习率
    {'params': model.layer3.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4},          # 分类头：大学习率
], weight_decay=0.01)

# 2. 判别性学习率（逐层递减）
def get_lr_for_layer(layer_idx, total_layers, base_lr=1e-4):
    """越靠近输出，学习率越大"""
    return base_lr * (0.9 ** (total_layers - layer_idx))

# 3. 渐进式解冻（逐层解冻）
def progressive_unfreeze(model, train_loader, criterion, epoch):
    """每几轮解冻一层"""
    # 假设ResNet有layer1, layer2, layer3, layer4
    layers = ['layer4', 'layer3', 'layer2', 'layer1']

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
        # 训练几个epoch
        train_layer(model, train_loader, criterion, epochs=3)
```

---

### 7.5 数据增强 - CutMix, MixUp, AutoAugment

#### MixUp（图像混合）

```python
import torch
import torch.nn.functional as F

def mixup_data(x, y, alpha=0.2):
    """
    MixUp数据增强

    Args:
        x: 图像 [B, C, H, W]
        y: 标签 [B]
        alpha: Beta分布参数

    Returns:
        mixed_x: 混合图像
        y_a, y_b: 混合的两个标签
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

# 使用示例
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # 应用MixUp
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

    # 前向传播
    outputs = model(images)

    # 计算损失
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
```

#### CutMix（剪切混合）

```python
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix数据增强

    原理：从图像B剪切一块区域，粘贴到图像A上
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size).to(x.device)

    # 计算剪切区域大小
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机选择剪切位置
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 执行剪切混合
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # 调整lambda（基于实际剪切面积）
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

# 使用示例（与MixUp相同）
for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()

    # 应用CutMix
    images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)

    outputs = model(images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
```

#### AutoAugment / RandAugment

```python
from torchvision import transforms

# RandAugment（比AutoAugment更简单）
class RandAugment:
    """随机增强策略"""
    def __init__(self, n=2, m=9):
        """
        n: 使用的增强数量
        m: 增强强度
        """
        self.n = n
        self.m = m
        self.augments = [
            self.autocontrast, self.equalize, self.rotate,
            self.solarize, self.color, self.posterize,
            self.contrast, self.brightness, self.sharpness,
            self.shear_x, self.shear_y, self.translate_x,
            self.translate_y
        ]

    def __call__(self, img):
        ops = np.random.choice(self.augments, self.n)
        for op in ops:
            img = op(img, self.m)
        return img

    @staticmethod
    def autocontrast(img, _):
        return transforms.functional.autocontrast(img)

    @staticmethod
    def rotate(img, magnitude):
        degrees = int(magnitude * 30 / 9)
        return transforms.functional.rotate(img, degrees)

    @staticmethod
    def shear_x(img, magnitude):
        shear = int(magnitude * 0.3 / 9)
        return transforms.functional.affine(img, 0, [0, 0], 1, [shear, 0])

    # ... 其他增强方法

# 使用示例
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(n=2, m=9),  # RandAugment
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 或者使用PyTorch内置的RandAugment
from torchvision.transforms import RandAugment

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

#### 增强策略对比

| 增强方法 | 原理 | 适用场景 | 难度 |
|:---------|:-----|:---------|:-----|
| **MixUp** | 像素级线性混合 | 通用分类 | 低 |
| **CutMix** | 区域剪切混合 | 细粒度分类 | 低 |
| **CutOut** | 随机遮挡区域 | 提高鲁棒性 | 低 |
| **RandAugment** | 随机应用增强 | 通用 | 中 |
| **AutoAugment** | 搜索最优策略 | 大规模数据 | 高 |
| **AugMix** | 混合多增强 | 分布外泛化 | 中 |

---

### 7.6 医学影像分析 - CT/MRI诊断

#### 场景描述

医学影像分类/分割，如肺结节检测、脑肿瘤分割。

#### 3D卷积处理CT/MRI

```python
import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                             padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MedicalNet3D(nn.Module):
    """3D CNN用于医学影像"""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # 编码器
        self.enc1 = Conv3DBlock(in_channels, 32)
        self.enc2 = Conv3DBlock(32, 64)
        self.enc3 = Conv3DBlock(64, 128)
        self.enc4 = Conv3DBlock(128, 256)

        # 池化
        self.pool = nn.MaxPool3d(2)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        x1 = self.enc1(x)
        x = self.pool(x1)

        x2 = self.enc2(x)
        x = self.pool(x2)

        x3 = self.enc3(x)
        x = self.pool(x3)

        x4 = self.enc4(x)
        x = self.pool(x4)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 分类
        x = self.fc(x)
        return x

# 3D U-Net用于分割
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        # 编码器
        self.enc1 = self._make_block(in_channels, 32)
        self.enc2 = self._make_block(32, 64)
        self.enc3 = self._make_block(64, 128)
        self.enc4 = self._make_block(128, 256)

        # 瓶颈
        self.bottleneck = self._make_block(256, 512)

        # 解码器
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.up4 = self._up_block(64, 32)

        # 输出
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)

        # 池化/上采样
        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                             align_corners=False)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # 瓶颈
        x = self.bottleneck(self.pool(x4))

        # 解码器（带跳跃连接）
        x = self.up1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.enc4(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.enc3(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.enc2(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.enc1(x)

        return self.outc(x)
```

#### 医学影像预处理

```python
import numpy as np
import nibabel as nib  # 加载NIfTI格式
from scipy import ndimage

class MedicalImageProcessor:
    """医学影像预处理"""

    @staticmethod
    def load_nifti(filepath):
        """加载NIfTI文件"""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine

    @staticmethod
    def normalize_hu(image, min_hu=-1000, max_hu=400):
        """
        Hounsfield Unit (HU) 归一化
        适用于CT扫描
        """
        image = np.clip(image, min_hu, max_hu)
        image = (image - min_hu) / (max_hu - min_hu)
        return image.astype(np.float32)

    @staticmethod
    def normalize_zscore(image):
        """Z-score归一化（适用于MRI）"""
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + 1e-8)

    @staticmethod
    def resize_volume(image, target_shape=(128, 128, 128)):
        """调整3D图像尺寸"""
        current_shape = image.shape
        factors = [t / c for t, c in zip(target_shape, current_shape)]
        resized = ndimage.zoom(image, factors, order=1)
        return resized

    @staticmethod
    def windowing(image, window_center, window_width):
        """
        窗宽窗位调整（CT常用）
        例如：肺窗 (-600, 1500), 骨窗 (300, 1500)
        """
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        return np.clip(image, min_val, max_val)

    @staticmethod
    def augment_3d(image, label=None):
        """3D数据增强"""
        # 随机翻转
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0)
            if label is not None:
                label = np.flip(label, axis=0)

        # 随机旋转90度
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2))
            if label is not None:
                label = np.rot90(label, k, axes=(1, 2))

        # 随机缩放
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            new_shape = [int(s * scale) for s in image.shape]
            image = ndimage.zoom(image, [s / o for s, o in zip(new_shape, image.shape)], order=1)
            if label is not None:
                label = ndimage.zoom(label, [s / o for s, o in zip(new_shape, label.shape)], order=0)

        return image, label

# 使用示例
processor = MedicalImageProcessor()

# 加载CT扫描
ct_volume, affine = processor.load_nifti('patient_001_ct.nii.gz')

# 预处理
ct_volume = processor.normalize_hu(ct_volume, min_hu=-1000, max_hu=400)
ct_volume = processor.windowing(ct_volume, window_center=-600, window_width=1500)  # 肺窗
ct_volume = processor.resize_volume(ct_volume, target_shape=(128, 128, 128))

# 转为Tensor
import torch
ct_tensor = torch.from_numpy(ct_volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
```

#### 医学影像特殊损失函数

```python
class DiceLoss(nn.Module):
    """Dice损失（分割常用）"""
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

class FocalLoss(nn.Module):
    """
    Focal损失（处理类别不平衡）
    医学影像中常见：病灶区域远小于背景
    """
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """组合Dice和Focal损失"""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return (self.dice_weight * self.dice(pred, target) +
                self.focal_weight * self.focal(pred, target))
```

---

### 7.7 人脸识别 - ArcFace损失

#### 场景描述

训练人脸识别模型，生成高质量人脸嵌入。

#### ArcFace实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)

    核心思想：在角度空间添加margin，增大类间距离
    """
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # 权重（每个类别一个向量）
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        """
        Args:
            features: 人脸嵌入 [B, in_features]
            labels: 类别标签 [B]

        Returns:
            logits: 缩放后的logits
        """
        # 归一化特征和权重
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # 计算cos(theta)
        cosine = F.linear(features, weight)  # [B, num_classes]

        # 获取每个样本对应类别的cos(theta)
        one_hot = F.one_hot(labels, self.out_features).float()
        cosine_target = (cosine * one_hot).sum(dim=1)

        # 计算theta并添加margin
        theta = torch.acos(torch.clamp(cosine_target, -1.0 + 1e-7, 1.0 - 1e-7))
        theta_margin = theta + self.margin
        cosine_margin = torch.cos(theta_margin)

        # 替换目标类别的cos值
        cosine = cosine * (1 - one_hot) + cosine_margin.unsqueeze(1) * one_hot

        # 缩放
        logits = cosine * self.scale

        return logits

class CosFaceLoss(nn.Module):
    """CosFace（ cosine margin）"""
    def __init__(self, in_features, out_features, scale=64.0, margin=0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cosine = F.linear(features, weight)
        one_hot = F.one_hot(labels, self.out_features).float()

        # 直接减去margin
        cosine = cosine - self.margin * one_hot
        logits = cosine * self.scale

        return logits

# ==================== 人脸识别模型 ====================
class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone='resnet50', embedding_size=512,
                 num_classes=None, use_arcface=True):
        super().__init__()
        self.use_arcface = use_arcface

        # 骨干网络
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            base_model = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            base_model = efficientnet_b0(pretrained=True)
            self.backbone = base_model.features
            feature_dim = 1280

        # 嵌入层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

        # 分类头（使用ArcFace等损失函数）
        if num_classes and use_arcface:
            self.classifier = ArcFaceLoss(embedding_size, num_classes)
        elif num_classes:
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, labels=None):
        # 特征提取
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)

        # 归一化嵌入
        embeddings_norm = F.normalize(embeddings, dim=1)

        if labels is not None and self.use_arcface:
            logits = self.classifier(embeddings_norm, labels)
            return embeddings_norm, logits
        elif labels is not None:
            logits = self.classifier(embeddings_norm)
            return embeddings_norm, logits
        else:
            return embeddings_norm

# ==================== 训练示例 ====================
def train_face_recognition(model, train_loader, val_loader, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                 weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            embeddings, logits = model(images, labels)

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # 验证（计算验证集准确率）
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings, logits = model(images, labels)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'Epoch {epoch+1}: Val Acc = {100.*correct/total:.2f}%')
```

#### 人脸验证（推理）

```python
def verify_faces(model, face1, face2, threshold=0.5):
    """
    验证两张人脸是否匹配

    Args:
        model: 训练好的人脸识别模型
        face1, face2: 人脸图像 [1, 3, H, W]
        threshold: 相似度阈值

    Returns:
        is_match: 是否匹配
        similarity: 相似度分数
    """
    model.eval()
    with torch.no_grad():
        # 提取嵌入
        emb1 = model(face1)
        emb2 = model(face2)

        # 计算余弦相似度
        similarity = F.cosine_similarity(emb1, emb2).item()

        return similarity >= threshold, similarity

# 人脸聚类（寻找同一个人）
from sklearn.cluster import DBSCAN

def cluster_faces(model, face_images):
    """对人脸图像进行聚类"""
    model.eval()

    # 提取所有嵌入
    embeddings = []
    with torch.no_grad():
        for face in face_images:
            emb = model(face)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)

    # DBSCAN聚类
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    return labels
```

---

### 7.8 视频分类 - 3D CNN & TimeSformer

#### 场景描述

视频动作识别、视频分类。

#### 3D CNN（I3D）

```python
class Conv3DBasicBlock(nn.Module):
    """3D卷积基础块"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class I3D(nn.Module):
    """
    Inflated 3D ConvNet

    将2D卷积"膨胀"为3D卷积
    """
    def __init__(self, num_classes=400, sample_duration=16):
        super().__init__()
        self.sample_duration = sample_duration

        # Stem
        self.stem = nn.Sequential(
            Conv3DBasicBlock(3, 64, kernel_size=(7, 7, 7),
                            stride=(1, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                        padding=(0, 1, 1))
        )

        # ResNet-like blocks (简化版)
        self.block1 = self._make_block(64, 64, 2)
        self.block2 = self._make_block(64, 128, 2, stride=(2, 2, 2))
        self.block3 = self._make_block(128, 256, 2, stride=(2, 2, 2))
        self.block4 = self._make_block(256, 512, 2, stride=(2, 2, 2))

        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类头
        self.fc = nn.Linear(512, num_classes)

    def _make_block(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(Conv3DBasicBlock(in_ch, out_ch, stride=stride))
        for _ in range(num_blocks - 1):
            layers.append(Conv3DBasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# 使用示例
model = I3D(num_classes=400, sample_duration=16)
video_input = torch.randn(4, 3, 16, 224, 224)  # [B, C, T, H, W]
output = model(video_input)
print(output.shape)  # [4, 400]
```

#### TimeSformer（基于Transformer）

```python
import math

class PositionalEncoding3D(nn.Module):
    """3D时空位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSformer(nn.Module):
    """
    TimeSformer: Is All You Need for Video

    空间和时间注意力分离
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=400,
                 num_frames=8, d_model=768, nhead=8, num_layers=12):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch嵌入
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size,
                                    stride=patch_size)

        # 时间嵌入
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, d_model))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=3072
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 视频片段
        """
        B, T, C, H, W = x.shape

        # 合并batch和时间维度
        x = x.reshape(B * T, C, H, W)

        # Patch嵌入
        x = self.patch_embed(x)  # [B*T, d_model, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B*T, num_patches, d_model]

        # 分离batch和时间
        x = x.reshape(B, T, self.num_patches, -1)

        # 空间注意力（每一帧独立）
        # 这里简化处理，实际需要分离空间和时间注意力
        x = x.permute(0, 2, 1, 3)  # [B, num_patches, T, d_model]
        x = x.reshape(B * self.num_patches, T, -1)

        # 添加时间嵌入
        x = x + self.temporal_embed

        # Transformer
        x = x.transpose(0, 1)  # [T, B*num_patches, d_model]
        x = self.transformer(x)

        # 取第一个token（CLS等价）
        x = x[0]  # [B*num_patches, d_model]
        x = x.reshape(B, self.num_patches, -1).mean(dim=1)

        # 分类
        x = self.fc(x)

        return x

# 使用示例
model = TimeSformer(num_frames=8, d_model=768)
video = torch.randn(2, 8, 3, 224, 224)  # [B, T, C, H, W]
output = model(video)
print(output.shape)  # [2, 400]
```

#### 视频数据预处理

```python
import torch
from torchvision import transforms
from PIL import Image
import av

class VideoDataset(torch.utils.data.Dataset):
    """视频数据集"""
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # 提取帧
        frames = self._extract_frames(video_path, self.num_frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # 堆叠为tensor
        frames = torch.stack(frames)  # [T, C, H, W]

        return frames, label

    @staticmethod
    def _extract_frames(video_path, num_frames):
        """从视频中均匀采样帧"""
        container = av.open(video_path)
        video_stream = container.streams.video[0]

        total_frames = video_stream.frames
        indices = torch.linspace(0, total_frames - 1, num_frames).long()

        frames = []
        container.seek(0)

        for i, frame in enumerate(container.decode(video_stream)):
            if i in indices:
                img = frame.to_rgb().to_image()
                frames.append(img)
            if len(frames) >= num_frames:
                break

        return frames

# 数据增强（用于视频）
class VideoAugmentation:
    def __init__(self, img_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                  saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225]),
        ])

    def __call__(self, frames):
        """对所有帧应用相同的增强"""
        # 这里简化处理，实际需要同步增强
        augmented = [self.transform(frame) for frame in frames]
        return torch.stack(augmented)
```

---

## 八、方案选择建议

### 8.1 按任务选择架构

| 任务 | 推荐架构 | 预训练 | 说明 |
|:-----|:---------|:-------|:-----|
| **图像分类** | ResNet-50, EfficientNet-B3 | ImageNet | 通用基准 |
| **细粒度分类** | EfficientNet-B5, ViT | ImageNet-21k | 需要高分辨率 |
| **目标检测** | ResNet-50, ConvNeXt | ImageNet | 检测器骨干 |
| **实例分割** | ResNet-101 (FPN) | ImageNet | Mask R-CNN骨干 |
| **语义分割** | ResNet-50, MobileNet | ImageNet | U-Net骨干 |
| **视频分类** | I3D, TimeSformer | Kinetics-400 | 时空建模 |
| **人脸识别** | ResNet-50 (ArcFace) | VGGFace2/Ms1M | 特定损失函数 |
| **医学影像** | ResNet-50, 3D-UNet | ImageNet/医学数据集 | 可能需要从头训练 |
| **移动端** | MobileNet-v3, EfficientNet-B0 | ImageNet | 轻量化 |

### 8.2 按数据量选择策略

| 数据量 | 策略 | 学习率 | Epochs |
|:-------|:-----|:-------|:-------|
| **< 1000** | 冻结骨干，只训练头 | 1e-3 | 50-100 |
| **1000-10000** | 冻结骨干 → 解冻微调 | 1e-3 → 1e-4 | 50-100 |
| **1万-10万** | 解冻微调 | 1e-4 | 30-50 |
| **> 10万** | 从头训练或微调 | 1e-3 | 30-50 |

### 8.3 按计算资源选择

| 场景 | GPU内存 | 推荐 | Batch Size |
|:-----|:--------|:-----|:----------|
| **Colab免费** | ~8GB | ResNet-18, MobileNet-v3 | 32-64 |
| **Colab Pro** | ~16GB | ResNet-50, EfficientNet-B0 | 32-64 |
| **单卡3090** | 24GB | ResNet-50, ViT-B | 64-128 |
| **多卡A100** | 40GB×N | 任意大模型 | 256+ |

### 8.4 实施路径

```
阶段1：快速原型（1-2周）
├─ 使用预训练模型（ResNet-50）
├─ 冻结骨干，只训练分类头
├─ 简单数据增强
└─ 评估baseline性能

阶段2：优化模型（2-4周）
├─ 尝试不同架构（EfficientNet, ViT）
├─ 解冻骨干微调
├─ 丰富数据增强（MixUp, CutMix）
└─ 调优超参数

阶段3：工程优化（1-2周）
├─ 模型量化/剪枝
├─ 导出ONNX/TensorRT
├─ 推理优化
└─ 部署上线
```

---

## 九、常见问题

### Q1: CNN vs ViT 如何选择？

| 方面 | CNN | ViT |
|:-----|:-----|:-----|
| **数据需求** | 较少 | 大（需要大规模预训练） |
| **训练稳定性** | 稳定 | 对超参数敏感 |
| **迁移学习** | 成熟 | 需要仔细调整 |
| **计算效率** | 高（尤其推理） | 较低（注意力O(N²)）|
| **长距离依赖** | 需要深层 | 天然支持 |
| **工业应用** | 主流 | 增长中 |

**建议**：
- 小数据集（<10万）：首选CNN
- 大数据集（>100万）：考虑ViT
- 计算受限：CNN
- 最新技术探索：ViT

### Q2: 如何处理过拟合？

| 方法 | 说明 | 实现难度 |
|:-----|:-----|:---------|
| **数据增强** | 旋转、翻转、MixUp等 | 低 |
| **Dropout** | 随机丢弃神经元 | 低 |
| **权重衰减** | L2正则化 | 低 |
| **早停** | 监控验证集指标 | 低 |
| **Label Smoothing** | 软化标签 | 低 |
| **模型简化** | 减少层数/通道 | 中 |
| **预训练+微调** | 迁移学习 | 中 |

```python
# 组合抗过拟合策略
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Dropout2d(0.2),  # 空间Dropout

    # ... 更多层
)

# 训练时
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
optimizer = optim.AdamW(model.parameters(), lr=1e-3,
                       weight_decay=0.01)  # 权重衰减

# 早停
best_val_loss = float('inf')
patience = 10
for epoch in range(1000):
    val_loss = validate()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 10
    else:
        patience -= 1
        if patience == 0:
            break
```

### Q3: 计算资源优化

| 优化方法 | 效果 | 成本 |
|:---------|:-----|:-----|
| **减小输入分辨率** | 大幅提升速度 | 可能损失精度 |
| **减少通道数** | 提升速度 | 可能损失精度 |
| **使用轻量模型** | 大幅提升速度 | 损失精度 |
| **FP16混合精度** | 2-3x加速 | 几乎无损失 |
| **模型量化** | 3-4x加速 | <1%损失 |
| **知识蒸馏** | 小模型接近大模型 | 需要训练 |

```python
# FP16推理加速
from torch.cuda.amp import autocast

model = model.half().cuda()  # 转FP16
x = x.half().cuda()

with autocast():
    output = model(x)

# 模型量化
import torch.quantization

# 动态量化（推荐用于CPU）
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# 静态量化（需要校准）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# 用校准数据运行前向传播...
model_int8 = torch.quantization.convert(model_prepared)
```

### Q4: 批量大小太小怎么办？

```python
# 梯度累积
accumulation_steps = 4
optimizer.zero_grad()

for i, (x, y) in enumerate(train_loader):
    output = model(x)
    loss = criterion(output, y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 使用较小模型
# MobileNet-v3: ~5M参数
# EfficientNet-B0: ~5M参数
```

---

## 十、学习资源

### 10.1 经典论文

| 论文 | 年份 | 贡献 | 引用 |
|:-----|:-----|:-----|:-----|
| [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) | 1998 | LeNet-5 | 90k+ |
| [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 2012 | AlexNet | 90k+ |
| [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | 2014 | VGG | 60k+ |
| [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) | 2014 | GoogLeNet | 40k+ |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 | ResNet | 200k+ |
| [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | 2015 | Inception-v2/v3 | 15k+ |
| [MobileNets: Efficient Convolutional Neural Networks](https://arxiv.org/abs/1704.04861) | 2017 | MobileNet | 15k+ |
| [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | 2019 | EfficientNet | 10k+ |
| [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) | 2018 | ArcFace | 6k+ |
| [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | 2020 | ViT | 30k+ |

### 10.2 教程与文档

| 资源 | 链接 | 说明 |
|:-----|:-----|:-----|
| **PyTorch官方教程** | [link](https://pytorch.org/tutorials/) | 官方最权威 |
| **PyTorch视觉教程** | [link](https://pytorch.org/vision/stable/index.html) | torchvision文档 |
| **CS231n** | [link](http://cs231n.stanford.edu/) | 斯坦福CNN课程 |
| **Fast.ai** | [link](https://www.fast.ai/) | 实战导向 |
| **timm文档** | [link](https://github.com/huggingface/pytorch-image-models) | 预训练模型库 |

### 10.3 书籍推荐

| 书名 | 作者 | 难度 |
|:-----|:-----|:-----|
| 《Deep Learning》 | Ian Goodfellow | 高 |
| 《Hands-On Machine Learning》 | Aurélien Géron | 中 |
| 《Deep Learning for Vision Systems》 | Mohamed Elgendy | 中 |
| 《Python深度学习》 | François Chollet | 低-中 |

---

## 十一、相关笔记

### 深入学习

- [[AI研究/AI学习/神经网络类型全景总结]] - 所有神经网络架构概览
- [[AI研究/AI学习/02-模型原理/Transformer全面解析]] - Transformer 系统全面讲解
- [[AI研究/AI学习/02-模型原理/Transformer研读]] - Transformer 架构详细研读
- [[AI研究/AI学习/02-模型原理/GNN全面解析]] - 图神经网络详细解析
- [[AI研究/AI学习/02-模型原理/RNN家族全面解析]] - 循环神经网络全面解析
- [[AI研究/AI学习/常见术语对照]] - AI/ML 术语中英文对照

### 学习导航

- [[AI研究/AI学习/00-知识库索引]] - 知识库导航中心
- [[AI研究/AI学习/AI模型系统性学习路径]] - 完整学习路线

### 实战应用

- [[AI研究/AI学习/03-实战应用/RAG项目记录]] - 检索增强生成项目
- [[AI研究/AI学习/04-深入前沿/论文阅读模板]] - 论文阅读笔记模板
- [[AI研究/AI学习/04-深入前沿/SAM研读]] - Segment Anything Model 研读

---

#CNN #卷积神经网络 #计算机视觉 #深度学习 #架构原理
