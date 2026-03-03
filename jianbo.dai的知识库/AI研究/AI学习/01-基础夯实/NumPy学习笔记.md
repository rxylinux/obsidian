---
title: NumPy 学习笔记
date: 2026-02-28
tags:
  - NumPy
  - Python
  - 数据处理
status: in-progress
---

# NumPy 学习笔记

> [!info] 学习目标
> - 掌握 NumPy 数组的创建和操作
> - 理解数组广播机制
> - 熟练使用数组索引和切片

> [!tip] 相关笔记
> - **数学基础**：[[AI研究/AI学习/01-基础夯实/数学基础笔记]] - 线性代数基础
> - **学习路径**：[[AI研究/AI学习/AI模型系统性学习路径]] - 第一周内容

---

## 基础概念

### 什么是 NumPy？

NumPy (Numerical Python) 是 Python 中用于科学计算的基础库，提供了：
- 高性能的多维数组对象 `ndarray`
- 用于数组运算的函数
- 线性代数、傅里叶变换等工具

---

## 数组的创建

### 从 Python 列表创建

```python
import numpy as np

# 一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]
print(arr1.shape)  # (5,)
print(arr1.dtype)  # int64

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2.shape)  # (2, 3)
```

### 使用 NumPy 函数创建

```python
# 全零数组
zeros = np.zeros((3, 4))  # shape (3, 4)

# 全一数组
ones = np.ones((2, 3))

# 单位矩阵
eye = np.eye(3)

# 等差数列
arange = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# 线性空间
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# 随机数组
random = np.random.rand(3, 3)  # 均匀分布 [0, 1)
randn = np.random.randn(3, 3)  # 标准正态分布
```

---

## 数组索引与切片

### 基本索引

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 单个元素
print(arr[0, 0])    # 1
print(arr[1, 2])    # 7

# 整行/整列
print(arr[0, :])    # [1, 2, 3, 4] 第一行
print(arr[:, 1])    # [2, 6, 10] 第二列
```

### 切片操作

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 基本切片 [start:stop:step]
print(arr[2:7])      # [2, 3, 4, 5, 6]
print(arr[::2])      # [0, 2, 4, 6, 8] 每隔一个
print(arr[::-1])     # [9, 8, ...] 反转

# 二维切片
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(arr2d[:2, :2])  # [[1, 2], [4, 5]] 左上 2x2
```

### 布尔索引

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 条件筛选
mask = arr > 5
print(arr[mask])      # [6, 7, 8, 9]

# 多条件
arr[(arr > 3) & (arr < 7)]  # [4, 5, 6] 且
arr[(arr < 3) | (arr > 7)]  # [1, 2, 8, 9] 或
```

---

## 数组运算

### 基本运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 四则运算（元素级）
print(a + b)   # [5, 7, 9]
print(a * b)   # [4, 10, 18]
print(a ** 2)  # [1, 4, 9] 幂运算

# 标量运算
print(a + 10)  # [11, 12, 13]
print(a * 2)   # [2, 4, 6]
```

### 广播机制 (Broadcasting)

> [!tip] 广播规则
> 1. 从右到左对齐形状
> 2. 维度为 1 或缺失时可以广播
> 3. 所有维度要么相同，要么为 1

```python
# 标量与数组
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr + 10)
# [[11, 12, 13],
#  [14, 15, 16]]

# 不同形状数组
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([10, 20, 30])      # (3,)
print(a + b)
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]
```

---

## 常用函数

### 统计函数

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 求和
print(arr.sum())           # 21
print(arr.sum(axis=0))     # [5, 7, 9] 列求和
print(arr.sum(axis=1))     # [6, 15] 行求和

# 其他统计
print(arr.mean())          # 平均值
print(arr.std())           # 标准差
print(arr.var())           # 方差
print(arr.min(), arr.max())# 最小值、最大值
print(arr.argmin(), arr.argmax())  # 最小/最大值索引
```

### 形状操作

```python
arr = np.arange(12)

# 改变形状
arr_reshaped = arr.reshape(3, 4)  # 3行4列
arr_flattened = arr_reshaped.flatten()  # 展平

# 转置
arr_t = arr_reshaped.T

# 拼接
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate([a, b])  # [1, 2, 3, 4, 5, 6]
np.vstack([a, b])       # 垂直堆叠
np.hstack([a, b])       # 水平堆叠
```

---

## 线性代数

```python
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])

# 矩阵乘法
print(a @ b)             # 或 np.dot(a, b)
# [[19, 22],
#  [43, 50]]

# 矩阵性质
print(np.linalg.det(a))  # 行列式
print(np.linalg.inv(a))  # 逆矩阵
print(np.linalg.eig(a))  # 特征值和特征向量
```

---

## 练习

### 练习1：创建和操作

```python
# 创建一个 5x5 的随机数组（值在 0-100 之间）
arr = np.random.randint(0, 100, (5, 5))

# 1. 取出前 3 行
# 2. 取出第 2 列
# 3. 找出所有大于 50 的数
# 4. 计算每行的平均值
```

### 练习2：广播应用

```python
# 标准化：将数据归一化到 [0, 1]
data = np.array([10, 20, 30, 40, 50])
normalized = (data - data.min()) / (data.max() - data.min())
```

### 练习3：矩阵运算

```python
# 实现线性回归 y = wx + b 的预测
X = np.array([[1], [2], [3], [4]])  # 特征
w = np.array([2.0])                 # 权重
b = 1.0                             # 偏置
y = X @ w + b                       # 预测
```

---

## 常见问题

### Q: NumPy 数组和 Python 列表的区别？

| 特性 | NumPy 数组 | Python 列表 |
|:-----|:-----------|:-----------|
| 内存 | 连续内存、高效 | 分散内存 |
| 运算 | 向量化、快 | 循环、慢 |
| 元素类型 | 统一类型 | 可混合 |
| 维度 | 支持多维 | 主要一维 |

### Q: 什么时候用 copy，什么时候用 view？

```python
arr = np.array([1, 2, 3, 4, 5])

# view = 引用，修改会影响原数组
view = arr[:2]
view[0] = 100
print(arr)  # [100, 2, 3, 4, 5]

# copy = 副本，独立
copy = arr[:2].copy()
copy[0] = 200
print(arr)  # [100, 2, 3, 4, 5] 不变
```

---

## 学习进度

> [!todo]- 学习清单
> - [ ] 数组创建与属性
> - [ ] 索引与切片
> - [ ] 广播机制
> - [ ] 常用函数
> - [ ] 线性代数基础
> - [ ] 完成练习

---

## 参考资料

- [NumPy 官方文档](https://numpy.org/doc/)
- [NumPy 快速入门](https://numpy.org/doc/stable/user/quickstart.html)
- 100 NumPy Exercises

---

#NumPy #Python #数据处理
