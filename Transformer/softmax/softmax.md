# Note: Softmax Operator & Optimization

## 1. Mathematical Definition

Softmax 将 $K$ 维的 Logits 向量 $\mathbf{z}$ 映射为概率分布 $\mathbf{s}$。

对于输入向量 $\mathbf{z} \in \mathbb{R}^K$，第 $i$ 个元素的输出为：

$$s_i = \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

- **约束**：$\sum s_i = 1$ 且 $s_i > 0$。
- **特性**：非逐元素（Element-wise）操作，具有全局耦合性（改变任意 $z_j$ 会影响所有 $s_i$）。

------

## 2. Engineering Implementation: Numerical Stability

### The Overflow Problem

在 FP32 中，$e^x$ 当 $x > 89$ 时会溢出（return `inf`）。直接计算会导致 `inf / inf = NaN`。

### The Shift Trick

利用 Softmax 的**平移不变性**：

$$\frac{e^{z_i}}{\sum e^{z_j}} = \frac{e^{z_i - C}}{\sum e^{z_j - C}} \quad \text{where } C = \max(\mathbf{z})$$

- 令 $z'_i = z_i - \max(\mathbf{z})$，则 $z'_i \le 0$。
- $e^{z'_i} \in (0, 1]$，保证分子分母都不溢出。
- 分母至少包含一项 $e^0=1$，不会除以零。

------

## 3. Theoretical Derivation: Backpropagation

### 3.1 Jacobian Matrix

我们需要求 $\frac{\partial \mathbf{s}}{\partial \mathbf{z}}$。由于分母的耦合性，分为两种情况：

1. $i = j$ (对角线):

   

   $$\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$$

2. $i \neq j$ (非对角线):

   

   $$\frac{\partial s_i}{\partial z_j} = -s_i s_j$$

Jacobian Matrix 形式:

$$J = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$$

### 3.2 Gradient Flow (VJP)

设 Loss 对 Softmax 输出的梯度为 $\mathbf{g} = \nabla_{\mathbf{s}} L$ (即 dout)。

利用链式法则计算 Loss 对输入 $\mathbf{z}$ 的梯度：

$$\begin{aligned} \nabla_{\mathbf{z}} L &= J^T \mathbf{g} \\ &= (\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T) \mathbf{g} \\ &= \mathbf{s} \odot \mathbf{g} - \mathbf{s}(\mathbf{s}^T \mathbf{g}) \end{aligned}$$

- $\mathbf{s} \odot \mathbf{g}$: 逐元素乘积。
- $\mathbf{s}^T \mathbf{g}$: 梯度在概率分布上的期望（标量/Batch wise标量）。
- **物理意义**: 梯度 = 原始梯度 - 梯度期望，再由概率门控。

### 3.3 Fused Gradient (with CrossEntropy)

若 $L$ 为 CrossEntropy Loss，则梯度极度简化：

$$\frac{\partial L}{\partial z_i} = s_i - y_i$$

注意: 在实际训练中，必须使用 LogSoftmax + NLLLoss 或 Fused Operator 以避免精度损失。

------

## 4. Vectorized Implementation (Python/NumPy)

```python
import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.softmax_out = None # Cache for backward
        
    def forward(self, x):
        """
        x: (N, D) Logits
        """
        # 1. Shift Trick for Stability
        # keepdims=True is crucial for broadcasting
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max
        
        # 2. Exp & Normalize
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        
        self.softmax_out = exp_x / sum_exp
        return self.softmax_out

    def backward(self, dout):
        """
        dout: (N, D) Gradient from upper layer
        returns: dx (N, D)
        Formula: dx = s * (dout - sum(s * dout))
        """
        # 1. Dot product term: (s^T * g)
        # Result shape: (N, 1) -> Represents weighted sum of gradients
        sum_s_dout = np.sum(self.softmax_out * dout, axis=1, keepdims=True)
        
        # 2. Broadcast subtraction & Element-wise multiplication
        # (N, D) - (N, 1) triggers broadcasting
        dx = self.softmax_out * (dout - sum_s_dout)
        
        return dx
```

------

## 5. Advanced Insights (Optimization & Acceleration)

### 5.1 The Memory Bottleneck (Standard Softmax)

Softmax 是典型的 **Memory Bound** 算子，而非 Compute Bound。

- **3-Pass 模式**: 标准实现需要对 HBM (显存) 进行 3 次全量读写：
  1. Read $x$ -> Find Max ($m$) -> Write $m$
  2. Read $x, m$ -> Compute Exp & Sum ($d$) -> Write $d$
  3. Read $x, m, d$ -> Compute Div -> Write Output
- **后果**: 当 $N$ (如 LLM Context Length) 很大时，内存带宽成为瓶颈，且中间结果占用大量显存。

### 5.2 Online Softmax (Streaming Calculation)

**核心思想**: 将 3-Pass 优化为 **1-Pass**。在一次遍历中，动态更新局部最大值 $m$ 和局部归一化因子 $d$。

推导公式:

假设当前处理到第 $k$ 个元素，局部状态为 $(m_k, d_k)$。当新元素 $x_{k+1}$ 到来时：

1. 更新最大值:

   $$m_{k+1} = \max(m_k, x_{k+1})$$

2. 更新分母 (Rescaling Trick):

   利用 $e^{x - m_{new}} = e^{x - m_{old}} \cdot e^{m_{old} - m_{new}}$，修正旧的累积和：

   $$d_{k+1} = d_k \cdot e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}$$

**Python Simulation**:

```python
def online_softmax_step(new_val, curr_m, curr_d):
    next_m = max(curr_m, new_val)
    scale = np.exp(curr_m - next_m) # Rescaling factor <= 1
    next_d = curr_d * scale + np.exp(new_val - next_m)
    return next_m, next_d
```

### 5.3 Connection to FlashAttention (Tiling & Fusion)

Online Softmax 是 **FlashAttention** 的数学基石。它允许我们将巨大的 Softmax 矩阵切分为小块 (Tiling)，在 **SRAM (片上高速缓存)** 中完成计算，无需回写 HBM。

在 Attention ($O = \text{softmax}(QK^T)V$) 计算中，我们甚至可以动态更新输出矩阵 $O$：

$$O_{k+1} = \text{diag}(scale) \cdot O_k + P_{k+1} V_{k+1}$$

- $P_{k+1}$: 当前块的局部概率。
- $scale$: 上述的缩放因子 $e^{m_k - m_{k+1}} \cdot (d_k / d_{k+1})$。
- **意义**: 彻底消除了 $O(N^2)$ 的显存读写，使得 LLM 长文本推理成为可能。

### 5.4 Quantization Challenges

- **问题**: Softmax 输出对 $e^x$ 极其敏感，数值范围动态极大（Outliers），难以直接做 INT8 量化。
- **策略**:
  1. **Mixed Precision**: 保持 Softmax 及其前后的 Accumulation 为 FP16/BF16/FP32。
  2. **Non-linear Quantization**: 使用特殊的量化表 (Lookup Table) 拟合指数分布。
  3. **I-BERT / Integer-only**: 使用 `poly` 近似 $e^x$ 实现全整型推理。