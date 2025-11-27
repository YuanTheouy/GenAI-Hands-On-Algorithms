# Normalization (LayerNorm & RMSNorm) 学习笔记

## 背景

- **核心问题**：在深度神经网络训练中，随着层数加深，内部节点激活值的分布会发生变化（Internal Covariate Shift），导致梯度消失或爆炸，且模型对初始化参数过于敏感，训练收敛困难。
- **通用解决方案**：将激活值 $x$ 转换为标准正态分布（或特定分布）$\hat{x}$，再引入可学习参数恢复表达能力。
  - 通用公式：$\hat{x} = \dfrac{x - \text{Stat}(x)}{\text{Scale}(x)}, \quad y = \gamma \odot \hat{x} + \beta$
- **NLP/LLM 领域的现状**：
  - **BatchNorm (BN)**：依赖 Batch 维度的统计量，受限于 Batch Size 大小，且无法有效处理变长序列（Padding 导致的统计量偏移），因此在 NLP 中被弃用。
  - **LayerNorm (LN)**：独立于 Batch，在样本内部（Feature 维度）进行归一化，成为 Transformer 的标准配置。
  - **RMSNorm**：LN 的简化变体，去除了去中心化（Mean Centering）操作，被 LLaMA、PaLM 等现代 LLM 广泛采用。

## 动机

在 Transformer 架构演进中，从 Post-LN 到 Pre-LN，再到 RMSNorm，核心驱动力是 **训练稳定性** 与 **计算效率** 的权衡。

- **LayerNorm vs BatchNorm**：
  - BN 假设同一特征通道在不同样本间同分布，但在 NLP 中，同一位置的 Token 语义差异极大。
  - LN 对每个 Token 独立计算，天然适合 RNN/Transformer 等序列模型，消除了 Training-Inference Gap。
- **RMSNorm vs LayerNorm**：
  - 研究表明，LayerNorm 的成功主要归功于 **Re-scaling（方差归一化）**，而非 **Re-centering（均值平移）**。
  - RMSNorm 省去了计算均值和减去均值的操作，减少了计算开销，且在深层网络中梯度更加稳定。

## 方法 (Implementation)

输入 $x \in \mathbb{R}^{B \times L \times D}$，归一化通常发生在最后一个维度 $D$ 上。

### 1. LayerNorm (LN)
包含去中心化（Mean Centering）和标准化（Scaling）。
1.  **计算均值**：$\mu = \frac{1}{N} \sum_{i=1}^N x_i$
2.  **计算方差**：$\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$ （注意是有偏估计）
3.  **归一化**：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4.  **仿射变换**：$y = \gamma \odot \hat{x} + \beta$

### 2. RMSNorm
仅保留标准化，假设输入均值为 0。
1.  **计算均方根 (RMS)**：$RMS(x) = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}$
2.  **归一化**：$\hat{x} = \frac{x}{RMS(x)}$
3.  **缩放**：$y = \gamma \odot \hat{x}$ （通常移除偏置项 $\beta$）

## Tensor 变化流程 (Data Flow)

假设 $B$=Batch, $L$=SeqLen, $D$=HiddenDim。

1.  **Input**: `[B, L, D]`
2.  **Statistics Calculation (Reduce)**:
    - LN Mean/Var: `x.mean(dim=-1, keepdim=True)` $\to$ `[B, L, 1]`
    - RMSNorm Square Mean: `x.pow(2).mean(dim=-1, keepdim=True)` $\to$ `[B, L, 1]`
3.  **Broadcasting (Alignment)**:
    - `x` (`[B, L, D]`) 与 统计量 (`[B, L, 1]`) 进行 element-wise 运算。
    - 广播机制自动将最后一维 `1` 扩展为 `D`。
4.  **Normalization**:
    - Output $\hat{x}$ Shape: `[B, L, D]`
5.  **Affine Transform**:
    - $\gamma, \beta$ Shape: `[D]`
    - PyTorch 自动广播：`[D]` $\to$ `[1, 1, D]` $\to$ `[B, L, D]` 与 $\hat{x}$ 运算。
    - **Output**: `[B, L, D]`

## 关键细节

### 1. 维度保持 (`keepdim=True`) 的必要性
- **问题**：若不保持维度，规约操作会压缩最后一维，Shape 变为 `[B, L]`。
- **后果**：直接与 `[B, L, D]` 的输入运算时，根据广播规则（右对齐），`[B, L]` 会尝试对齐输入的 `[L, D]`，导致维度错配报错或逻辑错误的列对齐。
- **解法**：必须保持为 `[B, L, 1]`，确保统计量在 Channel/Feature 维度上进行广播复制，实现正确的“减均值”和“除方差”。

### 2. 有偏估计 (Biased) vs 无偏估计 (Unbiased)
- **统计学定义**：无偏估计分母为 $N-1$ (Bessel's correction)，用于通过样本推断总体。
- **DL 场景**：LayerNorm 的对象是当前隐层的所有神经元。这 $N$ 个数值即为我们要标准化的“总体”，而非抽样。因此使用 **有偏估计**（分母为 $N$）。
- **PyTorch 陷阱**：`torch.var()` 和 `torch.std()` 默认是 `unbiased=True`。手动实现 LN 时必须指定 `unbiased=False`（或 `correction=0`），否则会导致与官方 API 输出微小的数值差异。

### 3. 数值稳定性 ($\epsilon$)
- 在分母中加入极小值 $\epsilon$ (e.g., $1e-5$ 或 $1e-6$) 防止除以零。
- **RMSNorm 优化**：工程实现常用 `rsqrt` (reciprocal square root) 算子：$x \times \text{rsqrt}(MeanSquare + \epsilon)$，在 GPU 上比先 `sqrt` 再除法更高效。

### 4. 仿射参数 ($\gamma, \beta$)
- **初始化**：$\gamma$ 初始化为 1，$\beta$ 初始化为 0。这意味着初始状态下 Norm 层是恒等变换（Identity Mapping），利于梯度的反向传播。
- **LLaMA 的做法**：RMSNorm 通常仅保留缩放因子 $\gamma$，去掉了位移因子 $\beta$，进一步精简参数。

## 意义

1.  **解耦与独立性 (Decoupling)**：
    LN/RMSNorm 使得每个样本的梯度计算不再依赖于 Batch 内的其他样本。这对于 Micro-Batch size 训练（如大规模模型分布式训练）和在线推理（Inference with Batch Size=1）至关重要。

2.  **梯度平滑 (Gradient Smoothing)**：
    归一化限制了激活值的幅度，避免了前向传播中的数值溢出，同时平滑了损失函数的几何曲面（Loss Landscape），使得可以使用更大的学习率（Learning Rate）加速收敛。

3.  **计算效率 (RMSNorm Advantage)**：
    RMSNorm 将计算复杂度从 $O(N)$ 略微降低（减少了均值计算和减法），但在大规模集群训练中，这种微小的算子级优化（配合 Fused Kernel）能显著减少显存带宽占用，提升整体 Training Throughput。