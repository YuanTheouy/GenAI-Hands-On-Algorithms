
# PositionEmbedding1 (Sinusoidal & RoPE) 学习笔记

## 背景

Transformer 模型的 Self-Attention 机制本质上是**置换不变的 (Permutation Invariant)**。为了让模型感知序列顺序，必须引入位置编码。

1.  **Sinusoidal PE (绝对位置)**：
    -   **核心公式**：
        $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) $$
        $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}) $$
    -   **机制**：将位置向量直接**加 (Additive)** 到 Input Embedding 上。
    -   **特点**：利用三角函数的和差化积性质，理论上具备相对位置表达能力（线性关系），但实际深度网络中常被破坏。

2.  **RoPE (旋转位置 - 相对位置)**：
    -   **核心公式** (复数域)：$f(x, m) = x \cdot e^{im\theta}$
    -   **核心目标**：寻找编码函数 $f$，使得 Attention Score 仅与相对距离 $m-n$ 有关：
        $$\langle f(q, m), f(k, n) \rangle = \text{Re}[q \cdot \bar{k} \cdot e^{i(m-n)\theta}]$$
    -   **机制**：在 Attention 计算前，对 Query 和 Key 向量进行**乘性 (Multiplicative)** 旋转变换。



## 动机

-   **Sinusoidal 的局限**：
    -   它是加性编码，位置信息容易随层数加深被稀释。
    -   外推能力 (Extrapolation) 较弱，训练长度之外的泛化性能差。
-   **RoPE 的优势**：
    -   **相对位置归纳偏置 (Inductive Bias)**：通过绝对位置输入实现相对位置的内积效果。
    -   **远程衰减**：随着相对距离增加，内积期望值自然衰减，符合语言模型直觉。
    -   **兼容性**：直接作用于 Linear Project 之后的 Q/K，与 Transformer 结构无缝衔接，是目前 LLaMA、Qwen 等主流大模型的标配。



## 方法 (Implementation)

### 1. 频率计算 (Common Step)
为了数值稳定性，通常在对数空间计算频率 $\theta_i$。
-   公式：$\theta_i = \text{base}^{-2i/d} = \exp(-\frac{2i}{d} \ln(\text{base}))$
-   PyTorch 实现：`torch.exp(torch.arange(0, d, 2) * -(math.log(base) / d))`

### 2. Sinusoidal 构造流程
利用广播机制生成全量矩阵。
1.  **Outer Product**：位置向量 `pos` $[L, 1]$ $\times$ 频率向量 `freq` $[1, D/2]$ $\to$ 角度矩阵 $[L, D/2]$。
2.  **Sin/Cos Interleave**：偶数维填 Sin，奇数维填 Cos。
3.  **Additive**：`output = input + pe`。

### 3. RoPE 构造流程
RoPE 需要将向量视为复数对 $(x_1, x_2)$ 进行旋转。
实数域实现公式：
$$
\begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \cos \theta + \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix} \sin \theta
$$
1.  **Pre-compute Frequencies**：计算 $\cos$ 和 $\sin$ 表。注意为了适配复数对逻辑，需要将频率重复拼接 `cat([freqs, freqs])`。
2.  **Apply Rotation**：执行 `x * cos + rotate_half(x) * sin`。



## Tensor 变化流程 (Data Flow)

假设 $L$=SeqLen, $D$=HeadDim (RoPE作用于head_dim), `base`=10000。

### 1. 频率与角度生成 (Outer Logic)
1.  **Theta**: `[D/2]` (例如 `[0.0, 2.0, ...] / D`)
2.  **Position**: `[L]` (`[0, 1, ..., L-1]`)
3.  **Freqs (Outer Product)**: `torch.outer(pos, theta)` $\to$ `[L, D/2]`
    -   *物理意义*：每个 Token 在每个频带上的旋转总角度。

### 2. RoPE 缓存构建 (Concatenation)
1.  **Emb**: `cat((freqs, freqs), dim=-1)` $\to$ `[L, D]`
    -   *目的*：前一半 $0 \dots d/2$ 对应实部，后一半 $d/2 \dots d$ 对应虚部，它们共享频率。
2.  **Cos/Sin Cache**: `emb.cos()` $\to$ `[L, D]`
3.  **Broadcasting Prep**: `[None, :, None, :]` $\to$ `[1, L, 1, D]`
    -   *适配目标*：Input Q/K 通常为 `[Batch, Seq, Head, Dim]`。

### 3. 运行时旋转 (Forward)
1.  **Input $x$**: `[B, L, H, D]`
2.  **Slicing**: 取 Cache 的前 $L$ 行 $\to$ `[1, L, 1, D]`
3.  **Rotate Half**:
    -   $x_1$: `x[..., :D/2]`
    -   $x_2$: `x[..., D/2:]`
    -   $res$: `cat((-x2, x1), dim=-1)` $\to$ `[B, L, H, D]`
4.  **Final**: `x * cos + res * sin` (Element-wise 乘法)



## 关键细节 (Engineering Best Practices)

### 1. `register_buffer` 的必要性

-   **问题**：PE 不是参数 (`nn.Parameter`)，不需要梯度，但它是模型状态的一部分。如果简单赋值 `self.pe = ...`：
    1.  `model.cuda()` 时，PE 会留在 CPU，导致 Device Mismatch 报错。
    2.  `torch.save()` 时，PE 不会进入 `state_dict`，模型加载后丢失。
-   **解法**：使用 `self.register_buffer('name', tensor)`。PyTorch 会自动管理其 Device 和持久化。

### 2. 广播 (Broadcasting) 与 右对齐原则

-   **原则**：PyTorch 两个 Tensor 运算时，从**最右侧维度**开始匹配。维度必须相等，或其中之一为 1。
-   **案例**：
    -   $x$: `[32, 50, 8, 64]` (B, L, H, D)
    -   $pe$: `[1, 50, 1, 64]`
    -   匹配过程：D(64=64) $\to$ H(8 vs 1, 广播) $\to$ L(50=50) $\to$ B(32 vs 1, 广播)。
-   **技巧**：使用 `x[None, :, None, :]` 替代繁琐的 `view` 或 `unsqueeze` 来快速插入维度。

### 3. `rotate_half` 的切片鲁棒性

-   **Ellipsis (`...`)**：在切片中使用 `...` 代表“前面所有的维度”。
    -   写法：`x[..., :d_half]`
    -   优势：函数变为 **Dimension Agnostic**。无论输入是 3维 `[L, D]` 还是 4维 `[B, L, H, D]` 甚至 5维，代码均无需修改。
-   **逻辑陷阱**：必须确保是 `cat((-x2, x1))`。顺序反了或者负号错了，几何意义就不是旋转，而是乱码。

### 4. 数值精度 (Float vs Half)

-   **计算时**：生成 PE 矩阵时（`arange`, `exp`, `outer`），务必强制使用 `float32`。
    -   原因：`float16` 在计算大数（位置索引）的指数时容易溢出或精度不足，导致长距离 Attention 坍塌。
-   **运行时**：计算完存入 Buffer 后，可以随模型权重转为 `float16` / `bfloat16` 进行推理。



## 意义

1.  **长窗口基础 (Long Context Foundation)**：
    RoPE 的旋转特性使得它比绝对位置编码更容易处理**外推 (Extrapolation)** 问题。结合 **NTK-Aware Scaled RoPE** 或 **YaRN** 等插值缩放技术，可以使 LLaMA 等模型在推理时处理远超训练长度的文本（如 128k context）。

2.  **KV Cache 显存优化前提**：
    RoPE 通常在 KV Cache 存入之前进行。一旦存入 Cache，Key 向量就已经包含了位置信息。这意味着我们在计算 Attention 时，不需要再额外加上位置矩阵，计算图更加高效。

3.  **计算图融合**：
    现代推理框架（vLLM, TensorRT-LLM）通常将 RoPE 的计算与其后的 `Permute` 或 `Copy` 操作融合为一个 Custom Kernel，因为 RoPE 是 Element-wise 操作，带宽是主要瓶颈，融合 Kernel 能大幅提升吞吐。