# PositionEmbedding2 (Learnable & ALiBi) 学习笔记

## 1. 背景 (Background)

在 RoPE 统治大模型之前，NLP 领域经历了两类重要的位置编码范式：基于参数学习的编码（Learnable）和基于偏置注入的线性注意力偏置（ALiBi）。

* **Learnable Absolute PE (可学习绝对位置)**：
    * **代表模型**：BERT, RoBERTa, GPT-2, GPT-3。
    * **机制**：不使用预设的三角函数，而是将位置 $0, 1, \dots, N$ 视为离散的索引，初始化一个可训练的 Embedding 矩阵 `nn.Embedding(max_len, dim)`。
    * **哲学**："Let the data speak"。不引入人为先验（Inductive Bias），让模型从海量数据中自适应地学习位置间的依赖关系。

* **Learnable Relative PE (可学习相对位置 - T5 Bias)**：
    * **代表模型**：T5, Swin Transformer。
    * **机制**：不给 Input 加任何位置向量。在 Attention $QK^T$ 计算时，加入一个仅与相对距离 $i-j$ 有关的可学习偏置 scalar/vector。
    * **特点**：通常配合**分桶 (Bucketing)** 策略，将远距离截断（如距离 > 128 的共享同一个偏置），以减少参数量并增强泛化。

* **ALiBi (Attention with Linear Biases)**：
    * **代表模型**：BLOOM, MPT。
    * **机制**：在 Attention Score 上直接加上一个**预设的、不可学习的**线性惩罚项。
    * **核心目标**：解决 "Train Short, Test Long" 的外推问题，利用强烈的**局部性归纳偏置 (Locality Inductive Bias)** 迫使模型关注邻近 Token。

---

## 2. 动机 (Motivation)

* **Learnable Absolute PE 的局限 (The Wall)**：
    * **无外推性**：如果定义 `max_len=1024`，推理时遇到第 1025 个 Token 会直接越界报错。模型对未见过的位置 Embedding 一无所知（通常对应参数保持为初始随机值或 0）。
    * **稀疏更新 (Sparse Update)**：在训练数据中，长序列往往少于短序列。导致尾部的位置 Embedding 更新频率远低于头部，学习不充分。

* **ALiBi 的突破**：
    * **平移不变性**：放弃绝对位置，惩罚项仅取决于 $|i-j|$。
    * **平滑外推**：不同于 Sinusoidal 在外推时遇到的 OOD (Out-of-Distribution) 波动，ALiBi 只是施加更大的负数惩罚。Softmax 对极大的负数不敏感（趋近于 0），这相当于一种软性的窗口截断，使得模型在处理未见过的长距离时表现稳定。

---

## 3. 方法 (Implementation)

### 3.1 Learnable Absolute PE 实现
本质是查表 (Lookup Table)。
* **初始化**：`self.wpe = nn.Embedding(config.max_position, config.hidden_size)`
* **Forward**：
    1.  生成索引：`ids = torch.arange(seq_len, device=device)`
    2.  查表：`pos_emb = self.wpe(ids)`
    3.  相加：`output = input_emb + pos_emb`

### 3.2 ALiBi 构造流程
ALiBi 不修改 Embedding，而是修改 Attention Score。
公式：$\text{Attn}(Q, K, V) = \text{Softmax}(QK^T + \mathbf{B})V$

1.  **Slopes 计算 (几何级数)**：
    * 为每个 Attention Head 分配一个斜率 $m$。
    * 公式（针对 $n$ 个 heads）：$m_i = 2^{-8 \cdot i / n}$。
    * 直觉：Slope 大的 Head 关注极短局部；Slope 小的 Head 捕捉长程依赖。

2.  **Distance Matrix 构造**：
    * 利用广播机制生成相对距离矩阵。
    * 对于 Causal LM，通常只关注 $j \le i$ (历史)，距离为 $-(i-j)$。

3.  **Bias 注入**：
    * `Bias = Distance_Matrix * Slope`。
    * 注意：Bias 必须是**负数**（或者是减去正数距离），以此降低远距离 Token 的注意力权重。

---

## 4. Tensor 变化流程 (Data Flow - ALiBi)

假设 $L$=SeqLen, $H$=NumHeads, $B$=BatchSize。

### 4.1 Slopes 生成
1.  **Base**: 计算几何级数比率 $2^{-8/H}$。
2.  **Vector**: `pow(base, arange(1, H+1))` $\to$ `[H]`。
3.  **Reshape**: `view(1, H, 1, 1)` $\to$ 准备广播。

### 4.2 Distance Matrix (Broadcasting)
1.  **Query Pos**: `arange(L)[:, None]` $\to$ `[L, 1]` (行索引 $i$)。
2.  **Key Pos**: `arange(L)[None, :]` $\to$ `[1, L]` (列索引 $j$)。
3.  **Delta**: `Key - Query` $\to$ `[L, L]`。
    * *下三角区域*：$j \le i$，值为 $0, -1, -2, \dots$ (符合 ALiBi 负惩罚需求)。
    * *上三角区域*：$j > i$，值为正数 (会被 Causal Mask 覆盖，忽略)。

### 4.3 Forward 计算
1.  **Scores**: `Q @ K.T` $\to$ `[B, H, L, L]`。
2.  **Bias 计算**: `Distance[1, 1, L, L] * Slopes[1, H, 1, 1]` $\to$ `[1, H, L, L]`。
    * 这里利用了广播机制，自动扩展 Batch 和 Head 维度。
3.  **Add**: `Scores + Bias` (注意 Bias 元素为负数)。
4.  **Causal Mask**: 对上三角填充 `-inf`。
5.  **Softmax**: 归一化。

---

## 5. 关键细节 (Engineering Best Practices)

### 5.1 Learnable PE 的扩容策略 (Hard Extension)
当需要微调一个 BERT/GPT 模型以支持更长上下文时（如 1k $\to$ 2k）：
* **错误做法**：直接 Resize Embedding 矩阵，新增加的 1k 参数随机初始化。这会导致 Loss 爆炸。
* **正确做法 (Copy/Interpolate)**：
    * **复制**：将前 1k 的权重复制给后 1k（周期性假设）。
    * **插值**：对原有权重进行线性或双线性插值，使其适应新的长度。这是一种常用的 Warm-start 技巧。

### 5.2 ALiBi 的高效实现
* **广播优于循环**：绝对不要写 `for head in heads` 循环来加 Bias。利用 Tensor 广播机制，ALiBi 的计算开销相对于 $QK^T$ 是可以忽略不计的。
* **Device 一致性**：在 `forward` 中动态生成 `arange` 或 `ones` 时，务必指定 `device=input.device`，否则会在多卡训练或 GPU 推理时崩溃。
* **FlashAttention 融合**：ALiBi 的 bias 规律非常简单（即 $val = m \cdot |i-j|$）。在 FlashAttention V2 等底层 Kernel 中，可以直接在 SRAM 计算 Attention Score 时实时算出这个 bias 并加上，**无需在显存中显式生成巨大的 `[B, H, L, L]` Bias 矩阵**，从而大幅节省显存带宽。

### 5.3 Learnable Relative PE (T5) 的 Log-binning
* T5 不直接对每个距离建模，而是对距离取对数分桶：
    * 桶索引 $B(d) \approx \lfloor C \cdot \log(d) \rfloor$。
    * 这使得模型对近距离位置敏感（区分距离 1 和 2），对远距离位置不敏感（距离 100 和 101 可能落入同一个桶，共享参数）。这符合人类对距离的感知规律（Weber's Law）。

---

## 6. 意义 (Significance)

1.  **外推能力的里程碑**：
    ALiBi 证明了不需要复杂的旋转数学，仅仅通过施加**静态的线性惩罚**，就能获得极强的外推能力（Train on 2k, Test on 8k+）。它挑战了 "Position Embedding 必须包含语义信息" 的传统观点，强调了 **Inductive Bias** 的重要性。

2.  **Learnable PE 的历史教训**：
    Learnable PE 的兴衰史教会了我们：在数据量极大且任务单一（如掩码建模）时，全参数学习（Fully Parametric）是可行的；但在需要泛化到 OOD 长度（外推）或需要高效推理时，**解析解 (Analytical Solution)** 和**数学先验**（如 RoPE 的旋转、ALiBi 的衰减）往往优于纯粹的学习。

3.  **对现代架构的启示**：
    尽管 LLaMA 选择了 RoPE，但 ALiBi 的思想（根据距离衰减 Attention）被后续许多工作继承，例如 **YaRN** (Yet another RoPE for NTK) 和 **XPos**，它们都在 RoPE 的基础上引入了类似 ALiBi 的衰减系数，以换取更稳定的长文本性能。