# KV Cache (Key-Value Cache) 学习笔记

## 背景

- **自回归生成的计算特性**：Transformer 解码是逐 Token 生成的（Token-by-Token）。生成第 $t$ 个 token 时，需要计算 $x_t$ 与历史所有序列 $x_{0:t-1}$ 的相关性。
- **原始计算的冗余性**：若不进行缓存，每次生成新 token，都需要重新映射整个序列 $X_{0:t}$ 来获得 $K$ 和 $V$。
  - **复杂度爆炸**：第 $t$ 步的计算量为 $O(t^2 \cdot d)$。随着序列长度 $T$ 增加，总计算量呈立方级增长。
  - **重复计算**：$K_{0:t-1}$ 和 $V_{0:t-1}$ 的数值在第 $t$ 步与第 $t-1$ 步是完全恒定的，重新计算是对算力的极大浪费。

## 动机

KV Cache 的本质是**以显存空间换取计算时间 (Space-Time Trade-off)**。

- **降维打击**：通过缓存历史层的 Key 和 Value 状态，将 decoding 阶段每一步的 Self-Attention 计算复杂度从 $O(t^2)$ 降低为 $O(t)$。
- **计算范式转移**：
  - 将推理过程从纯粹的 **Compute-Bound (算力受限)** 转化为 **Memory-Bound (带宽受限)**。
  - 这一转变直接决定了后续所有推理优化方向（如 Quantization, PagedAttention, Speculative Decoding）的核心逻辑。

## 方法 (Implementation)

核心逻辑在于**状态的增量更新 (Incremental Update)**。

1.  **Prefill 阶段 (Prompt)**：
    - 输入完整 Prompt，并行计算所有 Token 的 KV。
    - **全量写入**：将计算出的 $K_{prompt}, V_{prompt}$ 存入显存（HBM）。
2.  **Decoding 阶段 (Generation)**：
    - 输入仅为当前 Token $x_t$。
    - **增量计算**：只计算当前的 $k_t, v_t$。
    - **Cache Append**：将 $k_t, v_t$ 拼接到 HBM 中已有的 KV Cache 末尾。
    - **Attention**：读取完整的 Cache ($K_{0:t}, V_{0:t}$) 与当前的 $q_t$ 进行计算。

## Tensor 变化流程 (Data Flow)

假设 $B$=Batch, $L_{past}$=PastSeqLen, $D$=HeadDim, $H$=Heads。

**1. Prefill (一次性)**
- **Input**: `[B, L_prompt, EmbedDim]`
- **Compute**: 生成 `[B, H, L_prompt, D]` 的 K 和 V。
- **Store**: 写入显存，占用 $2 \times B \times H \times L_{prompt} \times D$。

**2. Decoding (第 $t$ 步)**
- **Input**: `[B, 1, EmbedDim]` (当前 Token)
- **Projection**:
  - $q_t \to$ `[B, H, 1, D]`
  - $k_t, v_t \to$ `[B, H, 1, D]`
- **Update Cache (Concatenate)**:
  - Load $K_{past}$ (`[B, H, L_past, D]`) from HBM.
  - $K_{new} = \text{Concat}(K_{past}, k_t) \to$ `[B, H, L_past + 1, D]`
  - Write $K_{new}$ back to HBM (或写入预分配槽位)。
- **Attention (GEMV)**:
  - `matmul(q_t, K_new.T)` $\to$ `[B, H, 1, L_past+1]` (Scores)
  - `matmul(Scores, V_new)` $\to$ `[B, H, 1, D]` (Output)

## 关键细节

### 1. 访存瓶颈 (The Memory Wall)
在 Decoding 阶段，算术强度 (Arithmetic Intensity) 极低。
- **计算量**：$2 \cdot B \cdot H \cdot L \cdot D$ (FLOPs)。
- **访存量**：$2 \cdot B \cdot H \cdot L \cdot D \cdot \text{sizeof(dtype)}$ (Bytes)。
- **比率 (Ops/Byte)**：$\approx 1 / \text{sizeof(dtype)}$。
- **结论**：A100/H100 的 Tensor Core 算力远超 HBM 带宽。推理速度完全取决于**搬运 KV Cache 的速度**，而非计算速度。

### 2. 显存碎片与 PagedAttention
传统的 KV Cache 实现通常要求显存物理连续（Contiguous）。
- **问题**：预分配可能导致显存浪费（OOM），动态分配导致内存碎片。
- **解法 (vLLM)**：引入操作系统中 **虚拟内存 (Virtual Memory)** 的概念。
  - 将 KV Cache 切分为非连续的 Blocks（例如每块存 16 个 token）。
  - 维护一个 **Block Table** 映射逻辑 Token 到物理 Block。
  - 彻底解决了显存碎片问题，大幅提升 Batch Size。

### 3. RoPE 的处理 (Positional Embedding)
KV Cache 存储的是**未旋转**还是**已旋转**的 Key？
- **通常做法**：存储**已旋转 (Rotated)** 的 Key。
- **隐忧**：RoPE 是基于绝对位置 $m$ 的。
  - $k_t$ 在进入 Cache 前，必须根据当前偏移量 `offset` 注入位置信息。
  - 如果使用 ALiBi 等相对位置编码，Attention Score 的计算逻辑需在读取 Cache 后动态调整偏置。

### 4. Prefill 的因果性 (Causality)
虽然 Prefill 是并行计算，但必须施加 **Causal Mask**。
- 若不加 Mask，Position 0 的 Key 会聚合 Position $L$ 的信息。
- 这会导致推理时的 Key 分布与训练时（Teacher Forcing）严重不一致（Distribution Shift），导致生成崩坏。

## 意义

1.  **大模型推理的基石**：
    没有 KV Cache，Transformer 的推理成本将随长度平方增长，ChatGPT 等长文本应用在经济上将不可行。

2.  **架构演进的原动力**：
    KV Cache 巨大的显存占用催生了一系列架构创新：
    - **MQA / GQA**：通过减少 KV 头数，直接物理减少 Cache 体积。
    - **MLA (Multi-Head Latent Attention)**：DeepSeek-V2/V3 提出的架构，通过低秩压缩 KV 进一步降低显存。

3.  **系统优化的核心靶点**：
    推理引擎（TensorRT-LLM, vLLM, SGLang）的 80% 优化工作（如 FlashAttention, Quantization, Offloading）都是围绕如何更高效地**读写、压缩、管理** KV Cache 展开的。