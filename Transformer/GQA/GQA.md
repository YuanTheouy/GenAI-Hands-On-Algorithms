# GQA (Grouped Query Attention) 学习笔记



## 背景

- **MHA核心流程**：输入嵌入 $h_t \in \mathbb{R}^d$，经投影矩阵生成查询 $q$、键 $k$、值 $v$，然后分头计算注意力并输出。
  - 公式：
    - $q_t = W_Q h_t, k_t = W_K h_t, v_t = W_V h_t$ （其中 $W_Q \in \mathbb{R}^{H d_h \times d}, W_{K,V} \in \mathbb{R}^{H d_h \times d}$）
    - 分头：$q_t \to [q_{t,1}; ...; q_{t,H}]$ 等
    - 注意力：$o_{t,i} = \sum \text{Softmax}(q_{t,i}^T k_{j,i} / \sqrt{d_h}) v_{j,i}$
    - 输出：$u_t = W_O [o_{t,1}; ...; o_{t,H}]$ （$W_O \in \mathbb{R}^{d \times H d_h}$）
  - 参数：$H$（头数）、$d_h$（每头维度）。
- **MHA的瓶颈**：推理时需缓存所有 KV，占用 $2 \times B \times L \times H \times d_h$ 显存。受限于显存带宽（Memory Bandwidth），KV Cache 成为大模型长序列推理的主要瓶颈。



## 动机

在 Transformer 架构中，GQA 旨在平衡 **MHA (Multi-Head Attention)** 的高质量与 **MQA (Multi-Query Attention)** 的高速度。

- **MHA**：$H_q = H_{kv}$。质量最高，但 KV Cache 显存占用巨大，推理吞吐受限。
- **MQA**：$H_{kv} = 1$。所有 Query 共享同一组 KV。显存占用极低，但模型表达能力下降明显。
- **GQA**：将 Query 的头数 $H_q$ 拆为 $G$ 个组，每个组共享一套 KV 参数。
  - 即 $H_{kv} = H_q / G$。
  - 这是一个折中方案：以牺牲微小的困惑度（Perplexity）为代价，大幅度降低显存占用并提升推理速度。



## 方法 (Implementation)

核心逻辑在于 **Broadcasting (广播/复制)**。

1. **独立投影**：Query 保持 $H_q$ 个头，Key/Value 投影为较少的 $H_{kv}$ 个头。
2. **Reshape & Group**：将 Query 分组，每组包含 $R = H_q / H_{kv}$ 个子头。
3. **Expand (关键步骤)**：将 $K, V$ 在 Head 维度复制 $R$ 次，使其物理形状与 $Q$ 对齐（或者在 CUDA Kernel 层面直接索引读取，避免物理复制）。
4. **计算 Attention**：执行标准的 Scaled Dot-Product Attention。
5. **Concat & Project**：合并所有头，经过 $W_O$ 输出。



## Tensor 变化流程 (Data Flow)

假设 $B$=Batch, $L$=SeqLen, $D$=HeadDim, $H_q$=QueryHeads, $H_{kv}$=KVHeads, $G$=GroupSize ($H_q/H_{kv}$)。

1. **Input**: `[B, L, EmbedDim]`
2. **Projection**:
   - $Q \to$ `[B, L, H_q * D]`
   - $K, V \to$ `[B, L, H_kv * D]`
3. **Split Heads (View + Transpose)**:
   - $Q \to$ `[B, H_q, L, D]`
   - $K, V \to$ `[B, H_kv, L, D]`
4. **Expand / Repeat (Align Layout)**:
   - $K, V \to$ `repeat_interleave(dim=1, repeats=G)` $\to$ `[B, H_q, L, D]`
   - *注：此时 K, V 的形状在逻辑上已经和 Q 完全一致。*
5. **Attention (Dot Product)**:
   - `matmul(Q, K.T)` $\to$ `[B, H_q, L, L]` (Scores)
   - `matmul(Weights, V)` $\to$ `[B, H_q, L, D]` (Output)
6. **Merge Heads (Transpose + Contiguous + View)**:
   - `[B, H_q, L, D]` $\xrightarrow{transpose}$ `[B, L, H_q, D]`
   - $\xrightarrow{contiguous}$ **物理重排内存**
   - $\xrightarrow{view}$ `[B, L, H_q * D]`



## 关键细节

### 1. View 与 Contiguous 的底层机制

- **View 的本质**：只修改 Metadata (Stride/Shape)，不修改 Storage。前提是内存必须符合 C-Contiguous（行优先）。
- **Transpose 的代价**：`x.transpose(1, 2)` 后，虽然逻辑形状变了，但 stride 变得不再连续。
  - 对于 GQA 输出 `[B, H, L, D]`，转置为 `[B, L, H, D]` 后，内存中 $H$ 和 $D$ 之间被 $L$ 维度的数据隔开（Stride Gap）。
  - **必须调用 `contiguous()`**：这会触发 Memory Copy，将分散的数据搬运到一块新的连续内存中，使得 $H$ 和 $D$ 物理相邻，从而允许 `view` 合并它们。



### 2. Torch.matmul 的维度规则

对于高维张量 `[B, H, L, D]`：

- 前面的维度 `[B, H]` 被视为 Batch 维度（并行循环）。
- 仅最后两维参与矩阵乘法。
- **第一步 (Q @ K.T)**: `(..., L, D) @ (..., D, L) -> (..., L, L)`。消去特征维 $D$，得到相似度矩阵。
- **第二步 (Attn @ V)**: `(..., L, L) @ (..., L, D) -> (..., L, D)`。消去源序列长 $L$，得到加权后的特征。



### 3. Einsum 表示 (GQA)

如果不想写复杂的 reshape/expand，可以用 `einsum` 描述 GQA 的逻辑（假设 $H$ 为总头数，$G$ 为组数，$K=H/G$ 为 KV 头数）：

```python
# Q: [B, G, K, L, D] (将头维度拆解为 组 x 组内头)
# K, V: [B, G, L, D] (每个组共享 1 个 KV 头)

# 计算 Scores: 广播 K 的 dim=2 (组内头维度)
scores = torch.einsum('bgkld, bgjd -> bgklj', Q, K) 
```



### 4. 数值稳定性 (Safe Softmax)

代码中 F.softmax(scores, dim=-1) 的底层实现通常采用 Safe Softmax 以防止 $e^{x}$ 上溢：

$$\text{Softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad \text{where } m = \max(x)$$

这是工程实现的标配。



## 意义

1. 突破 Memory Wall：

   LLM 推理是典型的 Memory-Bound（访存密集型） 任务。计算速度通常快于数据从显存搬运到芯片的速度。

   GQA 将 KV Cache 的大小降低了 $G$ 倍（例如 Llama-2-70B 中 $G=8$），这意味着：

   - **更大的 Batch Size**：同样的显存可以处理更多并发请求。
   - **更长的 Context Window**：支持处理更长的文本。
   - **更低的 Latency**：减少了加载 KV 数据所需的时间。

2. 性能与质量的帕累托最优：

   实验表明，GQA 的效果能够逼近 MHA，而速度能够逼近 MQA。它是目前大模型（Llama 3, Mistral 等）的标准配置。