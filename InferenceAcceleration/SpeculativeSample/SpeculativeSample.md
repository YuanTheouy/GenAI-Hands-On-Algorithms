# Speculative Sampling (投机采样) 学习笔记

## 背景

- **LLM 推理的瓶颈**：自回归（Auto-regressive）生成是串行的，即 $x_t$ 必须依赖 $x_{t-1}$。
  - **Memory Bound（显存带宽受限）**：对于大参数量的 Transformer（Target Model），每次 Forward Pass 的主要时间成本在于从显存读取巨大的权重矩阵（Weights）到计算单元。
  - **算力闲置**：由于 GPU 强大的并行计算能力，处理 1 个 Token 和并行处理 $K$ 个 Token 的时间消耗差异极小（只要未达到显存上限）。
- **核心矛盾**：Target Model 强大的并行计算能力被串行的生成逻辑所“闲置”，导致推理延迟（Latency）极高。

## 动机

Speculative Sampling 旨在利用 **“算力过剩”换取“时间”**。

- **核心思想**：
  引入一个轻量级的 **Draft Model ($M_d$)** 快速串行生成多个“猜测”Token，然后让笨重的 **Target Model ($M_t$)** 利用并行计算能力，在一次 Forward Pass 中验证所有猜测。
- **目标**：
  在保证输出分布与 Target Model **严格一致（无损）** 的前提下，显著减少 Target Model 的调用次数，从而提升推理速度。

## 方法 (Implementation)

核心逻辑遵循 **生成 (Draft) -> 验证 (Verify) -> 回退 (Rollback)** 的循环。

1.  **Drafting (猜测)**：
    使用低成本的 Draft Model ($M_d$) 串行生成 $K$ 个 Token 的短序列 $\tilde{x}_{1:K}$。
2.  **Verification (验证)**：
    将当前上下文与这 $K$ 个猜测拼接，一次性输入 Target Model ($M_t$)。
    利用 Causal Mask 的特性，$M_t$ 可以并行计算出每个位置 $i$ 的真实概率分布 $p_i(x)$。
3.  **Rejection Sampling (拒绝采样)**：
    按顺序逐个对比 $M_d$ 的生成概率 $q_i(\tilde{x}_i)$ 与 $M_t$ 的真实概率 $p_i(\tilde{x}_i)$。
    - **接受**：如果通过校验，保留该 Token。
    - **拒绝**：一旦某个 Token 被拒绝，丢弃其后所有猜测，并进行**重采样**。
4.  **Residual Sampling (修正重采样)**：
    在拒绝位置，从修正后的残差分布（Residual Distribution）中采样一个新的 Token，以保证数学上的分布一致性。

## 推导与计算流 (Derivation & Flow)

假设 $M_t$ 为 Target Model，$M_d$ 为 Draft Model，步长为 $K$。

1.  **Draft Phase**:
    $M_d$ 生成 $\tilde{x}_1, \dots, \tilde{x}_K$，并记录每一步的概率 $q_i(\tilde{x}_i)$。

2.  **Verify Phase**:
    $M_t$ 并行计算 $P(x | \text{context}, \tilde{x}_{<i})$，得到 $K$ 个目标分布 $p_1, \dots, p_K$。

3.  **Accept/Reject Loop** (对 $i = 1$ to $K$):
    - 采样随机数 $r \sim U[0, 1]$。
    - **接受准则**:
      $$r < \min\left(1, \frac{p_i(\tilde{x}_i)}{q_i(\tilde{x}_i)}\right)$$
      - 若满足：保留 $\tilde{x}_i$，继续检查 $i+1$。
      - 若不满足：**拒绝** $\tilde{x}_i$ 及其后的所有猜测 $\tilde{x}_{i+1:K}$。

4.  **Resample & Rollback (关键修正)**:
    - 若在第 $i$ 步拒绝，需要从**修正分布**中采样一个新的 Token $x'_i$：
      $$x'_i \sim \text{norm}(\max(0, p_i(x) - q_i(x)))$$
    - **终止**：将 $x'_i$ 加入序列，本轮循环结束（共生成 $i$ 个 Token）。
    - 若全部 $K$ 个都接受：额外多生成一个 Token $x_{K+1} \sim p_{K+1}(x)$（共生成 $K+1$ 个）。

## 关键细节

### 1. 无损性证明 (Lossless Guarantee)

为何简单的拒绝+修正能保证最终分布 $P(\text{out})$ 严格等于 $M_t$ 的分布 $p(x)$？
利用全概率公式分解：
$$P(\text{out}=x) = \underbrace{P(\text{accept } x)}_{\min(q(x), p(x))} + \underbrace{P(\text{reject}) \cdot P(\text{resample } x)}_{\beta \cdot \frac{\max(0, p(x)-q(x))}{\beta}}$$
$$= \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$
- **直觉**：接受步骤保留了 $p$ 和 $q$ 的公共部分（交集），修正步骤精确填补了 $p$ 比 $q$ 高出的部分（差集）。

### 2. 加速比公式 (Speedup Theory)

期望加速比 $E[\text{speedup}]$ 由接受率 $\alpha$ 和计算成本比率决定：
$$E[\text{speedup}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \times \frac{C_t}{C_t + K \cdot C_d}$$
- $\alpha$：平均接受率（Acceptance Rate）。
- $C_t$：Target Model 单次 Forward 时间。
- $C_d$：Draft Model 单次生成时间。
- **Trade-off**：$K$ 越大，潜在收益越高，但 Overhead ($K \cdot C_d$) 也越大。若 $\alpha$ 过低，不仅无加速，反而会变慢。

### 3. KV Cache 的管理

工程实现中最棘手的部分：
- **污染问题**：Target Model 在验证时，将猜测的 $K$ 个 Token 的 Key/Value 写入了 Cache。
- **回退操作**：一旦在第 $i$ 步拒绝，显存中第 $i$ 步之后的 KV Cache 变为脏数据。必须支持高效的 **Cache Truncation** 或使用 PagedAttention 机制丢弃无效页，否则显存操作的开销会抵消算法收益。

### 4. 为什么是 $\max(0, p-q)$？

这是为了处理 **Over-confidence** 和 **Under-confidence** 的非对称性：
- 当 $q(x) > p(x)$ (Over-confident)：通过拒绝率 $\frac{p}{q} < 1$ 剔除多余概率。
- 当 $q(x) < p(x)$ (Under-confident)：接受率全为 1，但总概率不够，必须通过拒绝后的 Resample 分布（即 $p-q$ 的正差值部分）补齐缺口。

## 意义

1.  **无需重新训练 (Plug-and-Play)**：
    不需要修改 Target Model 的权重，适用于任何现有的预训练大模型，只要能找到一个对应的“小模型”即可。
2.  **打破 Memory Wall**：
    将推理过程从 Memory Bound 向 Compute Bound 转移，更充分地榨干 GPU 的 Tensor Cores 性能。
3.  **精确性 (Exactness)**：
    不同于 Quantization（量化）或 Pruning（剪枝）等有损压缩，Speculative Sampling 是**数学上严格无损**的加速算法。