# PPO (Proximal Policy Optimization) 学习笔记

## 背景

- **Policy Gradient (PG) 核心流程**：目标是最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$。
  - **梯度公式 (REINFORCE)**：
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]$$
  - **组件**：
    - $\pi_\theta(a|s)$：策略网络（Actor）。
    - $A_t$：优势函数（Advantage），通常用 $Q(s,a) - V(s)$ 表示，衡量动作 $a$ 比平均情况好多少。
- **Vanilla PG 的瓶颈**：
  1. **Sample Efficiency 低**：On-policy 特性导致每采集一批数据只能更新一次参数，随后数据即作废（因为分布变了）。
  2. **训练极不稳定**：学习率 $\alpha$ 难以调节。步长太大导致策略坍塌（Policy Collapse），步长太小导致收敛极慢。

## 动机

在强化学习中，PPO 旨在平衡 **TRPO (Trust Region Policy Optimization)** 的理论稳定性与 **Vanilla PG** 的一阶优化效率。

- **TRPO**：强制约束策略更新前后的 KL 散度 $D_{KL}(\pi_{old} || \pi_{new}) \le \delta$。
  - **优点**：保证策略单调提升（Monotonic Improvement）。
  - **缺点**：需要计算 Hessian 矩阵（二阶优化）或使用共轭梯度法，计算代价极其昂贵。
- **PPO**：将 TRPO 的“硬约束”转化为一阶优化的“软约束”（Clipping 或 Penalty）。
  - **核心思想**：允许在同一批数据上进行多次 Epoch 更新（Importance Sampling），但限制每次更新幅度，防止新旧策略差异过大导致 Importance Weight 方差爆炸。

## 方法 (Implementation)

核心逻辑在于 **Conservative Update (保守更新)** 与 **Importance Sampling (重要性采样)**。

1. **构建比率 (Ratio)**：
   利用重要性采样，使得可以用旧策略 $\pi_{\theta_{old}}$ 采样的数据来计算新策略 $\pi_\theta$ 的梯度：
   $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
2. **Surrogate Objective (代理目标)**：
   PPO-Clip 的目标函数由两部分组成，取最小值以形成悲观下界（Pessimistic Bound）：
   $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
3. **Critic Update**：
   同时训练一个 Value Function $V_\phi(s)$ 来拟合回报，用于计算 Advantage。通常采用 MSE Loss。
4. **总 Loss**：
   $$L_{total} = -L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 S[\pi_\theta](entropy)$$

## 推导与计算流 (Derivation & Flow)

假设 $N$=BatchSize, $T$=TimeSteps。

1. **Sampling (Interaction)**:
   - 使用 $\pi_{\theta_{old}}$ 与环境交互，收集 Trajectory: $\{s_t, a_t, r_t, s_{t+1}, \log \pi_{old}(a_t|s_t)\}_{t=1}^T$。
2. **Advantage Estimation (GAE)**:
   - 计算 TD Error: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
   - 递归计算 GAE: $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$
   - **关键操作**: $\hat{A}_t$ 必须 **Detach**（停止梯度），视作常数标量。
3. **Surrogate Loss Computation**:
   - 前向传播新策略 $\pi_\theta(a_t|s_t)$。
   - 计算 $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{old})$。
   - **Case 1 ($A > 0$)**: 鼓励增加概率，但 $r_t$ 被 Clip 在 $1+\epsilon$。超过则梯度为 0。
   - **Case 2 ($A < 0$)**: 鼓励减少概率，但 $r_t$ 被 Clip 在 $1-\epsilon$。超过则梯度为 0。
4. **Optimization**:
   - 对 $L_{total}$ 执行 Adam 优化。通常在同一批数据上重复 $K$ 个 Epochs (e.g., $K=10$)。

## 关键细节

### 1. 自动微分与 Detach 的工程实现

在数学推导中：
$$\nabla_\theta J(\theta) \approx \sum A_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$
在代码实现（PyTorch）中，为了利用 AutoDiff，构造了等价的 Loss：
$$Loss = - \sum A_t \cdot \log \pi_\theta(a_t|s_t)$$
- **陷阱**：必须确保 $A_t$ 不在计算图中。如果 $A_t$ 来自 Critic 网络且未 `detach()`，梯度会错误地流回 Critic，导致训练崩溃。

### 2. GAE 的偏差-方差权衡 (Bias-Variance Tradeoff)

$\hat{A}_t^{GAE}(\gamma, \lambda)$ 中的 $\lambda$ 是核心调节参数：
- $\lambda = 1$ (Monte Carlo): 无偏估计，但方差极大（受长序列随机性影响）。
- $\lambda = 0$ (TD): 方差小，但偏差大（完全依赖 Critic 的准确性）。
- **PPO 默认**：$\lambda \approx 0.95$，在两者间取得平衡。

### 3. Advantage Normalization

为了配合 Clip 机制中的超参数 $\epsilon$（通常 0.2），必须保证 Advantage 的尺度一致。
- **操作**：在每个 Batch 内执行 $A = \frac{A - \mu}{\sigma + 10^{-8}}$。
- **原因**：如果 $A$ 的数值范围波动很大（如有时是 100，有时是 0.1），固定比例的 Clip 将失效或导致更新步长极不稳定。

### 4. 方差来源 (Instability Sources)

PPO 训练不稳定的主要数学来源：
- **KL 散度过大**：多次 Epoch 后 $\pi_\theta$ 偏离 $\pi_{old}$ 太远，导致 Importance Sampling 权重方差爆炸。
- **Critic 拟合差**：Explained Variance 低，导致 $\hat{A}_t$ 全是噪声。
- **Effective Sample Size 骤减**：大量样本触发 Clip 导致梯度为 0，有效样本数减少，梯度估计方差增大。

## 意义

1. **算法落地的首选 (The Default)**：
   相比 DQN（仅限离散）和 DDPG（超参数敏感），PPO 在连续控制和离散任务上都表现出极强的鲁棒性。它是 ChatGPT (RLHF) 阶段的核心算法。

2. **一阶优化的置信域近似**：
   PPO 用极其廉价的计算成本（仅需 element-wise 的 `min` 和 `clip`），近似实现了二阶优化（TRPO）的几何约束效果。

3. **解耦样本采集与优化**：
   通过 Importance Sampling 机制，打破了“采一次样只能更一次”的限制，极大提升了数据利用率（Data Efficiency）。