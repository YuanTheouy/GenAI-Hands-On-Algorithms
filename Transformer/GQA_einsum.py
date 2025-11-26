import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA_Einsum(nn.Module):
    def __init__(self, embed_dim, n_q_heads, n_kv_heads):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_dim // n_q_heads
        
        # 验证整除性
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        self.n_group = n_q_heads // n_kv_heads # Group Size (G)

        self.W_Q = nn.Linear(embed_dim, n_q_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_q_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [Batch, SeqLen, EmbedDim]
        B, L, _ = x.shape

        # 1. 投影 Projection
        # q: [B, L, H_q * D]
        # k, v: [B, L, H_kv * D]
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        # 2. 维度重塑 Reshape (关键步骤)
        # 我们需要显式地把 Query 的头维度拆解为 [KV头数, 组内头数]
        # H_q = H_kv * G
        
        # q: [B, L, H_kv, G, D]  <-- 拆分 H_q
        q = q.view(B, L, self.n_kv_heads, self.n_group, self.head_dim)
        
        # k, v: [B, L, H_kv, D]  <-- 保持原样，不需要物理复制！
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        v = v.view(B, L, self.n_kv_heads, self.head_dim)

        # 3. 计算 Attention Scores (Einsum Magic 1)
        # 语义解析:
        # b: Batch
        # q: Query Length
        # k: Key Length
        # h: KV Heads (组数)
        # g: Group Size (组内头数)
        # d: Head Dim
        
        
        # Q: [b, q, h, g, d]
        # K: [b, k, h, d]    <-- 注意这里没有 g 维度
        # 目标: [b, h, g, q, k] (Batch, Groups, SubHeads, Q_Len, K_Len)
        
        # Einsum 会自动将 K 在 g 维度上进行广播，因为它在方程右边缺失了 g
        scores = torch.einsum('bqhgd, bkhd -> bhgqk', q, k)
        
        # 4. Scaling & Softmax
        scores = scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1) # 在 k (Key Length) 维度归一化

        # 5. 计算 Output (Einsum Magic 2)
        # Weights: [b, h, g, q, k]
        # V:       [b, k, h, d]
        # 目标:     [b, q, h, g, d] (把 h 和 g 还原回去，准备合并)
        
        # 这里 Einsum 再次自动将 V 在 g 维度上广播       
        output = torch.einsum('bhgqk, bkhd -> bqhgd', attn_weights, v)

        # 6. 合并与输出
        # [B, L, H_kv, G, D] -> [B, L, H_kv * G * D] -> [B, L, H_q * D]
        # 注意：这里不需要 transpose 也不需要 contiguous，因为 einsum 输出已经是我们想要的顺序
        output = output.reshape(B, L, self.n_q_heads * self.head_dim)
        
        return self.W_O(output)