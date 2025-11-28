import torch
import torch.nn as nn
import math

class AlibiPositionalBias(nn.Module):
    def __init__(self, num_heads: int, causal:bool=True):
        super().__init__()
        self.num_heads = num_heads
        slopes = self._get_slops(num_heads)
        self.register_buffer("slopes", slopes)
        self.causal = causal

    def _get_slops(self, n:int) -> torch.Tensor:
        """
        计算 ALiBi 的斜率 m。
        参考论文公式或现有实现 (如 HuggingFace Bloom/各路实现)。
        
        Args:
            n: Attention Heads 的数量
        Returns:
            torch.Tensor: 形状为 (n,) 的张量，包含每个 head 的斜率
        """
        start = (2**(-8/n))
        slopes = torch.pow(start,torch.arange(1,n+1,dtype= torch.float32))
        return slopes



    def forward(self, attention_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        计算 Bias 矩阵并加到 attention_scores 上。
        
        Args:
            attention_scores: (Batch, Num_Heads, Seq_Len, Seq_Len) 
                              即 Q @ K.T / sqrt(d) 之后，Softmax 之前的结果
            seq_len: 当前序列长度 (通常取 attention_scores.size(-1))
            
        Returns:
            torch.Tensor: 加上 Bias 后的 scores
        """
        # TODO 2: 生成 ALiBi Bias Matrix 并相加
        # 1. 构造距离矩阵 distance_matrix: 形状 (Seq_Len, Seq_Len)
        #    对于 Causal LM，通常是对角线为 0，左下角为 -1, -2...
        #    注意：input 可能是 padding 过的，但在底层算 bias 时通常只看 max_seq_len 的网格
        
        # 2. 将 distance_matrix 与 self.slopes 相乘
        
        # 3. 调整形状以适配 (Batch, Num_Heads, Seq_Len, Seq_Len) 并执行加法
        
        # 1 构造 distance_matrix
        # attention scores B, h, l, l
        batch_size, num_heads, _ , _ = attention_scores.shape
        device = attention_scores.device

        q_pos = torch.arange(seq_len, dtype = torch.long, device=device)[:,None]
        k_pos = torch.arange(seq_len, dtype = torch.long, device=device)[None,:]

        distance_matrix = k_pos - q_pos # 下三角

        # 2 注入 slope
        # self.slopes 形状假设为 [num_heads]
        # 我们需要把它广播成 [1, num_heads, 1, 1] 以便和 [Batch, H, L, L] 相加
        # distance_matrix 形状 [L, L] -> [1, 1, L, L]
        alibi_bias = distance_matrix.float() * self.slopes.view(1,num_heads,1,1)


        # 4. 处理 Causal Mask (如果需要在这里做)
        # 上三角 (k > q) 的距离是正数，应该 Mask 掉
        if self.causal:
            mask = torch.triu(torch.ones(seq_len,seq_len,device=device),diagonal=1).bool()
            alibi_bias.masked_fill_(mask,-torch.inf)

        return attention_scores+alibi_bias


# --- 简单测试块 (面试时可口述这部分逻辑证明健壮性) ---
if __name__ == "__main__":
    num_heads = 4
    seq_len = 4
    model = AlibiPositionalBias(num_heads)
    
    # 模拟输入
    scores = torch.zeros(1, num_heads, seq_len, seq_len)
    out = model(scores, seq_len)
    
    print("Slopes:", model.slopes)
    # 检查第一个 Head 的左下角是否递减
    print(out[0, 0],f"\n",out[0,2])
     
     
    