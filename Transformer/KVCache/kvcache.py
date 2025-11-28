import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class KVCacheAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, input_seq_len, d_model).
               在 Decoding 阶段，input_seq_len 通常为 1。
            layer_past: Tuple of (past_k, past_v). 
               Each has shape (batch_size, num_heads, past_seq_len, head_dim).
        
        Returns:
            output: (batch_size, input_seq_len, d_model)
            present: (new_k, new_v), update cache for next step.
        """
        batch_size, input_seq_len, _ = x.size()
        
        # 1. Project Q, K, V
        # 此时形状通常是 (batch_size, input_seq_len, d_model)
        # 需要 view/transpose 成 (batch_size, num_heads, input_seq_len, head_dim)
        
        # x*w_q shape: B,L,D 
        q = self.W_q(x).view(batch_size,input_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(batch_size,input_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(batch_size,input_seq_len,self.num_heads,self.head_dim).permute(0,2,1,3)
        
        # 2. KV Cache Update Logic
        # 如果 layer_past 不为空，说明是 Decoding 阶段（或者 Prefill 的后续 chunk），需要拼接。
        # 如果为空，说明是 Prefill 的起始。
        
        # TODO 2: 更新 KV Cache (present_k, present_v)
        # 注意：这里需要处理 layer_past 是否为 None 的情况
        # present_k = ...
        # present_v = ...
        if layer_past is None:
            present_k = k
            present_v = v
        else:
            present_k = torch.cat((layer_past[0],k),dim=-2)
            present_v = torch.cat((layer_past[1],v),dim=-2)
        
        # 3. Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        # 注意维度匹配：
        # q: (B, H, 1/L_new, D)
        # k (present): (B, H, L_total, D)
        
        # TODO 3: 计算 attention output
        # attn_output = ...

        scores = (q @ present_k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        # prefill 阶段 需要因果注意力， decoding阶段不需要
        if input_seq_len>1:
            mask = torch.triu(torch.ones(input_seq_len,input_seq_len,device=x.device),diagonal=1).bool()
            scores.masked_fill_(mask,-torch.inf)

        attn_weights = F.softmax(scores,dim=-1) 
        attn_output = attn_weights @ present_v
        
        # 4. Output Projection
        # 还原形状 (B, H, L, D) -> (B, L, D*H) -> (B, L, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, input_seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output, (present_k, present_v)

# 测试代码框架（你可以用来验证你的实现）
def test_implementation():

    d_model = 64
    num_heads = 4
    model = KVCacheAttention(d_model, num_heads)
    
    # 1. Simulate Prefill: Input seq_len = 5
    x_prefill = torch.randn(2, 5, d_model)
    out_1, cache_1 = model(x_prefill, layer_past=None)
    
    print(f"Prefill Output Shape: {out_1.shape}") # Should be (2, 5, 64)
    print(f"Cache K Shape: {cache_1[0].shape}")   # Should be (2, 4, 5, 16)
    
    # 2. Simulate Decoding Step 1: Input seq_len = 1
    x_decode = torch.randn(2, 1, d_model)
    out_2, cache_2 = model(x_decode, layer_past=cache_1)
    
    print(f"Decode Output Shape: {out_2.shape}")  # Should be (2, 1, 64)
    print(f"New Cache K Shape: {cache_2[0].shape}") # Should be (2, 4, 6, 16)

test_implementation()