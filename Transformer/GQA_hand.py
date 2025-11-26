import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA(nn.Module):
    def __init__(self,embed_dim,n_q_heads,n_kv_heads):
        super().__init__()
        self.n_q_heads=n_q_heads
        self.n_kv_heads=n_kv_heads
        self.head_dim=embed_dim // n_q_heads

        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        self.n_group=n_q_heads // n_kv_heads

        self.W_Q = nn.Linear(embed_dim, n_q_heads*self.head_dim,bias=False)
        self.W_K = nn.Linear(embed_dim, n_kv_heads*self.head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, n_kv_heads*self.head_dim,bias=False)
        self.W_O = nn.Linear(n_q_heads * self.head_dim, embed_dim, bias=False)

    def forward(self,x):
        # x = [B,L,D]
        B, L, _ = x.shape
        
        # 1 投影
        q=self.W_Q(x) # BL(hq,hd)
        k=self.W_K(x) # BL(hkv,hd)
        v=self.W_V(x)

        # 2 Reshape 分头: [B,L,head,Dim] -> [B,head,L,dim]
        q = q.view(B,L,self.n_q_heads,self.head_dim).transpose(1,2) # B hq L hd
        k = k.view(B,L,self.n_kv_heads,self.head_dim).transpose(1,2) # B hkv L hd
        v = v.view(B,L,self.n_kv_heads,self.head_dim).transpose(1,2)
        
        # 3 处理 Group 复制
        # 把 B,n_kv_heads,L,head_dim 变为 B,n_q_heads,L,head_dim ，方便q点积
        k_expanded = torch.repeat_interleave(k,self.n_group,dim=1)
        v_expanded = torch.repeat_interleave(v,self.n_group,dim=1)

        # 4 标准attention计算
        # scores:[B,n_q_heads,L,L]
        scores = torch.matmul(q,k_expanded.transpose(-1,-2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim = -1)

        output = torch.matmul(attn_weights,v_expanded) # B hq L hd

        # reshape outputs
        output = output.transpose(1,2).contiguous().view(B,L,-1) # B L D

        return self.W_O(output)   

        