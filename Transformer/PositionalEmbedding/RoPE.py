import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        
        # 1. 检查 d_model 是否为偶数
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
            
        # 2. 计算 theta (频率)
        # 提示: torch.arange(0, d_model, 2)
        # power term = (arange / d_model)
        # theta = 1.0 / (base ** power)
        # theta = [0,……,max_len]
        power = torch.arange(0,d_model,2).float()
        theta = 1.0 / (base ** (power/d_model))
        
        # 3. 生成位置索引 t
        # 提示: arange max_len
        t = torch.arange(max_len).float() # arange 是 longint


        # 4. 计算外积 (m * theta)
        # 提示: use torch.outer
        # shape 变为 [max_len, d_model/2]
        freqs = torch.outer(t,theta) # [max_len,d_model/2]
        
        # 5. 拼接频率 (Concatenation)
        # 提示: torch.cat, 在最后一个维度拼接 freqs 和 freqs
        # shape 变为 [max_len, d_model]
        
        emb = torch.cat((freqs,freqs),dim=-1)
        
        # 6. 计算 cos/sin 并调整维度
        # 提示: emb.cos(), 然后使用 [None, :, None, :] 进行维度扩展
        # 目标: [1, max_len, 1, d_model]
        self.cos_cached = emb.cos()[None,:,None,:]
        self.sin_cached = emb.sin()[None,:,None,:]
        # 7. 注册 buffer (千万别忘了这一步)
        # 提示: self.register_buffer(name_str, tensor)
        # 注意: 如果这里不写，上面赋值的 self.cos_cached 只是普通属性
        self.register_buffer("cos_cached",self.cos_cached)
        self.register_buffer("sin_cached",self.sin_cached)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        # 8. 返回切片
        # 提示: 只要前 seq_len 个位置
        return (
            self.cos_cached[:, :seq_len, :,:],
            self.sin_cached[:, :seq_len, :,:]
        )

# 别忘了辅助函数 apply_rope (只需写出公式)
def apply_rope(x, cos, sin):
    # 假设 rotate_half 已经有了
    return x*cos+rotate_half(x)*sin


def rotate_half(x):
    # x shape :B,L,h,d
    d_half = x.shape[-1] // 2
    
    x1=x[:,:,:,:d_half] 
    x2=x[:,:,:,d_half:]

    return torch.cat((-x2,x1),dim=-1)
