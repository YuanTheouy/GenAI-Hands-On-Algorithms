import torch
import torch.nn as nn

class RMSnorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        # x shape batch_size seq_len dim

        # 1 mean squere and dont minus mean
        ms = x.pow(2).mean(dim=-1,keepdim=True)

        # 2 RMS
        x_norm = x * torch.rsqrt(ms + self.eps)

        # 3 scale
        return self.gamma * x_norm
    