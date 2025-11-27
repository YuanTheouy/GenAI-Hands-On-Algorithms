import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        # x shape [batch_size , seq lenth, dim]

        # 1 计算mean
        mean = x.mean(dim=-1, keep_dim = True)
        
        # 2 var
        var = x.var(dim=-1, keepdim=True, unbiassed=False)

        # 3 Normalizatioin
        x_norm = (x-mean)/torch.sqrt(var + self.eps)

        return self.gamma*x_norm+self.beta
    
    