import torch
import torch.nn as nn
import math

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        # 1、验证偶数
        if d_model%2 != 0:
            raise ValueError(f"Cannot use SPE with odd d_model")
        
        # 2、计算分母频率项
        # exp(-2i/d * ln(10000))
        # torch.arange(0,d,2) 生成 [0,2,4,……,d-2] 即 2i
        div_term = torch.exp(
            torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)
        )

        # 3、位置索引
        # shape [max_len,1]
        # torch.arange(max_len) 获取 [0,1,2,……,max_len-1] shape max_len
        # .float() arange 默认 证书类型
        # .unsqueeze(1) 在第1维度插入一个维度， [max_len,1]
        position = torch.arange(max_len).float().unsqueeze(1)

        # 4、计算PE矩阵
        # 广播机制 [max_len,1]*[d_model/2] -> [max_len,d_model/2]
        pe = torch.zeros(max_len,d_model)

        # 偶数索引 sin    奇数索引 cos
        #  position*div_term 是参数matrix
        # python 切片是 [start:end:step]
        # pe[:,0::2] 第一维度 ： 表示所有行，第二位 0：：2 表示从0开始，步长为2
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        # 5、增加batch维度，，并注册为Buffer
        pe = pe.unsqueeze(0)
        # 使得PE不是Parameter，不参与梯度计算
        # 但会在state_dict中，并随着模型移动到GPU
        self.register_buffer('pe',pe)

    def forward(self,x):
        # shape x: B,L,D
        # 不需要 .to(x.device)，因为 self.pe 已经随模型在对应设备上了
        return x+self.pe[:,:x.size(1),:]