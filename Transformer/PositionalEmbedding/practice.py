import torch

'''
题目 1：广播加法 (Level: 简单)
场景： 给每一个 Token 的 Embedding 加上一个特定的偏置向量。

输入 x: 形状 [Batch, Seq_Len, Dim]

输入 bias: 形状 [Dim]

要求： 返回 x + bias。

难点： 甚至不需要 unsqueeze，想一想为什么？
'''

# x shape B L D
# bias shape D
# return x + bias

def tokenAddBias(x,bias):
    return x + bias  # 因为tensor从右向左匹配

# 正规版
def tokenAddBias(x,bias):
    bias = bias[None,None,:]
    return x+bias


'''
题目 2：多头注意力拆分 (Level: 核心/中等)
场景： 这是 Transformer 中最经典的一步。把宽向量拆成多头。

输入 x: 形状 [Batch, Seq_Len, Dim]。假设 Dim = Heads * Head_Dim。

要求： 将其转换为 [Batch, Heads, Seq_Len, Head_Dim]。

提示： 需要用到 view (或 reshape) 和 transpose (或 permute)。注意顺序！
'''

# x shape B L D, D = h * d

def MHA_split(x,heads):
    B,L,D = x.shape
    h = heads
    d = D // h
    x = x.view(B,L,h,d) # BLD -> BLhd
    x = x.permute(0,2,1,3) # BLhd -> BhLd
    return x


'''
题目 3：因果掩码 (Causal Mask) (Level: 进阶/切片)
场景： GPT 生成时，不能看后面的词。需要生成一个下三角矩阵。

输入 size: 一个整数，比如 4。

要求： 生成一个 [1, 1, size, size] 的矩阵。对角线及以下是 0 (或1)，对角线以上是 -inf。

提示： torch.triu (上三角) 或 torch.tril (下三角)。利用 masked_fill。

'''

def CausalMask(size):
    x = torch.tril()
    # 这个我不行了
    pass

def CausalMask(size):
    # 1. 生成全1矩阵
    # 2. 取出右上角(不包含对角线)，作为"要遮挡的部分"
    # 3. 转为 bool 或者直接用 1/0 做 mask
    # 4. 填充 -inf
    matrix = torch.ones(size,size)  # 全1矩阵
    matrix_triu=torch.triu(matrix,diagonal=1) # 对角线及之下全0,其余全1,用于写-inf
    mask = matrix_triu.bool() #用bool来当mask
    masked_matrix=matrix.masked_fill(mask, -torch.inf) # matrix 现在下三角全1,其余全-inf
    return masked_matrix[None,None,:,:]


def CausalMask(size):
    mask = torch.zeros(1,1,size,size)

    ones = torch.ones(size,size,dtype=bool) # 全1 bool矩阵
    mask_cond = torch.triu(ones,diagonal=1) # 对角线及之下为0，其余为1
    
    mask.masked_fill(mask_cond, -float('inf'))

    return mask


'''
3. 手撕挑战：RoPE 的两个关键函数请尝试实现以下两个函数。
挑战 A：rotate_half(x)这是 RoPE 的灵魂。
将输入 $x$ 的后半部分取负，并交换前后两部分。
输入: x [Batch, Seq, Head, Dim] (假设 Dim 是偶数)
逻辑: 把 Dim 拆成两半 x1 和 x2。
目标: 返回拼接后的 [-x2, x1]。
提示: 使用 x[..., :half] 切片。

挑战 B：apply_rope(x, cos, sin)
输入:x: [Batch, Seq, Head, Dim] (Query 或 Key)
cos, sin: [1, Seq, 1, Dim] (预计算好的位置编码，已经广播好了)
公式: $x \cdot \cos + \text{rotate\_half}(x) \cdot \sin$
要求: 直接写出这一行代码。
'''


def rotate_half(x):
    # shape x: B,L,h,d
    B,L,h,d = x.shape
    d_half = d // 2 
    
    x1 = x[:,:,:,:d_half-1]
    x2 = x[:,:,:,d_half:]
    # 拼接是啥公式啊？我不太明白？
    x_rotated = torch.cat(x1,x2,dim=3)

    return x_rotated

def apply_rope(x,cos,sin):
    # shape x:BLhd cos,sin:1,L,1d
    return x*cos + rotate_half(x)*sin


def rotate_half(x):
    d_half = x.shape[-1] // 2
    # 按这个意思，这个切片是左闭右开的？
    x1 = x[:,:,:,:d_half]
    x2 = x[:,:,:,d_half:]

    return torch.cat((-x2,x1),dim=-1) #放一个元组，然后写拼接的维度
