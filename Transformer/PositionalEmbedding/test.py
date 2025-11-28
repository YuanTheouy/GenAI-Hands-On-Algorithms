import torch
seq_len = 4
device = "cuda:0"

q_pos = torch.arange(seq_len, dtype = torch.long, device=device)[:,None]
k_pos = torch.arange(seq_len, dtype = torch.long, device=device)[None,:]

matrix = k_pos - q_pos

print(matrix)
 

 
