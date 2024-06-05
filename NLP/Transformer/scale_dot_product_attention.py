# scaled dot-product attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def attention(query, key, value, mask):
    sqrt_dim_head = query.shape[-1] ** 0.5
    
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head
    
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
        
    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)

        
    

