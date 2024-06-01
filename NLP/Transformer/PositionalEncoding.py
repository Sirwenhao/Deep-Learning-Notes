# 24/5/31 @author:WH

import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, max_positions, dim_embed, drop_prob):
        super().__init__()
        
        assert dim_embed % 2 == 0
        
        # 生成一个max_positions行一列, 所有值在0到max_positions-1之间的矩阵
        position = torch.arange(max_positions).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / dim_embed))
        
        pe = torch.zeros(max_positions, dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term) # 取出偶数位元素
        # print('pe[:, 0::2]:', pe[:, 0::2][0])
        # print('pe[:, 0::2].shape:', pe[:, 0::2].shape)  # torch.Size([100, 256])
        pe[:, 1::2] = torch.cos(position * div_term) # 取出奇数位元素
        
        # 扩充batch维度
        pe = pe.unsqueeze(0)
        
        # 整个学习阶段，位置信息是不变的，固定为不可学习的数据
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        # 计算每个batch的最大句子长度
        max_squence_length = x.size(1)
        
        # 词向量中添加位置信息
        x = x + self.pe[:, :max_squence_length]
        x = self.dropout(x)
        return x
        
        
if __name__ == "__main__":
    max_len = 100
    embed_size = 512
    drop_prob = 0.5
    PE = PositionalEncoding(max_len, embed_size, drop_prob)
    # print(PE)
