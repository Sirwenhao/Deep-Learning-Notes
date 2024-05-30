# 24/5/30   @author:WH

import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super(TransformerEmbedding, self).__init__()
        self.embed_size = embed_size
        
        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        # 位置编码
        self.position_encoding = self.create_position_encoding(max_len, embed_size)
        
    def create_position_encoding(self, max_len, embed_size):
        position_encoding = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                position_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size))) 
                if i+1 < embed_size:
                    position_encoding[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/embed_size)))
        return position_encoding.unsqueeze(0)
    
    def forward(self, x):
        # x是输入的词索引序列，形状为(batch_size, seq_len)
        
        seq_len = x.size(1)
        
        # 获取词嵌入
        word_embeddings = self.word_embedding(x)
        
        # 添加位置编码
        position_embeddings = self.position_encoding[:, :seq_len, :].to(x.device)
        
        embeddings = word_embeddings + position_embeddings
        return embeddings
        

if __name__ == "__main__":
    
    """
        max_len: 表示所有的输入文本被统一的长度
        embed_size: 是每一个词被向量化后的向量长度
    """
    
    vocab_size = 10000
    embed_size = 512
    max_len = 100
    
    # 实力化
    embedding_layer = TransformerEmbedding(vocab_size, embed_size, max_len)
    
    # 示例输入
    input_indices = torch.randint(0, vocab_size, (32, 20))
    
    output_embeddings = embedding_layer(input_indices)
    
    print(output_embeddings.shape)
    