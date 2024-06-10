import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
  def __init__(self, dim_embed, epsilon=1e-6):
    super(LayerNormalization, self).__init__()
    self.epsilon = epsilon
    self.gamma = nn.Parameter(torch.ones(dim_embed))
    self.beta = nn.Parameter(torch.zeros(dim_embed))
    
  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    normalized_x = (x - mean) / (std + self.epsilon)
    return self.gamma * normalized_x + self.beta
  
  
if __name__ == "__main__":
  dim_embed = 512
  layer_norm = LayerNormalization(dim_embed)
  x = torch.rand(2, 3, dim_embed)
  print(f'x: {x}')
  output = layer_norm(x)
  print(f'output: {output}')