import torch
from fvcore.nn import FloopCountAnalysis
from vit_model import Attention

def main():
    # self-attention
    a1 = Attention(dim=512, num_heads=5)
    a1.proj = torch.nn.Identity()
    
    # Multi-head Attention
    a2 = Attention(dim=512, num_heads=8)
    
    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512))
    
    flops1 = FloopCountAnalysis(a1, t)
    print(f'Self-Attention FLOPs:{flops1.total()}')
    
    flops2 = FloopCountAnalysis(a2, t)
    print(f'Multi_Head FLOPS:{flops2.total()}')
    
if __name__ == "__main__":
    main()
