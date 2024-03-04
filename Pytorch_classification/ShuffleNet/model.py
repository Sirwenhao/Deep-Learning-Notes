# 24/01/28 author:WH

from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn

def channel_shuffule(x: Tensor, groups: int):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # reshape
    # [batch_size, num_channel, height, width] -> [batch_size, groups, channel_per_groups, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten 
    x = x.view(batch_size, -1, height, width)
    
    return x


  
class InvertedResidual(nn.Module):
    def __init__(self, input_c, output_c, stride):
        super(InvertedResidual, self).__init__()
        
        if stride not in [1, 2]:
            raise ValueError("illega; stride value.")
        self.stride = stride
        
        assert output_c % 2 == 0
        branch_features = output_c // 2
        # stride=1时，input_channel应该是branch_features的两倍
        # python中'<<'是位运算，可以理解为x2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)
        
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                
            )
