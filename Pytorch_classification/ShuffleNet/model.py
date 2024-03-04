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


    
