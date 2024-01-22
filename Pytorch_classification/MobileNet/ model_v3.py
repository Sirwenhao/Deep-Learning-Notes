# 24/01/22 author:WH

from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 100%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNActivation(nn.Sequential):
    def __init__(self, 
                in_planes: int,
                out_planes: int,
                kernrlz_size: int = 3,
                stride: int = 1,
                groups: int = 1,
                norm_layer: Optional[Callable[.../, nn.Module]] = None,
                activation_layer: Optional[Callable[..., nn.Module]] = None):
        
    
