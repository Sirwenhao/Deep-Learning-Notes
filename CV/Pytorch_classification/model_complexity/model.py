from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor

from utils import *

def drop_path(x, drop_prob=0., training=False):
    if drop_path == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size = 3,
                 stride = 1,
                 groups = 1,
                 norm_layer=None,
                 activation_layer=None):
        super(ConvBNAct, self).__init__()
        
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU # alias Swish
            
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_planes=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = norm_layer(out_planes)
        self.act = activation_layer()
        
    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)
        return result
            
    def complexity(self, x):
        cx = conv2d_cx(cx,
                       in_c=self.conv.in_channels,
                       out_c=self.conv.out_channels,
                       k=self.conv.kernel_size[0],
                       stride=self.conv.stride[0],
                       groups=self.conv.groups,
                       bias=False,
                       trainable=self.conv.weight.requires_grad)
        cx = norm2d_cx(cx, self.conv.out_channels, trainable=self.bn.weight.requires_grad)
        
        return cx
            
    