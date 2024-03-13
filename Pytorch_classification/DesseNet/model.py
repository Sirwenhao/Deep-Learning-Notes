import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

class _DenseLayer(nn.Module):
    def __init__(self, input_c, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module("normal", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=input_c, out_channels=bn_size, kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate), growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dro_rate = drop_rate
        self.memory_efficient = memory_efficient
    
    def bn_function(self, inputs):
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output
    
    @staticmethod
    def any_requires_grad(self, inputs):
        for tensor in inputs:
            if tensor.requires_grad:
                return True
            
        return False
    
    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs):
        def closure(*inp):
            return self.bn_function(inp)
        return cp.ckeckpoints(closure, *input)
        
        
    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
            
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")
            
            bottleneck_ouput = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_ouput = self.bn_function(prev_features)
            
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_ouput)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training = self.training)
        return new_features
    
class _DenseBlock(nn.ModuleDict):
    _version=2
    def __init__(self, num_layers, input_c, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, memory_efficient=memory_efficient)
            self.add_module("denselayer%d" %(i+1), layer)
            
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
class _Transition(nn.Sequential):
    def __init__(self, input_c,):
            
        
    


