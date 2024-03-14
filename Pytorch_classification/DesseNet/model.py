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
    def __init__(self, input_c, output_c):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AdaptiveAvgPool2d(kernel_size=2, stride=2))
        
class DenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super(DenseNet, self).__init__()
        
        # first conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
            
        # each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, input_c=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module("denseblock %d" %(i+1), block)
            num_features = num_features + num_layers*growth_rate
            
            if i != len(block_config)-1:
                trans = _Transition(input_c=num_features, output_c=num_features//2)
                self.features.add_module("transition %d" %(i+1), trans)
                num_features = num_features // 2
                
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d)


