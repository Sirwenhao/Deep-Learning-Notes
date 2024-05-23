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
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
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
            layer = _DenseLayer(input_c + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size,drop_rate=drop_rate, memory_efficient=memory_efficient)
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
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
        
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
                
        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)
                
                
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
        
def densenet121(**kwargs):
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **kwargs)


def densenet169(**kwargs: Any) -> DenseNet:
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    **kwargs)
    
    
def densenet161(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    **kwargs)
    
    
    
def load_state_dict(model, weights_path):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    """
    此处的pattern语句是一个用于解析神经网络层参数的正则表达式:
    1、'^'表示匹配字符串的开始
    2、'(.*denselayer\d+\.(?:norm|relu|conv))'为匹配的对象
        - '.*'表示匹配零个或者多个任意字符
        - 'denselayer': 匹配文本"denselayer"
        - '\d+': 匹配一个或者多个数字
        - '\.': 匹配一个点字符
        - '(?:norm|relu|conv)': 非捕获组，匹配 "norm"、"relu" 或 "conv" 中的一个
        - 总结起来，这部分匹配了形如 "denselayerX.norm"、"denselayerX.relu" 或 "denselayerX.conv" 的子字符串，其中 X 是一个或多个数字
    3、'\.((?:[12])\.(?:weight|bias|running_mean|running_var))'为匹配的对象
        - '\.': 匹配一个点字符
        - '((?:[12])\.(?:weight|bias|running_mean|running_var))'
        - '(?:[12])'：非捕获组，匹配 "1" 或 "2" 中的一个
        - '\.'：匹配一个点字符
        - '(?:weight|bias|running_mean|running_var)'：非捕获组，匹配 "weight"、"bias"、"running_mean" 或 "running_var" 中的一个。
        - 总结起来，这部分匹配了形如 ".1.weight"、".2.bias"、".1.running_mean" 或 ".2.running_var" 的子字符串
    4、'$': 匹配字符串的结束
    """
    state_dict = torch.load(weights_path)
    
    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000
    
    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]
                
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)
    print("Successfully load pretrain-weights.")
            
            
        
    
    
    
    
    
