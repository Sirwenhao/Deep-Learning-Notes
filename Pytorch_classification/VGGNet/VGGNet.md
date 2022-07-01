## VGGNet

主要对应教程为：WZMIAOMIAO，对应训练任务为：花分类，五类别

### model.py

主要学习model.py中网络以配置文件来写的这种写法，具体代码如下：

```python
# 采用写配置文件的方式，将网络对应层的类型及其参数写入到sequential容器中
def make_features(cfg: list):
    # 空列表layers用于存放每一层创建的结构
    layers = []
    # 需要注意输入输出的通道数会随着不同的卷积层中卷积核的参数而改变，因此需要单独写出此参数
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

# 其对应的配置文件是一个列表，将相应参数写到配置文件中进行读取并对应于卷积或者池化操作
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```

