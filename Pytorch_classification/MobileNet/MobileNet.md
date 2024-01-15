### MobileNet

24/01/15. Author:WH

MobileNet论文链接：https://arxiv.org/pdf/1704.04861.pdf

MobileNet网络的相关结构及其中所使用的两个超参数（width multiplier和resolution multiplier）在论文中的Section3中有详细介绍

##### 网络结构中的两个亮点

- depthwise separable filters（使用此结构的核心目的：减少参数量）
- shrinking hyperparameters
  - width multiplier
  - resolution multiplier

##### 原论文中关于depthwise separable convolution的定义：

- depth wise separable convolution：which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution

and 1x1 convolution called a pointwise convolution

##### 两种不同的convolution的作用域

- depthwise convolution applies a single filter each **input channel**（先）
- point wise convolution **then** applies a 1x1 convolution to combine the outputs the depthwise convolution（后）

##### 关于standard convolution以及depthwise convolution、pointwise convolution的示意图

![image-20240115231846650](https://gitee.com/Sirwenhao/typora-illustration/raw/master/image-20240115231846650.png)