### MobileNet

24/01/15. author:WH

《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》

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

##### 参数$\alpha$

- The role of the width multiplier $\alpha$ is to thin a network uninformly at each layer

- Usage: For a given layer and width multiplier $\alpha$, the number of input channels *M* becomes $\alpha M$ and the number of output channels *N* becomes $\alpha N$

- 数值上的量级对比：The computational cost of a depthwise separable convolution with width multiplier $\alpha$  ($\alpha \in (0, 1]$)is:
  $$
  D_ {K}\cdot  D_ {K} \cdot \alpha M \cdot D_ {F} \cdot D_ {F} + \alpha M \cdot \alpha N  \cdot D_ {F}  \cdot  D_ {F}
  $$

参数$\rho$

- The second hyper-parameter to reduce the computational cost of a neural network is a resolution multiplier $\rho$

- Usage: We apply this s to the input image and the internal representation of every layer is subsequently reduced by the same multiplier

- 数值上的量级对比：We can now express the computational cost for the core layers of our network as depthwise separable convolutions with width multiplier $\alpha$ and resolution multiplier $\rho$​：
  $$
    D_ {K}  \cdot  D_ {K}  \cdot \alpha  M  \cdot \rho D_ {F} \cdot \rho D_ {F} + \alpha M \cdot  \alpha  N  \cdot \rho D_ {F} \cdot \rho D_ {F}
  $$
  