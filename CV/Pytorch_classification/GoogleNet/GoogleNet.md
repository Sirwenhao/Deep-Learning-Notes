###GoogLeNet

- GoogleNet论文链接：https://arxiv.org/pdf/1409.4842.pdf

GoogLeNet与2014年有Google团队提出《Going deeper with convolutions》，斩获同年ImageNet竞赛分类赛道（Classification Task）中的第二名，该网络与以往的卷积神经网络相比有以下四个亮点：

- 引入了inception结构（旨在融合不同尺度的特征信息）
- 使用$1\times1$卷积进行降维以及映射处理
- 添加了两个辅助分类器帮助训练（相比于AlexNet和VGG都只有一个输出层，GoogLeNet有三个输出层）
- 丢弃全连接层，使用平均池化（大大减少了模型参数）

##### GoogLeNet中的最大亮点：Inception结构

![Alt text](https://gitee.com/Sirwenhao/typora-illustration/raw/master/image.png)

其拼接是按照深度这个维度进行的，每个分支的特征矩阵的高和宽必须是一致的

##### GoogLeNet中的亮点之二：$1\times1$卷积

使用$1\times1$卷积核降低计算量，同时改变深度维度（由所使用的$1\times1$卷积核的个数所决定）

##### GoogLeNet中的亮点之三：辅助分类器（Auxiliary  Classifier）

其中辅助分类器一来源于结构中Inception4(a)的输出，另一个辅助分类器的输出来源于结构中的Inception4(d)

![image-20240113132252043](https://gitee.com/Sirwenhao/typora-illustration/raw/master/image-20240113132252043.png)