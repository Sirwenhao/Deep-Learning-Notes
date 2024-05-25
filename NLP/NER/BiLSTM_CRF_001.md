## BiLSTM-CRF模型详解一

@author: WH  24/5/23

本文将尽可能通俗的介绍NER经典模型BiLSTM+CRF相关的知识细节

1. #### 本文目录：

- 介绍NER相关任务中，BiLSTM+CRF模型的具体工作流
- 简要介绍相关部件的工作原理（详见本文后续）

##### 1.1 基本情况介绍

本文的任务的数据中，主要的实体信息是地址数据和人名等。其中：

- 地址数据主要包含street、city、region、postcode等
- 人名数据主要是person、organization等

以人名数据为例，采用BIO标记方法会有五类标签：

- B-Person
- I-Person
- B-Organization
- I-Organization
- O

假设一个句子有5个对应的字符组成，这5个字符表示为$x=(w_0, w_1, w_2, w_3, w_4)$，其中$[w_0, w_1]$为**人名**实体，$[w_3]$为**组织**实体，其他的字符标签均为"O"。

##### 1.2 BiLSTM-CRF模型

以当前5输入的句子为例，对应的模型结构图如下：



![img](https://gitee.com/Sirwenhao/typora-illustration/raw/master/20181227235235.png)

其整体的流程总结如下：

- 输入是一段包含实体信息的自然语言，输出是对应于这段自然语言的实体标签
- BiLSTM可以学习不同的word之间的上下文关系，CRF可以学习到一种更为合理的词间关系

##### 1.3 如果没有CRF层是否可以实现实体的识别？

先说答案，可以。

但是没有CRF层的模型，会给出一些不合理的结果。

![img](https://gitee.com/Sirwenhao/typora-illustration/raw/master/20181227235450.jpg)

如上图，没有CRF层同样可以为每一个word预测出对应的实体标签，BiLSTM模型会为每一个word预测出一个标签预测分值向量，可以从其中选出概率分值最大的作为最终的标签。此图中可以得到对应的预测标签为：$w_0$—B-Person，$w_1$—I-Person，$w_2$—O，$w_3$—B-Organization，$w_4$—O。

以上是较为理想的预测情况，如下：

![img](https://gitee.com/Sirwenhao/typora-illustration/raw/master/20181227235531.png)

在此实例中，对应的预测标签为$w_0$—I-Organization，$w_1$—I-Person，$w_2$—O，$w_3$—I-Organization，$w_4$—I-person。显然，标签序列"I-Organization I-Person"及"B-Organization I-Person"明显是错误的。

##### 1.4 CRF层能从训练数据中学习到约束规则

这便是CRF层的必要之处，可以确保预测后的标签是合理的情况。在训练过程中，CRF层可以自动的学习到对应的一些约束规则，如：

- 句子中的第一个词总是以标签“B-”或者“O”开始，而非“I-”
- 标签“B-label1 I-label2 I-label3 I-”中，label1、label2、label3应该属于同一类实体。如“B-Person I-Person”为合理序列，“B-PersonI-Organization”非法
- 标签序列“O I-label”是不合理的，实体中的首个标签必是“B-”而非其他，

通过以上约束，可以尽可能保证BiLSTM-CRF模型预测出的实体的标签为合理的标签。



##### Reference：

[1] Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. and Dyer, C., 2016. Neural architectures for named entity recognition. arXiv preprint arXiv:1603.01360. https://arxiv.org/abs/1603.01360

[2] https://lonepatient.top/2018/07/06/CRF-Layer-on-the-Top-of-BiLSTM%20--1.html