## BiLSTM+CRF模型详解一

@author: WH  24/5/23

本文将尽可能通俗的介绍NER经典模型BiLSTM+CRF相关的知识细节

本文目录：

- 介绍NER相关任务中，BiLSTM+CRF模型的具体工作流
- 简要介绍相关部件的工作原理（详见本文后续）

1. ##### 基本情况介绍

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

   