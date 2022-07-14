## Objection_Detection合辑

2022/7/8  笔记来源：WZMIAOMIAO，author:WH

### Two-Stage：

- R-CNN(Region with CNN feature)
- Fast R-CNN
- Faster R-CNN

### One-Stage:

- YOLO v1-v4
- YOLO v5

### R-CNN

> 对应论文：《*Rich feature hierarchies for accurate object detection and semantic segmentation*》(https://arxiv.org/pdf/1311.2524.pdf)

根据原论文RCNN的算法流程可以分为四步：

1. 每张图像上生成1K~2K个候选区域（使用Selective Search方法）
2. 对于每个候选区域使用深度网络提取特征
3. 特征送入到SVM分类器，判断是否属于该类别
4. 使用回归其精细修正候选框位置

具体来讲：

1. 候选区域生成是利用SS算法通过图像分割的方法得到一些原始的区域，然后基于合并策略对于这些区域进行合并，得到一个层次化的区域结构。

   - 合并规则：优先合并颜色（颜色直方图）、纹理（梯度直方图）相近的、总面积小的区域等

2. 对于每个候选框进行特征提取，使用CNN网络进行特征提取。具体来讲，将2000个候选区域缩放为$227\times227$大小的区域，使用预先训练好的AlexNet网络提取特征，其输出维度是4096维，因此最后2000个候选区域所产生的特征向量维度为$2000\times4096$.

3. 将特征送入到每一维的SVM分类器，进行类别判定。将$2000\times4096$维特征与20个SVM组成的权值矩阵$4096\times20$相乘，获得$2000\times20$维矩阵中每一列即即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框。关于矩阵$2000\times20$的解释：矩阵中的第一列就表示所有候选框框出的部分为某一种类别的概率，第一列的第一个元素就表示第一个框框出的部分为相应类别的概率。

4. 有关于非极大值抑制：使用非极大值抑制剔除重叠建议框，一个重要的概念：IOU(Intersection over Union)，具体计算为：$IOU=(A\cup B)/(A\cap B)$。此处非极大值抑制是对于矩阵$2000\times20$的每一列进行处理的。

   ![image-20220714112649962](https://gitee.com/sirwenhao/images/raw/master/image-20220714112649962.png)

5. 使用回归器精细修正候选框位置：对NMS处理后的候选框进行进一步筛选。用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

6. ![image-20220714112736866](https://gitee.com/sirwenhao/images/raw/master/image-20220714112736866.png)

R-CNN存在的问题：

1. 测试速度慢，使用SS算法提取候选框时，候选框之间存在大量的重复冗余
2. 训练速度慢，过程繁琐复杂
3. 训练所需空间大，SVM和bbox回归训练都需要从候选框中提取特征。一方面冗余重复特征多，另一方面特征文件的存取需要占用大量的内存空间。

2022/7/9  author:WH

### Fast R-CNN

Fast R-CNN与R-CNN同样都使用VGG16作为CNN backbone。较R-CNN，训练时间快9倍，测试推理时间快213倍，准确率从62%提升至66%（在Pascal VOC数据集上）

> 对应论文为：《Fast R-CNN》(https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

Fast R-CNN算法流程可大致分为三个步骤：

- 一张图像生成1K~2K个候选区域（使用Selective Search方法）
- 将图像输入网络得到相应的特征图，将SS算法生成的候选框投影到特征图上获得相应的特征矩阵
- 将每个特征矩阵通过ROI(Region of Interest) pooling层缩放到$7\times7$大小的特征图，接着将特征图展平，通过一系列全连接层得到预测的结果。相比于R-CNN，这一版本直接使用全连接层进行回归预测，不需要训练SVM和回归器。

具体结构图如下：

![image-20220714113244188](https://gitee.com/sirwenhao/images/raw/master/image-20220714113244188.png)

有关于Fast R-CNN的一些重要知识点：

- Fast R-CNN相较于R-CNN一个比较大的改变是：特征提取，首先整张图象送入到网络中，得到对应的特征图；然后通过每个候选区域的原图与特征图之间的对应关系，直接在特征图中获取其特征矩阵，从而可以使特征矩阵免于重复计算，其算法速度的大幅提升也主要是来源于此。
- Fast R-CNN的训练数据是2K个候选框中随机采样到的一部分，且这部分采样数据又可以划分为正样本和负样本。其中正样本是指：候选框中确实存在所需检测目标的样本，负样本：是指候选框中没有所需检测目标的样本（准确来说是所需检测目标所占比例不超过一定标准）。
- 设置正负样本是为了防止数据不平衡的情况下，网络预测的结构更加偏向于占据主导地位的某一类样本
- Fast R-CNN中选取正负样本的标准：GT bbox与proposal bbox的IOU大于等于0.5即被选取作为正样本，IOU在$[0.1,0.5)$被选取作为负样本
- 训练样本的候选框通过RoI Pooling Layer缩放到统一的尺寸。对于所有的候选区，将其划分为$7\times7$大小的区域，然后对每个区域执行最大池化下采样，从而得到大小为$7\times7$大小的特征矩阵

![image-20220714104136684](https://gitee.com/sirwenhao/images/raw/master/image-20220714104136684.png)

![image-20220714104335248](https://gitee.com/sirwenhao/images/raw/master/image-20220714104335248.png)

有关于损失函数：

- 其中$p$是分类器预测的softmax概率分布$p=(p_0,...,p_k)$，其中$p_0$是指分类器预测结果为背景的概率。根据softmax函数的特点，所有这21个概率之和为1.
- $u$对应目标的真是类别标签，分类损失计算的是预测类别和真实类别的偏差
- $t^u$对应的边界框回归器预测的对应的类别$u$的回归参数$(t_x^u,t_y^u,t_w^u,t_h^u)$
- $v$对应真实目标边界框的回归参数$(v_x,x_y,v_w,v_h)$
- $[u\geqslant1]$表示艾弗森括号，具体理解为：$u\ge1$时整个式子的值为1，反之为0。在此问题中，$u$表示的是目标的真实标签值，若$u\geq1$则表示候选区域确实属于所需检测的某一个类别，即对应于正样本，因此才会有边界框损失函数。反之，对应于负样本，就不需要计算边界框损失函数了，因此此时的$[u\geq1]=0$。

Fast R-CNN结构

![image-20220714112832570](https://gitee.com/sirwenhao/images/raw/master/image-20220714112832570.png)

### Fastr R-CNN

Faster R-CNN基于Fast R-CNN提出，其骨干网络仍然是基于VGG16的，无论是在推理速度还是准确率方面都有了显著性提升。并且，获得了2015年的ILSVRC以及COCO竞赛中的多项第一名。

> 对应论文为：Ren, Shaoqing, et al. “Faster R-CNN: Towards real-time object detection with region proposal networks.” Advances in Neural Information Processing Systems. 2015.（https://arxiv.org/pdf/1506.01497.pdf）

Faster R-CNN的算法流程具体也可以分为三个步骤：

- 将图像输入网络中获取相应的特征图
- 使用RPN网络结构生成候选框，将RPN结构生成的候选框投影到特征图上获取相应的特征矩阵
- 将每个特征矩阵通过ROI pooling层缩放到$7\times7$大小的特征图，接着将特征图展平并通过一系列全连接层得到预测结果

具体结构图如下：

![在这里插入图片描述](https://gitee.com/sirwenhao/images/raw/master/20191105195816143.png)

简单来说，可以将其理解为`RPN+Fast R-CNN`，其核心部分的RPN结构替代了原有的SS算法。在本篇论文中，目标检测的四个基本步骤（候选区域生成、特征提取、分类、位置精修）被统一到一个深度网络框架之中。

![image-20220714155639579](https://gitee.com/sirwenhao/images/raw/master/image-20220714155639579.png)

关于上图的解释：

- 2K scores是针对每一个anchor box生成两个概率值，当前anchor为前景的概率和当前anchor为背景的概率
- 4K coordinates表示对于每一个anchor都会生成四个边界框回归参数，每四个一组
- 此处的256-d是根据具体使用的网络backbone而言的，其中的256是其对应的channel
- 特征图到原图的映射：特征图上的每一个pixel都是由原图经过一系列卷积操作所得，根据特征图与原图的大小比例可以找出特征图上一个pixel在原图上所对应的pixel的位置
- Faster R-CNN中对应的anchor总共有9种类型，涵盖三种面积尺度：$\{128^2,256^2,512^2\}$，三种比列：$\{1:1, 1:2, 2:1\}$，即特征图上的每个位置对应到原图上都有9个与之对应的anchor
- 对于一个$1000\times6000\times3$的图像，大约有$60\times40\times9(20K)$个anchor，忽略掉跨越边界的anchor，还剩下约$6K$个anchor。然后使用RPN生成的候选框回归参数，对anchor的位置进行调整生成候选框，但是候选框之间存在重叠部分。因此，进一步采用非极大值抑制筛选掉存在大量重叠的候选框，这一步的非极大值抑制在原论文中有两种设定方式。通过上述操作，最终每张图片可以得到大约2k个候选框
- 对于每一张图片，随机采样256个anchor进行计算。采样的原则是正负样本比例控制在1:1，如果正样本比例不足$\frac{1}{2}$则使用负样本进行填充
  - 关于正负样本的定义：
    - 正样本：(1)anchor/anshors with the heighest Intersection-over-Union(IoU) overlap with a ground-truth box，(2)an anchor that has an IoU overlap heigher than 0.7 with any ground-truth box.
    - 负样本：We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes.

有关于损失函数：$L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_ip_i^*L_{reg}(t_i,t_i^*)$

$i$表示mini-batch中的第i个anchor，$p_i$表示第$i$个anchor预测为真实标签值的概率，当第$i$个anchor为前景时，$p_i^*$为1反之为0。$t_i$表示预测的bounding box的坐标，$t_i^*$为第$i$个ground-truth box对应的边界回归参数，$N_{cls}$表示一个mini-batch中所有样本数量256，$N_{reg}$表示anchor位置的个数（不是anchor的个数，一个位置对应多个anchor）约为2400。$\lambda$是平衡参数，用于平衡分类损失和边界框回归损失。

Faster R-CNN的训练：可以直接采用RPN loss+Fast R-CNN loss的联合训练方法

![image-20220714172249385](https://gitee.com/sirwenhao/images/raw/master/image-20220714172249385.png)

原论文中使用的是分别训练RPN loss和Fast R-CNN loss的方法：

1. 利用ImageNet预训练分类模型初始化前置卷积层参数，并开始单独训练RPN网络参数
2. 冻结RPN网络的卷积层以及全连接层参数，再利用ImageNet与训练分类模型初始化前置卷积层网络参数，并利用Fast R-CNN生成的目标建议框去训练Fast R-CNN网络参数
3. 冻结Fast R-CNN训练好的前置卷积层的参数，微调RPN网络的卷积层和全连接层的参数
4. 冻结前置卷积网络层的参数，微调Fast R-CNN网络的全连接层参数。最后RPN和Fast R-CNN共享前置卷积层参数，构成一个统一的网络。

Faster R-CNN结构

![image-20220714173743153](https://gitee.com/sirwenhao/images/raw/master/image-20220714173743153.png)
