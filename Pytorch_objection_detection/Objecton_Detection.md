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

对应论文：《*Rich feature hierarchies for accurate object detection and semantic segmentation*》（https://arxiv.org/pdf/1311.2524.pdf）。根据原论文RCNN的算法流程可以分为四步：

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

   ![image-20220708205648099](https://gitee.com/sirwenhao/images/raw/master/image-20220708205648099.png)

5. 使用回归器精细修正候选框位置：对NMS处理后的候选框进行进一步筛选。用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

6. <center> R-CNN框架 </center>
   <center>Region proposal(Selective Search)</center>
   <center>Feature Extraction(CNN)</center>
   <center>Classfication(SVM)  and   Bounding-box regression(regression)</center>

R-CNN存在的问题：

1. 测试速度慢，使用SS算法提取候选框时，候选框之间存在大量的重复冗余
2. 训练速度慢，过程繁琐复杂
3. 训练所需空间大，SVM和bbox回归训练都需要从候选框中提取特征。一方面冗余重复特征多，另一方面特征文件的存取需要占用大量的内存空间。

2022/7/9  author:WH

### Fast R-CNN

对应论文《Fast R-CNN》(https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)，Fast R-CNN与R-CNN同样都使用VGG16作为CNN backbone。较R-CNN，训练时间快9倍，测试推理时间快213倍，准确率从62%提升至66%（在Pascal VOC数据集上）

Fast R-CNN算法流程可大致分为三个步骤：

- 一张图像生成1K~2K个候选区域（使用Selective Search方法）
- 将图像输入网络得到相应的特征图，将SS算法生成的候选框投影到特征图上获得相应的特征矩阵
- 将每个特征矩阵通过ROI(Region of Interest) pooling层缩放到$7\times7$大小的特征图，接着将特征图展平，通过一系列全连接层得到预测的结果。相比于R-CNN，这一版本直接使用全连接层进行回归预测，不需要训练SVM和回归器。

具体结构图如下：

![image-20220709145317443](https://gitee.com/sirwenhao/images/raw/master/image-20220709145317443.png)

有关于Fast R-CNN的一些重要知识点：

- Fast R-CNN相较于R-CNN一个比较大的改变是：特征提取，首先整张图象送入到网络中，得到对应的特征图；然后通过每个候选区域的原图与特征图之间的对应关系，直接在特征图中获取其特征矩阵，从而可以使特征矩阵免于重复计算，其算法速度的大幅提升也主要是来源于此。
- Fast R-CNN的训练数据是2K个候选框中随机采样到的一部分，且这部分采样数据又可以划分为正样本和负样本。其中正样本是指：候选框中确实存在所需检测目标的样本，负样本：是指候选框中没有所需检测目标的样本（准确来说是所需检测目标所占比例不超过一定标准）。
- 设置正负样本是为了防止数据不平衡的情况下，网络预测的结构更加偏向于占据主导地位的某一类样本
- Fast R-CNN中选取正负样本的标准：GT bbox与proposal bbox的IOU大于等于0.5即被选取作为正样本，IOU在$[0.1,0.5)$被选取作为负样本
- 训练样本的候选框通过RoI Pooling Layer缩放到统一的尺寸。对于所有的候选区，将其划分为$7\times7$大小的区域，然后对每个区域执行最大池化下采样，从而得到大小为$7\times7$大小的特征矩阵

![image-20220710162931160](https://gitee.com/sirwenhao/images/raw/master/image-20220710162931160.png)

![image-20220710164217840](https://gitee.com/sirwenhao/images/raw/master/image-20220710164217840.png)

![image-20220710164249571](https://gitee.com/sirwenhao/images/raw/master/image-20220710164249571.png)

![image-20220710164350137](https://gitee.com/sirwenhao/images/raw/master/image-20220710164350137.png)