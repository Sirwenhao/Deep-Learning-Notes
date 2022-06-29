## AlexNet

### 目录：

- AlexNet网络的基本结构
- 使用AlexNet对花进行分类任务
- split_data.py脚本的问题与issue

### AlexNet网络的基本结构

有关AlexNet网络结构细节，原博主(https://github.com/WZMIAOMIAO)已经讲解的非常详细，具体不在赘述

### 使用AlexNet对花进行分类任务

此处有关split_data.py脚本有一点点问题，具体如下：

```python
# 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 2022/6/29补充，author:WH
        # 建立保存验证集中对应每个类别的文件夹
        mk_file(os.path.join(val_root, cla))
```

具体的数据集划分步骤：

- 数据集下载链接：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- 流程：在data_set文件夹下创建新文件夹"flower_data"，将下载好的数据集解压到flower_data文件夹下面
- 在data_set文件夹内单击右键，在终端中打开并执行命令`python split_data.py`进行验证集和训练集的划分