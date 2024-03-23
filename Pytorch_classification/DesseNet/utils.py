import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_split_data(root, val_rate=0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.lisrtdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序 
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indics.items()), indent=4)
    with open('class_indices.json', "w") as json_file:
        json_file.write(json_str)
        
    train_images_path = [] # 存储训练集所有的图片的路径
    train_images_label = [] # 存储训练集图片对应的索引信息
    val_images_path = [] # 存储验证集所有图片路径
    val_images_label = [] # 存储验证集所有图片对应的索引信息
    every_class_num = [] # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        images.sort()
        images_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))
        
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(images_class)
                
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    
    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
    # 设置x坐标
    plt.xlabel('image class')
    # 设置y坐标
    plt.ylabel('number of images')
    # 设置柱状图的标题
    plt.title('flower class distribution')
    plt.show()
    
    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    
    json_path = ''
    assert os.path.exists(json_path), json_path + "does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)
    
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


