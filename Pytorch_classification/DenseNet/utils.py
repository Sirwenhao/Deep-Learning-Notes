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
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序 
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
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
            if img_path in val_path: # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(images_class)
            else: # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(images_class)
                
                
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
    
    json_path = r'data_set/class_indices.json'
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


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        
        loss = loss_function(pred, labels.to(device))
        loss.backward() 
        mean_loss  = (mean_loss * step + loss.detach()) / (step+1) # update mean loss
        
        data_loader.desc = "[epoch {}] meand loss {}".format(epoch, round(mean_loss.item(), 1))
        
        # torch.isfinite(loss)函数用于检查张量loss中的所有元素是否为有限值，即非NaN和无穷大，返回值为True或False 
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)
            
        optimizer.step()
        optimizer.zero_grad()
    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    
    # 验证样本总个数
    total_num = len(data_loader.dataset)
    
    # 用于存储预测正确的样本个数
    sun_num = torch.zeros(1).to(device)
    
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
    return sum_num.item() / total_num
