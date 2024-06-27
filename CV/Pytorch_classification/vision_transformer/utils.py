import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_split_data(root, val_rate = 0.2):
    random.seed(0) # 保证随机可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，确保顺序一致
    flower_class.sort()
    # 声称类别名称以及对应的数字索引
    class_indices = dict((k, v) for k,v in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".jpeg", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.split(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))
        
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)
                
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    
    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class), every_class_num, align='center'))
        plt.xticks(range(len(flower_class)), flower_class)
        for i,v in enumerate(every_class_num):
            plt.text(x=i, y=v+5,s=str(v), ha='center')
            
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()
        
    return train_images_path, train_images_label, val_images_path, val_images_label
    
    
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    
    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + "does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)
    
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([]) # 去掉x轴刻度
            plt.yticks([]) # 去掉y轴刻度
            plt.imshow(img.astype('uint8'))
        plt.show()
        
        
def write_pickle(list_info, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)
        
def rad_pickle(file_name):
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list
    
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred, labels.to(device)).sum()
        
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)
        
        if not torch.isfinite(loss):
            print("Warning: non-finiti loss, ending training", loss)
            sys.exit(1)
            
        optimizer.step()
        optimizer.zero_grad()
        
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@ torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    
    model.eval()
    
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        
        data_loader.desc = "[Valid epoch {}] loss: {:.3f}, acc: {:,3f}".format(epoch, accu_loss.item()/(step+1), accu_num()/sample_num)
        
    return accu_loss.item()/(step+1), accu_num.item()/sample_num
            
        
        
    
    