# 24/3/17

import os
import math
import argparse
import torch

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import densenet121, load_state_dict
from my_dataset import MyDataset
from utils import read_split_data, train_one_epoch, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writter = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedits("./weights")
        
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataset(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=data_transform["train"])
    
    # 实例化验证数据集
    val_dataset = MyDataset(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,8]) # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data_DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=nw,
                                             collate_fn=val_dataset.collate_fn)
    
    # 载入权重
    model = densenet121(num_classes=args.num_classes).to(device)

