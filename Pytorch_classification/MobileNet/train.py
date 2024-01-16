# 24/01/16

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import MobileNetV2

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")

    batch_size = 16
    epochs = 5

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterVrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    data_root =   #daraser path
    image_path = os.path.join(data_root, "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key,val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) # number of workers
    print(f"Using {nw} dataloader workers every process")

    trian_loader = torch.utils.data.Dataloader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.Dataloader(validate_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=nw)
    print(f"Using {train_num} images for trainning, using {val_num} for validation")

    # create model
    net = MobileNetV2(num_classes=5)

    # load pretrained weights
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = ''
    assert os.path.exists(model_weight_path), f"file {model_weight_path} does not exist."
    pre_weights = torch.load(model_v2, map_location='cpu')  # 返回值为一个字典元素，键是参数的名称、值是对应参数的张量

    # delete classifier weithts
    # 下述代码使用了字典推导式, numel()是统计张量的元素数量
    pre_dict = {k:v for k,v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

