# 24/01/07 author:WH

import os
import torch
import torch.nn as nn
from model import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(f'using {device} device')

    # load pretraion weights
    model_weights_path = "/Users/WH/Desktop/pytorch_classification/resnet34-333f7ec4.pth"
    assert os.path.exists(model_weights_path), "file {} does not exist.".format(model_weights_path)

    #以下两种方法均涉及到对原模型结构的最后一层的修改

    # # 方法一
    # net = resnet34()
    # net.load_state_dict(torch.load(model_weights_path, map_location=device))
    # # 修改最后一层的全连接层结构
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)

    # 方法二
    net = resnet34(num_classes=5)
    pre_weights = torch.load(model_weights_path, map_location=device)
    del_key = []
    for key,_ in pre_weights.items():
        if 'fc' in key:
            del_key.append(key) 
    for key in del_key:
        del pre_weights[key]

    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == "__main__":
    main()



