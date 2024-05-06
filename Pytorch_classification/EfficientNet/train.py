# 24/5/6   author:WH
import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_schedular

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate



