import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from torchvision import transforms, datasets
# from model import MobileNetV2


class ConfusionMatrix(object):
    def __init__(self, num_classes, labels):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        
    def update(self, preds, labels):
        for p,t in zip(preds, labels):
            self.matrix[p, t] += 1
            
    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is ", acc)
        
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            #self.matrix[i, :]为矩阵matrix的行 
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            




# matix = np.eye(5)
# matix[0, 3] = 6
# print(matix)

# print(matix[0, :])
# print(matix[1, :])
