import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from torchvision import transforms, datasets
from model import MobileNetV2


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
            Precision = round(TP / (TP+FP), 3) if TP+FP !=0 else 0.
            Recall = round(TP / (TP+FN), 3) if TP+FN != 0 else 0.
            Specificity = round(TN / (TN+FP), 3) if TN+FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table) 
            
    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cn.Blues)
        
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Lables')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[x, y])
                plt.text(x, y, info, verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = '/Users/WH/Desktop/pytorch_classification/flower_data/'
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)
    
    validate_dateset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)
    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dateset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = MobileNetV2(num_classes=5)
    # load model weigths
    model_weight_path = '/Users/WH/Desktop/pytorch_classification/mobilenet_v2.pth'
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)
    
    # read class_indict
    json_label_path = '/Users/WH/Desktop/Deep-Learning-for-image-processing/Pytorch_classification/ConfusionMatrix/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_imags, val_labels = val_data
            outputs = net(val_imags).to(device)
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
            
    



# matix = np.eye(5)
# matix[0, 3] = 6
# print(matix)

# print(matix[0, :])
# print(matix[1, :])
