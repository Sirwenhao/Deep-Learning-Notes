import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2

def main():
    device = torch.device("duda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image
    img_path = "/Users/WH/Desktop/pytorch_classification/flower_imgs/daisy.jpg" # 待预测的图像路径
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    # read class_indict
    json_path = '/Users/WH/Desktop/Deep-Learning-for-image-processing/Pytorch_classification/MobileNet/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    
    with open(json_path, "r") as f:
        class_indict = json.load(f)
        
    # create model
    model = MobileNetV2(num_classes=5).to(device)
    # load model weights
    model_weight_path = "/Users/WH/Desktop/Deep-Learning-for-image-processing/Pytorch_classification/MobileNet/MobileNetv2_retrain.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        print(model(img.to(device)))  # torch.Size([1, 5]) 1行5列, 对应的是5个类别每个类的预测结果
        output = torch.squeeze(model(img.to(device))).cpu()
        # print(output) # torch.Size([5]) 
        predict = torch.softmax(output, dim=0)
        print(predict) # tensor类型，对应的是将每个预测数值转换之后的概率
        predict_cla = torch.argmax(predict).numpy()
        print(predict_cla) # 0，torch,argmax()的作用是找到预测结果最大值对应的索引
        
    print_res = "class: {} prob: {:.3f}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10} prob:{:.3f}".format(class_indict[str(i)],
                                                predict[i].numpy()))
    plt.show()
    
if __name__ == '__main__':
    main()