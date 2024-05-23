# 244/4/25 author:WH

import os
import json
import torch

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = 'B0'
    
    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image
    img_path = r''
    assert os.path.exists(img_path). "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.show(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    # read class_indict
    json_path = r''
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    
    with open(json_path, 'r') as f:
        class_indict = json.load(f)
        
    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = r""
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        
    print_res = "class: {:10} prob:{:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10} prob:{:.3}".format(class_indict[str(i)], predict[i].numpy()))
        plt.show()
        
if __name__ == '__main__':
    main()
    



