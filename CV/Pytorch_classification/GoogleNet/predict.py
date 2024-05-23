# 24/2/28

import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # load image
    img_path = "data_set/camelia.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # expand batch dimension
    # [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)
    
    # read class_indict
    json_path = 'data_set/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exists.".format(json_path)
    
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    # create model
    # 参数aux_logits决定是否添加辅助分类器
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
    
    # load model weights
    weights_path = ''
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.eval()
    with torch.no_grad():
        # predict class 
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        
    print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}  prob: {:3}".format(class_indict[str(i)],
                                                predict[i].numpy()))
    plt.show()
    
if __name__ == '__main__':
    main()

