# 24/01/09  author::WH

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image
    img_path = 'data_set/tulip.jpg'
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.show(img)
    # [N, C, H, W] N为batch_size,此处应该等于1
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'data_set/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weigths_path = "./ResNet34_retrain.pth"
    assert os.path.exists(weigths_path), "file: '{}' does not exist.".format(weigths_path)
    model.load_state_dict(torch.load(weigths_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_cla)],
                                               predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))

if __name__ == '__main__':
    main()
    



