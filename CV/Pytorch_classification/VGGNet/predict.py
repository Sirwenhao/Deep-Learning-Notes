# 2022/7/1  author:WH
# 参考WZMIAOMIAO

import os
import json
import torch
from PIL import Image
from model import vgg
from torchvision import transforms
import matplotlib as plt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "./tulip.jpg" # 存放待识别的图片的路径
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_dict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_dict = json.load(f)

    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.loda(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class:{} prob:{:.3}".format(class_dict[str(predict_cla)],
                                             predict[predict_cla].numpy())

    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.3}".format(class_dict[str(i)],
                                              predict[i].numpy()))

    plt.show()

if __name__ == "__main__":
    main()