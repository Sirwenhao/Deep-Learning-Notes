# 24/01/09 author:WH
# 测试一批图像数据的分类结果

import os
import json
import torch
from PIL import Image
from torchvision import transforms

from model import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    #load image
    # 给定需要进行批量测试的图像文件夹路径
    imgs_root = '/Users/WH/Desktop/pytorch_classification/flower_imgs'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' does not exist."
    # 读取指定文件夹下所有的jpg文件
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg") or i.endswith(".jpeg")]
    # print("img_path_list", img_path_list)

    # read class_dict
    json_path = '/Users/WH/Desktop/Deep-Learning-for-image-processing/data_set/class_indices.json'
    assert os.path.exists(json_path), f"file: {json_path} does not exist."


    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)
    
    # load model weights
    weights_path = '/Users/WH/Desktop/Deep-Learning-for-image-processing/Pytorch_classification/ResNet/ResNet34_retrain.pth'
    assert os.path.exists(weights_path), f"file: {weights_path} does not exits."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 6 # 每次进行预测时打包的图片数量,我这里只有5张图片。。。
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids+1)*batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
if __name__ == '__main__':
    main()
