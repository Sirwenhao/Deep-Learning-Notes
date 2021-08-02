import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('palne', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truch')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim = 0) # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
    #     predict = torch.max(outputs, dim = 1)[1].data.numpy()
    # print(classes[int(predict)])

    # 使用softmax函数，将输出转化为概率分布的方法
        predict = torch.softmax(outputs, dim=1)
    print(predict)

if __name__ == '__main__':
    main()