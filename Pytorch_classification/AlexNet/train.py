import matplotlib
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/AlexNetTrain')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.getcwd()) #获取当前train.py文件所在的位置
print(os.getcwd())
#data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) #get data root path
# os.getcwd()表示获取当前文件所在的目录，“../..”是指获取上上层文件夹的目录，”..“代表上一层，四个点就代表上上层
#image_path = data_root + "\\data_set\\flower_data\\" #flower dataset path
image_path = 'data_set\\flower_data/'
print(image_path)
train_dataset = datasets.ImageFolder(root = image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
#{'diasy':0, 'dandelion:1', 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, shuffle = True,
                                           num_workers = 0)

#get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

#create grid of images
img_grid = torchvision.utils.make_grid(images)
print(type(img_grid))

# #show images
# plt.imshow(img_grid, one_channel = True)

#write to tensorboard
writer.add_image('flower_images', img_grid)

validate_dataset = datasets.ImageFolder(root = image_path + "val",
                                        transform = data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size = 4, shuffle = True,
                                              num_workers = 0)

test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()

# def imshow(img):
#     img = img / 2 + 0.5 #unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes = 5, init_weights = True).to(device)

init_img = torch.zeros((1, 3, 224, 224), device=device)
writer.add_graph(net, init_img)


loss_function = nn.CrossEntropyLoss()
#pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr = 0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0
for epoch in range(10):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start = 0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        #print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "*" * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end = "")
    print(time.perf_counter()-t1)

    #validate
    net.eval()
    acc = 0.0 # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs =net(test_images.to(device))
            predict_y = torch.max(outputs, dim = 1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f'%
              (epoch + 1,running_loss / step, val_accurate))

    #add loss, acc and lr into tensorboard
    tags = ["running_loss", "val_accurate", "learning_rate"]
    writer.add_scalar(tags[0], running_loss / step, epoch)
    writer.add_scalar(tags[1],  val_accurate, epoch)
    writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)


print('Finished Training')