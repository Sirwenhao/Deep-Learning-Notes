from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDastaSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图，L为灰度图
        if img.mode != 'RGB':
            raise ValueError("image: {} is not RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        
        if self.transform is not None:
            img = self.images_class[item]
            
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)     # torch.as_tensor()创建张量，共享底层数据
        return images, labels
