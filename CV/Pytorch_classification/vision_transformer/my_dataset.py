# 2024/6/26

from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images_path, images_class, transform):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
        