import torch
from torch.utils import data
from utility import separate_dataset_and_classes

class AlexNetDataLoader(data.Dataset):
    """
        Class for loading the dataset from a specific path
    """
    def __init__(self, input_path: str = "/home/zeeshan/AlexNet-Pytorch/tiny-imagenet-200", type_dataset: str = "train", img_type: str = "JPEG", transform = None  ):
        self.images, self.classes = separate_dataset_and_classes(input_path, type_dataset, img_type)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.transform:
            self.images[idx] = self.transform(self.images[idx])
        return self.images[idx], self.classes[idx]
