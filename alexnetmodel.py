import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['AlexNet']

class AlexNet(nn.Module):
    """
    Model proposed by the original ALexNet Paper
    """
    def __init__(self, num_classes: int) -> None:
        """
            Initializing the architecture

            Args:
                num_classes(int): Number of classes to predict from this model
        """

        super.__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, bias = True)
        self.relu = nn.ReLU()
        self.LRN1 = nn.LocalResponseNorm(k = 2, size = 5, alpha = 0.0001, beta = 0.75)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2, bias = True)
        self.LRN2 = nn.LocalResponseNorm(k = 2, size = 5, alpha = 0.0001, beta = 0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1, bias = True)

        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1, bias = True)

        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1, bias = True)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.dropout = nn.Dropout(p = 0.5, inplace = True)
        self.fc1 = nn.Linear(in_features = (256 * 6 * 6), out_features = 4096)
        
        self.fc2 = nn.Linear(in_features = 4096, out_features = 4096)

        self.fc3 = nn.Linear(in_features = 4096, out_features = num_classes)

    
    def forward(self, x: Tensor) -> Tensor:
        """
        Passing the image through the architecture

        Args:
            x (tensor): Input image/tensor
        
        Return:
            x (tensor): Output Tensor
        """

        x = self.conv1(x)
        x = self.relu(x)
        x = self.LRN1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.LRN2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.maxpool3(x)

        x = x.view(-1, 256 * 6 * 6)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x        





