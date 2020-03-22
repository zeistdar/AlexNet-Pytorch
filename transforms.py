import numpy as np
import torch
from torch import Tensor

def RandomCrop(object):
    """
        Randomly crop parts  of the image with the specific output dimension
        Args:
            output_size (int or tuple): The required output size
    """
    def __init__(self, output_size: (int , tuple)) -> Tensor:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.random = np.random

    def __call__(self, img: Tensor) -> None:
        height, width = img.shape[:2]
        required_height, required_width = self.output_size[:2]

        new_width = self.random.randint(0, width - required_height + 1)
        new_height = self.random.randint(0, height - required_height + 1)
        new_img = img[new_height: new_height + required_height, new_width: new_width + required_width]

        return new_img



def HorizontalFlip(object):
    """
        Random horizontal flip of the image
        Args:
            prob (float): Probability at the end the image is flipped
    """
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        self.random = np.random.random_sample()
    
    def __call__(self, img: Tensor) -> Tensor:
        if self.random < self.prob:
            img = img[:, ::-1]
        return img

        
def VerticalFlip(object):
    """
        Random Vertical flip of the image
        Args:
            prob (float): Probability at the end the image is flipped
    """
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob
        self.random - np.random.random_sample()
    
    def __call__(self, img: Tensor) -> Tensor:
        if self.random < self.prob:
            img = img[::-1, :]
        return img

