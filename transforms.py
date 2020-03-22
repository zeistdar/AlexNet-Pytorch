import numpy as np
import torch
from torch import Tensor

def RandomCrop(object):
    """
        Randomly crop parts  of the image with the specific output dimension
        Args:
            output_size (int or tuple): The required output size
    """
    def __init__(self, output_size: (int , tuple)) -> np.ndarray:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.random = np.random

    def __call__(self, img: np.ndarray) -> None:
        height, width = img.shape[:2]
        required_height, required_width = self.output_size

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
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
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
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.random < self.prob:
            img = img[::-1, :]
        return img


def CenterCrop(object):
    """
        Center Crop of the image
        Args:
            output_size (int or tuple): The required output size
    """
    def __init__(self, output_size: (int, tuple)) -> None:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        required_height, required_width = self.output_size

        if height == required_height and width == required_width:
            return img
        assert (height > required_height and width > required_width)

        new_width = int(round(width - required_width/ 2.))
        new_height = int(round(height - required_height/ 2.))
        img = img[new_height: new_height + required_height, new_width: new_width + required_width]

        return img
    

def ToTensor(object):
    """
        Convert image to Tensor
    """
    def __init__(self, is_image_channel_first: bool = False):
        self. is_image_channel_first = is_image_channel_first = False

    def __call__(self, img: np.ndarray) -> Tensor:
        if not self.is_image_channel_first:
            img = img.traspose((2, 0, 1))
        return img


def Normalize(object)
    """
        Normalize the image with the provided mean and std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        assert isinstance(img, np.ndarray)

        return (img - self.mean) / self.std
            
        



