import os
import glob
from skimage import io, transform
from sklearn import preprocessing

def label_encoder(X):
    """
        Unique label generater in integer for classes based on the string or interger names
    """
    le = preprocessing.LabelEncoder()
    le = le.fit_transform(X)
    return le



def separate_dataset_and_classes(input_directory: str = "/home/zeeshan/AlexNet-Pytorch/tiny-imagenet-200", type_dataset: str = "train", img_type: str = "JPEG"):
    """
        Args:
            input_directory(string): root path to the directory containing the classes
            type(string): Is the dataset for training, validation or test. Options = ['train', 'test', 'val']

        Returns:
            X(Tensor) and Y(class)
    """
    print("Reading {}...".format(input_directory))

    full_path = os.path.join(input_directory, type_dataset)
    all_files = glob.glob(full_path + '/*/*/*.' + img_type)
    if not all_files:
        return None, None
    
    image_X = []
    class_Y = []

    for img in all_files:
        image_class = os.path.split(os.path.dirname(img))[-2]
        image_array = io.imread(img)
        image_X.append(image_array)
        class_Y.append(image_class)
    class_Y = label_encoder(class_Y)
    
    return image_X, class_Y
        





    





