import os
import glob
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder

def label_encoder(X):
    """
        Unique label generater in integer for classes based on the string or interger names
    """
    le = LabelEncoder()
    X = le.fit_transform(X)
    return X, le



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
    class_Y, enc = label_encoder(class_Y)
    
    return image_X, class_Y, enc

def separate_classes_val_test( enc: LabelEncoder, input_directory: str = "/home/zeeshan/AlexNet-Pytorch/tiny-imagenet-200", type_dataset: str = "val", img_type: str = "JPEG", annotation_filename: str = "val_annotations.txt"):
    """
        Args:
            input_directory(string): root path to the directory containing the classes
            type(string): Is the dataset for training, validation or test. Options = ['train', 'test', 'val']

        Returns:
            X(Tensor) and Y(class)
    """
    print("Reading {}...".format(input_directory))


    full_path = os.path.join(input_directory, type_dataset)
    all_files = glob.glob(full_path + '/*/*.' + img_type)
    text_file = glob.glob(full_path + '/' + annotation_filename)
    annotated_file = open(text_file[0], 'r') 
    lines = annotated_file.readlines()
    class_name = []
    image_X = [] 
    for line in lines:
        full_line = line.split()
        class_name.append(full_line[1])

    Y = enc.transform(class_name)

    for img in all_files:
        image_array = io.imread(img)
        image_X.append(image_array)
    
    return image_X, Y

    
