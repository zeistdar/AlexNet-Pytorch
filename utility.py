import os
import glob
from skimage import io, transform
from sklearn.preprocessing import LabelEncoder
import config
import cv2
import pandas as pd
import numpy as np

def label_encoder(X) -> (np.ndarray, LabelEncoder):
    """
        Unique label generater in integer for classes based on the string or interger names
    """
    le = LabelEncoder()
    X = le.fit_transform(X)
    return X, le



def separate_dataset_and_classes(type_dataset: str = "train") -> (np.ndarray, np.ndarray, LabelEncoder):
    """
        Args:
            input_directory(string): Path to the directory containing the folders train, val and te
            type(string): Is the dataset for training, validation or test. Options = ['train', 'test', 'val']

        Returns:
            X(Tensor) and Y(class)
    """
    print("Reading {}...".format(config.INPUT_DATASET_PATH))


    full_path = os.path.join(config.INPUT_DATASET_PATH, type_dataset)
    all_files = glob.glob(full_path + '/*/*.' + config.IMG_TYPE)
    if not all_files:
        return None, None, None
    
    image_X = []
    class_Y = []

    for img in all_files:
        image_class = img.split('/')[-2]
        image_array = io.imread(img)
        resized_image_array = cv2.resize(image_array, (config.IMG_DIM, config.IMG_DIM), interpolation= cv2.INTER_NEAREST)
        if resized_image_array.ndim == 3:
            image_X.append(resized_image_array)
            class_Y.append(image_class)
    class_Y, enc = label_encoder(class_Y)
    return image_X, class_Y, enc

def separate_classes_val_test( enc: LabelEncoder, type_dataset: str = "val") -> (np.ndarray, np.ndarray):
    """
        Args:
            enc (LabelEncoder): Scikit learns label encoder fitted on the train classnames dataset
            type(string): Is the dataset for training, validation or test. Options = ['test', 'val']

        Returns:
            X(numpy) and Y(class)
    """
    print("Reading Validation/test  {}...".format(config.INPUT_DATASET_PATH))


    full_path = os.path.join(config.INPUT_DATASET_PATH, type_dataset)
    text_file = glob.glob(config.ANNOTATION_FILEPATH_VAL + '/' + config.ANNOTATION_FILENAME_VAL)
    if type_dataset == 'test':
        text_file = glob.glob(config.ANNOTATION_FILEPATH_TEST + '/' + config.ANNOTATION_FILENAME_TEST)
    annotated_file = open(text_file[0], 'r') 
    lines = annotated_file.readlines()
    class_name = []
    image_X = [] 
    for i, line in enumerate(lines):
        if i == 0:
            continue
        full_line = line.split(',')
        class_name_split = line[0].split(' ')
        actual_class = class_name_split[0]
        filename = os.path.join(full_path, full_line[0] + '.' + config.IMG_TYPE)
        image_array = io.imread(filename)
        resized_image_array = cv2.resize(image_array, (config.IMG_DIM, config.IMG_DIM), interpolation= cv2.INTER_NEAREST)
        if resized_image_array.ndim == 3:
            image_X.append(resized_image_array)
            class_name.append(actual_class)
    Y = enc.transform(class_name)
    
    return image_X, Y