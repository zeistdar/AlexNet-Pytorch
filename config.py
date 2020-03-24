PARAMETERS = {
    "NUMBER_EPOCHS": 100,
    "BATCH_SIZE": 128,
    "MOMENTUM": 0.9,
    "INPUT_WIDTH": 227,
    "INPUT_HEIGHT": 227,
    "NUM_CLASSES": 3,
    "LEARNING_RATE": 0.001,
    "LEARNING_RATE_DECAY": 0.005,
    "KEEP_PROB": 0.5,
    "LR_SC_STEP_SIZE": 30,
    "LR_SC_GAMMA": 0.1,
    "SHUFFLE": True,
    "NUM_WORKERS": 8
}

INPUT_DIR = '/'
LOG_DIR_PATH = 'tensorboard_output'
IMG_DIM = 227

INPUT_DATASET_PATH = '/home/zeeshan/AlexNet-Pytorch/tiny-imagenet-200'
IMG_TYPE = 'JPEG'

ANNOTATION_FILEPATH_VAL = '/home/zeeshan/AlexNet-Pytorch/'
ANNOTATION_FILEPATH_TEST = '/home/zeeshan/AlexNet-Pytorch/'


ANNOTATION_FILENAME_VAL = "LOC_val_solution.csv"
ANNOTATION_FILENAME_TEST = "LOC_test_solutions.csv"

DEVICE_IDS = [0, 1, 2, 3]