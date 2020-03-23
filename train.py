import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
import config
import alexnetdataloader
import alexnetmodel
from transforms import CenterCrop, HorizontalFlip, VerticalFlip, ToTensor, Normalize 
from torchvision import transforms
from torch.utils import data
from utility import separate_dataset_and_classes, separate_classes_val_test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':


    alexnet = alexnetmodel.AlexNet(num_classes = config.PARAMETERS['NUM_CLASSES'])
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=config.DEVICE_IDS)
    transformations = transforms.Compose([
        CenterCrop(64),
        HorizontalFlip(),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]
    )

    train_X, train_Y, enc = separate_dataset_and_classes()
    val_X, val_Y = separate_classes_val_test(enc = enc)

    train_dataset = alexnetdataloader.AlexNetDataLoader(train_X, train_Y, transform = transformations)
    val_dataset = alexnetdataloader.AlexNetDataLoader(val_X, val_Y, None)



    dataloader = data.DataLoader(
        train_dataset,
        batch_size = config.PARAMETERS['BATCH_SIZE'],
        shuffle = config.PARAMETERS['BATCH_SIZE'],
        num_workers = config.PARAMETERS['NUM_WORKERS']
    )

    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size = config.PARAMETERS['BATCH_SIZE'],
        shuffle = config.PARAMETERS['BATCH_SIZE'],
        num_workers = config.PARAMETERS['NUM_WORKERS']
    )

    optimizer = optim.SGD(
        params = alexnet.parameters(),
        lr = config.PARAMETERS['LEARNING_RATE'],
        momentum = config.PARAMETERS['MOMENTUM'],
        weight_decay = config.PARAMETERS['LEARNING_RATE_DECAY']
    )

    lrScheduler = optim.lr_scheduler.StepLR(optimizer, config.PARAMETERS['LR_SC_STEP_SIZE'], config.PARAMETERS['LR_SC_GAMMA'])

    for epoch in range(config.PARAMETERS['NUMBER_EPOCHS']):
        lrScheduler.step()
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = alexnet(imgs)
            loss = F.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()