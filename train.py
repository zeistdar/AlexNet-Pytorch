import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
import config
import alexnetdataloader
import alexnetmodel
from transforms import CenterCrop, HorizontalFlip, VerticalFlip, ToTensor, Normalize 
from torchvision import transforms as tf
from torch.utils import data
from utility import separate_dataset_and_classes, separate_classes_val_test
from tensorboardX import SummaryWriter
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':


    tbwiter = SummaryWriter(log_dir= config.LOG_DIR_PATH)

    print("Initializing AlexNet model")
    alexnet = alexnetmodel.AlexNet(num_classes = config.PARAMETERS['NUM_CLASSES'])
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=config.DEVICE_IDS)
    print("Initilaizing transfrmations to apply on the image")
    transformations = tf.Compose([
        CenterCrop(64),
        # HorizontalFlip(),
        ToTensor(),
        Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
    )

    print("Generating train images and labels")
    train_X, train_Y, enc = separate_dataset_and_classes()
    print("Generating train images and labels")
    val_X, val_Y = separate_classes_val_test(enc = enc)

    print("Initilazing data loader for Alexnet")
    train_dataset = alexnetdataloader.AlexNetDataLoader(train_X, train_Y, transform = transformations)
    val_dataset = alexnetdataloader.AlexNetDataLoader(val_X, val_Y, None)

    
    print(len(train_dataset))
    print(len(val_dataset))
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size = config.PARAMETERS['BATCH_SIZE'],
        shuffle = True,
        num_workers = config.PARAMETERS['NUM_WORKERS']
    )

    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size = config.PARAMETERS['BATCH_SIZE'],
        shuffle = True,
        num_workers = config.PARAMETERS['NUM_WORKERS']
    )

    optimizer = optim.SGD(
        params = alexnet.parameters(),
        lr = config.PARAMETERS['LEARNING_RATE'],
        momentum = config.PARAMETERS['MOMENTUM'],
        weight_decay = config.PARAMETERS['LEARNING_RATE_DECAY']
    )

    lrScheduler = optim.lr_scheduler.StepLR(optimizer, config.PARAMETERS['LR_SC_STEP_SIZE'], config.PARAMETERS['LR_SC_GAMMA'])
    logging.info("Strating Training")
    for epoch in range(config.PARAMETERS['NUMBER_EPOCHS']):
        total_train_loss = 0
        total_val_loss = 0
        total_steps = 0
        val_accuracy = 0

        for imgs, labels in train_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = alexnet(imgs)
            loss = F.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_steps += 1
            if total_steps == 10:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == labels)
                    tbwiter.add_scalar('Train Loss', loss.item(), total_steps)
                    tbwiter.add_scalar('Train Accuracy', accuracy.item(), total_steps)
                    print("Epoch {} \t Train Loss {} \t Train Accuracy {}".format(epoch + 1, total_train_loss, accuracy.item()))
                    for val_imgs, val_labels in val_dataloader:
                        val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)

                        val_output = alexnet(val_imgs)
                        val_loss = F.cross_entropy(val_output, val_labels)

                        total_val_loss += val_loss.item()
                        _, val_preds = torch.max(val_output, 1)
                        val_accuracy = torch.sum(val_preds == val_labels)
                    tbwiter.add_scalar('Validation Loss', loss.item(), total_steps)
                    tbwiter.add_scalar('Validation Accuracy', val_accuracy.item(), total_steps)
                    print("Epoch {} \t Validation Loss {} \t Validation Accuracy {}".format(epoch + 1, total_val_loss, val_accuracy))
            
        print("Total Valid loss", total_val_loss)
        lrScheduler.step()