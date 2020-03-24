# Pytorch Implementation of AlexNet paper
Implementation of the [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Alex Krizhevsky. This was the first major architecture that made a breakthrough in the imagenet classification.

# Dataset

The dataset used for this implementation is `IMAGENET` dataset. The Imagenet dataset is around 167 GB. This repsoirtory used one from the kaggle competition ImageNet localization Challange. There are 1000 classes in the dataset. You can change the the number of classes that the network outputs in the `config.py` file. You have to provide the path where your train, test and the val folders are: This respository assumes the following strcture of the path.

```
-path_to_train_test_val_folder
---train
------class 1
---------filename1.JPEG

---val
------class 1
---------filename1.JPEG

---test
------class 1
---------filename1.JPEG
```

For validation you can also provide a csv file which links the filename to a class. In that case the structure is assumed as. You can provide the file path in the `config.py` file. The format of the csv file is as under.
```
filename,predictionstring
filename1,class_1
```
And for the actual files path folder
```
-path_to_train_test_val_folder
---val
------filename1.JPEG
------filename2.JPEG
```

# Dependencies

To meet all the dependdencies type the following command in terminal.
`python3 install -r requirements.txt`


# Training
`python3 train.py`

# Tuning parameters
You can tune parameters from the `config.py` file to try out different variations of the training loop.
