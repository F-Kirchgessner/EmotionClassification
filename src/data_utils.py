"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def get_CK():
    # load images into one big numpy array and load labels
    # make random mask for the training data set of 1000 pictures
    # somehow order the rest of the pics into validation Data
    # return test_data, val_data
    labels = np.loadtxt('../data/CK/labels.csv', delimiter=',')[:, 1]
    images = np.array([np.array(Image.open(fname))[np.newaxis, :, :]
                       for fname in np.sort(os.listdir('../data/CK/pics'))])
    images /= 255.0
    mean_image = np.mean(images, axis=0)
    images -= mean_image

    # TODO seperate into training and validation data
    training_mask = np.sort(np.random.choice(range(labels.shape[0]), 1000, replace=False))

    # return training and validation data
    return (CK_Data(X_train, y_train), CK_Data(X_val, y_val))


class CK_Data(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.y)
