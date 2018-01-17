"""Data utility functions."""
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def get_CK():
    # load images into one big numpy array and load labels
    # make random mask for the training data set of 1000 pictures
    # somehow order the rest of the pics into validation Data
    # return test_data, val_data

    labels = np.array(np.loadtxt(ABS_PATH + '/../data/CK/labels.csv',
                                 delimiter=',')[:, 1], dtype=np.int)

    # vgg_face base_model assume three input color channels, try to find more elegant solution than to copy the greyscale image to all three channels
    images = np.array([[np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64)]
                       for fname in np.sort(os.listdir(ABS_PATH + '/../data/CK/pics'))])
    images /= 255.0
    images -= np.mean(images, axis=0)

    np.random.seed(0)  # split dataset the same way every time
    training_mask = np.sort(np.random.choice(range(labels.shape[0]), 1100, replace=False))
    validation_mask = np.sort(np.setdiff1d(list(range(labels.shape[0])), training_mask))

    # return training and validation data
    return CK_Data(images[training_mask], labels[training_mask]), CK_Data(images[validation_mask], labels[validation_mask])


def get_pics():
    np.random.seed()
    filenames = np.sort(os.listdir(ABS_PATH + '/../data/CK/pics')
                        )[np.random.choice(range(1100, 1245), 5)]
    test_pics = np.array([[np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                           np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname),
                                    dtype=np.float64),
                           np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64)]
                          for fname in filenames])
    return test_pics, filenames


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


class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples
