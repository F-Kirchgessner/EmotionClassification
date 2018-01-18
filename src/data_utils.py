"""Data utility functions."""
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# path where this file is located
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def get_Dataset():
    # combine all existing Datasets get_XY() functions
    # concatenate CK, ISED
    CK_train, CK_val = get_Some_Dataset('CK', 1100)
    ISED_train, ISED_val = get_Some_Dataset('ISED', 350)
    dataset_train = data.ConcatDataset([CK_train, ISED_train])
    dataset_val = data.ConcatDataset([CK_val, ISED_val])

    return dataset_train, dataset_val


def get_Some_Dataset(DataSetName, numberOfTrainPics):
    # load images into one big numpy array and load labels
    # make random mask for the training data set of given number of pictures
    # somehow order the rest of the pics into validation Data
    # return test_data, val_data

    labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s/labels.csv' % DataSetName, delimiter=',')[:, 1], dtype=np.int)

    # vgg_face base_model assume three input color channels, try to find more elegant solution than to copy the greyscale image to all three channels
    images = np.array([[np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64)]
                       for fname in np.sort(os.listdir(ABS_PATH + '/../data/%s/pics' % DataSetName))])
    images /= 255.0
    images -= np.mean(images, axis=0)

    np.random.seed(0)  # split dataset the same way every time
    training_mask = np.sort(np.random.choice(range(labels.shape[0]), numberOfTrainPics, replace=False))
    validation_mask = np.sort(np.setdiff1d(list(range(labels.shape[0])), training_mask))

    # return training and validation data using the below defined class Data()
    return Data(images[training_mask], labels[training_mask]), Data(images[validation_mask], labels[validation_mask])


# Try make get_pics(Data()) more general! Maybe this doesn't work!!
def get_pics(train_data, val_data):
    amount_example_pics = 5

    # throws: 'RandomSampler' object has no attribute '__getitem__'
    #sample_dataset = data.sampler.RandomSampler(data.ConcatDataset([train_data, val_data]))
    # return random_set.__getitem__(np.array(range(amount_example_pics)))[0], amount_example_pics

    np.random.seed()

    # choose 5 random pics
    filenames = np.sort(os.listdir(ABS_PATH + '/../data/CK/pics'))[np.random.choice(range(1245), amount_example_pics)]
    example_labels = [int(s.split('.')[0]) for s in filenames]
    example_labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s/labels.csv' % 'CK', delimiter=',')[:, 1], dtype=np.int)[example_labels - 1]
    test_pics = [[np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                  np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname),
                           dtype=np.float64),
                  np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64)]
                 for fname in filenames]
    return test_pics, example_labels, filenames, amount_example_pics


class Data(data.Dataset):

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
