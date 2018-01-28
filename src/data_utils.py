"""Data utility functions."""
import os
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import scipy.ndimage

# path where this file is located
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


def get_Dataset():
    # combine all existing Datasets get_XY() functions
    # concatenate CK, ISED, AN
    # IMPORTANT: Emotion labels indices in labels.csv either need to correspond to order in pics folder or they need to be present in every pictures name like this: 000001.png // amount of zeroes or .png/.jpg shoudln't matter. 	For latter IndexInPicName = True.
    # use either get_Huge_Dataset(DataSetNamePics, DataSetNameLabels, RGBDimensions, numberTrain) or get_Some_Dataset(DataSetName, numberTrain)
    # if you don't want to split dataset with get_Huge_Dataset() use for numberTrain = 0

    #CK_train, CK_val = get_Huge_Dataset('CK/pics/', 'CK/labels.csv', 1, 1100, False)
    #ISED_train, ISED_val = get_Huge_Dataset('ISED/pics/', 'ISED/labels.csv', 1, 350, False)
    #AN_train, AN_val = get_Huge_Dataset('AN/training/', 'AN/training_labels.csv', 3, 0,
    #                                    True), get_Huge_Dataset('AN/validation/', 'AN/validation_labels.csv', 3, 0, True)
    AN_train, AN_val = get_Huge_Dataset('AN/validation/', 'AN/validation_labels.csv', 3, 500, True)

    #dataset_train = data.ConcatDataset([CK_train, ISED_train, AN_train])
    #dataset_val = data.ConcatDataset([CK_val, ISED_val, AN_val])

    return AN_train, AN_val


def get_Some_Dataset(DataSetName, numberTrain):
    # load images into one big numpy array and load labels
    # make random mask for the training data set of given number of pictures
    # somehow order the rest of the pics into validation Data
    # return test_data, val_data

    labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s/labels.csv' % DataSetName, delimiter=',', usecols=1), dtype=np.int)

    # vgg_face base_model assume three input color channels, try to find more elegant solution than to copy the greyscale image to all three channels
    images = np.array([[np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64),
                        np.array(Image.open(ABS_PATH + '/../data/%s/pics/' % DataSetName + fname), dtype=np.float64)]
                       for fname in np.sort(os.listdir(ABS_PATH + '/../data/%s/pics' % DataSetName))])
    images /= 255.0
    images -= np.mean(images, axis=0)

    np.random.seed(0)  # split dataset the same way every time
    training_mask = np.sort(np.random.choice(range(labels.shape[0]), numberTrain, replace=False))
    validation_mask = np.sort(np.setdiff1d(list(range(labels.shape[0])), training_mask))

    # return training and validation data using the below defined class Data()
    return Data(images[training_mask], labels[training_mask]), Data(images[validation_mask], labels[validation_mask])


# Try make get_pics(Data()) more general! Maybe this doesn't work!!
def get_pics(train_data, val_data):
    amount_example_pics = 10

    # throws error: 'RandomSampler' object has no attribute '__getitem__'
    #sample_dataset = data.sampler.RandomSampler(data.ConcatDataset([train_data, val_data]))
    # return random_set.__getitem__(np.array(range(amount_example_pics)))[0], amount_example_pics

    np.random.seed()

    # choose 5 random pics
    filenames = np.sort(os.listdir(ABS_PATH + '/../data/CK/pics'))[np.random.choice(range(1245), amount_example_pics)]
    example_labels = [int(s.split('.')[0]) - 1 for s in filenames]
    example_labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s/labels.csv' % 'CK', delimiter=',', usecols=1), dtype=np.int)[example_labels]
    test_pics = [[np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                  np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64),
                  np.array(Image.open(ABS_PATH + '/../data/CK/pics/' + fname), dtype=np.float64)]
                 for fname in filenames]
    return test_pics, example_labels, filenames, amount_example_pics


def get_Huge_Dataset(DataSetNamePics, DataSetNameLabels, RGBDimensions, numberTrain, IndexInPicName):
    # Main Difference to get_Some_Dataset: load images with Solver when calling __getitem__()
    if 'AN' in DataSetNamePics:
        labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s' % DataSetNameLabels, usecols=1, skiprows=1), dtype=np.int)
    else:
        labels = np.array(np.loadtxt(ABS_PATH + '/../data/%s' % DataSetNameLabels, delimiter=',', usecols=1), dtype=np.int)
    data_path = ABS_PATH + '/../data/%s' % DataSetNamePics
    data_files = np.sort(os.listdir(data_path))

    if numberTrain != 0:
        np.random.seed(0)  # split dataset the same way every time
        training_mask = np.sort(np.random.choice(range(labels.shape[0]), numberTrain, replace=False))
        validation_mask = np.sort(np.setdiff1d(list(range(labels.shape[0])), training_mask))

        # return training and validation data using the below defined class Huge_Dataset()
        return Huge_Dataset(data_path, data_files[training_mask], labels[training_mask], RGBDimensions, IndexInPicName), Huge_Dataset(data_path, data_files[validation_mask], labels[validation_mask], RGBDimensions, IndexInPicName)
    else:
        return Huge_Dataset(data_path, data_files, labels, RGBDimensions, IndexInPicName)


def load_image(data_path, data_filename, dimension, mean, index):
    # open image in numpy array and check if file has only one dimension, we need 3 dimensions for our neural net! If it doesn't open, give dataloader an already used picture and do not put it in self.indices!
    try:
        img = np.array(Image.open(data_path + data_filename), dtype=np.float64)
    except:
        try:
            img = scipy.ndimage.imread(data_path + data_filename).astype(float)
        except:
            return index, True
    if dimension == 1:
        image = np.array([img, img, img])
    else:
        image = np.moveaxis(img, 2, 0)
    image /= 255.0
    image -= mean
    return image, False


def get_label_index(IndexInPicName, index, data_files, labels):
    if IndexInPicName:
        index_from_filename = int(data_files[index].split('.')[0])
        return labels[index_from_filenames]
    else:
        return labels[index]


class Huge_Dataset(data.Dataset):
    def __init__(self, data_path, data_files, labels, RGBDimensions, IndexInPicName):
        self.data_path = data_path
        self.data_files = data_files
        self.labels = labels
        self.RGBDimensions = RGBDimensions
        self.IndexInPicName = IndexInPicName

        # over the course of calling __getitem__() with the Solver, refine self.mean
        self.mean = 0.0
        self.indices = []

    def __getitem__(self, index):
        image, error = load_image(self.data_path, self.data_files[index], self.RGBDimensions, self.mean, index)
        if error:
            # generate random index that has already been used, only going to be called in first epoch
            try:
                rand = random.randint(0, len(self.indices) - 1)
            except:  # if self.indices has no entries yet, just take first element.
                if not 0 in self.indices:
                    self.indices.append(0)
                rand = 0
            image = load_image(self.data_path, self.data_files[self.indices[rand]], self.RGBDimensions, self.mean, self.indices[rand])[0]
            image = torch.from_numpy(image)
            return image, get_label_index(self.IndexInPicName, self.indices[rand], self.data_files, self.labels)

        # update mean
        if not index in self.indices:
            self.mean = (len(self.indices) * self.mean + image) / (len(self.indices) + 1)
            self.indices.append(index)
            self.indices.sort()

        image = torch.from_numpy(image)

        return image, get_label_index(self.IndexInPicName, index, self.data_files, self.labels)

    def __len__(self):
        return len(self.data_files)


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
