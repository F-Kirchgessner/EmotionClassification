from random import shuffle
import numpy as np
import cv2

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(model.parameters(), **self.optim_args)
        criterion = torch.nn.CrossEntropyLoss()
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        num_iterations = num_epochs * iter_per_epoch

        for epoch in range(num_epochs):
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels.type(torch.LongTensor))

                pic = cv2.imread('vgg_face_model/candice.png')
                pic = cv2.resize(pic, (200, 200), interpolation = cv2.INTER_LINEAR)
                pic = pic[np.newaxis,:,:,:]
                x = Variable(torch.Tensor(pic))
                x = x.permute(0,3,1,2)

                print(x)

                if model.is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()

                # print statistics
                if i % 50 == 50 - 1:
                    self.train_loss_history.append(loss.data[0])

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels.data)
                    train_acc = 100*correct/total

                    self.train_acc_history.append(train_acc)

                    print('[%d, %5d] training:   loss: %.3f,  acc: %.3f' %(epoch + 1, i + 1, loss.data[0], train_acc))


            # Validation
            correctVal = 0
            totalVal = 0
            runningLossVal = 0.0
            for i, data in enumerate(val_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels.type(torch.LongTensor))

                if model.is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                runningLossVal += loss.data[0]

                _, predicted = torch.max(outputs.data, 1)
                totalVal += labels.size(0)
                correctVal += torch.sum(predicted == labels.data)
                trainAccVal = 100*correctVal/totalVal

            # print statistics
            self.val_loss_history.append(runningLossVal)
            self.val_acc_history.append(trainAccVal)
            print('[%d, %5d] validation: loss: %.3f, acc: %.3f' %(epoch + 1, i + 1, runningLossVal, trainAccVal))

            
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')