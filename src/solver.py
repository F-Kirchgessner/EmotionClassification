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
                 loss_func=torch.nn.CrossEntropyLoss(), data='CK'):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        # um den vielen neutralen Bildern entgegenzuwirken
        if data == 'CK':
            weight = torch.Tensor([0.06070826, 0.4, 1.0, 0.30508475, 0.72,
                                   0.26086957, 0.64285714, 0.21686747])
        elif data == 'ISED':
            weight = torch.Tensor([0.0,  0.0, 0.0, 0.18691589, 0.0,
                                   0.53037383, 0.11214953, 0.17056075])
        if torch.cuda.is_available():
            weight = weight.cuda()
        self.loss_func = torch.nn.CrossEntropyLoss(weight=weight)

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

        # filter out frozen grads of base_model for optimizer
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model = model.cuda()

        if log_nth != 0:
            print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs.float()), Variable(targets.long())
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           train_loss))

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0, not relevant in our use case
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = Variable(inputs.float()), Variable(targets.long())
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.data.cpu().numpy())

                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets >= 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)

            model.train()
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))

        if log_nth != 0:
            print('FINISH.')
