from random import shuffle
import numpy as np
import cv2
from src.weight_compensation import get_AN_val_compensation_weights, get_AN_train_compensation_weights
import datetime

import torch
from torch.autograd import Variable
import torchvision

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

useTensorboard = False

# For people that have their torch.cuda.is_available() = True, yet their GPU is too old...SAD!
# IMPORTANT: Remember to set TRUE when needed!!!
GPU_Computing = True

# path of this file
ABS_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

try:
    from tensorboardX import SummaryWriter
except ImportError:
    useTensorboard = False


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
        # one weight that correspends for unequal amount of emotion labels in AN, CK and ISED datasets
        # weight for one emotion is small if we have a lot of pictures with that emotion label
        # Old labels: 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise --> DO NOT USE!
        # New labels: 0=neutral, 1=happy, 2=sad, 3=surprise, 4=fear, 5=disgust, 6=anger, 7=contempt
        AN_train_compensation_weights = torch.Tensor(get_AN_train_compensation_weights())
        AN_val_compensation_weights = torch.Tensor(get_AN_val_compensation_weights())
        if torch.cuda.is_available() and GPU_Computing:
            AN_train_compensation_weights = torch.FloatTensor(AN_train_compensation_weights).cuda()
            AN_val_compensation_weights = torch.FloatTensor(AN_val_compensation_weights).cuda()
        self.loss_func_train = torch.nn.CrossEntropyLoss(weight=AN_train_compensation_weights)
        self.loss_func_val = torch.nn.CrossEntropyLoss(weight=AN_val_compensation_weights)

        self._reset_histories()

        # Logger
        # tensorboard --logdir=runs --reload_interval=5
        # ssh -L 16006:127.0.0.1:6006 clientname@serverip
        self.numValExamples = 5
        # New labels: 0=neutral, 1=happy, 2=sad, 3=surprise, 4=fear, 5=disgust, 6=anger, 7=contempt
        self.emotions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}
        if useTensorboard:
            self.writer = SummaryWriter()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, val_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        if useTensorboard:
            # Log validation examples
            for inputs, targets in val_loader:
                self.writer.add_image('Validation examples', torchvision.utils.make_grid(inputs[:self.numValExamples].float(), normalize=True), 0)
                break

        # filter out frozen grads of base_model for optimizer
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        print("%s batches per epoch with %s images per batch." %(iter_per_epoch, train_loader.batch_size))

        if torch.cuda.is_available() and GPU_Computing:
            model = model.cuda()

        for epoch in range(num_epochs):
            # TRAINING
            train_loss = 0

            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs.float()), Variable(targets.long())
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func_train(outputs, targets)
                loss.backward()
                optim.step()

                train_loss = loss.data.cpu().numpy()
                self.train_loss_history.append(train_loss)
                if useTensorboard:
                    self.writer.add_scalar('Training loss', train_loss, i + epoch * iter_per_epoch)
                if log_nth and (i + epoch * iter_per_epoch) % log_nth == 0:
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
                    if useTensorboard:
                        self.writer.add_scalar('Training accuracy', train_acc, i + epoch * iter_per_epoch)
                    if log_nth:
                        print('[Iteration %d/%d, Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (i + epoch * iter_per_epoch,
                                                                           iter_per_epoch * num_epochs,
                                                                           epoch + 1,
                                                                           num_epochs,
                                                                           train_acc,
                                                                           train_loss))

                if val_nth and (i + epoch * iter_per_epoch) % (val_nth) == 0:
                    self.runValidation(model, val_loader, i, epoch, iter_per_epoch, num_epochs)
            self.runValidation(model, train_loader, iter_per_epoch, epoch, iter_per_epoch, num_epochs)
            self.runValidation(model, val_loader, iter_per_epoch, epoch, iter_per_epoch, num_epochs)
            self.saveModel(model, epoch)
            self.savePerformance(epoch)


    def runValidation(self, model, some_loader, iter, epoch, iter_per_epoch, num_epochs):
        # VALIDATION
        val_losses = []
        val_scores = []
        model.eval()
        logOnce = False
        for j, (inputs, targets) in enumerate(some_loader, 1):
            x, tar = Variable(inputs.float()), Variable(targets.long())
            if model.is_cuda:
                x, tar = x.cuda(), tar.cuda()

            output = model.forward(x)
            loss = self.loss_func_val(output, tar)
            val_losses.append(loss.data.cpu().numpy())

            _, preds = torch.max(output, 1)

            # Only allow images/pixels with target >= 0 e.g. for segmentation
            targets_mask = tar >= 0
            scores = np.mean((preds == tar)[targets_mask].data.cpu().numpy())
            val_scores.append(scores)

            if not logOnce and useTensorboard:
                logOnce = True
                prediction = ''
                output = output.cpu().data.numpy()
                preds = preds.cpu().data.numpy()
                
                if 'val' in some_loader:
                    for i in range(self.numValExamples):
                        prediction += '%s: Truth=%s, Pred=%s, N=%.2e, A=%.2e, C=%.2e, D=%.2e, F=%.2e, H=%.2e, Sad=%.2e, Sur=%.2e  \n' % (
                            i, self.emotions[targets[i]], self.emotions[preds[i]], list(output[i])[0], list(output[i])[1], list(output[i])[2], list(output[i])[3], list(output[i])[4], list(output[i])[5], list(output[i])[6], list(output[i])[7])
                    self.writer.add_text('Validation predictions', prediction, epoch + 1)
             
            # stop after 4000 pictures (divided by batch size) if train_loader is being used
            if 'train' in some_loader and j >= (4000 / 25):
                break   
                
        model.train()
        if 'val' in some_loader:
            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if useTensorboard:
                self.writer.add_scalar('Validation loss', val_loss, iter + epoch * iter_per_epoch)
                self.writer.add_scalar('Validation accuracy', val_acc, iter + epoch * iter_per_epoch)

                for name, param in model.named_parameters():
                    if not "base" in name:
                        self.writer.add_histogram(name, param, epoch + 1)

            print('[Iteration %d/%d, Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (iter + epoch * iter_per_epoch,
                                                               iter_per_epoch * num_epochs,
                                                               epoch + 1,
                                                               num_epochs,
                                                               val_acc,
                                                               val_loss))
        if 'train' in some_loader:
            train_acc, train_loss = np.mean(val_scores), np.mean(val_losses)
            self.train_acc_history.append(train_acc)
            self.train_loss_history.append(train_loss)
            if useTensorboard:
                self.writer.add_scalar('Total Training loss', val_loss, iter + epoch * iter_per_epoch)
                self.writer.add_scalar('Total Training accuracy', val_acc, iter + epoch * iter_per_epoch)


            print('[Iteration %d/%d, Epoch %d/%d] TRAIN   acc/loss: %.3f/%.3f' % (iter + epoch * iter_per_epoch,
                                                               iter_per_epoch * num_epochs,
                                                               epoch + 1,
                                                               num_epochs,
                                                               train_acc,
                                                               train_loss))

    def savePerformance(self, epochs):
        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss_history, '-', label='train_loss')
        x = np.linspace(0, len(self.train_loss_history), len(self.val_loss_history))
        plt.plot(x, self.val_loss_history, '-o', label='val_loss')
        plt.legend(loc='upper right')
        plt.title('Training vs Validation loss | %d Epochs' % epochs)
        plt.xlabel('iteration')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(self.train_acc_history, '-o', label='train_acc=%.4f' % (self.train_acc_history[-1]))
        plt.plot(self.val_acc_history, '-o', label='val_acc=%.4f' % (self.val_acc_history[-1]))
        plt.legend(loc='upper left')
        plt.title('Training vs Validation accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.gca().yaxis.grid(True)

        plt.gcf().set_size_inches(15, 15)
        plt.tight_layout()
        plt.savefig(ABS_PATH + '../output/performance_{}.png'.format(timestamp))
        plt.gcf().clear()

    def saveModel(self, model, epoch):
        # Save model after each epoch
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = "models/model_{}".format(timestamp)
        model.save(fileName + "_e%s.model" %epoch)
