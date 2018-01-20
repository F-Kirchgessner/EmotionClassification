"""
Simple FC-Layer on top of vgg_face_model.base_model for testing purposes of emotion classification in still images on sorted CK+ Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import vgg_face_model.base_model as base_model

# labels=np.loadtxt('labels.csv', delimiter=',')
# labels[1] -> [2,3]


class SimpleEmoClassifier(nn.Module):
    def __init__(self, weight_scale=0.001):
        super(SimpleEmoClassifier, self).__init__()

        self.base = base_model.base_model
        for param in self.base.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(32768, 8, bias=True)
        nn.init.normal(self.fc1.weight.data, std=weight_scale)

    def forward(self, x):

        x.data = x.data.float()
        x = self.base.forward(x)
        #print(x.size, x.data.shape)
        x.data = x.data.float()
        return self.fc1(x.view(x.size(0), -1))

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)