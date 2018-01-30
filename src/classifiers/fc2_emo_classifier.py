"""
2 FC-Layers on top of vgg_face_model.base_model for testing purposes of emotion classification in still images on sorted AN Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import vgg_face_model.base_model as base_model

# labels=np.loadtxt('labels.csv', delimiter=',')
# labels[1] -> [2,3]


class FC2EmoClassifier(nn.Module):
    def __init__(self, weight_scale=0.001):
        super(SimpleEmoClassifier, self).__init__()

        self.base = base_model.base_model
        for param in self.base.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(32768, 500)
        self.fc2 = nn.Linear(500, 300, bias=True)
        self.fc3 = nn.Linear(300, 8, bias=True)

        nn.init.normal(self.fc2.weight.data, std=weight_scale)
        nn.init.normal(self.fc3.weight.data, std=weight_scale)

    def forward(self, x):

        x.data = x.data.float()
        x = self.base.forward(x)
        x.data = x.data.float()
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

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
