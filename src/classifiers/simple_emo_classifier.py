"""
Simple FC-Layer on top of vgg_face_model.base_model for testing purposes of emotion classification in still images on sorted CK+ Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import base_model

# labels=np.loadtxt('labels.csv', delimiter=',')
# labels[1] -> [2,3]


class SimpleEmoClassifier(nn.Module):
    def __init__(self):
        super(SimpleEmoClassifier, self).__init__()

        self.base = base_model.base_model
        for param in base.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(25088, 8)

    def forward(self, x):

        x = self.base.forward(x)
        return self.fc1(x.view(x.size(0), -1))

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
