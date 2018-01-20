"""
Simple FC-Layer on top of vgg_face_model.base_model for testing purposes of emotion classification in still images on sorted CK+ Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import vgg_face_model.base_model as base_model

import dlib
import numpy as np

# labels=np.loadtxt('labels.csv', delimiter=',')
# labels[1] -> [2,3]


class LandmarkEmoClassifier(nn.Module):
    def __init__(self, weight_scale=0.001):
        super(SimpleEmoClassifier, self).__init__()

        self.base = base_model.base_model
        for param in self.base.parameters():
            param.requires_grad = False

        #self.fc1 = nn.Linear(32768, 8, bias=True)
        #self.fc1 = nn.Linear(32768, 500)
        self.fc1 = nn.Linear(32904, 300)
        self.fc2 = nn.Linear(300, 8, bias=True)
        #self.fc2 = nn.Linear(200, 200)
        #self.fc3 = nn.Linear(200, 8, bias=True)

        nn.init.normal(self.fc2.weight.data, std=weight_scale)

    def forward(self, x):
        predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        points = []
        for i in range(len(x)):
            points.append([])
            shape = predictor(((x[i][0].cpu().data.numpy() + 0.5) * 255).astype(np.uint8), dlib.rectangle(left=0, top=0, right=256, bottom=256))
            for p in range(68):
                points[i].append((shape.part(p).x / 255.0) - 0.5)
                points[i].append((shape.part(p).y / 255.0) - 0.5)

            """
            # Display faces and landmarks
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(((x[i][0].cpu().data.numpy() + 0.5) * 255).astype(np.uint8))

            win.add_overlay(shape)
            dlib.hit_enter_to_continue()
            """

        x = self.base.forward(x)
        x.data = x.data.float()
        x = x.view(x.size(0), -1)

        x = torch.cat((x, torch.FloatTensor(points).cuda()), 1)

        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)

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
