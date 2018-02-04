"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmoClassifier(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(512, 7, 7), num_filters=128, kernel_size=5,
                 weight_scale=0.001, num_classes=8, dropout=0.5, padding=2):   
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scale for the convolution weights initialization
        - dropout: Probability of an element to be zeroed.
        """
        super(CNNEmoClassifier, self).__init__()
        channels, height, width = input_dim
    
        self.base = base_model.base_model
        for param in self.base.parameters():
            param.requires_grad = False
        
        #use pytorchs conv2d layer, what is output channels --> number of filters??!!, dim(w1) = (F, C, HH, WW)
        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size, bias = True, padding=padding)
        nn.init.normal(self.conv1.weight.data, std = weight_scale)
             
        #input dim (Height, width) are dependent of padding and kernel size, ignoring stride and dilation
        height = height + 2*padding - (kernel_size - 1)
        width = width + 2*padding - (kernel_size - 1)
        fc_dim = num_filters * height * width 
        
        #dropout operation, relu again has no parameters
        self.dropout = nn.Dropout(p = dropout)
        
        #linear layer might not take conv dimensions as input dims --> view(-1) for input tensor; linear op does not need num_images!!
        self.fc1 = nn.Linear(in_features = fc_dim, out_features = num_classes, bias = True)
        nn.init.normal(self.fc1.weight.data, std = weight_scale)
 
        #second fc layer operation and parameters
        #self.fc2 = nn.Linear(hidden_dim, num_classes, bias = True)
        #nn.init.normal(self.fc2.weight.data, std = weight_scale)
        

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x.data = x.data.float()
        x = self.base.forward(x)
        x.data = x.data.float()
        
        x = self.conv1(x)
        x = F.relu(x)     
        print(x.shape)
        
        #flatten x, except first dim --> num_images
        x = x.view(x.data.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(x)  
        x = self.fc1(x)

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
