import torch
import numpy as np
import scipy.ndimage
from torch.autograd import Variable
import base_model
import test_model

# for testing purposes put base_model from vgg_face back together with its fc_layers in test_model
m1=base_model.base_model
m2=test_model.test_model
model=torch.nn.Sequential(m1, m2)
model.eval()
m1.eval()

# Load picture and prepare for impact
pic=scipy.ndimage.imread('candice.png')
pic = pic[np.newaxis,:,:,:]
pic = Variable(torch.Tensor(pic.astype(float))) #If error occurs because some elements are not float : pic = Variable(torch.Tensor(pic.astype(float)))
pic = pic.permute(0,3,1,2)

# classify picture
out=m1.forward(pic).data.numpy()

# Get Top 5
print('\nTop 5:', np.argsort(-out)[:,:5][0])
print(out.shape)
# candice.png should put 283 in the top 5



