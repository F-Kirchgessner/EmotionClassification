import torch
import numpy as np
import scipy.ndimage
import VGG_FACE
net = VGG_FACE.VGG_FACE
net.load_state_dict(torch.load('VGG_FACE.pth'))
net.eval()
pic=scipy.ndimage.imread('candice.png')
from torch.autograd import Variable
pic = pic[np.newaxis,:,:,:]
pic = Variable(torch.Tensor(pic))
pic = pic.permute(0,3,1,2)

out=net.forward(pic).data.numpy()
#Get Top 5
print(np.argsort(-out)[:,:5])
print(-np.sort(-out)[:,:5])





