import torch
from torch.autograd import Variable
import numpy as np
import vgg_face_model.VGG_FACE

def runCNN(pic):
	net = vgg_face_model.VGG_FACE.VGG_FACE
	net.load_state_dict(torch.load('vgg_face_model/VGG_FACE.pth'))
	net.eval()

	pic = pic[np.newaxis,:,:,:]
	pic = Variable(torch.Tensor(pic))
	pic = pic.permute(0,3,1,2)
	
	out=net.forward(pic).data.numpy()
	#Get Top 5
	results = np.argsort(-out)[:,:5]
	print(results)
	print(-np.sort(-out)[:,:5])
	
	return results