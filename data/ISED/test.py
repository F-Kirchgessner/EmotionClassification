import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# path where this file is located
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
data_files = np.sort(os.listdir(ABS_PATH + '/test'))
img = np.array(Image.open(ABS_PATH + '/test/0000.jpg'), dtype=np.float64)
