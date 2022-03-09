import torch
import numpy as np
from models import CNN
import utils

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(np.array([1,2]))