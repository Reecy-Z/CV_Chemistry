import torch
import numpy as np
from models import CNN

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(a)

b = np.arange(1080000).reshape(2,4,3,300,300)
b = torch.from_numpy(a)
# torch.Size([2, 4, 3, 300, 300])
model = CNN()
output = model(a)
print(output.size())