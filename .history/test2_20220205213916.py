import torch
import numpy as np
from models import CNN

# a = np.arange(2160000).reshape(2,4,3,300,300)
# a = torch.from_numpy(a)

# b = np.arange(1080000).reshape(4,3,300,300)
# b = torch.from_numpy(b)
# # torch.Size([2, 4, 3, 300, 300])
# model = CNN()
# output = model(b.to(torch.float))
# print(output.size())

a = np.arange(12).reshape(3,4)
a = torch.from_numpy(a)
print(a.size())
print(a)
a = a.transpose(0,1)
print(a)