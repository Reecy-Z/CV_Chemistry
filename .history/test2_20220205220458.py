from statistics import mode
import torch
import numpy as np
from models import CNN

model = CNN()

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(a)
output = model(a)
print(output_cat.size())
# torch.Size([2, 4, 3, 300, 300])
# model = CNN()
# output = model(a.to(torch.float))
# print(output.size())
