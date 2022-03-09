import torch
import numpy as np
from models import CNN

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(a)
a = a.transpose(0,1)
for i in a:
    print(i.size())
# torch.Size([2, 4, 3, 300, 300])
# model = CNN()
# output = model(a.to(torch.float))
# print(output.size())
