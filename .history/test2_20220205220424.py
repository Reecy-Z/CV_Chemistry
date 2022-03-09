import torch
import numpy as np
from models import CNN

model = CNN()

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(a)
a = a.transpose(0,1)
for index,i in enumerate(a):
    output = model(i.to(torch.float))
    if index == 0:
        output_cat = output
    else:
        output_cat = torch.cat((output_cat,output),dim = 1)
    print(output_cat.size())
# torch.Size([2, 4, 3, 300, 300])
# model = CNN()
# output = model(a.to(torch.float))
# print(output.size())
