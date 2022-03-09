from statistics import mode
import torch
import numpy as np
from models import CNN

a = torch.LongTensor([1,2,3,4,5])
print(a.size())
a = a.unsqueeze(0)
print(a.size())
a = a.unsqueeze(1)
print(a.size())
a = a.expand(1,2,5)