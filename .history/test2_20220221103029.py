from statistics import mode
import torch
import numpy as np
from models import CNN

a = torch.LongTensor([[2,3,4,5,0]])
a = a.eq(0)
a = a.unsqueeze(1).expand(1,5,5)
a = a.unsqueeze(1).repeat(1,8,5,5)
print(a,a.size())