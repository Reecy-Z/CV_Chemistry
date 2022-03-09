import torch
import numpy as np
from torch import nn
a = torch.LongTensor([[1,2,3,4],[5,6,7,8]])
print(a.size())
embedding = nn.Embedding(5,3)
