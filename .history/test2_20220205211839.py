import torch
import numpy as np
from models import CNN
import utils

a = ([1],[2])
c = ([3],[4])
b = []
b.append(a)
b.append(c)
a[0].append(2)
print(len(a[0]))