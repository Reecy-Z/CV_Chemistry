import torch
import numpy as np
from torch import nn
a = torch.LongTensor([[1,2,3,4],[5,6,7,8]])
print(a.size())
embedding = nn.Embedding(5,3)

sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
# Transformer Parameters
# Padding Should be Zero index
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
input_batch = [[src_vocab[n] for n in sentences[0].split()]]
print(torch.LongTensor(input_batch).size())