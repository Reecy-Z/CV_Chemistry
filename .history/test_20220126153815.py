import torch
import numpy as np
from torch import nn
a = torch.LongTensor([[1,2,3,4],[5,6,7,8]])
# print(a.size())
embedding = nn.Embedding(5,3)

d_model = 6
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
# Transformer Parameters
# Padding Should be Zero index
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
src_len = len(src_vocab)
input_batch = [[src_vocab[n] for n in sentences[0].split()]]
input = torch.LongTensor(input_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

# print(get_sinusoid_encoding_table(src_len+1, d_model))

def changshi(a,b):
    def add(a,b):
        return (a+b)
    def minus(a,b):
        return (a-b)

print(changshi.add(3,2))