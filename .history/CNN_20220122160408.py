from cgi import test
from nis import cat
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from xgboost import train

def get_molecule_dict(category):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    return category_dict

file = '19_science_total.csv'
split = 600

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

data = pd.read_csv(file)

Catalyst = get_molecule_dict('Catalyst')
Imine = get_molecule_dict('Imine')
Thiol = get_molecule_dict('Thiol')

train_fea = []
train_label = []
for i in range(split):
    train_reaction = []

    catalyst = data['Catalyst'][i]
    imine = data['Imine'][i]
    thiol = data['Thiol'][i]

    catalyst_img = Image.open('./Catalyst\\' + Catalyst[catalyst] + '.png')
    imine_img = Image.open('./Imine\\' + Imine[imine] + '.png')
    thiol_img = Image.open('./Thiol\\' + Thiol[thiol] + '.png')

    catalyst_data = transform(catalyst_img)
    imine_data = transform(imine_img)
    thiol_data = transform(thiol_img)

    train_reaction.append(catalyst_data)
    train_reaction.append(imine_data)
    train_reaction.append(thiol_data)

    train_fea.append(train_reaction)
    train_label.append(data['Output'][i])
    if i == 0:
        break

test_fea = []
for i in range(split,1075):
    test_reaction = []

    catalyst = data['Catalyst'][i]
    imine = data['Imine'][i]
    thiol = data['Thiol'][i]

    catalyst_img = Image.open('./Catalyst\\' + Catalyst[catalyst] + '.png')
    imine_img = Image.open('./Imine\\' + Imine[imine] + '.png')
    thiol_img = Image.open('./Thiol\\' + Thiol[thiol] + '.png')

    catalyst_data = transform(catalyst_img)
    imine_data = transform(imine_img)
    thiol_data = transform(thiol_img)

    test_reaction.append(catalyst_data)
    test_reaction.append(imine_data)
    test_reaction.append(thiol_data)

    train.append(train_reaction)
    if i == 0:
        break

