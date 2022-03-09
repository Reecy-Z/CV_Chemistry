import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

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

data = pd.read_csv(file)

Catalyst = get_molecule_dict('Catalyst')
Imine = get_molecule_dict('Imine')
Thiol = get_molecule_dict('Thiol')

for i in range(split):
    catalyst = data['Catalyst'][i]
    imine = data['Imine'][i]
    thiol = data['Thiol'][i]

    catalyst_image = 