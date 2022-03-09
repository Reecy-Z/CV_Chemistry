import imp
from attr import dataclass
import torch
import numpy as np
import pandas as pd

def deduplicationt_molecule(category):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    category_df = pd.DataFrame(list(category_dict.items()))
    category_df.to_csv(category + '_dict.csv', header=None, index=None)
    return category_dict

file = '19_science_total.csv'
data = pd.read_csv(file)



Catalyst = deduplicationt_molecule('Catalyst')
Imine = deduplicationt_molecule('Imine')
Thiol = deduplicationt_molecule('Thiol')