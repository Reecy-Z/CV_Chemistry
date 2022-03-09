from email import header
from tkinter import N
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

file = '19_science_total.csv'
data = pd.read_csv(file)

def deduplicationt_molecule(category):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    # category_df = pd.DataFrame(category_dict.values(),index=category_dict.keys())
    category_df = pd.DataFrame.from_dict(category_dict, orient ='columns', columns=None)
    category_df.to_csv(category + '_dict.csv', header=None)
    return category_dict

Catalyst = deduplicationt_molecule('Catalyst')
print(Catalyst)
Imine = deduplicationt_molecule('Imine')
Thiol = deduplicationt_molecule('Thiol')

def save_images(category_li,category):
    for mol_smi in category_li:            
        mol = Chem.MolFromSmiles(mol_smi)
        rdDepictor.Compute2DCoords(mol)
        Draw.MolToFile(mol, './' + category + '//' + mol_smi + '.png')

# save_images(Catalyst,'Catalyst')
save_images(Imine,'Imine')
# save_images(Thiol,'Thiol')