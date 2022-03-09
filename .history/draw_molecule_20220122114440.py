import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

file = '19_science_total.csv'
data = pd.read_csv(file)

def deduplicationt_molecule(category):
    category_li = []
    for mol in data[category]:
        if mol not in category_li:
            category_li.append(mol)
    return category_li

Catalyst = deduplicationt_molecule('Catalyst')
Imine = deduplicationt_molecule('Imine')
Thiol = deduplicationt_molecule('Thiol')

def save_images(category_li,category):
    for mol_smi in category_li:
        mol = Chem.MolFromSmiles(mol_smi)
        rdDepictor.Compute2DCoords(mol)
        Draw.MolToFile(mol, './' + category + '//' + mol_smi + '.png')

save_images(Catalyst,'Catalyst')
# save_images(Imine,'Imine')
save_images(Thiol,'Thiol')