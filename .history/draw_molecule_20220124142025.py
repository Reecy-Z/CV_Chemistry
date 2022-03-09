import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

file = './18 science\\18_science_total.csv'
data = pd.read_csv(file)

def deduplicationt_molecule(category,dir):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    category_df = pd.DataFrame(list(category_dict.items()))
    category_df.to_csv('./' + dir + '\\' + category + '_dict.csv', header=None, index=None)
    return category_dict

# 19 science
# Catalyst = deduplicationt_molecule('Catalyst')
# Imine = deduplicationt_molecule('Imine')
# Thiol = deduplicationt_molecule('Thiol')

# 18 science
Ligand = deduplicationt_molecule('Ligand','18 science')
Additive = deduplicationt_molecule('Additive','18 science')
Base = deduplicationt_molecule('Ligand','18 science')
Aryl_halide = deduplicationt_molecule('Aryl halide','18 science')


def save_images(category_dict,category,dir):
    for mol_smi in category_dict.keys():            
        mol = Chem.MolFromSmiles(mol_smi)
        rdDepictor.Compute2DCoords(mol)
        Draw.MolToFile(mol, './' + dir + '\\' + category + '\\' + str(category_dict[mol_smi]) + '.png')

# 19 science
# save_images(Catalyst,'Catalyst','19 science')
# save_images(Imine,'Imine','19 science')
# save_images(Thiol,'Thiol','19 science')