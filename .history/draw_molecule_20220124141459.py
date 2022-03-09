import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

file = '19_science_total.csv'
data = pd.read_csv(file)

def deduplicationt_molecule(category,dir):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    category_df = pd.DataFrame(list(category_dict.items()))
    category_df.to_csv(category + '_dict.csv', header=None, index=None)
    return category_dict

Catalyst = deduplicationt_molecule('Catalyst')
Imine = deduplicationt_molecule('Imine')
Thiol = deduplicationt_molecule('Thiol')

def save_images(category_dict,category,dir):
    for mol_smi in category_dict.keys():            
        mol = Chem.MolFromSmiles(mol_smi)
        rdDepictor.Compute2DCoords(mol)
        Draw.MolToFile(mol, './' + dir + '\\' + category + '\\' + str(category_dict[mol_smi]) + '.png')

save_images(Catalyst,'Catalyst','18 science')
save_images(Imine,'Imine','18 science')
save_images(Thiol,'Thiol','18 science')