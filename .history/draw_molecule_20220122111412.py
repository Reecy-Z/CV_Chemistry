import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

file = '19_science_total.csv'
data = pd.read_csv(file)

# mol = Chem.MolFromSmiles('CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(=O)C)[NH3+])C(=O)OCC')

# rdDepictor.Compute2DCoords(mol)

# Draw.MolToFile(mol, 'mol.png')