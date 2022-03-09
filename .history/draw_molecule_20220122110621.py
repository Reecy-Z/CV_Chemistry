from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

mol = Chem.MolFromSmiles('CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(=O)C)[NH3+])C(=O)OCC')

rdDepictor.Compute2DCoords(mol)

Draw.MolToFile(mol, 'mol.png')