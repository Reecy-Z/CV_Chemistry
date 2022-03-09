from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults

opts = DrawingOptions()
m = Chem.MolFromSmiles('OC1C2C1CC2')
opts.includeAtomNumbers=False
opts.bondLineWidth=2.8
draw = Draw.MolToImage(m, options=opts)
draw.save('mol10.jpg')