import pandas as pd

file = '19_science_total.csv'
data = pd.read_csv(file)

def deduplicationt_molecule(category):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    return category_dict

Catalyst = deduplicationt_molecule('Catalyst')

file_cat = 'Catalyst_dict.csv'
cat_dict = pd.read_csv(file_cat,header=None)
for i,mol in enumerate(cat_dict[0]):
    index = cat_dict[1][i]
    if Catalyst[mol] == index:
        print('Pass')
    else:
        print(index)
    

