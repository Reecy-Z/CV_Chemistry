import pandas as pd
file = 'Catalyst_dict.csv'

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
print(Catalyst)
