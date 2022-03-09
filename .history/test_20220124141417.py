import pandas as pd
file = './18 science\\18_science_total.csv'
data = pd.read_csv(file)
columns = data.columns.values.tolist()
print(columns)
