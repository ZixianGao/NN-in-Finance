import pandas as pd

data = pd.read_csv('data.csv')
data = data.drop(columns=['y'])
data.to_csv('data_without_y.csv', index=False)