import pandas as pd

df = pd.read_csv(r'./dataset.csv')
df.set_index('track_id', inplace=True)
print(df.head())