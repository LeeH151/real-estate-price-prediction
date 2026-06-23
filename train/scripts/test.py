import pandas as pd

df = pd.read_csv("data/dataset.csv")
print(df.shape)

# debug thêm
print(df.head())
print(df.tail())