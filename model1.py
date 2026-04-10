import pandas as pd

# Data Cleaning
df = pd.read_csv("AAPL.csv")
df = df.drop(columns=["Open", "High", "Low"])
df = df.iloc[2:]
df.rename(columns={"Price": 'Date'}, inplace=True)

