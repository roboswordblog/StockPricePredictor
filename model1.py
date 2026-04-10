import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data Cleaning
df = pd.read_csv("AAPL.csv")
df = df.drop(columns=["Open", "High", "Low"])
df = df.iloc[2:]
df.rename(columns={"Price": 'Date'}, inplace=True)

# convert it into a tensor
data = torch.tensor(df.values, dtype=torch.float32)

# create the model class
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
