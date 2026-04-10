import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data Cleaning
df = pd.read_csv("AAPL.csv")
df = df.drop(columns=["Open", "High", "Low"])
df = df.iloc[2:]
# df.rename(columns={"Price": 'Date'}, inplace=True)
# df = df.drop(columns=["Date"])
# convert it into a tensor
# data = torch.tensor(df.values, dtype=torch.float32)

# create the model class
class Model(nn.Module):
    def __init__(self, in_features=2, h1=8, h2=9):
        super().__init__(self)
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.out(x)
        return x

def getPast5():
    pass

# Seed
torch.manual_seed(41)

model = Model()


