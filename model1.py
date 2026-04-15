import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

df = pd.read_csv("AAPL.csv")
df = df.drop(columns=["Open", "High", "Low"])
df = df.iloc[2:].reset_index(drop=True)

def custom_func(x):
    x = str(x)
    return float(x.split("-")[0] + x.split("-")[1] + x.split("-")[2])

df.rename(columns={"Price": "Date"}, inplace=True)
df["Date"] = df["Date"].apply(custom_func)
df["Close"] = df["Close"].astype(float)

def getPast5(index):
    values = []
    for i in range(5, 0, -1):
        values.append(float(df.iloc[index - i]["Close"]))
    return values
def getPast5Volumes(index):
    values = []
    for i in range(5, 0, -1):
        values.append(float(df.iloc[index - i]["Volume"]))
    return values

X = []
y = []

for i in range(5, len(df)):
    X.append(getPast5(i))
#    X.append(getPast5(i))
    y.append(float(df.iloc[i]["Close"]))


X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

torch.manual_seed(41)
model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {100-(loss.item()/1000)}")

with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)

print(f"Accuracy: {100-(test_loss.item()/1000)}")

with torch.no_grad():
    train_pred = model(torch.tensor([int(input("5 days ago: ")),int(input("4 days ago: ")),int(input("3 days ago: ")),int(input("Day before yesterday: ")),int(input("Yesterday: "))]))
