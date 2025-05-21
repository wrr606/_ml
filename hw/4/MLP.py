import torch
import torch.nn as nn
import torch.optim as optim

X = torch.eye(10)

Y = torch.tensor([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1],  # 9
], dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    print(output)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

pred = model(X).round()
print("預測結果：")
for i in range(10):
    print(f"{i}: {pred[i].int().tolist()}")