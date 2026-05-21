import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MahjongMLP(nn.Module):
    def __init__(self, input_dim=187, num_classes=34):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------

# load dataset
X = np.load("dataset_X.npy")
y = np.load("dataset_y.npy")

# normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert to torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

# model
model = MahjongMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()

print("Test Accuracy:", acc.item())


# 5000 samples
# 10 epochs:  .18199 accuracy
# 30 epochs:  .23299 accuracy
# 50 epochs:  .23600 accuracy
# 75 epochs:  .24799 accuracy
# 100 epochs: .25100 accuracy