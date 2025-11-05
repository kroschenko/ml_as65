import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("seeds_dataset.txt", delim_whitespace=True, header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)


torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, hidden_size=7):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, x):
        return self.model(x)


hidden_size = 7
# hidden_size = 14
model = MLP(hidden_size=hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    y_logits = model(X_test)
    y_pred = torch.argmax(y_logits, dim=1)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
