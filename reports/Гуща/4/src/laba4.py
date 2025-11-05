import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

data = pd.read_csv('winequality-white.csv', sep=';')
data['good'] = (data['quality'] >= 7).astype(int)

X = data.drop(['quality', 'good'], axis=1).values
y = data['good'].values

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

class WineQualityNet(nn.Module):
    def __init__(self, input_dim):
        super(WineQualityNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
)

    def forward(self, x):
        return self.model(x)

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = WineQualityNet(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, X_train, y_train, epochs=50):
    for epoch in range(epochs):
        model.train()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

train_model(model, X_train, y_train, epochs=50)

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    acc = accuracy_score(y_test.numpy(), preds.numpy())
    f1 = f1_score(y_test.numpy(), preds.numpy())
    precision = precision_score(y_test.numpy(), preds.numpy())
    recall = recall_score(y_test.numpy(),preds.numpy())
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Precision: { precision: .4f}, Recall: { recall: .4f}  ")
    return acc, f1

print("Оценка после 50 эпох:")
evaluate_model(model, X_test, y_test)

model2 = WineQualityNet(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, X_train, y_train, epochs=50)

print("Оценка после 100 эпох:")
evaluate_model(model, X_test, y_test)