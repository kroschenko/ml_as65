import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Импорт библиотек и подготовка данных
df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)
Y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

input_dim = X_train.shape[1]


# 2. Определение архитектуры нейронной сети
class MLP_NoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 24)
        self.relu = nn.ReLU()
        self.output = nn.Linear(24, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.output(x)
        return x


class MLP_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 24)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(24, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


# 3. Инициализация модели, функции потерь и оптимизатора
criterion = nn.BCEWithLogitsLoss()

def train_and_evaluate(model, name, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Написание цикла обучения (Training Loop)
    for epoch in range(1, epochs + 1):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    # 5. Оценка модели (Evaluation)
    model.eval()
    with torch.no_grad():
        logits = model(X_test).numpy()
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy().astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\nРезультаты модели {name}:")
    print("===================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    return acc, prec, rec, f1



model1 = MLP_NoDropout()
results1 = train_and_evaluate(model1, "MLP")

model2 = MLP_Dropout()
results2 = train_and_evaluate(model2, "MLP + Dropout 0.2")
