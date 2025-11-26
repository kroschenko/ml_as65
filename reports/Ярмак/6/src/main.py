import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


a = 0.3
b = 0.5
c = 0.05
d = 0.5

n_inputs = 8
n_hidden = 3
n_epochs = 2000


def func(x):
    return a * np.cos(b * x) + c * np.sin(d * x)
 

x = np.linspace(0, 30, 500)
y = func(x)

X, Y = [], []
for i in range(len(y) - n_inputs):
    X.append(y[i:i + n_inputs])
    Y.append(y[i + n_inputs])
X, Y = np.array(X), np.array(Y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

torch.manual_seed(42)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)


class JordanRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(JordanRNN, self).__init__()
        self.hidden = nn.Linear(n_inputs + 1, n_hidden)
        self.out = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y_prev):
        x_combined = torch.cat((x, y_prev), dim=1)
        h = self.sigmoid(self.hidden(x_combined))
        y = self.out(h)
        return y


def evaluate_model(model, X, Y):
    y_prev = torch.zeros((X.shape[0], 1))
    preds = []
    for i in range(len(X)):
        pred = model(X[i:i+1], y_prev[i:i+1])
        preds.append(pred.item())
        if i + 1 < len(X):
            y_prev[i+1] = pred
    return np.array(preds)


# best l rate
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
best_lr = None
best_loss = float('inf')

print("Подбор оптимального α:")
for lr in learning_rates:
    model = JordanRNN(n_inputs, n_hidden)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    y_prev = torch.zeros_like(Y_train)
    for epoch in range(500):
        pred = model(X_train, y_prev)
        loss = criterion(pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_prev = pred.detach()

    test_pred = evaluate_model(model, X_test, Y_test)
    mse = np.mean((test_pred - Y_test.flatten().numpy()) ** 2)

    print(f"  α={lr:.3f} -> MSE={mse:.6f}")
    if mse < best_loss:
        best_loss = mse
        best_lr = lr

print(f"\nОптимальное значение α: {best_lr:.3f}, минимальная ошибка: {best_loss:.6f}\n")


# model training
model = JordanRNN(n_inputs, n_hidden)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

losses = []
y_prev = torch.zeros_like(Y_train)

for epoch in range(n_epochs):
    pred = model(X_train, y_prev)
    loss = criterion(pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_prev = pred.detach()
    losses.append(loss.item())


# 4 pred func plot
train_pred = pred.detach().numpy()
plt.figure(figsize=(10,5))
plt.plot(Y_train.numpy(), label="Эталон")
plt.plot(train_pred, label="Прогноз")
plt.title("График прогнозируемой функции на участке обучения")
plt.legend()
plt.grid(True)
plt.show()


# 5 train res
train_results = pd.DataFrame({
    "Эталонное значение": Y_train.numpy().flatten(),
    "Полученное значение": train_pred.flatten(),
})
train_results["Отклонение"] = train_results["Полученное значение"] - train_results["Эталонное значение"]
print("\nРезультаты обучения:")
print(train_results.head(10))


plt.figure(figsize=(8,4))
plt.plot(losses)
plt.title("График изменения ошибки")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.grid(True)
plt.show()


# 6 pred res
test_pred = evaluate_model(model, X_test, Y_test)
test_results = pd.DataFrame({
    "Эталонное значение": Y_test.numpy().flatten(),
    "Полученное значение": test_pred.flatten(),
})
test_results["Отклонение"] = test_results["Полученное значение"] - test_results["Эталонное значение"]

print("\nРезультаты прогнозирования:")
print(test_results.head(10))
