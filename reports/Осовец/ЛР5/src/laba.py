import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Вариант 3 ---
b = 0.3
c = 0.07
d = 0.3
INPUT_SIZE = 10
HIDDEN = 4
EPOCHS = 500

# --- Функция ---
def f(x):
    return np.cos(b * x) + c * np.sin(d * x)

# --- Данные ---
x = np.linspace(0, 20, 300)
y = f(x)

# График функции
plt.figure()
plt.plot(x, y)
plt.title("y = cos(0.3x) + 0.07*sin(0.3x)")
plt.show()

# --- Создание выборки ---
def create_dataset(y, window):
    X, Y = [], []
    for i in range(len(y) - window):
        X.append(y[i:i + window])
        Y.append(y[i + window])
    return np.array(X), np.array(Y)

X, Y = create_dataset(y, INPUT_SIZE)

# train/test = 80/20
split = int(len(X) * 0.8)
X_train = torch.tensor(X[:split], dtype=torch.float32)
Y_train = torch.tensor(Y[:split], dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X[split:], dtype=torch.float32)
Y_test = torch.tensor(Y[split:], dtype=torch.float32).view(-1, 1)

# --- Модель ---
model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_SIZE, HIDDEN),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN, 1)
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Обучение ---
losses = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# График ошибки
plt.figure()
plt.plot(losses)
plt.title("Ошибка обучения")
plt.show()

# --- Результаты обучения ---
print("\nОБУЧЕНИЕ (первые 10 значений)")
print("Истинное | Предсказанное | Ошибка")

with torch.no_grad():
    train_pred = model(X_train)

for i in range(10):
    real = Y_train[i].item()
    pred = train_pred[i].item()
    err = pred - real
    print(f"{real:.6f} | {pred:.6f} | {err:.6f}")

# --- Результаты прогноза ---
print("\nПРОГНОЗ (первые 10 значений)")
print("Истинное | Предсказанное | Ошибка")

with torch.no_grad():
    test_pred = model(X_test)

for i in range(10):
    real = Y_test[i].item()
    pred = test_pred[i].item()
    err = pred - real
    print(f"{real:.6f} | {pred:.6f} | {err:.6f}")
