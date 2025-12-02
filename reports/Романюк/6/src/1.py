import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a, b, c, d = 0.1, 0.5, 0.09, 0.5

t = np.linspace(0, 20, 400)
y = a*np.sin(b*t) + c*np.cos(d*t)

seq_len = 5

X, Y = [], []
for i in range(len(y) - seq_len):
    X.append(y[i:i+seq_len])
    Y.append(y[i+seq_len])

X = np.array(X)
Y = np.array(Y)

class JordanRNN:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, 1) * 0.1
        self.Why = np.random.randn(1, hidden_size) * 0.1

        self.h = np.zeros((hidden_size, 1))
        self.y_prev = np.zeros((1, 1))

        self.lr = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def forward(self, x):
        x = x.reshape(-1, 1)
        self.h = self.sigmoid(self.Wxh @ x + self.Whh @ self.y_prev)
        return self.Why @ self.h

    def train_step(self, x, target):
        y_pred = self.forward(x)
        error = y_pred - target

        dWhy = error @ self.h.T
        dh = (self.Why.T @ error) * self.dsigmoid(self.h)
        dWxh = dh @ x.reshape(1, -1)
        dWhh = dh @ self.y_prev.T

        self.Why -= self.lr * dWhy
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh

        self.y_prev = y_pred.copy()

        return (error**2).item()

rnn = JordanRNN(input_size=seq_len, hidden_size=3)
errors = []

for epoch in range(300):
    err = 0
    rnn.y_prev = np.zeros((1,1))
    for i in range(len(X)):
        err += rnn.train_step(X[i], Y[i])
    errors.append(err / len(X))

pred_train = []
rnn.y_prev = np.zeros((1,1))

for i in range(len(X)):
    pred_train.append(rnn.forward(X[i]).item())

pred_train = np.array(pred_train)

df_train = pd.DataFrame({
    "Эталонное значение": Y[:10],
    "Полученное значение": pred_train[:10],
    "Отклонение": pred_train[:10] - Y[:10]
})

print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (ПЕРВЫЕ 10) ===\n")
print(df_train.to_string(index=False))

future_steps = 30
input_seq = list(Y[-seq_len:])
pred_future = []

rnn.y_prev = np.zeros((1,1))

for _ in range(future_steps):
    x_arr = np.array(input_seq[-seq_len:])
    y_hat = rnn.forward(x_arr).item()
    pred_future.append(y_hat)
    input_seq.append(y_hat)

t_future = np.linspace(t[-1], t[-1] + (20/400)*future_steps, future_steps)
y_future = a*np.sin(b*t_future) + c*np.cos(d*t_future)

df_forecast = pd.DataFrame({
    "Эталонное значение": y_future[:10],
    "Полученное значение": np.array(pred_future[:10]),
    "Отклонение": np.array(pred_future[:10]) - y_future[:10]
})

print("\n=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ (ПЕРВЫЕ 10) ===\n")
print(df_forecast.to_string(index=False))

plt.figure(figsize=(10,5))
plt.plot(Y, label="target")
plt.plot(pred_train, label="prediction")
plt.legend()
plt.title("Функция на обучении")
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(errors)
plt.title("Ошибка по итерациям")
plt.grid()
plt.show()
