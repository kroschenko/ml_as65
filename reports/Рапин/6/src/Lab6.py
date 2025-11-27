import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b, c, d = 0.3, 0.5, 0.05, 0.5

np.random.seed(42)
N_total = 1200            # всего точек
x = np.linspace(0, 60, N_total)
y = a * np.cos(b * x) + c * np.sin(d * x)

noise_level = 0.0
y_noisy = y + noise_level * np.random.randn(N_total)

# Разбиение на обучение/тест
train_ratio = 0.7
N_train = int(N_total * train_ratio)
y_train = y_noisy[:N_train]
y_test = y_noisy[N_train:]

window = 8

def make_sequences(series, window):
    X, T = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])   # 8 прошлых значений
        T.append(series[i])              # целевая — следующее
    X = np.array(X)          # shape: [N, 8]
    T = np.array(T).reshape(-1, 1)  # shape: [N, 1]
    return X, T

X_train, T_train = make_sequences(y_train, window)
X_test, T_test = make_sequences(y_test, window)

class JordanRNN:
    def __init__(self, input_size=8, hidden_size=3, output_size=1, lr=0.01):
        self.input_size = input_size
        self.context_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Инициализация весов
        self.Wxh = np.random.randn(input_size + self.context_size, hidden_size) * 0.1
        self.bh = np.zeros((1, hidden_size))

        self.Why = np.random.randn(hidden_size, output_size) * 0.1
        self.by = np.zeros((1, output_size))

        self.context = np.zeros((1, self.context_size))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1.0 - a)

    def forward(self, x_t):
        inp = np.concatenate([x_t, self.context], axis=1)  # shape (1, 9)

        h_net = inp @ self.Wxh + self.bh
        h_act = self.sigmoid(h_net)

        y_net = h_act @ self.Why + self.by
        y_hat = y_net

        self.context = y_hat.copy()

        cache = {
            "inp": inp,
            "h_net": h_net,
            "h_act": h_act,
            "y_net": y_net,
            "y_hat": y_hat
        }
        return y_hat, cache

    def backward(self, cache, t_t):
        y_hat = cache["y_hat"]
        h_act = cache["h_act"]
        inp = cache["inp"]

        dy = (y_hat - t_t)  # shape (1,1)

        dWhy = h_act.T @ dy                            # (3,1)
        dby = dy                                       # (1,1)

        dh = dy @ self.Why.T                           # (1,3)
        dh_net = dh * self.sigmoid_deriv(h_act)        # (1,3)

        dWxh = inp.T @ dh_net                          # (9,3)
        dbh = dh_net                                   # (1,3)

        # Обновление весов
        self.Why -= self.lr * dWhy
        self.by  -= self.lr * dby
        self.Wxh -= self.lr * dWxh
        self.bh  -= self.lr * dbh
        loss = 0.5 * float(np.sum((y_hat - t_t) ** 2))
        return loss

    def reset_context(self):
        self.context = np.zeros((1, self.context_size))

    def fit(self, X, T, epochs=50, verbose=True):
        losses = []
        for epoch in range(1, epochs + 1):
            self.reset_context()
            epoch_loss = 0.0
            for i in range(len(X)):
                x_t = X[i].reshape(1, -1)
                t_t = T[i].reshape(1, -1)
                y_hat, cache = self.forward(x_t)
                loss = self.backward(cache, t_t)
                epoch_loss += loss
            epoch_loss /= len(X)
            losses.append(epoch_loss)
            if verbose and (epoch % max(1, (epochs // 10)) == 0 or epoch == 1):
                print(f"Epoch {epoch:3d}/{epochs}, loss = {epoch_loss:.6f}")
        return losses

    def predict_sequence(self, X):
        self.reset_context()
        preds = []
        for i in range(len(X)):
            x_t = X[i].reshape(1, -1)
            y_hat, _ = self.forward(x_t)
            preds.append(y_hat.item())
        return np.array(preds).reshape(-1, 1)

# Обучение модели
model = JordanRNN(input_size=window, hidden_size=3, output_size=1, lr=0.01)

epochs = 1000
losses = model.fit(X_train, T_train, epochs=epochs, verbose=True)

# Предсказания на обучении и тесте
y_train_pred = model.predict_sequence(X_train)
y_test_pred = model.predict_sequence(X_test)

# График прогнозируемой функции на участке обучения
plt.figure(figsize=(10, 5))
plt.plot(range(len(T_train)), T_train, label='Эталон (train)', linewidth=2)
plt.plot(range(len(y_train_pred)), y_train_pred, label='Прогноз (train)', linewidth=2)
plt.title('Прогноз на участке обучения')
plt.xlabel('Шаг')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График изменения ошибки в зависимости от итерации
def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed

# Сглаженные потери
smoothed_losses = smooth_curve(losses, factor=0.9)

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), smoothed_losses, label='MSE', color='darkorange', linewidth=2)
plt.title('Изменение ошибки (MSE) в зависимости от итерации')
plt.xlabel('Эпоха')
plt.ylabel('Средняя ошибка')
plt.grid(True)

max_loss = max(smoothed_losses)
plt.ylim(0, max_loss * 1.1)

plt.tight_layout()
plt.legend()
plt.show()

# Таблицы результатов

def make_result_table(target, pred):
    df = pd.DataFrame({
        "Эталонное значение": target.flatten(),
        "Полученное значение": pred.flatten()
    })
    df["Отклонение"] = df["Полученное значение"] - df["Эталонное значение"]
    return df

train_table = make_result_table(T_train, y_train_pred)
print("\nРезультаты обучения:")
print(train_table.head(10).to_string(index=False))

test_table = make_result_table(T_test, y_test_pred)
print("\nРезультаты прогнозирования:")
print(test_table.head(10).to_string(index=False))

train_mse = np.mean((T_train - y_train_pred) ** 2)
test_mse = np.mean((T_test - y_test_pred) ** 2)
print(f"Итоговая MSE на обучении: {train_mse:.6f}\nИтоговая MSE на тесте: {test_mse:.6f}")
