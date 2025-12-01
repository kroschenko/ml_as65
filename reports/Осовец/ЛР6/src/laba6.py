import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Параметры варианта 3
# -----------------------------
b = 0.3
c = 0.07
d = 0.3

INPUT_SIZE = 10
HIDDEN_SIZE = 4
LR = 0.001
EPOCHS = 600

# -----------------------------
# Функция
# -----------------------------
def func(x):
    return np.cos(b * x) + c * np.sin(d * x)

# -----------------------------
# Сигмоида
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

# -----------------------------
# Данные
# -----------------------------
x = np.linspace(0, 30, 400)
y = func(x)

plt.plot(x, y)
plt.title("Исходная функция")
plt.show()

# -----------------------------
# Формирование выборки
# -----------------------------
def create_dataset(series, window):
    X, Y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        Y.append(series[i+window])
    return np.array(X), np.array(Y).reshape(-1, 1)

X_all, Y_all = create_dataset(y, INPUT_SIZE)

split = int(len(X_all) * 0.8)

X_train = X_all[:split]
Y_train = Y_all[:split]

X_test = X_all[split:]
Y_test = Y_all[split:]

# -----------------------------
# Нормализация
# -----------------------------
mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-8

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

my = Y_train.mean()
sy = Y_train.std() + 1e-8

Y_train = (Y_train - my) / sy
Y_test = (Y_test - my) / sy

# -----------------------------
# Мультирекуррентная ИНС
# -----------------------------
np.random.seed(42)

W_in = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
W_rec = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.5
b_h = np.zeros((1, HIDDEN_SIZE))

W_out = np.random.randn(HIDDEN_SIZE, 1) * 0.5
b_out = np.zeros((1, 1))


def forward(X):
    h_prev = np.zeros((1, HIDDEN_SIZE))
    hs = []
    ys = []

    for i in range(len(X)):
        x_t = X[i:i+1]
        h = sigmoid(x_t @ W_in + h_prev @ W_rec + b_h)
        y_t = h @ W_out + b_out

        hs.append(h)
        ys.append(y_t)

        h_prev = h.copy()

    return hs, ys


# -----------------------------
# Обучение
# -----------------------------
errors = []

for epoch in range(EPOCHS):

    hs, ys = forward(X_train)
    Y_pred = np.vstack(ys)

    error = Y_pred - Y_train
    mse = np.mean(error**2)
    errors.append(mse)

    dW_out = np.zeros_like(W_out)
    db_out = np.zeros_like(b_out)
    dW_in = np.zeros_like(W_in)
    dW_rec = np.zeros_like(W_rec)
    db_h = np.zeros_like(b_h)

    dh_next = np.zeros((1, HIDDEN_SIZE))

    for t in reversed(range(len(X_train))):
        x_t = X_train[t:t+1]
        h_t = hs[t]

        dy = (Y_pred[t:t+1] - Y_train[t:t+1])

        dW_out += h_t.T @ dy
        db_out += dy

        dh = dy @ W_out.T + dh_next
        dh_raw = dh * sigmoid_derivative(h_t)

        dW_in += x_t.T @ dh_raw
        db_h += dh_raw

        if t > 0:
            h_prev = hs[t-1]
        else:
            h_prev = np.zeros((1, HIDDEN_SIZE))

        dW_rec += h_prev.T @ dh_raw
        dh_next = dh_raw @ W_rec.T

    # обновление весов
    W_out -= LR * dW_out
    b_out -= LR * db_out
    W_in -= LR * dW_in
    W_rec -= LR * dW_rec
    b_h -= LR * db_h

# -----------------------------
# График ошибки
# -----------------------------
plt.plot(errors)
plt.title("Ошибка обучения (MSE)")
plt.show()

# -----------------------------
# Результаты обучения
# -----------------------------
hs_train, ys_train_norm = forward(X_train)
Y_train_pred = np.vstack(ys_train_norm) * sy + my
Y_train_real = Y_train * sy + my

print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ (первые 10):")
print("Истинное   Предсказанное   Ошибка")

for i in range(10):
    real = Y_train_real[i][0]
    pred = Y_train_pred[i][0]
    err = pred - real
    print(real, pred, err)

# -----------------------------
# Результаты прогнозирования
# -----------------------------
hs_test, ys_test_norm = forward(X_test)
Y_test_pred = np.vstack(ys_test_norm) * sy + my
Y_test_real = Y_test * sy + my

print("\nРЕЗУЛЬТАТЫ ПРОГНОЗА (первые 10):")
print("Истинное   Предсказанное   Ошибка")

for i in range(10):
    real = Y_test_real[i][0]
    pred = Y_test_pred[i][0]
    err = pred - real
    print(real, pred, err)

# -----------------------------
# График прогноза
# -----------------------------
plt.plot(Y_test_real[:150], label="Истинные")
plt.plot(Y_test_pred[:150], label="Прогноз")
plt.legend()
plt.title("Прогноз на тестовой выборке")
plt.show()
