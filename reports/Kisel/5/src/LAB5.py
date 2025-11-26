import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Генерация функции y(x) = a*cos(bx) + c*sin(dx)
def generate_series(a=0.4, b=0.2, c=0.07, d=0.2, N=300, x_max=50):
    x = np.linspace(0, x_max, N)   # равномерно расположенные 300 точек от 0 до 50
    y = a * np.cos(b * x) + c * np.sin(d * x)
    return x, y

# Формирование данных типа "окно"
def prepare_windows(y, window=8):
    X = []#входы
    Y = []#выходы
    for i in range(window, len(y)):
        X.append(y[i-window:i])   # [0:8], [1:9], [2:10], [3:11]
        Y.append(y[i]) # [9], [10], [11], [12]
    return np.array(X), np.array(Y).reshape(-1, 1)  #reshape из вектора в матрицу 

# Стандартизатор
class SimpleScaler:
    def fit(self, X):# Только вычисляет параметры стандартизации
        self.mean = X.mean(axis=0)#считает среднее
        self.std = X.std(axis=0)#считает стандартное отклонение
        self.std[self.std == 0] = 1#защита от деления на ноль (если признак константный)
    def transform(self, X): # Применяет формулу на основы подготовленных параметров
        return (X - self.mean) / self.std #Формула стандартизации 
    def fit_transform(self, X): # И подготавливает и рассчитывает 
        self.fit(X)
        return self.transform(X)


# Параметры функции
a, b, c, d = 0.4, 0.2, 0.07, 0.2

# Генерируем ряд
x_vals, y_vals = generate_series(a=a, b=b, c=c, d=d, N=360, x_max=60)

# Подготавливаем окна
window = 8
X_all, Y_all = prepare_windows(y_vals, window=window)
# Разделяем train/test (70% train)
train_size = int(0.7 * len(X_all))
X_train, Y_train = X_all[:train_size], Y_all[:train_size]
X_test, Y_test = X_all[train_size:], Y_all[train_size:]

# Стандартизация признаков (по колонкам)
scaler = SimpleScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- Инициализация параметров сети ----
in_dim = window     # 8
hidden = 3
out_dim = 1

rng = np.random.RandomState(42) #фиксированный генератор случайных чисел
W1 = rng.normal(scale=0.5, size=(in_dim, hidden))   # 8x3   rng.normal → берём случайные числа из нормального распределения
b1 = np.zeros((1, hidden))                          # 1x3
W2 = rng.normal(scale=0.5, size=(hidden, out_dim))  # 3x1
b2 = np.zeros((1, out_dim))                         # 1x1
#Вход (8) —W1,b1→ Скрытый (3) —W2,b2→ Выход (1)

# Активация сигмоида и её производная
def sigmoid(z):#применяется в скрытом слое
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_deriv(z):#для обратного распространения ошибки и понимания как нужно изменить наши веса
    s = sigmoid(z)
    return s * (1.0 - s)

# Обучение (батчовый градиентный спуск)
lr = 0.01 #скорость обучения
epochs = 1000

loss_history = []

for epoch in range(1, epochs+1):
    # Forward (batch)
    #Считаем вход в скрытый слой
    Z1 = X_train_s.dot(W1) + b1 #Z1 = X * W1 + b1       # (N, hidden)
    H = sigmoid(Z1)#активации скрытых нейронов         # (N, hidden)
    Y_hat = H.dot(W2) + b2                 # (N, 1) линейный выход

    # Loss (MSE)
    # Вычисление ошибки
    diff = Y_hat - Y_train                 # (N,1)
    loss = 0.5 * np.mean(diff**2)
    loss_history.append(loss)

    # Backprop (batch derivatives) Насколько нужно изменить каждый вес, чтобы ошибка уменьшилась
    # Обратное распределение 
    N = X_train_s.shape[0]
    dY = diff / N                         #производная ошибки по выходу
    # Grad W2: H^T * dY
    dW2 = H.T.dot(dY)      #dW2 = H * dY             # (hidden x 1)
    db2 = np.sum(dY, axis=0, keepdims=True)  # (1 x 1)

    # Backprop into hidden
    #ошибка скрытого слоя
    dH = dY.dot(W2.T)                      # (N x hidden)
    dZ1 = dH * sigmoid_deriv(Z1)          # (N x hidden)
    dW1 = X_train_s.T.dot(dZ1)            # (in_dim x hidden)
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1 x hidden)

    # Update weights(new_weight = old_weight − lr * gradient)
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0 or epoch==1:
        print(f"[MLP] Epoch {epoch}/{epochs}  Loss={loss:.6f}")

# Оценка на train и test
def predict_mlp(X_s):
    #Мы уже обучили сеть, у нас есть обученные веса
    #Это та же часть forward-pass   
    Z1 = X_s.dot(W1) + b1
    H = sigmoid(Z1)
    Y_hat = H.dot(W2) + b2
    return Z1, H, Y_hat


Y_train_pred = predict_mlp(X_train_s)[2]  # берем только Y_hat
Y_test_pred = []

for i in range(len(X_test_s)):
    _, _, y_pred = predict_mlp(X_test_s[i].reshape(1, -1))  # одно окно
    Y_test_pred.append(y_pred.item())  

# после цикла преобразуем список в массив NumPy
Y_test_pred = np.array(Y_test_pred)

# считаем ошибки
mse_train = np.mean((Y_train_pred - Y_train)**2)
mse_test = np.mean((Y_test.flatten() - Y_test_pred)**2)
print(f"\nMLP Results: MSE_train={mse_train:.6f}, MSE_test={mse_test:.6f}")



# Графики
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("MLP: Loss per epoch")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1,2,2)
# График: реальная и предсказанная на обучающем участке (первые 100 точек)
idx = np.arange(len(Y_train))
plt.plot(idx, Y_train.flatten(), label='Target (train)', alpha=0.7)
plt.plot(idx, Y_train_pred.flatten(), label='Predicted (train)', alpha=0.7)
plt.legend()
plt.title("MLP: train target vs predicted")
plt.tight_layout()
plt.show()

# Таблица результатов 
df_res_train = pd.DataFrame({
    "target": Y_train.flatten(),
    "pred": Y_train_pred.flatten(),
    "error": (Y_train_pred - Y_train).flatten()
})
print("\nПример первых 10 строк таблицы результатов (train):")
print(df_res_train.head(10).to_string(index=False))

# Таблица результатов для теста
df_res_test = pd.DataFrame({
    "target": Y_test.flatten(),
    "pred": np.array(Y_test_pred).flatten(),
    "error": (np.array(Y_test_pred) - Y_test.flatten())
})
print("\nTest MSE sample:")
print(df_res_test.head(10).to_string(index=False))

# график прогноза на тестовом участке вместе с эталоном
plt.figure(figsize=(8,4))
plt.plot(Y_test.flatten(), label='Target (test)')
plt.plot(Y_test_pred.flatten(), label='Predicted (test)')
plt.title("MLP: test target vs predicted")
plt.legend()
plt.show()
