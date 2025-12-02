
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
def y(x):
    return 0.3*np.cos(0.3*x) + 0.07*np.sin(0.3*x)

x = np.linspace(0, 20, 100)
y_data = y(x)
plt.plot(x, y_data, label='y', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Пример графика')
plt.legend()
plt.grid(True)
plt.show()

input_size = 10
hidden_size = 4
train_size = 80

def make_dataset(y, window):
    X, Y = [], []
    for i in range(len(y) - window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    return np.array(X), np.array(Y)

X, Y = make_dataset(y_data, input_size)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]
Y_train = Y_train.reshape(-1,1)

Wx = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
Wh = np.random.uniform(-0.5, 0.5, (hidden_size, hidden_size))
b_h = np.zeros((1, hidden_size))

W_out = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
b_out = np.zeros((1, 1))

# Активация
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Прямой проход RNN
def rnn_forward(x_seq):
    x_seq = x_seq.reshape(1, -1)
    h = np.zeros((1, hidden_size))
    h = sigmoid(x_seq @ Wx + h @ Wh + b_h)
    y_pred = h @ W_out + b_out
    return h, y_pred

# Обучение RNN
def train_rnn(X, Y, lr=0.01):
    global Wx, Wh, b_h, W_out, b_out
    total_loss = 0
    for i in range(len(X)):
        x_seq = X[i]
        y_true = Y[i:i+1]
        h, y_pred = rnn_forward(x_seq)

        delta_out = y_pred - y_true
        total_loss += delta_out**2

        delta_h = delta_out @ W_out.T
        delta_h_raw = delta_h * h * (1 - h)

        # Обновление весов
        x_seq_2d = x_seq.reshape(1,-1)
        Wx -= lr * x_seq_2d.T @ delta_h_raw
        Wh -= lr * np.zeros_like(h).T @ delta_h_raw  # h_prev=0 для одного шага
        b_h -= lr * delta_h_raw
        W_out -= lr * h.T @ delta_out
        b_out -= lr * delta_out
    return total_loss / len(X)

# Обучение сети
epochs = 2000
loss_history = []
for epoch in range(epochs):
    loss = train_rnn(X_train, Y_train)
    loss_history.append(loss)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

# Прогноз на тестовой выборке
y_test_pred = []
X_seq = X_test[0]
for i in range(len(Y_test)):
    h, y_pred = rnn_forward(X_seq)
    y_test_pred.append(y_pred.item())
    X_seq = np.append(X_seq[1:], y_pred.item())

# Сравнение
print("    Эталон | Предсказано | Отклонение")
for y_true, y_pred in zip(Y_test, y_test_pred):
    print(f"{y_true:.4f}     | {y_pred:.4f}       | {(y_true-y_pred):.4f}")

# График MSE
mse = (Y_test - np.array(y_test_pred))**2
plt.plot(mse)
plt.title("Изменение ошибки MSE")
plt.xlabel("Итерация")
plt.ylabel("Ошибка")
plt.grid()
plt.show()

# Сравнение прогнозов и истинных значений
plt.plot(Y_test, label="Истинные значения")
plt.plot(y_test_pred, label="Прогноз")
plt.title("Прогноз на тестовой выборке")
plt.legend()
plt.grid()
plt.show()
