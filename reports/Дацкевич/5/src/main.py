import numpy as np
import matplotlib.pyplot as plt

# Параметры варианта 4
a, b, c, d = 0.4, 0.4, 0.08, 0.4

# Генерация данных
x = np.arange(0, 30.1, 0.1)
y = a * np.cos(b * x) + c * np.sin(d * x)

# Разделение на обучающую и тестовую выборки
train_size = 200  # [0, 20] с шагом 0.1
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Создание dataset с окном = 6
def create_dataset(data, window_size=6):
    X, Y = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        Y.append(data[i])
    return np.array(X), np.array(Y)


window_size = 6
X_train, Y_train = create_dataset(y_train, window_size)
X_test, Y_test = create_dataset(y_test, window_size)

# Нормализация
mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
Y_train = (Y_train - mean) / std
X_test = (X_test - mean) / std
Y_test = (Y_test - mean) / std


# ИНС с одним скрытым слоем
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.W1) + self.b1)
        self.output = np.dot(self.hidden, self.W2) + self.b2
        return self.output

    def backward(self, X, y, output, lr=0.01):
        m = X.shape[0]
        dZ2 = output - y
        dW2 = (1 / m) * np.dot(self.hidden.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * self.hidden * (1 - self.hidden)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs=1000, lr=0.01):
        errors = []
        for epoch in range(epochs):
            output = self.forward(X)
            error = np.mean((output - y) ** 2)
            errors.append(error)
            self.backward(X, y, output, lr)
        return errors


# Создание и обучение сети
nn = NeuralNetwork(input_size=6, hidden_size=2, output_size=1)
errors = nn.train(X_train, Y_train.reshape(-1, 1), epochs=5000, lr=0.1)

# Прогноз
train_predict = nn.forward(X_train)
test_predict = nn.forward(X_test)

# Обратная нормализация
train_predict = train_predict * std + mean
test_predict = test_predict * std + mean
Y_train_orig = Y_train * std + mean
Y_test_orig = Y_test * std + mean

# 4. График прогнозируемой функции на участке обучения
plt.figure(figsize=(10, 4))
plt.plot(y_train[window_size:], label='Эталон', linewidth=2)
plt.plot(train_predict, label='Прогноз ИНС', linestyle='--')
plt.title('График прогнозируемой функции на участке обучения')
plt.xlabel('Время')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 5. Результаты обучения
print("Результаты обучения:")
print("Эталонное значение | Полученное значение | Отклонение")
print("-" * 55)
for i in range(10):
    etalon = Y_train_orig[i]
    predicted = train_predict[i, 0]
    deviation = abs(etalon - predicted)
    print(f"{etalon:16.4f} | {predicted:19.4f} | {deviation:10.4f}")

# График изменения ошибки
plt.figure(figsize=(10, 4))
plt.plot(errors)
plt.title('График изменения ошибки в зависимости от итерации')
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

# 6. Результаты прогнозирования
print("\nРезультаты прогнозирования:")
print("Эталонное значение | Полученное значение | Отклонение")
print("-" * 55)
for i in range(10):
    etalon = Y_test_orig[i]
    predicted = test_predict[i, 0]
    deviation = abs(etalon - predicted)
    print(f"{etalon:16.4f} | {predicted:19.4f} | {deviation:10.4f}")
