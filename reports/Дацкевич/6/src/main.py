import numpy as np
import matplotlib.pyplot as plt


class ElmanRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.1

        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))

        self.context = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def linear(self, x):
        return x

    def forward(self, inputs):
        self.inputs = inputs
        hidden_input = (
            self.W_ih @ inputs
            + self.W_hh @ self.context
            + self.b_h
        )
        self.hidden = self.sigmoid(hidden_input)
        self.context = self.hidden.copy()

        output_input = self.W_ho @ self.hidden + self.b_o
        self.output = self.linear(output_input)
        return self.output

    def backward(self, target, learning_rate=0.1):
        output_error = self.output - target

        dW_ho = output_error @ self.hidden.T
        db_o = output_error

        hidden_error = (self.W_ho.T @ output_error) * self.sigmoid_derivative(self.hidden)
        dW_ih = hidden_error @ self.inputs.T
        dW_hh = hidden_error @ self.context.T
        db_h = hidden_error

        self.W_ho -= learning_rate * dW_ho
        self.W_ih -= learning_rate * dW_ih
        self.W_hh -= learning_rate * dW_hh
        self.b_o -= learning_rate * db_o
        self.b_h -= learning_rate * db_h

        return np.mean(output_error**2)


# Целевая функция
def target_function(t):
    return 0.5 * np.sin(t) + 0.3 * np.cos(2 * t)


t = np.linspace(0, 4 * np.pi, 100)
series = target_function(t)

# Формирование выборки
X, y = [], []
for i in range(len(series) - 6):
    X.append(series[i:i + 6])
    y.append(series[i + 6])

X = np.array(X)
y = np.array(y)

split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Инициализация и обучение сети
rnn = ElmanRNN(6, 2, 1)
epochs = 1000
errors = []

for epoch in range(epochs):
    epoch_error = 0
    for i in range(len(X_train)):
        rnn.context = np.zeros((2, 1))
        inputs = X_train[i].reshape(-1, 1)
        target = np.array([[y_train[i]]])

        output = rnn.forward(inputs)
        error = rnn.backward(target, 0.4)
        epoch_error += error

    avg_error = epoch_error / len(X_train)
    errors.append(avg_error)

# Результаты обучения
print("Результаты обучения:")
print("Эталонное значение | Полученное значение | Отклонение")
print("-" * 55)

train_results = []
for i in range(10):
    rnn.context = np.zeros((2, 1))
    inputs = X_train[i].reshape(-1, 1)
    prediction = rnn.forward(inputs)[0, 0]
    target = y_train[i]
    deviation = abs(target - prediction)
    train_results.append((target, prediction, deviation))
    print(f"{target:16.4f} | {prediction:19.4f} | {deviation:10.4f}")

print("\nРезультаты прогнозирования:")
print("Эталонное значение | Полученное значение | Отклонение")
print("-" * 55)

test_results = []
for i in range(10):
    rnn.context = np.zeros((2, 1))
    inputs = X_test[i].reshape(-1, 1)
    prediction = rnn.forward(inputs)[0, 0]
    target = y_test[i]
    deviation = abs(target - prediction)
    test_results.append((target, prediction, deviation))
    print(f"{target:16.4f} | {prediction:19.4f} | {deviation:10.4f}")

# Визуализация
plt.figure(figsize=(12, 8))

# График прогнозируемой функции на обучении
plt.subplot(2, 2, 1)
t_train = t[6:6 + len(y_train)]
plt.plot(t_train, y_train, label='Эталон')
train_preds = []
for i in range(len(X_train)):
    rnn.context = np.zeros((2, 1))
    inputs = X_train[i].reshape(-1, 1)
    pred = rnn.forward(inputs)[0, 0]
    train_preds.append(pred)
plt.plot(t_train, train_preds, label='Прогноз')
plt.title('График прогнозируемой функции на участке обучения')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

# Ошибка по эпохам
plt.subplot(2, 2, 2)
plt.plot(errors)
plt.title('Изменение ошибки в зависимости от итерации')
plt.xlabel('Итерация')
plt.ylabel('Ошибка')
plt.grid(True)

# Результаты обучения (10 значений)
plt.subplot(2, 2, 3)
train_targets = [r[0] for r in train_results]
train_preds = [r[1] for r in train_results]
plt.plot(train_targets, label='Эталон')
plt.plot(train_preds, label='Прогноз')
plt.title('Результаты обучения (первые 10 значений)')
plt.xlabel('Номер измерения')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

# Результаты прогнозирования (10 значений)
plt.subplot(2, 2, 4)
test_targets = [r[0] for r in test_results]
test_preds = [r[1] for r in test_results]
plt.plot(test_targets, label='Эталон')
plt.plot(test_preds, label='Прогноз')
plt.title('Результаты прогнозирования (первые 10 значений)')
plt.xlabel('Номер измерения')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
