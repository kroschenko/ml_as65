import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        # Инициализация весов маленькими случайными значениями
        self.W1 = np.random.randn(n_inputs, n_hidden) * 0.1
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_outputs) * 0.1
        self.b2 = np.zeros((1, n_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Прямой проход
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)  # сигмоид для скрытого слоя
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2  # линейная активация для выходного слоя
        return self.output

    def backward(self, X, y, output, learning_rate):
        # Обратное распространение ошибки
        m = X.shape[0]

        # Градиенты для выходного слоя (линейная активация)
        dZ2 = output - y.reshape(-1, 1)
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Градиенты для скрытого слоя (сигмоидная активация)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Обновление весов
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            # Прямой проход
            output = self.forward(X)

            # Расчет среднеквадратичной ошибки
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            losses.append(loss)

            # Обратное распространение и обновление весов
            self.backward(X, y, output, learning_rate)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}')

        return losses


def generate_data(a, b, c, d, x_range=(0, 50), step=0.5):
    """Генерация данных по заданной функции: y = a*cos(dx) + c*sin(dx)"""
    x = np.arange(x_range[0], x_range[1], step)
    # ИСПРАВЛЕНИЕ: используем только параметры a, c, d согласно формуле из задания
    y = a * np.cos(d * x) + c * np.sin(d * x)
    return x, y


def create_dataset(data, n_inputs=10):
    """Создание обучающей выборки методом скользящего окна"""
    X, Y = [], []
    for i in range(len(data) - n_inputs):
        X.append(data[i:i + n_inputs])
        Y.append(data[i + n_inputs])
    return np.array(X), np.array(Y)


# Основная программа
def main():
    # Параметры по варианту №6
    a, b, c, d = 0.2, 0.6, 0.05, 0.6
    n_inputs = 10
    n_hidden = 4
    n_outputs = 1
    print(f"Функция: y = {a}*cos({d}x) + {c}*sin({d}x)")
    print("Активация: скрытый слой - сигмоид, выходной - линейный\n")

    print("Генерация данных...")
    x, y = generate_data(a, b, c, d)

    print("Создание обучающей выборки...")
    X, Y = create_dataset(y, n_inputs)

    # Разделение на обучающую и тестовую выборки (70%/30%)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Нормализация данных
    X_mean, X_std = X_train.mean(), X_train.std()
    Y_mean, Y_std = Y_train.mean(), Y_train.std()

    X_train_norm = (X_train - X_mean) / X_std
    Y_train_norm = (Y_train - Y_mean) / Y_std
    X_test_norm = (X_test - X_mean) / X_std
    Y_test_norm = (Y_test - Y_mean) / Y_std

    # Создание и обучение сети
    print("\nСоздание нейронной сети...")
    nn = NeuralNetwork(n_inputs, n_hidden, n_outputs)

    print("Начало обучения...")
    losses = nn.train(X_train_norm, Y_train_norm, epochs=10000, learning_rate=0.1)

    # Прогнозирование
    train_predictions_norm = nn.forward(X_train_norm)
    train_predictions = train_predictions_norm * Y_std + Y_mean

    test_predictions_norm = nn.forward(X_test_norm)
    test_predictions = test_predictions_norm * Y_std + Y_mean

    # Результаты обучения
    print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    train_results = pd.DataFrame({
        'Эталон': Y_train,
        'Прогноз': train_predictions.flatten(),
        'Отклонение': np.abs(Y_train - train_predictions.flatten())
    })
    print("Первые 10 записей:")
    print(train_results.head(10).round(6))

    # Результаты прогнозирования
    print("\n=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ===")
    test_results = pd.DataFrame({
        'Эталон': Y_test,
        'Прогноз': test_predictions.flatten(),
        'Отклонение': np.abs(Y_test - test_predictions.flatten())
    })
    print("Первые 10 записей:")
    print(test_results.head(10).round(6))

    # Визуализация
    plt.figure(figsize=(15, 10))

    # График 1: Функция и прогноз на участке обучения
    plt.subplot(2, 2, 1)
    train_indices = range(n_inputs, n_inputs + len(Y_train))
    plt.plot(train_indices, Y_train, 'b-', label='Эталон', linewidth=2, alpha=0.7)
    plt.plot(train_indices, train_predictions, 'r--', label='Прогноз', linewidth=1.5)
    plt.title('Прогнозируемая функция на участке обучения')
    plt.xlabel('Номер точки')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 2: Изменение ошибки
    plt.subplot(2, 2, 2)
    plt.plot(losses, 'g-', linewidth=1)
    plt.title('Изменение ошибки в процессе обучения')
    plt.xlabel('Итерация')
    plt.ylabel('Среднеквадратичная ошибка')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # График 3: Результаты прогнозирования (тест)
    plt.subplot(2, 2, 3)
    test_indices = range(n_inputs + len(Y_train), n_inputs + len(Y_train) + len(Y_test))
    plt.plot(test_indices, Y_test, 'b-', label='Эталон', linewidth=2, alpha=0.7)
    plt.plot(test_indices, test_predictions, 'r--', label='Прогноз', linewidth=1.5)
    plt.title('Результаты прогнозирования (тестовая выборка)')
    plt.xlabel('Номер точки')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 4: Исходная функция
    plt.subplot(2, 2, 4)
    plt.plot(x, y, 'purple', linewidth=2, alpha=0.8)
    plt.title(f'Исходная функция\ny = {a}·cos({d}x) + {c}·sin({d}x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Статистика
    print(f"\n=== СТАТИСТИКА ===")
    print(f"Финальная ошибка обучения: {losses[-1]:.8f}")
    print(f"Среднее отклонение на обучении: {train_results['Отклонение'].mean():.6f}")
    print(f"Среднее отклонение на тесте: {test_results['Отклонение'].mean():.6f}")
    print(f"Максимальное отклонение на тесте: {test_results['Отклонение'].max():.6f}")


if __name__ == "__main__":
    main()
