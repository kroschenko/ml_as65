import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    a, b, c, d = 0.2, 0.6, 0.05, 0.6
    x = np.linspace(0, 20, 400)
    y = a * np.cos(d * x) + c * np.sin(d * x)
    return x, y

def create_dataset(data, n_inputs=10):
    X, Y = [], []
    for i in range(len(data) - n_inputs):
        X.append(data[i:i + n_inputs])
        Y.append(data[i + n_inputs])
    return np.array(X), np.array(Y)


# 3. МУЛЬТИРЕКУРРЕНТНАЯ СЕТЬ
class SimpleRecurrentNetwork:
    def __init__(self, n_inputs=10, n_hidden=4):
        self.W_in = np.random.randn(n_inputs, n_hidden) * 0.1
        self.W_rec = np.random.randn(n_hidden, n_hidden) * 0.01
        self.W_out = np.random.randn(n_hidden, 1) * 0.1
        self.b_hidden = np.zeros((1, n_hidden))
        self.b_out = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X_batch, return_states=False):
        batch_size = X_batch.shape[0]
        h_prev = np.zeros((batch_size, self.W_rec.shape[0]))

        h = self.sigmoid(
            np.dot(X_batch, self.W_in) +
            np.dot(h_prev, self.W_rec) +
            self.b_hidden
        )

        y = np.dot(h, self.W_out) + self.b_out

        if return_states:
            return y, h
        return y

    def train(self, X_train, y_train, epochs=3000, lr=0.01):
        X_mean, X_std = X_train.mean(), X_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()

        X_train_norm = (X_train - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std

        losses = []

        for epoch in range(epochs):
            predictions, h = self.forward(X_train_norm, return_states=True)

            error = predictions - y_train_norm.reshape(-1, 1)
            loss = np.mean(error ** 2)
            losses.append(loss)

            dW_out = np.dot(h.T, error) / len(X_train_norm)
            db_out = np.mean(error, axis=0, keepdims=True)

            dh = np.dot(error, self.W_out.T) * h * (1 - h)

            dW_in = np.dot(X_train_norm.T, dh) / len(X_train_norm)
            dW_rec = np.dot(h.T, dh) / len(X_train_norm)
            db_hidden = np.mean(dh, axis=0, keepdims=True)

            self.W_in -= lr * dW_in
            self.W_rec -= lr * dW_rec
            self.W_out -= lr * dW_out
            self.b_hidden -= lr * db_hidden
            self.b_out -= lr * db_out

            if epoch % 500 == 0:
                print(f"Эпоха {epoch}, Ошибка: {loss:.6f}")

        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

        return losses

    def predict(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        predictions_norm = self.forward(X_norm)
        predictions = predictions_norm * self.y_std + self.y_mean
        return predictions


# 4. ОСН. ПРОГ
def main():
    print("=== ЛАБОРАТОРНАЯ РАБОТА №6 ===")
    print("Мультирекуррентная сеть\n")

    x, y = generate_data()
    X, Y = create_dataset(y, n_inputs=10)

    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    print("Начало обучения...")
    rnn = SimpleRecurrentNetwork(n_inputs=10, n_hidden=4)
    losses = rnn.train(X_train, y_train, epochs=3000, lr=0.01)

    y_train_pred = rnn.predict(X_train)
    y_test_pred = rnn.predict(X_test)

    # РЕЗУЛЬТАТЫ
    print("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    for i in range(5):
        print(f"Эталон: {y_train[i]:.6f}, Прогноз: {y_train_pred[i][0]:.6f}, "
              f"Отклонение: {abs(y_train[i] - y_train_pred[i][0]):.6f}")

    print("\n=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ===")
    for i in range(5):
        print(f"Эталон: {y_test[i]:.6f}, Прогноз: {y_test_pred[i][0]:.6f}, "
              f"Отклонение: {abs(y_test[i] - y_test_pred[i][0]):.6f}")

    train_errors = abs(y_train - y_train_pred.flatten())
    test_errors = abs(y_test - y_test_pred.flatten())

    print(f"\n=== СТАТИСТИКА ===")
    print(f"Среднее отклонение на обучении: {train_errors.mean():.6f}")
    print(f"Среднее отклонение на тесте: {test_errors.mean():.6f}")
    print(f"Максимальное отклонение на тесте: {test_errors.max():.6f}")
    print(f"Финальная ошибка обучения: {losses[-1]:.6f}")

    # ГРАФИКИ
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(losses, 'r-')
    axes[0, 0].set_title('Изменение ошибки обучения')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Ошибка')
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')

    train_indices = range(len(y_train))
    axes[0, 1].plot(train_indices, y_train, 'b-', label='Эталон', alpha=0.7)
    axes[0, 1].plot(train_indices, y_train_pred, 'r--', label='Прогноз', alpha=0.9)
    axes[0, 1].set_title('Прогноз на обучающей выборке')
    axes[0, 1].set_xlabel('Номер точки')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    test_indices = range(len(y_test))
    axes[1, 0].plot(test_indices, y_test, 'b-', label='Эталон', alpha=0.7)
    axes[1, 0].plot(test_indices, y_test_pred, 'r--', label='Прогноз', alpha=0.9)
    axes[1, 0].set_title('Прогноз на тестовой выборке')
    axes[1, 0].set_xlabel('Номер точки')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(x, y, 'purple', linewidth=2)
    axes[1, 1].set_title('Исходная функция')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
