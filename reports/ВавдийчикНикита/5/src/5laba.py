import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# Генерация данных согласно варианту
def generate_data(x):
    a, b, c, d = 0.1, 0.1, 0.05, 0.1
    return a * np.cos(b * x) + c * np.sin(d * x)


# Функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Защита от переполнения


def sigmoid_derivative(x):
    return x * (1 - x)


# Создание датасета для прогнозирования
def create_dataset(data, look_back=6):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        Y.append(data[i])
    return np.array(X), np.array(Y)


# Инициализация весов
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# Прямое распространение
def forward_propagation(X, W1, b1, W2, b2):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_output = np.dot(hidden_output, W2) + b2
    return hidden_output, final_output


# Обучение нейронной сети
def train_neural_network(X_train, Y_train, X_test, Y_test, learning_rate=0.01, epochs=10000):
    input_size = X_train.shape[1]
    hidden_size = 2
    output_size = 1

    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    train_errors = []
    test_errors = []
    best_weights = None
    best_error = float('inf')

    for epoch in range(epochs):
        # Прямое распространение
        hidden_output, final_output = forward_propagation(X_train, W1, b1, W2, b2)

        # Вычисление ошибки
        train_loss = np.mean((final_output - Y_train.reshape(-1, 1)) ** 2)
        train_errors.append(train_loss)

        # Прямое распространение на тестовых данных для мониторинга
        _, test_output = forward_propagation(X_test, W1, b1, W2, b2)
        test_loss = np.mean((test_output - Y_test.reshape(-1, 1)) ** 2)
        test_errors.append(test_loss)

        # Сохранение лучших весов
        if test_loss < best_error:
            best_error = test_loss
            best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

        # Обратное распространение ошибки
        d_final_output = 2 * (final_output - Y_train.reshape(-1, 1)) / X_train.shape[0]
        dW2 = np.dot(hidden_output.T, d_final_output)
        db2 = np.sum(d_final_output, axis=0, keepdims=True)

        d_hidden_output = np.dot(d_final_output, W2.T)
        d_hidden_input = d_hidden_output * sigmoid_derivative(hidden_output)
        dW1 = np.dot(X_train.T, d_hidden_input)
        db1 = np.sum(d_hidden_input, axis=0, keepdims=True)

        # Обновление весов
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 1000 == 0:
            print(f'Эпоха {epoch}, Ошибка обучения: {train_loss:.6f}, Ошибка теста: {test_loss:.6f}')

    return best_weights, train_errors, test_errors


# Основная программа
def main():
    print("Лабораторная работа №5: Нелинейные ИНС в задачах регрессии")
    print("=" * 60)

    # Генерация данных
    x = np.linspace(0, 200, 1000)
    y = generate_data(x)

    # Создание датасета
    X, Y = create_dataset(y, look_back=6)

    # Разделение на обучающую и тестовую выборки
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    # Обучение нейронной сети
    print("\nНачало обучения...")
    best_weights, train_errors, test_errors = train_neural_network(
        X_train, Y_train, X_test, Y_test,
        learning_rate=0.1, epochs=5000  # Уменьшил для скорости
    )

    W1, b1, W2, b2 = best_weights

    # Прогнозирование на обучающей выборке
    _, train_predictions = forward_propagation(X_train, W1, b1, W2, b2)
    train_predictions = train_predictions.flatten()

    # Прогнозирование на тестовой выборке
    _, test_predictions = forward_propagation(X_test, W1, b1, W2, b2)
    test_predictions = test_predictions.flatten()

    # Визуализация результатов
    plt.figure(figsize=(15, 10))  # ИСПРАВЛЕНО: добавлены скобки

    # 1. График исходной функции и прогноза на всем интервале
    plt.subplot(2, 2, 1)
    plt.plot(x[6:], y[6:], label='Исходная функция', alpha=0.7)

    # Прогноз на обучающей выборке
    train_x_indices = np.arange(6, 6 + len(train_predictions))
    plt.plot(x[train_x_indices], train_predictions, label='Прогноз (обучение)', alpha=0.8)

    # Прогноз на тестовой выборке
    test_x_indices = np.arange(6 + len(train_predictions), 6 + len(train_predictions) + len(test_predictions))
    plt.plot(x[test_x_indices], test_predictions, label='Прогноз (тест)', alpha=0.8)

    plt.axvline(x=x[6 + len(train_predictions)], color='red', linestyle='--', label='Граница train/test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('График прогнозируемой функции')
    plt.legend()
    plt.grid(True)

    # 2. График ошибки обучения
    plt.subplot(2, 2, 2)
    plt.plot(train_errors, label='Ошибка обучения')
    plt.plot(test_errors, label='Ошибка теста')
    plt.xlabel('Итерация')
    plt.ylabel('MSE')
    plt.title('Изменение ошибки в процессе обучения')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    # 3. График только на участке обучения
    plt.subplot(2, 2, 3)
    train_range = slice(6, 6 + len(train_predictions))
    plt.plot(x[train_range], y[train_range], label='Эталон', marker='o', markersize=2)
    plt.plot(x[train_range], train_predictions, label='Прогноз', marker='x', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Участок обучения: эталон vs прогноз')
    plt.legend()
    plt.grid(True)

    # 4. График только на участке тестирования
    plt.subplot(2, 2, 4)
    test_range = slice(6 + len(train_predictions), 6 + len(train_predictions) + len(test_predictions))
    plt.plot(x[test_range], y[test_range], label='Эталон', marker='o', markersize=2)
    plt.plot(x[test_range], test_predictions, label='Прогноз', marker='x', markersize=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Участок тестирования: эталон vs прогноз')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Таблица результатов обучения (первые 10 строк)
    print("\nРезультаты обучения (первые 10 записей):")
    train_table = []
    for i in range(min(10, len(Y_train))):
        train_table.append([
            i + 1,
            f"{Y_train[i]:.6f}",
            f"{train_predictions[i]:.6f}",
            f"{abs(Y_train[i] - train_predictions[i]):.6f}"
        ])

    print(tabulate(train_table,
                   headers=['№', 'Эталонное значение', 'Полученное значение', 'Отклонение'],
                   tablefmt='grid'))

    # Таблица результатов прогнозирования (первые 10 строк)
    print("\nРезультаты прогнозирования (первые 10 записей):")
    test_table = []
    for i in range(min(10, len(Y_test))):
        test_table.append([
            i + 1,
            f"{Y_test[i]:.6f}",
            f"{test_predictions[i]:.6f}",
            f"{abs(Y_test[i] - test_predictions[i]):.6f}"
        ])

    print(tabulate(test_table,
                   headers=['№', 'Эталонное значение', 'Полученное значение', 'Отклонение'],
                   tablefmt='grid'))

    # Статистика ошибок
    train_mae = np.mean(np.abs(Y_train - train_predictions))
    test_mae = np.mean(np.abs(Y_test - test_predictions))

    train_mse = np.mean((Y_train - train_predictions) ** 2)
    test_mse = np.mean((Y_test - test_predictions) ** 2)

    print(f"\nСтатистика ошибок:")
    print(f"Обучающая выборка - MAE: {train_mae:.6f}, MSE: {train_mse:.6f}")
    print(f"Тестовая выборка - MAE: {test_mae:.6f}, MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
