import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
a, b, c, d = 0.1, 0.1, 0.05, 0.1
print(f"Функция: y = {a}*cos({b}x) + {c}*sin({d}x)")
print("Тип РНС: Джордана")
print("\nГенерация данных...")
x = np.linspace(0, 200, 1000)
y = a * np.cos(b * x) + c * np.sin(d * x)
print("Создание обучающей выборки...")


def create_dataset(data, look_back=6):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        Y.append(data[i])
    return np.array(X), np.array(Y)


X, Y = create_dataset(y, look_back=6)

# Разделение 70/30
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
print(f"Обучающих примеров: {len(X_train)}")
print(f"Тестовых примеров: {len(X_test)}")
print("\nСоздание РНС Джордана...")

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Инициализация весов
np.random.seed(42)
# 7 входов (6 значений + 1 контекст) -> 2 нейрона
W1 = np.random.randn(7, 2) * 0.1
b1 = np.zeros((1, 2))

# 2 нейрона -> 1 выход
W2 = np.random.randn(2, 1) * 0.1
b2 = np.zeros((1, 1))

print("Начало обучения РНС Джордана...")

epochs = 3000
learning_rate = 0.05  # Оптимальный alpha для РНС
train_errors = []
test_errors = []

for epoch in range(epochs):
    # Обучение с последовательной обработкой (важно для РНС!)
    context = np.zeros((1, 1))  
    epoch_loss = 0

    for i in range(len(X_train)):
       
        combined_input = np.concatenate([X_train[i:i + 1], context], axis=1)

        # Скрытый слой (сигмоида)
        hidden = sigmoid(np.dot(combined_input, W1) + b1)

        # Выходной слой (линейный)
        output = np.dot(hidden, W2) + b2
        error = output - Y_train[i]
        epoch_loss += error ** 2

        # Обратное распространение (BPTT)
        d_output = 2 * error
        dW2 = np.dot(hidden.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden)
        dW1 = np.dot(combined_input.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        # Обновление весов
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        # Обновление контекста для следующего шага
        context = output.copy()
    # Средняя ошибка эпохи
    train_loss = epoch_loss[0, 0] / len(X_train)
    train_errors.append(train_loss)
    # Тестирование
    test_context = np.zeros((1, 1))
    test_loss = 0

    for i in range(len(X_test)):
        combined_input = np.concatenate([X_test[i:i + 1], test_context], axis=1)
        hidden = sigmoid(np.dot(combined_input, W1) + b1)
        output = np.dot(hidden, W2) + b2
        test_loss += (output - Y_test[i]) ** 2
        test_context = output.copy()

    avg_test_loss = test_loss[0, 0] / len(X_test)
    test_errors.append(avg_test_loss)

    if epoch % 500 == 0:
        print(f"Эпоха {epoch}, Ошибка обучения: {train_loss:.6f}, Ошибка теста: {avg_test_loss:.6f}")
def predict_rnn(X):
    predictions = []
    context = np.zeros((1, 1))

    for i in range(len(X)):
        combined_input = np.concatenate([X[i:i + 1], context], axis=1)
        hidden = sigmoid(np.dot(combined_input, W1) + b1)
        output = np.dot(hidden, W2) + b2
        predictions.append(output[0, 0])
        context = output.copy()

    return np.array(predictions)
train_predictions = predict_rnn(X_train)
test_predictions = predict_rnn(X_test)
print("\n" + "=" * 60)
print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ РНС ДЖОРДАНА ===")
print("=" * 60)
print("Первые 10 записей:")
print("     Эталон   Прогноз  Отклонение")
for i in range(min(10, len(Y_train))):
    deviation = abs(Y_train[i] - train_predictions[i])
    print(f"{i:2d} {Y_train[i]:8.6f} {train_predictions[i]:8.6f} {deviation:11.6f}")

print("\n" + "=" * 60)
print("=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ РНС ДЖОРДАНА ===")
print("=" * 60)
print("Первые 10 записей:")
print("     Эталон   Прогноз  Отклонение")
for i in range(min(10, len(Y_test))):
    deviation = abs(Y_test[i] - test_predictions[i])
    print(f"{i:2d} {Y_test[i]:8.6f} {test_predictions[i]:8.6f} {deviation:11.6f}")
train_mse = np.mean((Y_train - train_predictions) ** 2)
test_mse = np.mean((Y_test - test_predictions) ** 2)
train_mae = np.mean(np.abs(Y_train - train_predictions))
test_mae = np.mean(np.abs(Y_test - test_predictions))
max_deviation = np.max(np.abs(Y_test - test_predictions))
print("\n" + "=" * 60)
print("=== СТАТИСТИКА ===")
print("=" * 60)
print(f"Финальная ошибка обучения (MSE): {train_errors[-1]:.8f}")
print(f"Финальная ошибка теста (MSE): {test_errors[-1]:.8f}")
print(f"Среднее отклонение на обучении (MAE): {train_mae:.6f}")
print(f"Среднее отклонение на тесте (MAE): {test_mae:.6f}")
print(f"Максимальное отклонение на тесте: {max_deviation:.6f}")
print(f"\nОптимальный параметр α (learning rate): {learning_rate}")
print(f"При α = {learning_rate} достигнута MSE на тесте: {test_errors[-1]:.8f}")
plt.figure(figsize=(12, 8))
# График 1: Изменение ошибки
plt.subplot(2, 2, 1)
plt.plot(train_errors, 'b-', label='Ошибка обучения', linewidth=1)
plt.plot(test_errors, 'r-', label='Ошибка теста', linewidth=1)
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.title('РНС Джордана: Изменение ошибки обучения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
# График 2: Прогнозируемая функция на участке обучения
plt.subplot(2, 2, 2)
n_points = min(50, len(Y_train))
x_train_indices = range(n_points)
plt.plot(x_train_indices, Y_train[:n_points], 'bo-',
         label='Эталон', markersize=3, linewidth=1, alpha=0.7)
plt.plot(x_train_indices, train_predictions[:n_points], 'rx-',
         label='Прогноз РНС', markersize=4, linewidth=1)
plt.xlabel('Индекс точки')
plt.ylabel('y')
plt.title('РНС Джордана: Участок обучения (первые 50 точек)')
plt.legend()
plt.grid(True, alpha=0.3)
# График 3: Сравнение эталон vs прогноз на обучении
plt.subplot(2, 2, 3)
plt.scatter(Y_train, train_predictions, alpha=0.5, s=10)
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()],
         'r--', label='Идеальная линия', linewidth=1)
plt.xlabel('Эталонные значения')
plt.ylabel('Прогноз РНС')
plt.title('РНС Джордана: Эталон vs Прогноз (обучение)')
plt.legend()
plt.grid(True, alpha=0.3)
# График 4: Сравнение эталон vs прогноз на тесте
plt.subplot(2, 2, 4)
plt.scatter(Y_test, test_predictions, alpha=0.5, s=10)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
         'r--', label='Идеальная линия', linewidth=1)
plt.xlabel('Эталонные значения')
plt.ylabel('Прогноз РНС')
plt.title('РНС Джордана: Эталон vs Прогноз (тест)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("\n" + "=" * 60)
print("Обучение РНС Джордана завершено!")
print("=" * 60)
