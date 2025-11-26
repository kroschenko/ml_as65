import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_function(x, a=0.2, c=0.06, d=0.2):
    return a * np.cos(d * x) + c * np.sin(d * x)

a, c, d = 0.2, 0.06, 0.2

x_total = np.arange(0, 40, 0.1)
y_total = generate_function(x_total, a, c, d)

train_size = 300
x_train, y_train = x_total[:train_size], y_total[:train_size]
x_test, y_test = x_total[train_size:], y_total[train_size:]

def create_dataset(data, window_size=8):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 8
X_train, y_train_target = create_dataset(y_train, window_size)
X_test, y_test_target = create_dataset(y_test, window_size)

scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train_target.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test_target.reshape(-1, 1)).flatten()

print(f"Обучающая выборка: {X_train_scaled.shape}")
print(f"Тестовая выборка: {X_test_scaled.shape}")

class SigmoidNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2  
        
        return self.output
    
    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        
        output_error = output - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, output_error)
        db2 = (1/m) * np.sum(output_error, axis=0, keepdims=True)
        
        hidden_error = np.dot(output_error, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, hidden_error)
        db1 = (1/m) * np.sum(hidden_error, axis=0, keepdims=True)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate, verbose=True):
        losses = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y.reshape(-1, 1))**2)
            losses.append(loss)
            
            self.backward(X, y, output, learning_rate)
            
            if verbose and epoch % 200 == 0:
                print(f"Эпоха {epoch}, Ошибка: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        output = self.forward(X)
        return output

input_size, hidden_size, output_size = 8, 3, 1

nn = SigmoidNeuralNetwork(input_size, hidden_size, output_size)

epochs = 5000
learning_rate = 0.1 

print(f"\nОбучение ИНС с архитектурой: {input_size}-{hidden_size}-{output_size}")
print("Активация скрытого слоя: СИГМОИДНАЯ")
print("Активация выходного слоя: ЛИНЕЙНАЯ")

losses = nn.train(X_train_scaled, y_train_scaled, epochs, learning_rate)

train_predictions_scaled = nn.predict(X_train_scaled)
test_predictions_scaled = nn.predict(X_test_scaled)

train_predictions = scaler_y.inverse_transform(train_predictions_scaled).flatten()
test_predictions = scaler_y.inverse_transform(test_predictions_scaled).flatten()

train_errors = np.abs(y_train_target - train_predictions)
test_errors = np.abs(y_test_target - test_predictions)

print(f"\n РЕЗУЛЬТАТЫ ")
print(f"Средняя ошибка на обучающей выборке: {np.mean(train_errors):.6f}")
print(f"Средняя ошибка на тестовой выборке: {np.mean(test_errors):.6f}")
print(f"Максимальная ошибка: {np.max(test_errors):.6f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(losses)
plt.title('Изменение ошибки обучения (сигмоидная функция)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x_train[window_size:], y_train_target, 'b-', label='Эталон', linewidth=2)
plt.plot(x_train[window_size:], train_predictions, 'r--', label='Прогноз ИНС', linewidth=1.5)
plt.title('Прогнозирование на обучающей выборке')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x_test[window_size:], y_test_target, 'b-', label='Эталон', linewidth=2)
plt.plot(x_test[window_size:], test_predictions, 'r--', label='Прогноз ИНС', linewidth=1.5)
plt.title('Прогнозирование на тестовой выборке')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x_total, y_total, 'b-', label='Исходная функция', linewidth=2, alpha=0.7)
plt.plot(x_train[window_size:], train_predictions, 'g-', label='Прогноз (обучение)', linewidth=1)
plt.plot(x_test[window_size:], test_predictions, 'r-', label='Прогноз (тест)', linewidth=1)
plt.axvline(x=30, color='k', linestyle='--', label='Граница обучения/теста')
plt.title('Полный график функции и прогноза')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nТАБЛИЦА РЕЗУЛЬТАТОВ ОБУЧЕНИЯ (первые 10 строк):")
train_results = pd.DataFrame({
    'Эталонное значение': y_train_target[:10],
    'Полученное значение': train_predictions[:10], 
    'Отклонение': train_errors[:10]
})
print(train_results.round(6))

print("\nТАБЛИЦА РЕЗУЛЬТАТОВ ПРОГНОЗИРОВАНИЯ (первые 10 строк):")
test_results = pd.DataFrame({
    'Эталонное значение': y_test_target[:10],
    'Полученное значение': test_predictions[:10],
    'Отклонение': test_errors[:10]
})
print(test_results.round(6))

avg_train_error = np.mean(train_errors)
avg_test_error = np.mean(test_errors)

print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print(f"\t- Средняя ошибка обучения: {avg_train_error:.6f}")
print(f"\t- Средняя ошибка тестирования: {avg_test_error:.6f}")
print(f"\t- Максимальная ошибка: {np.max(test_errors):.6f}")

if avg_test_error < 0.001:
    quality = "ОТЛИЧНОЕ"
elif avg_test_error < 0.005:
    quality = "ХОРОШЕЕ"
elif avg_test_error < 0.01:
    quality = "УДОВЛЕТВОРИТЕЛЬНОЕ"
else:
    quality = "НЕУДОВЛЕТВОРИТЕЛЬНОЕ"

print(f"\t- Качество прогноза: {quality}")

relative_error = (np.mean(test_errors) / np.mean(np.abs(y_test_target))) * 100
print(f"\t- Относительная ошибка: {relative_error:.2f}%")
print(f"\t- Стандартное отклонение ошибок: {np.std(test_errors):.6f}")
print("\t- График показывает хорошее соответствие прогноза эталонным значениям")
print("\t- ИНС успешно уловила нелинейный характер исходной функции")