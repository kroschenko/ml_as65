# 1. Импорт библиотек и подготовка данных
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Загрузка датасета
digits = load_digits()
X, y = digits.data, digits.target

# Покажем пример изображения
plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f'Цифра: {digits.target[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 2. Модель с одним скрытым слоем (32 нейрона)
model_one_layer = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

# 3. Модель с двумя скрытыми слоями (по 32 нейрона)
model_two_layers = MLPClassifier(
    hidden_layer_sizes=(32, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

# 4. Обучение моделей
model_one_layer.fit(X_train, y_train)
model_two_layers.fit(X_train, y_train)

# 5. Оценка моделей
y_pred_one = model_one_layer.predict(X_test)
y_pred_two = model_two_layers.predict(X_test)

accuracy_one = accuracy_score(y_test, y_pred_one)
accuracy_two = accuracy_score(y_test, y_pred_two)

print(f"Точность с одним скрытым слоем: {accuracy_one:.4f}")
print(f"Точность с двумя скрытыми слоями: {accuracy_two:.4f}")

# Визуализация примеров предсказаний
plt.figure(figsize=(12, 6))

# Примеры правильных предсказаний
correct_indices = np.where(y_pred_one == y_test)[0][:4]

for i, idx in enumerate(correct_indices):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.title(f'Правильно: {y_test[idx]}')
    plt.axis('off')

# Примеры неправильных предсказаний
wrong_indices = np.where(y_pred_one != y_test)[0]
if len(wrong_indices) > 0:
    for i, idx in enumerate(wrong_indices[:4]):
        plt.subplot(2, 4, i + 5)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f'Ошибка: {y_test[idx]}→{y_pred_one[idx]}')
        plt.axis('off')

plt.tight_layout()
plt.show()

if accuracy_two > accuracy_one:
    print("Точность улучшилась с добавлением второго скрытого слоя")
elif accuracy_one > accuracy_two:
    print("Точность ухудшилась с добавлением второго скрытого слоя")
else:
    print("Точность осталась одинаковой")