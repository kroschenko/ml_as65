import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('california_housing.csv')

print("=== ЗАДАНИЕ 1: Загрузка данных ===")
print(f"Размерность данных: {data.shape}")
print(f"Первые 5 строк:\n{data.head()}")
print(f"Информация о данных:\n{data.info()}")
print(f"Пропущенные значения:\n{data.isnull().sum()}")
print("\nРазделение данных на обучающую и тестовую выборки")

X = data[['median_income']]
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

print("\n=== ЗАДАНИЕ 2: Обучение модели линейной регрессии ===")

model = LinearRegression()
model.fit(X_train, y_train)

print("Модель успешно обучена!")
print(f"Коэффициент (наклон): {model.coef_[0]:.4f}")
print(f"Пересечение: {model.intercept_:.4f}")

print("\n=== ЗАДАНИЕ 3: Предсказания для тестовой выборки ===")

y_pred = model.predict(X_test)

print(f"Первые 5 предсказаний: {y_pred[:5]}")
print(f"Первые 5 реальных значений: {y_test.values[:5]}")

print("\n=== ЗАДАНИЕ 4: Оценка качества модели ===")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(12, 8))

plt.scatter(X_test, y_test, alpha=0.5, color='blue', label='Реальные значения')

x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=3, label='Линия регрессии')

plt.xlabel('Медианный доход (median_income)', fontsize=12)
plt.ylabel('Медианная стоимость дома (median_house_value)', fontsize=12)
plt.title('Зависимость стоимости дома от дохода населения\nЛинейная регрессия', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR²: {r2:.4f}',
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()