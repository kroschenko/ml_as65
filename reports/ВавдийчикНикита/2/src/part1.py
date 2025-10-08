import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv('california_housing.csv')

# Разделение на признаки и целевую переменную
X = df[['median_income']]
y = df['median_house_value']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания для тестовой выборки
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Реальные значения')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Медианный доход')
plt.ylabel('Медианная стоимость дома')
plt.title('Зависимость стоимости дома от дохода')
plt.legend()
plt.grid(True)
plt.show()