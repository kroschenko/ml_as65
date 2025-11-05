import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Загрузка данных
df = pd.read_csv('BostonHousing.csv')

# 2. Подготовка данных - ВСЕ ПРИЗНАКИ для обучения
X = df.drop(['MEDV', 'CAT. MEDV'], axis=1)
y = df['MEDV']

# 3. Обучение модели на ВСЕХ признаках
model = LinearRegression()
model.fit(X, y)

# 4. Предсказания и метрики
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')

# 5. Визуализация - независимая для RM
plt.figure(figsize=(10, 6))
plt.scatter(df['RM'], df['MEDV'], alpha=0.6, color='blue', s=30)

# Отдельная регрессия только для RM для красивого графика
rm_model = LinearRegression()
rm_model.fit(df[['RM']], y)
rm_pred = rm_model.predict(df[['RM']])

plt.plot(df['RM'], rm_pred, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Среднее количество комнат (RM)', fontsize=12)
plt.ylabel('Медианная стоимость дома (MEDV)', fontsize=12)
plt.title('Зависимость стоимости дома от количества комнат', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
