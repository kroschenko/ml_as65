import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Загрузка данных
df = pd.read_csv('Fish.csv')

# 2. Подготовка признаков и целевой переменной
X = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df['Weight']

# 3. Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)

# 4. Предсказание и оценка качества
y_pred = model.predict(X)

# Расчет метрик
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 5. Диаграмма рассеяния для Length3 и Weight с линией регрессии
plt.figure(figsize=(8, 6))
plt.scatter(df['Length3'], df['Weight'], alpha=0.7)
plt.xlabel('Length3')
plt.ylabel('Weight')
plt.title('Length3 vs Weight')

# Линия регрессии для Length3
x_line = pd.DataFrame({'Length3': np.linspace(df['Length3'].min(), df['Length3'].max(), 100)})
model_length3 = LinearRegression()
model_length3.fit(df[['Length3']], y)
y_line = model_length3.predict(x_line)

plt.plot(x_line['Length3'], y_line, color='red', linewidth=2)
plt.show()