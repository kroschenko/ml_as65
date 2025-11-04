import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df_happiness = pd.read_csv('World_Happiness_Report.csv')  # замените на ваш путь к файлу

features = ['GDP per capita', 'Social support', 'Healthy life expectancy']
X = df_happiness[features]
y = df_happiness['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

y_pred = model_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== РЕГРЕССИЯ: World Happiness Report ===")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Коэффициенты: {model_reg.coef_}")
print(f"Intercept: {model_reg.intercept_:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test['GDP per capita'], y_test, alpha=0.7, label='Фактические значения')
plt.scatter(X_test['GDP per capita'], y_pred, alpha=0.7, label='Предсказанные значения', color='red')

x_line = np.linspace(X_test['GDP per capita'].min(), X_test['GDP per capita'].max(), 100)

temp_X = X_test.copy()
temp_X['Social support'] = X_test['Social support'].mean()
temp_X['Healthy life expectancy'] = X_test['Healthy life expectancy'].mean()
y_line = model_reg.predict(pd.DataFrame({
    'GDP per capita': x_line,
    'Social support': [X_test['Social support'].mean()] * 100,
    'Healthy life expectancy': [X_test['Healthy life expectancy'].mean()] * 100
}))

plt.plot(x_line, y_line, color='green', linewidth=2, label='Линия регрессии')
plt.xlabel('GDP per capita')
plt.ylabel('Score (Оценка счастья)')
plt.title('Зависимость оценки счастья от ВВП на душу населения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
