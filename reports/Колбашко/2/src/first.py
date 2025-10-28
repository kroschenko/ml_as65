import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("world_happiness_report.csv")

X = data[['GDP per capita', 'Social support', 'Healthy life expectancy']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

gdp = data['GDP per capita'].values.reshape(-1, 1)
gdp_sorted_idx = np.argsort(gdp.flatten())
gdp_sorted = gdp[gdp_sorted_idx]

X_line = pd.DataFrame({
    'GDP per capita': gdp_sorted.flatten(),
    'Social support': [data['Social support'].mean()] * len(gdp_sorted),
    'Healthy life expectancy': [data['Healthy life expectancy'].mean()] * len(gdp_sorted)
})

y_line = model.predict(X_line)

plt.figure(figsize=(8, 6))
plt.scatter(data['GDP per capita'], data['Score'], color='blue', alpha=0.6, label='Данные')
plt.plot(gdp_sorted, y_line, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('GDP per capita')
plt.ylabel('Score')
plt.title('Зависимость счастья от GDP per capita')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
