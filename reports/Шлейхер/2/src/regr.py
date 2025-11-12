import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных выбор признаков
df = pd.read_csv("world_happiness_report.csv")

X = df[["GDP per capita", "Social support", "Healthy life expectancy"]]
y = df["Score"]

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)

# MSE R²
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Зависимость Score от GDP per capita
plt.figure(figsize=(8, 5))
sns.regplot(x=df["GDP per capita"], y=df["Score"], line_kws={"color": "red"})
plt.title("Зависимость счастья от ВВП на душу населения")
plt.xlabel("GDP per capita")
plt.ylabel("Score")
plt.grid(True)
plt.tight_layout()
plt.show()
