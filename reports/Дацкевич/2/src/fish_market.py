import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
df = pd.read_csv('auto-mpg.csv')
print(df.shape)

# обработка нулей
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['cylinders', 'horsepower', 'weight', 'mpg'])
print(df.shape)

# обучение
X = df[['cylinders', 'horsepower', 'weight']]
y = df['mpg']
model=LinearRegression().fit(X,y)
y_m = model.predict(X)

#4. признаки
mse=mean_squared_error(y,y_m)
print(mse)
r2=r2_score(y,y_m)
print(r2)

#5. визуализация
plt.scatter(df['horsepower'], df['mpg'], color='blue',  label='Фактические данные')
mean_cylinders = np.mean(df['cylinders'])
mean_weight = np.mean(df['weight'])
sorted_hp = np.sort(df['horsepower'])
#значения
hp_line = pd.DataFrame({
    'cylinders': mean_cylinders,
    'horsepower': sorted_hp,
    'weight': mean_weight
})
plt.plot(sorted_hp, model.predict(hp_line), color='red', label='Линия регрессии')
plt.title('зависимость mpg от horsepower с линией регрессии')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.grid(True)
plt.show()
