import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

file_path = '/content/iris.csv'

# 1. Загрузка  и проверка
df = pd.read_csv(file_path)
print("Пропущенных значений :")
print(df.isnull().sum())
# 2. Количество образцов каждого вида
print("Количество образцов каждого вида:")
counts = {}  # пустой словарь для подсчёта

for item in df['variety']:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1
for key, value in counts.items():
    print(f"{key}: {value}")
# 3. Парные диаграммы рассеяния
sns.pairplot(df, hue='variety')
plt.show()
# 4. Средние значения
mean= df.groupby('variety').mean()
print(mean)
# 5. Ящик с усами
sns.boxplot(x='variety', y='petal.length', data=df)
plt.show()
# 6. Стандартизация данных
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

print(df_scaled)