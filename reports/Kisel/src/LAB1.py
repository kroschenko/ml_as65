import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  

#Заголовки в csv-файле(Задаем вручную)
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Загрузка и первичный просмотр данных
os.chdir("d:/Универ/ОМО/ОМО2025/Лаба1/")
df = pd.read_csv("pima-indians-diabetes.csv", comment="#", names=columns, header=None)
print("Первые 5 строк:") 
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистика по числовым признакам:")
print(df.describe())

# Замена скрытых пропусков (нулей) на медиану
cols = ['Glucose', 'BloodPressure', 'SkinThickness']
for c in cols:
    median = df[c].median()
    df[c] = df[c].replace(0, median)

# Визуализация распределений BMI и Age 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['BMI'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Распределение BMI')
plt.xlabel('BMI')
plt.ylabel('Количество')

plt.subplot(1, 2, 2)
df['Age'].hist(bins=20, color='salmon', edgecolor='black')
plt.title('Распределение возраста')
plt.xlabel('Возраст')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()

# Матрица корреляции (Glucose, BMI, Age, Outcome)
subset = df[['Glucose', 'BMI', 'Age', 'Outcome']]
corr = subset.corr()
print("\nМатрица корреляции:")
print(corr)

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Матрица корреляции')
plt.show()

# Круговая диаграмма распределения Outcome
outcome_counts = df['Outcome'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Без диабета', 'С диабетом'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Распределение наличия диабета')
plt.show()

# Стандартизация признаков (кроме Outcome)
scaler = StandardScaler()
features = df.drop('Outcome', axis=1)
scaled_features = scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled['Outcome'] = df['Outcome']

print("\nСтандартизированные данные:")
print(df_scaled.head())
