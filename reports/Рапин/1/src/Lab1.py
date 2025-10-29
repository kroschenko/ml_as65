import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv("Titanic-Dataset.csv")

print("Первые 5 строк:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())

print("\nКоличество выживших и погибших:")
print(data['Survived'].value_counts())

data['Survived'].value_counts().plot(kind='bar')
plt.title("Выжившие (1) и погибшие (0)")
plt.xlabel("Статус")
plt.ylabel("Количество")
plt.show()

print("\nПропуски в Age до обработки:", data['Age'].isnull().sum())

median_age = data['Age'].median()
data['Age'] = data['Age'].fillna(median_age)

print("Пропуски в Age после обработки:", data['Age'].isnull().sum())

# Сохраняем исходные данные перед one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

print("\nДанные после One-Hot Encoding:")
print(data_encoded.head())

plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Распределение возраста пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.show()

data_encoded['FamilySize'] = data_encoded['SibSp'] + data_encoded['Parch']

print("\nПервые строки с новым признаком FamilySize:")
print(data_encoded[['SibSp', 'Parch', 'FamilySize']].head())

# СТАНДАРТИЗАЦИЯ
# Выбираем только числовые признаки для стандартизации
numeric_columns = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']

# Создаем копию данных для стандартизации
data_for_scaling = data_encoded.copy()

# Стандартизируем только числовые признаки
scaler = StandardScaler()
data_for_scaling[numeric_columns] = scaler.fit_transform(data_for_scaling[numeric_columns])

print("\nСтандартизированные данные (первые 5 строк):")
print(data_for_scaling.head())

print("\nСтатистика стандартизированных числовых признаков:")
print(data_for_scaling[numeric_columns].describe())