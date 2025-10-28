import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

print("\nДанные после One-Hot Encoding:")
print(data.head())

plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Распределение возраста пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.show()

data['FamilySize'] = data['SibSp'] + data['Parch']

print("\nПервые строки с новым признаком FamilySize:")
print(data[['SibSp', 'Parch', 'FamilySize']].head())

scaler = StandardScaler()
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

print("\nСтандартизированные данные (первые 5 строк):")
print(data[numeric_features].head())
