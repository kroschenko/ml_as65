import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv("Titanic-Dataset.csv")

# Просмотр данных
print("Первые 5 записей:")
print(df.head())

print("\nСведения о данных:")
print(df.info())

print("\nРаспределение по выживаемости:")
survival_counts = df['Survived'].value_counts()
print(survival_counts)

# Визуализация выживаемости
plt.figure(figsize=(8, 5))
survival_counts.plot(kind='bar', color=['lightcoral', 'lightgreen'])
plt.title("Распределение пассажиров по выживаемости")
plt.xlabel("Выжил (1) / Не выжил (0)")
plt.ylabel("Число пассажиров")
plt.xticks(rotation=0)
plt.show()

# Обработка пропущенных значений в возрасте
print(f"\nПропущенных значений в возрасте до обработки: {df['Age'].isna().sum()}")

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

print(f"Пропущенных значений в возрасте после обработки: {df['Age'].isna().sum()}")

# Анализ категориальных переменных перед преобразованием
print("\n--- Анализ категориальных переменных ---")
print("Уникальные значения Sex:", df['Sex'].unique())
print("Распределение Sex:")
print(df['Sex'].value_counts())

print("\nУникальные значения Embarked:", df['Embarked'].unique())
print("Распределение Embarked:")
print(df['Embarked'].value_counts())

# Преобразование категориальных переменных - создаем все категории явно
print("\n--- Преобразование категориальных переменных ---")
categorical_cols = ['Sex', 'Embarked']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print("\nДанные после преобразования категориальных переменных:")
print(df[['Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].head(10))

# Объяснение преобразованных переменных
print("\n--- Объяснение преобразованных переменных ---")
print("Sex_male = 1 если мужчина, 0 если женщина")
print("Sex_female = 1 если женщина, 0 если мужчина")
print("Embarked_C = 1 если порт Cherbourg, иначе 0")
print("Embarked_Q = 1 если порт Queenstown, иначе 0") 
print("Embarked_S = 1 если порт Southampton, иначе 0")

# Распределение возраста
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=15, color='lightblue', edgecolor='navy', alpha=0.7)
plt.title("Распределение возраста пассажиров")
plt.xlabel("Возраст")
plt.ylabel("Частота")
plt.grid(axis='y', alpha=0.3)
plt.show()

# Создание нового признака
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 для учета самого пассажира

print("\nПример данных с новым признаком размера семьи:")
print(df[['SibSp', 'Parch', 'FamilySize']].head(8))

# Дополнительный анализ: выживаемость по полу и порту посадки
print("\n--- Дополнительный анализ ---")
print("Выживаемость по полу:")
survival_by_sex = df.groupby('Sex_female')['Survived'].mean()
print(survival_by_sex)

print("\nВыживаемость по порту посадки:")
if 'Embarked_C' in df.columns:
    survival_by_embarked = df.groupby([df['Embarked_C'], df['Embarked_Q'], df['Embarked_S']])['Survived'].mean()
    print("C (Cherbourg):", df[df['Embarked_C'] == 1]['Survived'].mean())
    print("Q (Queenstown):", df[df['Embarked_Q'] == 1]['Survived'].mean())
    print("S (Southampton):", df[df['Embarked_S'] == 1]['Survived'].mean())

# Проверка итоговых столбцов
print("\n--- Итоговые столбцы в DataFrame ---")
print(f"Всего столбцов: {len(df.columns)}")
print("Столбцы:", list(df.columns))
