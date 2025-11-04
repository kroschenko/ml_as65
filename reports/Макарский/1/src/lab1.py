import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('german_credit.csv')

print("=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Размер датасета: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о типах данных:")
print(df.info())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

print("\nПроверка пропущенных значений:")
print(df.isnull().sum())

print("\n=== РАСПРЕДЕЛЕНИЕ ЦЕЛИ КРЕДИТА ===")
purpose_counts = df['purpose'].value_counts()
print("Распределение целей кредита:")
print(purpose_counts)

plt.figure(figsize=(12, 6))
top_5_purposes = purpose_counts.head(5)
plt.bar(top_5_purposes.index, top_5_purposes.values)
plt.title('5 самых популярных целей кредита')
plt.xlabel('Цель кредита')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n=== ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ===")

print("Уникальные значения personal_status_sex:")
print(df['personal_status_sex'].unique())

df['sex_encoded'] = df['personal_status_sex'].apply(
    lambda x: 1 if 'male' in x.lower() else 0
)

print(f"\nРаспределение по полу:")
print(df['sex_encoded'].value_counts())

print("\nУникальные значения housing:")
print(df['housing'].unique())

housing_encoded = pd.get_dummies(df['housing'], prefix='housing')
df = pd.concat([df, housing_encoded], axis=1)

print("\nРезультат One-Hot Encoding для housing:")
print(housing_encoded.head())

print("\n=== СРАВНЕНИЕ СУММ КРЕДИТОВ ===")
plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='credit_amount', data=df)
plt.title('Сравнение сумм кредитов у "хороших" и "плохих" заемщиков')
plt.xlabel('Класс заемщика (0-хороший, 1-плохой)')
plt.ylabel('Сумма кредита')
plt.show()

good_borrowers = df[df['default'] == 0]['credit_amount']
bad_borrowers = df[df['default'] == 1]['credit_amount']

print(f"Средняя сумма кредита для хороших заемщиков: {good_borrowers.mean():.2f}")
print(f"Средняя сумма кредита для плохих заемщиков: {bad_borrowers.mean():.2f}")
print(f"Медианная сумма кредита для хороших заемщиков: {good_borrowers.median():.2f}")
print(f"Медианная сумма кредита для плохих заемщиков: {bad_borrowers.median():.2f}")

print("\n=== СВОДНАЯ ТАБЛИЦА ===")
pivot_table = df.pivot_table(
    values=['age', 'duration_in_month'],
    index='credit_history',
    aggfunc={'age': 'mean', 'duration_in_month': 'mean'}
).round(2)

print("Средний возраст и длительность кредита по кредитной истории:")
print(pivot_table)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

pivot_table['age'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Средний возраст по кредитной истории')
ax1.set_ylabel('Возраст')
ax1.tick_params(axis='x', rotation=45)

pivot_table['duration_in_month'].plot(kind='bar', ax=ax2, color='lightcoral')
ax2.set_title('Средняя длительность кредита по кредитной истории')
ax2.set_ylabel('Длительность (месяцы)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n=== НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ ===")

numeric_columns = ['age', 'credit_amount', 'duration_in_month']

print("Исходные статистики:")
print(df[numeric_columns].describe())

scaler_standard = StandardScaler()
df_standardized = df.copy()
df_standardized[numeric_columns] = scaler_standard.fit_transform(df[numeric_columns])

print("\nПосле стандартизации (Z-score):")
print(df_standardized[numeric_columns].describe())

scaler_minmax = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_columns] = scaler_minmax.fit_transform(df[numeric_columns])

print("\nПосле нормализации (Min-Max):")
print(df_normalized[numeric_columns].describe())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, col in enumerate(numeric_columns):

    axes[0, i].hist(df[col], bins=20, alpha=0.7, color='blue')
    axes[0, i].set_title(f'{col} (исходный)')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('Частота')


    axes[1, i].hist(df_normalized[col], bins=20, alpha=0.7, color='green')
    axes[1, i].set_title(f'{col} (нормализованный)')
    axes[1, i].set_xlabel(col)
    axes[1, i].set_ylabel('Частота')

plt.tight_layout()
plt.show()

print("\n=== МАТРИЦА КОРРЕЛЯЦИЙ ===")
numeric_df = df[['age', 'credit_amount', 'duration_in_month', 'default']]
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Матрица корреляций числовых признаков')
plt.show()

print("\n=== РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
default_distribution = df['default'].value_counts()
print(default_distribution)

plt.figure(figsize=(8, 6))
plt.pie(default_distribution.values, labels=['Хорошие заемщики', 'Плохие заемщики'],
        autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Распределение заемщиков по кредитоспособности')
plt.show()

df_processed = df_normalized.copy()
print(f"\nОбработанный датасет сохранен. Размер: {df_processed.shape}")

print("\n=== ИТОГОВАЯ ИНФОРМАЦИЯ ===")
print(f"Исходный размер данных: {df.shape}")
print(f"Обработанный размер данных: {df_processed.shape}")
print(f"Количество числовых признаков: {len(numeric_columns)}")
print(f"Количество категориальных признаков после кодирования увеличено")

with open('credit_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЕТ ПО АНАЛИЗУ GERMAN CREDIT DATA\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Размер исходного датасета: {df.shape}\n")
    f.write(f"Целевая переменная распределение:\n{default_distribution}\n\n")
    f.write("Топ-5 целей кредита:\n")
    for purpose, count in top_5_purposes.items():
        f.write(f"  {purpose}: {count}\n")
    f.write(f"\nСредний возраст: {df['age'].mean():.2f}\n")
    f.write(f"Средняя сумма кредита: {df['credit_amount'].mean():.2f}\n")
    f.write(f"Средняя длительность: {df['duration_in_month'].mean():.2f}\n")

print("\nАнализ завершен! Отчет сохранен в файл 'credit_analysis_report.txt'")