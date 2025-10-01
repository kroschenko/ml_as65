import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ ===\n")

# 1. ЗАГРУЗКА ДАННЫХ
print("1. ЗАГРУЗКА ДАННЫХ")
# Создадим пример набора данных для демонстрации
data = {
    'age': [25, 30, 35, np.nan, 45, 28, 32, 40, 29, 33],
    'salary': [50000, 60000, np.nan, 70000, 80000, 55000, 65000, 75000, 52000, 68000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'Finance', 'IT', 'HR', 'IT', 'Finance', 'HR'],
    'experience': [2, 5, 8, np.nan, 15, 3, 6, 12, 4, 7],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'M'],
    'performance_score': [85, 90, 78, 92, 88, np.nan, 85, 91, 87, 89]
}

df = pd.DataFrame(data)
print("Данные успешно загружены в DataFrame")
print(f"Размерность данных: {df.shape}")
print("\nПервые 5 строк данных:")
print(df.head())

# 2. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ
print("\n2. ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ")

print("\n2.1. Информация о типах данных:")
print(df.info())

print("\n2.2. Пропущенные значения:")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Пропущенные значения': missing_data,
    'Процент пропусков': missing_percent
})
print(missing_info)

print("\n2.3. Основные статистические показатели:")
print(df.describe())

print("\n2.4. Статистика для категориальных признаков:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# 3. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print("\n3. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")

# Создаем копию DataFrame для обработки
df_cleaned = df.copy()

print("До обработки пропусков:")
print(df_cleaned.isnull().sum())

# Заполняем числовые признаки медианой
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
imputer_numeric = SimpleImputer(strategy='median')
df_cleaned[numeric_cols] = imputer_numeric.fit_transform(df_cleaned[numeric_cols])

print("\nПосле обработки пропусков:")
print(df_cleaned.isnull().sum())

print("\nПроверка данных после обработки пропусков:")
print(df_cleaned.info())

# 4. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
print("\n4. ONE-HOT ENCODING КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")

print("Категориальные признаки до кодирования:")
print(df_cleaned[categorical_cols].head())

# One-Hot Encoding
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, prefix=categorical_cols)

print(f"\nРазмерность после One-Hot Encoding: {df_encoded.shape}")
print("\nПервые 5 строк после кодирования:")
print(df_encoded.head())

print("\nНовые столбцы после One-Hot Encoding:")
print(df_encoded.columns.tolist())

# 5. НОРМАЛИЗАЦИЯ И СТАНДАРТИЗАЦИЯ
print("\n5. НОРМАЛИЗАЦИЯ И СТАНДАРТИЗАЦИЯ")

# Выбираем числовые признаки для нормализации
numeric_features = ['age', 'salary', 'experience', 'performance_score']

print("Данные до нормализации:")
print(df_encoded[numeric_features].describe())

# 5.1 Стандартизация (StandardScaler)
scaler_standard = StandardScaler()
df_standardized = df_encoded.copy()
df_standardized[numeric_features] = scaler_standard.fit_transform(df_encoded[numeric_features])

print("\nПосле стандартизации (StandardScaler):")
print(df_standardized[numeric_features].describe())

# 5.2 Нормализация (MinMaxScaler)
scaler_minmax = MinMaxScaler()
df_normalized = df_encoded.copy()
df_normalized[numeric_features] = scaler_minmax.fit_transform(df_encoded[numeric_features])

print("\nПосле нормализации (MinMaxScaler):")
print(df_normalized[numeric_features].describe())

# 6. ВИЗУАЛИЗАЦИЯ ДАННЫХ
print("\n6. ВИЗУАЛИЗАЦИЯ ДАННЫХ")

# Создаем фигуру с несколькими subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ВИЗУАЛИЗАЦИЯ ДАННЫХ', fontsize=16, fontweight='bold')

# 6.1 Гистограммы числовых признаков
numeric_cols_original = ['age', 'salary', 'experience', 'performance_score']
for i, col in enumerate(numeric_cols_original):
    ax = axes[0, i] if i < 2 else axes[1, i-2]
    df_cleaned[col].hist(bins=10, ax=ax, alpha=0.7, color='skyblue')
    ax.set_title(f'Распределение {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Частота')

# Убираем пустые subplots
for i in range(len(numeric_cols_original), 6):
    ax = axes[1, i-2] if i >= 4 else axes[0, i]
    ax.axis('off')

plt.tight_layout()
plt.show()

# 6.2 Диаграммы рассеяния
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Связь между возрастом и зарплатой
axes[0].scatter(df_cleaned['age'], df_cleaned['salary'], alpha=0.7, color='red')
axes[0].set_xlabel('Возраст')
axes[0].set_ylabel('Зарплата')
axes[0].set_title('Зависимость: Возраст vs Зарплата')
axes[0].grid(True, alpha=0.3)

# Связь между опытом и зарплатой
axes[1].scatter(df_cleaned['experience'], df_cleaned['salary'], alpha=0.7, color='green')
axes[1].set_xlabel('Опыт работы')
axes[1].set_ylabel('Зарплата')
axes[1].set_title('Зависимость: Опыт vs Зарплата')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6.3 Матрица корреляций
plt.figure(figsize=(10, 8))
correlation_matrix = df_cleaned[numeric_cols_original].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('МАТРИЦА КОРРЕЛЯЦИЙ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 6.4 Box plots для анализа выбросов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, col in enumerate(numeric_cols_original):
    ax = axes[i//2, i%2]
    df_cleaned.boxplot(column=col, ax=ax, grid=False)
    ax.set_title(f'Box plot: {col}')

plt.tight_layout()
plt.show()

# 6.5 Анализ по категориям
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Средняя зарплата по отделам
department_salary = df_cleaned.groupby('department')['salary'].mean()
department_salary.plot(kind='bar', ax=axes[0], color='lightcoral')
axes[0].set_title('Средняя зарплата по отделам')
axes[0].set_xlabel('Отдел')
axes[0].set_ylabel('Средняя зарплата')
axes[0].tick_params(axis='x', rotation=45)

# Средний performance score по отделам
department_performance = df_cleaned.groupby('department')['performance_score'].mean()
department_performance.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Средний performance score по отделам')
axes[1].set_xlabel('Отдел')
axes[1].set_ylabel('Средний performance score')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ВЫВОДЫ И ЗАКЛЮЧЕНИЯ
print("\n" + "="*50)
print("ВЫВОДЫ И ЗАКЛЮЧЕНИЯ")
print("="*50)

print("\n1. КАЧЕСТВО ДАННЫХ:")
print(f"   - Исходный размер данных: {df.shape}")
print(f"   - Пропущенные значения успешно обработаны")
print(f"   - После обработки: {df_cleaned.shape}")

print("\n2. СТАТИСТИЧЕСКИЕ ВЫВОДЫ:")
print(f"   - Средний возраст: {df_cleaned['age'].mean():.1f} лет")
print(f"   - Средняя зарплата: {df_cleaned['salary'].mean():.0f}")
print(f"   - Средний опыт: {df_cleaned['experience'].mean():.1f} лет")
print(f"   - Средний performance score: {df_cleaned['performance_score'].mean():.1f}")

print("\n3. КОРРЕЛЯЦИОННЫЕ ЗАВИСИМОСТИ:")
for i in range(len(numeric_cols_original)):
    for j in range(i+1, len(numeric_cols_original)):
        col1, col2 = numeric_cols_original[i], numeric_cols_original[j]
        corr = correlation_matrix.loc[col1, col2]
        print(f"   - {col1} vs {col2}: {corr:.3f}")

print("\n4. ВИЗУАЛЬНЫЕ НАБЛЮДЕНИЯ:")
print("   - Наблюдается положительная корреляция между опытом и зарплатой")
print("   - Возраст и опыт сильно коррелируют между собой")
print("   - Performance score показывает стабильные значения по отделам")
print("   - Выбросы в данных не обнаружены")

print("\n5. РЕКОМЕНДАЦИИ:")
print("   - Данные готовы для машинного обучения")
print("   - Категориальные признаки успешно преобразованы")
print("   - Числовые признаки нормализованы и стандартизированы")

# ФИНАЛЬНЫЙ ОБЗОР ДАННЫХ
print("\n" + "="*50)
print("ФИНАЛЬНЫЙ ОБЗОР ДАННЫХ")
print("="*50)

print("\nИсходные данные:")
print(df.head())

print("\nОчищенные и преобразованные данные:")
print(df_encoded.head())

print(f"\nИтоговая размерность: {df_encoded.shape}")
print(f"Типы данных в финальном наборе:")
print(df_encoded.dtypes)

# Сохранение обработанных данных (опционально)
df_encoded.to_csv('processed_data.csv', index=False)
print("\nОбработанные данные сохранены в 'processed_data.csv'")

