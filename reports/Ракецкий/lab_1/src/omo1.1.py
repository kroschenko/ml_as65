import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler	
# 1. Загрузка данных и информация о типах столбцов
wine_data = pd.read_csv('winequality-white.csv', sep=';')
print(wine_data.dtypes)
# 2. Преобразование целевой переменной quality в категориальную
def categorize_quality(quality):
    if quality <= 4:
        return "плохое"
    elif quality <= 6:
        return "среднее"
    else:
        return "хорошее"

wine_data['quality_category'] = wine_data['quality'].apply(categorize_quality)
# 3. Столбчатая диаграмма количества вин каждой категории качества
wine_data['quality_category'].value_counts().plot(kind='bar')
plt.title('Количество вин по категориям качества')
plt.xlabel('Категория качества')
plt.ylabel('Количество')
plt.show()
# 4. Проверка корреляции между fixed acidity и pH
correlation = wine_data['fixed acidity'].corr(wine_data['pH'])
print(f"Корреляция между fixed acidity и pH: {correlation:.4f}")
plt.scatter(wine_data['fixed acidity'], wine_data['pH'])
plt.title('Зависимость между fixed acidity и pH')
plt.xlabel('Fixed Acidity')
plt.ylabel('pH')
plt.show()
# 5. Поиск признака с наибольшим количеством выбросов
numeric_columns = wine_data.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = numeric_columns.drop('quality')

plt.figure(figsize=(12, 6))
wine_data[numeric_columns].boxplot()
plt.xticks(rotation=45)
plt.title('Ящики с усами для числовых признаков')
plt.show()
# 6. Стандартизация всех числовых признаков
scaler = StandardScaler()
wine_data_standardized = wine_data.copy()
wine_data_standardized[numeric_columns] = scaler.fit_transform(wine_data_standardized[numeric_columns])
print("Стандартизация выполнена")