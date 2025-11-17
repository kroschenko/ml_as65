import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка данных и ознакомление с ними
print("=== 1. ЗАГРУЗКА ДАННЫХ ===")
iris = load_iris()
X = iris.data
y = iris.target

# Создаем DataFrame для анализа
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in y]

print("Первые 5 строк данных:")
print(iris_df.head())

print("\nСтатистика данных:")
print(iris_df.describe())

print("\nРаспределение по классам:")
print(iris_df['species'].value_counts())

# Простая визуализация
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='species')
plt.title('Визуализация данных ирисов')
plt.show()

# 2. Разделение выборки
print("\n=== 2. РАЗДЕЛЕНИЕ ДАННЫХ ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Обучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")

# 3. Обучение моделей
print("\n=== 3. ОБУЧЕНИЕ МОДЕЛЕЙ ===")

# Поиск оптимального k для k-NN
print("Поиск оптимального k для k-NN...")
best_k = 1
best_score = 0

for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Оптимальное k: {best_k} (точность: {best_score:.4f})")

# Создаем и обучаем модели
models = {
    f'k-NN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Обучена модель: {name}")

# 4. Оценка точности
print("\n=== 4. ОЦЕНКА ТОЧНОСТИ ===")
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"Точность: {accuracy:.4f}")
    print("Отчет классификации:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Сравнение результатов
print("\n=== 5. СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")

# Визуализация сравнения моделей
plt.figure(figsize=(8, 5))
models_names = list(results.keys())
accuracies = list(results.values())

bars = plt.bar(models_names, accuracies, color=['lightblue', 'lightgreen', 'salmon'])
plt.ylabel('Точность (accuracy)')
plt.title('Сравнение точности моделей')
plt.ylim(0.8, 1.0)

# Добавляем значения на столбцы
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Вывод лучшей модели
best_model = max(results, key=results.get)
print(f"\nЛучшая модель: {best_model}")
print(f"Точность: {results[best_model]:.4f}")

# Дополнительно: важность признаков для Decision Tree
print("\nВажность признаков (Decision Tree):")
dt_model = models['Decision Tree']
feature_importance = pd.DataFrame({
    'Признак': iris.feature_names,
    'Важность': dt_model.feature_importances_
}).sort_values('Важность', ascending=False)

print(feature_importance)
