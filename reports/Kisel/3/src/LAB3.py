import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import seaborn as sns

# Переходим в директорию с файлом
os.chdir("d:/Универ/ОМО/ОМО2025/Лаба3/")

# Загружаем данные
columns = [
    'area',
    'perimeter',
    'compactness',
    'length_of_kernel',
    'width_of_kernel',
    'asymmetry_coefficient',
    'length_of_kernel_groove',
    'class'
]

df = pd.read_csv("seeds_dataset.txt", delim_whitespace=True, names=columns)

print("🔹 Первые 5 строк:")
print(df.head())
print("\n🔹 Информация о данных:")
print(df.info())
print("\n🔹 Проверка на пропуски:")
print(df.isnull().sum())
print("\n🔹 Статистика по числовым признакам:")
print(df.describe())


# Разделяем признаки и целевую переменную
X = df.drop('class', axis=1)
y = df['class']

# Разделяем выборку на обучающую и тестовую (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Стандартизация (обучаем scaler только на обучающей выборке)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализируем модели
knn = KNeighborsClassifier(n_neighbors=5)
tree = DecisionTreeClassifier(random_state=42)
svm = SVC(kernel='rbf', random_state=42)

# Обучаем модели на обучающей выборке
knn.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

# Делаем предсказания на тестовой выборке
y_pred_knn = knn.predict(X_test_scaled)
y_pred_tree = tree.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)

# Вычисляем accuracy для каждой модели
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Выводим результаты
print("Точность (Accuracy) каждой модели:")
print(f"KNN: {acc_knn:.3f}")
print(f"Decision Tree: {acc_tree:.3f}")
print(f"SVM: {acc_svm:.3f}")

# Определим лучшую модель
best_model = max(
    [('KNN', acc_knn), ('Decision Tree', acc_tree), ('SVM', acc_svm)],
    key=lambda x: x[1]
)
print(f"\nЛучшая модель: {best_model[0]} (accuracy = {best_model[1]:.3f})")

# Проверяем значения k от 1 до 30
k_values = range(1, 31)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)

# Строим график
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title('Зависимость точности модели KNN от количества соседей (k)')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Находим лучшее значение k
best_k = k_values[accuracies.index(max(accuracies))]
best_acc = max(accuracies)
print(f"Лучшее k: {best_k}, Accuracy = {best_acc:.3f}")

# Предсказания лучшей модели (допустим, KNN с оптимальным k)
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred_best = best_knn.predict(X_test_scaled)

# Применяем PCA (уменьшаем размерность до 2 признаков)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)

# Строим DataFrame для удобства
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['Predicted class'] = y_pred_best

# Визуализация
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Predicted class',
    data=pca_df,
    palette='Set1',
    alpha=0.8
)
plt.title('Визуализация семян после PCA (раскраска по предсказанным классам)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend(title='Класс пшеницы')
plt.show()
