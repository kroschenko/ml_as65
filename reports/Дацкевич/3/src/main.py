# 1. Загружаем датасет digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


print("=== ЗАГРУЗКА ДАТАСЕТА DIGITS ===")
digits = load_digits()
X = digits.data  # данные (1797 изображений, каждое 8x8 = 64 пикселя)
y = digits.target  # метки (цифры от 0 до 9)
print(f"Размер данных: {X.shape}")
print(f"Размер меток: {y.shape}")
print(f"Классы: {np.unique(y)}")

# Покажем пример цифры 
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='binary')
    plt.title(f"Цифра: {digits.target[i]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 2. Разделяем на обучающую и тестовую выборки print("\n=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Обучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")

# 3. Обучаем три модели
print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")

models = {
    'k-NN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'SVM': SVC(random_state=42, kernel='rbf')
}

# Словарь для хранения результатов
results = {}

# 4. Обучение и оценка для каждой модели
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"МОДЕЛЬ: {name}")
    print(f"{'='*60}")

    # Обучение модели
    model.fit(X_train, y_train)

    # Предсказания на тестовой выборке     
    y_pred = model.predict(X_test)

    # Сохраняем результаты     
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy_score(y_test, y_pred)
    }

    # Classification report для каждой модели
    print(f"ОТЧЕТ ПО КЛАССИФИКАЦИИ ДЛЯ {name}:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ОБЩАЯ ТОЧНОСТЬ: {results[name]['accuracy']:.4f}")

# 5. Сравнение моделей
print(f"\n{'='*70}")
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print(f"{'='*70}")

# Сортируем модели по точности
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
print("РЕЙТИНГ МОДЕЛЕЙ ПО ТОЧНОСТИ:")
for i, (name, result) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {result['accuracy']:.4f}")

# Определяем лучшую модель
best_model_name, best_result = sorted_results[0]
print(f"\n  ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с точностью {best_result['accuracy']:.4f}")

# Демонстрация работы лучшей модели на нескольких примерах
print(f"\n=== ДЕМОНСТРАЦИЯ РАБОТЫ ЛУЧШЕЙ МОДЕЛИ ({best_model_name}) ===")
best_model = best_result['model']

# Берем несколько случайных примеров из тестовой выборки
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), 5, replace=False)
plt.figure(figsize=(12, 3))
for i, idx in enumerate(sample_indices):     # Берем пример из тестовой выборки
    test_image = X_test[idx].reshape(8, 8)
    true_label = y_test[idx]

    # Предсказываем цифру
    predicted_label = best_model.predict([X_test[idx]])[0]

    # Визуализируем
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_image, cmap='binary')
    plt.title(f"Истина: {true_label}\nПредсказано: {predicted_label}", color='green' if true_label == predicted_label else 'red')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Анализ ошибок
print("\n=== АНАЛИЗ ОШИБОК ===")
wrong_predictions = []
for i in range(len(X_test)):
    if y_test[i] != results[best_model_name]['predictions'][i]:
        wrong_predictions.append({             'index': i,
            'true': y_test[i],
            'predicted': results[best_model_name]['predictions'][i]
        })
print(f"Количество ошибок у лучшей модели: {len(wrong_predictions)}")
print(f"Точность: {(1 - len(wrong_predictions)/len(X_test)):.4f}")
if wrong_predictions:
    print("\nПримеры ошибок:")
    for i, error in enumerate(wrong_predictions[:3]):
        print(f"  Пример {error['index']}: истинная цифра {error['true']}, предсказана {error['predicted']}")
