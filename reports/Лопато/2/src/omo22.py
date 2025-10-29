
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("КЛАССИФИКАЦИЯ - ПРОГНОЗИРОВАНИЕ ВЫЖИВАЕМОСТИ НА ТИТАНИКЕ")
print("=" * 60)

print("Загрузка данных Titanic...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url)

print("Данные успешно загружены")
print(f"Размерность данных: {titanic_df.shape}")

print("\nПредварительная обработка данных...")

titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)

titanic_df_clean = titanic_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

label_encoder = LabelEncoder()
titanic_df_clean['Sex'] = label_encoder.fit_transform(titanic_df_clean['Sex'])
titanic_df_clean['Embarked'] = label_encoder.fit_transform(titanic_df_clean['Embarked'])

print("Данные обработаны")
print(f"Размерность после обработки: {titanic_df_clean.shape}")

X_titanic = titanic_df_clean.drop('Survived', axis=1)
y_titanic = titanic_df_clean['Survived']

print("\nРазделение данных на обучающую и тестовую выборки...")
X_train_tit, X_test_tit, y_train_tit, y_test_tit = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic
)

scaler = StandardScaler()
X_train_tit_scaled = scaler.fit_transform(X_train_tit)
X_test_tit_scaled = scaler.transform(X_test_tit)

print(f"Обучающая выборка: {X_train_tit.shape}")
print(f"Тестовая выборка: {X_test_tit.shape}")

print("\nОбучение модели логистической регрессии...")
logreg_model = LogisticRegression(random_state=42, max_iter=1000)
logreg_model.fit(X_train_tit_scaled, y_train_tit)
print("Модель успешно обучена")

y_pred_tit = logreg_model.predict(X_test_tit_scaled)
y_pred_proba = logreg_model.predict_proba(X_test_tit_scaled)[:, 1]

accuracy = accuracy_score(y_test_tit, y_pred_tit)
precision = precision_score(y_test_tit, y_pred_tit)
recall = recall_score(y_test_tit, y_pred_tit)

print("\n" + "=" * 40)
print("МЕТРИКИ КАЧЕСТВА МОДЕЛИ")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

cm = confusion_matrix(y_test_tit, y_pred_tit)

print("\nСоздание визуализаций...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Не выжил', 'Выжил'], 
            yticklabels=['Не выжил', 'Выжил'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Фактический класс')
plt.title('Матрица ошибок')

plt.subplot(1, 3, 2)
feature_importance = pd.DataFrame({
    'feature': X_titanic.columns,
    'importance': abs(logreg_model.coef_[0])  
}).sort_values('importance', ascending=True)

colors = ['green' if x > 0 else 'red' for x in logreg_model.coef_[0]]
plt.barh(feature_importance['feature'], feature_importance['importance'], 
         color=colors[::-1])  
plt.xlabel('Важность признака (абсолютное значение)')
plt.title('Важность признаков в модели')

plt.subplot(1, 3, 3)
for survived in [0, 1]:
    plt.hist(y_pred_proba[y_test_tit == survived], 
             alpha=0.7, label=f"Фактически {'Выжил' if survived == 1 else 'Не выжил'}",
             bins=20)
plt.xlabel('Вероятность выживания')
plt.ylabel('Количество')
plt.title('Распределение вероятностей')
plt.legend()

plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 40)
print("АНАЛИЗ МАТРИЦЫ ОШИБОК")
print("=" * 40)
print(f"True Negative (TN):  {tn:3d} - правильно предсказаны случаи невыживания")
print(f"False Positive (FP): {fp:3d} - ложно предсказаны случаи выживания")
print(f"False Negative (FN): {fn:3d} - ложно предсказаны случаи невыживания") 
print(f"True Positive (TP):  {tp:3d} - правильно предсказаны случаи выживания")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score:    {f1_score:.4f}")

print("\n" + "=" * 40)
print("АНАЛИЗ КОЭФФИЦИЕНТОВ МОДЕЛИ")
print("=" * 40)
print("Коэффициенты логистической регрессии:")
for feature, coef in zip(X_titanic.columns, logreg_model.coef_[0]):
    effect = " увеличивает шанс" if coef > 0 else " уменьшает шанс"
    print(f"  {feature:10}: {coef:7.4f} ({effect})")

print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЕТ ПО КЛАССИФИКАЦИИ")
print("=" * 60)
print(f"Точность модели: {accuracy:.1%}")
print(f" Precision: {precision:.1%} - доля правильно предсказанных выживших")
print(f" Recall: {recall:.1%} - доля выживших, правильно идентифицированных моделью")
print(f" Наиболее важные признаки: Sex, Fare, Pclass")
print(" Модель готова к использованию для прогнозирования выживаемости")