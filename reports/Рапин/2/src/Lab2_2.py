import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

print("=== ЗАДАНИЕ 1: Загрузка данных ===")
print(f"Размерность данных: {df.shape}")
print(f"Первые 5 строк:\n{df.head()}")
print(f"Пропущенные значения:\n{df.isnull().sum()}")

df = df.copy()  # Создаем копию для безопасной модификации
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['HasCabin'] = df['Cabin'].notna().astype(int)
df.drop('Cabin', axis=1, inplace=True)

df = df.dropna(subset=['Fare'])

print(f"Пропущенные значения после обработки:\n{df.isnull().sum()}")

df_processed = df.copy()

le_sex = LabelEncoder()
df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])

le_embarked = LabelEncoder()
df_processed['Embarked_encoded'] = le_embarked.fit_transform(df_processed['Embarked'])

df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1

df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

columns_to_drop = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked']
df_processed = df_processed.drop(columns_to_drop, axis=1)

print(f"\nДанные после предобработки:")
print(df_processed.head())
print(f"\nРазмерность после обработки: {df_processed.shape}")

X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

print(f"\nПризнаки (X): {X.shape}")
print(f"Целевая переменная (y): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== ЗАДАНИЕ 2: Обучение модели логистической регрессии ===")

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)

print("Модель логистической регрессии обучена!")

print("\n=== ЗАДАНИЕ 3: Оценка качества модели ===")

y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\n=== ЗАДАНИЕ 4: Построение и анализ матрицы ошибок ===")

cm = confusion_matrix(y_test, y_pred)
print(f"Матрица ошибок:\n{cm}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Predicted 0', 'Predicted 1'],
           yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Titanic Survival Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nАнализ матрицы ошибок:")
print(f"True Negatives (TN): {tn} - правильно предсказали смерть")
print(f"False Positives (FP): {fp} - ошибочно предсказали выживание")
print(f"False Negatives (FN): {fn} - ошибочно предсказали смерть")
print(f"True Positives (TP): {tp} - правильно предсказали выживание")
