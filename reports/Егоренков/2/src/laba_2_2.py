import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('mushrooms.csv')

X = df.drop('class', axis=1)
y = df['class']


le = LabelEncoder()
y_encoded = le.fit_transform(y)

onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = onehot_encoder.fit_transform(X)

feature_names = []
for i, col in enumerate(X.columns):
    categories = onehot_encoder.categories_[i][1:]
    for cat in categories:
        feature_names.append(f"{col}_{cat}")

print(f"\nКоличество признаков после One-Hot Encoding: {X_encoded.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = ...

cm = confusion_matrix(y_test, y_pred)

print(f"Точность: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Специфичность: {specificity}")

print("\nДетальный отчет:")
print(classification_report(y_test, y_pred, target_names=['Съедобный', 'Ядовитый']))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['e', 'p'], yticklabels=['e', 'p'])
plt.title('Матрица ошибок')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.savefig('confusion_matrix.png')

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': abs(model.coef_[0])
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.xlabel('Важность признака')
plt.title('Топ-10 признаков по важности')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png')

sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    print(f"Истинный класс: {y_test[idx]}, Предсказание: {y_pred[idx]}, Вероятность: {y_pred_proba[idx]:.4f}")