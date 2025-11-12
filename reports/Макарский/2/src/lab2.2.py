import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


df_churn = pd.read_csv('Telco-Customer-Churn.csv')

df_churn = df_churn.drop('customerID', axis=1)
df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'], errors='coerce')
df_churn['TotalCharges'] = df_churn['TotalCharges'].fillna(df_churn['TotalCharges'].median())

categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_churn[col] = le.fit_transform(df_churn[col].astype(str))
    label_encoders[col] = le

X = df_churn[categorical_features + numerical_features]
y = df_churn['Churn'].map({'Yes': 1, 'No': 0})
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_clf = LogisticRegression(random_state=42, max_iter=1000)
model_clf.fit(X_train, y_train)

y_pred = model_clf.predict(X_test)
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)

print("\n=== КЛАССИФИКАЦИЯ: Telco Customer Churn ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (для 'Yes'): {precision:.4f}")
print(f"Recall (для 'Yes'): {recall:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No', 'Yes'],
           yticklabels=['No', 'Yes'])
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.title('Матрица ошибок для прогнозирования оттока клиентов')
plt.show()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model_clf.coef_[0])
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Важность признака (абсолютное значение коэффициента)')
plt.title('Важность признаков в модели логистической регрессии')
plt.tight_layout()
plt.show()
