import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Загрузка данных
df = pd.read_csv("bank.csv", delimiter=';')

# 2. Преобразование категориальных признаков
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Преобразование целевой переменной
le_y = LabelEncoder()
df['y'] = le_y.fit_transform(df['y'])

# 3. Подготовка данных
X = df.drop('y', axis=1)
y = df['y']

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Обучение модели логистической регрессии
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 5. Предсказания и метрики
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 6. Матрица ошибок
matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(matrix)