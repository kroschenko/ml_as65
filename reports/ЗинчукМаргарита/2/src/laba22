import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Загрузка данных
data = pd.read_csv('breast_cancer.csv')

# Подготовка данных
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Заполняем пропуски нулями
X = X.fillna(0)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказания
predictions = model.predict(X_test)

# Метрики
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))

# Матрица ошибок
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
