import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#1. Загрузка
df = pd.read_csv("pima-indians-diabetes.csv", skiprows=9, header=None)
df.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
]
print(df.head())
print("Размер данных:", df.shape)

#2. разделение признаков и целевой
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#3. Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled)

#4. разделение на обучающую и тестовую выборку и обучение
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

#5. Предсказание и рассчет
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#6.матрица
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(f"Ложноположительные (FP): {matrix[0][1]}")
print(f"Ложноотрицательные (False Negatives): {matrix[1][0]}")