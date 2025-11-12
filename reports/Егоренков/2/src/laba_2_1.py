import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

df = pd.read_csv('student-por.csv', sep=';')

required_columns = ['studytime', 'failures', 'G1', 'G2', 'G3']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Отсутствуют необходимые столбцы: {missing_columns}")

print("Количество пропущенных значений по столбцам:\n", df.isnull().sum())

X = df[['studytime', 'failures', 'G1', 'G2']]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"Коэффициент детерминации (R^2): {r2:.2f}")

coefficients = pd.Series(model.coef_, index=X.columns)
print("Коэффициенты модели:\n", coefficients)
print("Важность признаков по абсолютным значениям:\n", coefficients.abs().sort_values(ascending=False))

corr_matrix = df[['studytime', 'failures', 'G1', 'G2', 'G3']].corr()
print("Корреляционная матрица:\n", corr_matrix)

G2_min, G2_max = X_test['G2'].min(), X_test['G2'].max()
G2_range = np.linspace(G2_min, G2_max, 100)

mean_studytime = X_test['studytime'].mean()
mean_failures = X_test['failures'].mean()
mean_G1 = X_test['G1'].mean()

X_line = pd.DataFrame({
    'studytime': [mean_studytime] * 100,
    'failures': [mean_failures] * 100,
    'G1': [mean_G1] * 100,
    'G2': G2_range
})

G3_pred_line = model.predict(X_line)

plt.figure(figsize=(8, 6))
plt.scatter(X_test['G2'], y_test, color='blue', alpha=0.5, label='Реальные значения G3')
plt.plot(G2_range, G3_pred_line, color='red', linewidth=2, label='Линия регрессии по G2')

plt.xlabel('G2 (Pre-final grade)')
plt.ylabel('G3 (Final grade)')
plt.title('Зависимость G3 от G2 при фиксированных средних по другим признакам')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_dependence.png")

residuals = y_test - y_pred
print("Среднее значение остатков:", residuals.mean())
print("Стандартное отклонение остатков:", residuals.std())

predictions_df = pd.DataFrame({
    'Реальное G3': y_test.values,
    'Предсказанное G3': y_pred
})
print("Первые 10 предсказаний:\n", predictions_df.head(10))

plt.figure(figsize=(8, 4))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Реальные значения G3')
plt.ylabel('Остатки (ошибки)')
plt.title('Распределение остатков')
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_residuals.png")

print("Описание остатков:\n", residuals.describe())

sample_students = pd.DataFrame({
    'studytime': [2, 3],
    'failures': [0, 1],
    'G1': [10, 12],
    'G2': [10, 14]
})
sample_predictions = model.predict(sample_students)
print("Прогноз для гипотетических студентов:\n", sample_students)
print("Предсказанные G3:\n", sample_predictions)

print("\nФункционал завершен. Проверьте изображение 'regression_dependence.png' для визуализации.")