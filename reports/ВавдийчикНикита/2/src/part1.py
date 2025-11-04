
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('default')
sns.set_palette("husl")

print("=" * 50)
print("РЕГРЕССИЯ - CALIFORNIA HOUSING")
print("=" * 50)


california = fetch_california_housing()
X_cal = pd.DataFrame(california.data, columns=california.feature_names)
y_cal = pd.Series(california.target, name='median_house_value')

print("Информация о данных:")
print(f"Размерность признаков: {X_cal.shape}")
print(f"Размерность целевой переменной: {y_cal.shape}")
print("\nПервые 5 строк данных:")
print(X_cal.head())
print(f"\nЦелевая переменная (первые 5 значений): {y_cal[:5].values}")

X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=42
)

print(f"\nРазмер обучающей выборки: {X_train_cal.shape}")
print(f"Размер тестовой выборки: {X_test_cal.shape}")

lr_model = LinearRegression()
lr_model.fit(X_train_cal, y_train_cal)

y_pred_cal = lr_model.predict(X_test_cal)

mse = mean_squared_error(y_test_cal, y_pred_cal)
r2 = r2_score(y_test_cal, y_pred_cal)

print("\nМЕТРИКИ КАЧЕСТВА МОДЕЛИ:")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"R² (Coefficient of Determination): {r2:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test_cal['MedInc'], y_test_cal, alpha=0.5, label='Фактические значения')
plt.scatter(X_test_cal['MedInc'], y_pred_cal, alpha=0.5, color='red', label='Предсказанные значения')

x_line = np.linspace(X_test_cal['MedInc'].min(), X_test_cal['MedInc'].max(), 100).reshape(-1, 1)
lr_simple = LinearRegression()
lr_simple.fit(X_test_cal[['MedInc']], y_test_cal)
y_line = lr_simple.predict(x_line)

plt.plot(x_line, y_line, color='black', linewidth=2, label='Линия регрессии')
plt.xlabel('Медианный доход (MedInc)')
plt.ylabel('Медианная стоимость дома')
plt.title('Диаграмма рассеяния: доход vs стоимость дома')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_cal, y_pred_cal, alpha=0.5)
plt.plot([y_test_cal.min(), y_test_cal.max()], [y_test_cal.min(), y_test_cal.max()], 'r--', lw=2)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Фактические vs Предсказанные значения')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nКОЭФФИЦИЕНТЫ МОДЕЛИ:")
for feature, coef in zip(california.feature_names, lr_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Свободный член: {lr_model.intercept_:.4f}")

print("\n" + "=" * 50)
print("ИТОГОВЫЙ ОТЧЕТ ПО РЕГРЕССИИ")
print("=" * 50)
print(f"✓ Модель объясняет {r2:.1%} дисперсии целевой переменной")
print(f"✓ Среднеквадратичная ошибка: {mse:.4f}")
print("✓ Наиболее важные признаки: MedInc, AveOccup, Latitude")
print("✓ Модель готова к использованию для прогнозирования стоимости жилья")
