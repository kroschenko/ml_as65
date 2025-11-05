import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

sns.set(style="whitegrid")

os.chdir("d:/Универ/ОМО/ОМО2025/Лаба2/")
df = pd.read_csv("winequality-white.csv", sep=';')

print("Первые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистика по числовым признакам:")
print(df.describe())
print("\nПропуски по столбцам:")
print(df.isnull().sum())

# COUNTPLOT — количество наблюдений в каждой ка тегории quality
plt.figure(figsize=(8, 5))
quality_order = sorted(df['quality'].unique())
ax = sns.countplot(x='quality', data=df, palette='viridis', order=quality_order, edgecolor='black')
plt.title('Распределение качества белых вин')
plt.xlabel('Оценка качества (quality)')
plt.ylabel('Количество образцов')

# Подписи над столбиками (значения count)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=9)
plt.show()

# BOXPLOT — распределение alcohol в каждой категории quality
plt.figure(figsize=(9, 5))
ax2 = sns.boxplot(x='quality', y='alcohol', data=df, palette='coolwarm', order=quality_order,
                  showmeans=True, meanprops={"marker":"D", "markeredgecolor":"black", "markerfacecolor":"white"})
plt.title('Зависимость содержания алкоголя от качества вина')
plt.xlabel('Оценка качества (quality)')
plt.ylabel('Содержание алкоголя (%)')
plt.show()

plt.figure(figsize=(8, 5))

# Диаграмма рассеяния + линия регрессии
sns.regplot(
    x='alcohol',
    y='quality',
    data=df,
    scatter_kws={'alpha': 0.5, 'color': 'lightblue'},
    line_kws={'color': 'red'}                        
)

plt.title('Связь между содержанием алкоголя и качеством вина')
plt.xlabel('Содержание алкоголя (%)')
plt.ylabel('Оценка качества вина')
plt.show()

# Разделяем признаки (X) и целевую переменную (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Разделяем выборку на обучающую и тестовую (80% / 20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Вычисляем метрики качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Среднеквадратичная ошибка (MSE):", round(mse, 3))
print("Коэффициент детерминации (R²):", round(r2, 3))

# Смотрим, какие признаки сильнее влияют на качество
coefficients = pd.DataFrame({'Признак': X.columns, 'Коэффициент': model.coef_})
print("\nВлияние признаков на качество вина:")
print(coefficients.sort_values(by='Коэффициент', ascending=False))


# Создаём новый бинарный столбец: 1 — хорошее, 0 — плохое
df['good'] = (df['quality'] >= 7).astype(int)
print(df[['quality', 'good']].head(10))

# Разделяем признаки (X) и целевую переменную (y)
X = df.drop(['quality', 'good'], axis=1)
y = df['good']

# Разделяем выборку на обучающую и тестовую (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Обучаем модель логистической регрессии
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Предсказания на тестовых данных
y_pred = log_model.predict(X_test)

# Метрики качества классификации
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("\nМетрики качества модели:")
print(f"Accuracy (доля правильных предсказаний): {acc:.3f}")
print(f"Precision (точность для класса 'хорошее'): {prec:.3f}")
print(f"Recall (полнота для класса 'хорошее'): {rec:.3f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Плохое", "Хорошее"])
disp.plot(cmap="Blues")
plt.title("Матрица ошибок (Confusion Matrix)")
plt.show()