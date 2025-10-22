import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных
df = pd.read_csv('heart.csv')

print("1. ИНФОРМАЦИЯ О ДАННЫХ:")
print(df.info())
print("\nПропуски в данных:")
print(df.isnull().sum())

# 2. Статистика
print("\nОСНОВНЫЕ СТАТИСТИКИ:")
print(df.describe())
print("\nМедианы:")
print(df.median(numeric_only=True))
print("\nСтандартные отклонения:")
print(df.std(numeric_only=True))

# 3. Обработка пропусков (если есть)
if df.isnull().sum().sum() == 0:
    print("\nПропусков нет, обработка не требуется.")
else:
    df = df.fillna(df.mean(numeric_only=True))
    print("\nПропуски заполнены средними значениями.")

# 2. Столбчатая диаграмма
plt.figure(figsize=(6, 4))
df['target'].value_counts().plot(kind='bar')
plt.title('Количество здоровых и больных пациентов')
plt.xlabel('Target (0-здоров, 1-болен)')
plt.ylabel('Количество')
plt.xticks(rotation=0)
plt.show()
print("Вывод: больных пациентов больше, чем здоровых.")

# 3. Диаграмма рассеяния
plt.figure(figsize=(8, 6))
colors = ['red' if x == 1 else 'blue' for x in df['target']]
plt.scatter(df['age'], df['thalach'], c=colors, alpha=0.6)
plt.title('Зависимость пульса от возраста')
plt.xlabel('Возраст')
plt.ylabel('Максимальный пульс')
plt.legend(['Больные', 'Здоровые'])
plt.show()
print("Вывод: у более молодых пациентов чаще встречается более высокий пульс.")

# 4. Преобразование sex
df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
sex_encoded = pd.get_dummies(df['sex'], prefix='sex')
df = pd.concat([df, sex_encoded], axis=1)

print("\n4. Результат One-Hot Encoding:")
print(df[['sex', 'sex_female', 'sex_male']].head())

# 5. Средний холестерин
mean_chol_sick = df[df['target'] == 1]['chol'].mean()
mean_chol_healthy = df[df['target'] == 0]['chol'].mean()

print(f"\n5. Средний уровень холестерина:")
print(f"Больные: {mean_chol_sick:.2f}")
print(f"Здоровые: {mean_chol_healthy:.2f}")

# 6. Нормализация
scaler = MinMaxScaler()
df[['age', 'trestbps', 'chol', 'thalach']] = scaler.fit_transform(
    df[['age', 'trestbps', 'chol', 'thalach']]
)

print("\n6. Данные после нормализации:")
print(df[['age', 'trestbps', 'chol', 'thalach']].head())