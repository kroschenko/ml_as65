import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# загрузка данных
df = pd.read_csv("heart.csv")

print("1. ИНФОРМАЦИЯ О ДАННЫХ:")
df_info = df.info()
print(df_info)

print("\nПропуски в данных:")
print(df.isna().sum())

print("\nОСНОВНЫЕ СТАТИСТИКИ:")
print(df.describe())

print("\nМедианы:")
print(df.median(numeric_only=True))

print("\nСтандартные отклонения:")
print(df.std(numeric_only=True))

# проверка пропусков
if df.isna().sum().sum() == 0:
    print("\nПропусков нет, обрабатывать не нужно.")
else:
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("\nПропуски заменены средними значениями.")

# график количества пациентов
plt.figure(figsize=(6, 4))
df["target"].value_counts().plot.bar()
plt.title("Количество здоровых и больных пациентов")
plt.xlabel("Target (0-здоров, 1-болен)")
plt.ylabel("Количество")
plt.xticks(rotation=0)
plt.show()
print("Вывод: больных пациентов больше, чем здоровых.")

# график пульс vs возраст
plt.figure(figsize=(8, 6))
plt.scatter(df["age"], df["thalach"],
            c=["red" if t == 1 else "blue" for t in df["target"]],
            alpha=0.6)
plt.title("Зависимость пульса от возраста")
plt.xlabel("Возраст")
plt.ylabel("Максимальный пульс")
plt.legend(["Больные", "Здоровые"])
plt.show()
print("Вывод: у молодых пациентов чаще выше пульс.")

# кодирование пола
df["sex"] = df["sex"].map({0: "female", 1: "male"})
df = pd.concat([df, pd.get_dummies(df["sex"], prefix="sex")], axis=1)

print("\n4. Результат One-Hot Encoding:")
print(df[["sex", "sex_female", "sex_male"]].head())

# сравнение холестерина
chol_sick = df.loc[df["target"] == 1, "chol"].mean()
chol_healthy = df.loc[df["target"] == 0, "chol"].mean()

print("\n5. Средний уровень холестерина:")
print(f"Больные: {chol_sick:.2f}")
print(f"Здоровые: {chol_healthy:.2f}")

# нормализация признаков
scaler = MinMaxScaler()
cols = ["age", "trestbps", "chol", "thalach"]
df[cols] = scaler.fit_transform(df[cols])

print("\n6. Данные после нормализации:")
print(df[cols].head())