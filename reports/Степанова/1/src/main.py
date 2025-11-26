import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1 Загрузите данные
df = pd.read_csv("auto-mpg.csv")
print("Первые строки:\n", df.head(), "\n")

# 2 Преобразуйте столбец horsepower в числовой формат и заполните пропуски средним значением
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
mean_hp = df["horsepower"].mean()
df["horsepower"] = df["horsepower"].fillna(mean_hp)
print(f"Среднее значение horsepower (замена пропусков): {mean_hp:.2f}\n")

# 3 Постройте диаграмму рассеяния weight mpg
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="weight", y="mpg", color="steelblue", edgecolor="black")
plt.title("Зависимость расхода топлива (mpg) от веса автомобиля (weight)")
plt.xlabel("weight")
plt.ylabel("mpg")
plt.show()

# 4 Преобразуйте категориальный признак origin 
# 1-usa 2-eu 3-japan
df["origin"] = df["origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
df = pd.get_dummies(df, columns=["origin"], drop_first=False)
print("Последние колонки после кодирования:\n", df.columns.tolist()[-5:], "\n")

# 5 Создайте новый признак age
df["age"] = 1983 - (1900 + df["model year"])
print("Пример возрастов автомобилей:\n", df[["model year", "age"]].head(), "\n")

# 6 Визуализируйте распределение количества цилиндров 
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="cylinders", color="skyblue", edgecolor="black")
plt.title("Распределение количества цилиндров")
plt.xlabel("Количество цилиндров")
plt.ylabel("Количество автомобилей")
plt.show()
