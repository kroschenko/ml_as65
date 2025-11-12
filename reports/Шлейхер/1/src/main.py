import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv("german_credit.csv")
print("Информация о данных:")
print(df.info(), "\n")



print("Цели кредита:")
print(df["purpose"].value_counts(), "\n")

plt.figure(figsize=(8, 4))
df["purpose"].value_counts().head(5).plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("5 самых популярных целей кредита")
plt.xlabel("Цель кредита")
plt.ylabel("Кол-во заемщиков")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



df["sex"] = df["personal_status_sex"].apply(lambda x: 0 if "female" in x.lower() else 1)
print("Уникальные значения sex:", df["sex"].unique(),)

df["housing"] = df["housing"].map({"own": 2, "rent": 1, "for free": 0})
print("Уникальные значения housing:", df["housing"].unique(), "\n")



plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="default", y="credit_amount", hue="default", palette="pastel", legend=False)
plt.title("Суммы кредитов у хороших и плохих заемщиков")
plt.xlabel("Кредитоспособность (0 = хорошая, 1 = плохая)")
plt.ylabel("Сумма кредита")
plt.tight_layout()
plt.show()



pivot = df.pivot_table(
    values=["age", "duration_in_month"],
    index="credit_history",
    aggfunc="mean"
)
print("Средний возраст и длительность кредита по кредитной истории:")
print(pivot, "\n")



scaler = MinMaxScaler()
num_cols = ["age", "credit_amount", "duration_in_month"]
df_norm = df.copy()
df_norm[num_cols] = scaler.fit_transform(df[num_cols])

print("Первые строки после нормализации числовых признаков:")
print(df_norm[num_cols].head())
