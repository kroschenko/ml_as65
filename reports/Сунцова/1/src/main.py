import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1
df = pd.read_csv("pima-indians-diabetes.csv", comment="#", header=None)
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

print("Статистические характеристики:")
print(df.describe(), "\n")


# 2
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness"]
for col in cols_to_fix:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)

print("Проверка нулевых значений после замены:")
print(df[cols_to_fix].isin([0]).sum(), "\n")


# 3
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df["BMI"], bins=20, color="skyblue", edgecolor="black")
plt.title("Распределение BMI")
plt.xlabel("BMI")
plt.ylabel("Количество")

plt.subplot(1, 2, 2)
plt.hist(df["Age"], bins=20, color="salmon", edgecolor="black")
plt.title("Распределение возраста")
plt.xlabel("Age")
plt.ylabel("Количество")

plt.tight_layout()
plt.show()


# 4
corr_matrix = df[["Glucose", "BMI", "Age", "Outcome"]].corr()

plt.figure(figsize=(5, 4))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
plt.colorbar(label="Коэффициент корреляции")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Матрица корреляции")
plt.show()


# 5
outcome_counts = df["Outcome"].value_counts()
labels = ["Без диабета", "С диабетом"]
plt.figure(figsize=(5, 5))
plt.pie(outcome_counts, labels=labels, autopct="%.1f%%", startangle=90, colors=["lightgreen", "lightcoral"])
plt.title("Распределение наличия диабета")
plt.show()


# 6
scaler = StandardScaler()
features = df.drop("Outcome", axis=1)
scaled_features = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled["Outcome"] = df["Outcome"]

print("Первые строки стандартизированных данных:")
print(df_scaled.head())
