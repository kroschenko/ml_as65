import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ds_fileName = "datasets/medical_cost_personal_dataset.csv"
df = pd.read_csv(ds_fileName)


# 1
df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.head())


# 2
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# 3
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")


# 4
plt.figure(figsize=(7,5))
sns.scatterplot(x=df["bmi"], y=df["charges"], alpha=0.6)
sns.regplot(x=df["bmi"], y=df["charges"], scatter=False, color="red")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()
