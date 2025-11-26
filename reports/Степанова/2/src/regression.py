import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("CarPrice_Assignment.csv")

X = df[["horsepower", "citympg", "enginesize", "carwidth", "carlength", "carheight"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(7, 5))
sns.regplot(x=df["horsepower"], y=df["price"], line_kws={"color": "red"})
plt.title("Зависимость цены от л.с.")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.grid(True)
plt.show()
