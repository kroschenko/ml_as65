import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_csv("CarPrice_Assignment.csv")

features = ['horsepower', 'citympg', 'enginesize', 'curbweight', 'carwidth', 'carlength']

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2: {r2:.3f}")
print(f"MAE: {mae:.2f}")


plt.figure(figsize=(10, 7))
sns.regplot(x=df['horsepower'], y=df['price'], line_kws={"color": "red"})
plt.title("Price dependence on horsepower")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.grid(True)
plt.show()
