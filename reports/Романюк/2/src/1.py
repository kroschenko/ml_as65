
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("C:\\ОМО\\2lab\\student-mat.csv", sep=';')

features = ['studytime', 'failures', 'G1', 'G2']
target = 'G3'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", round(mae, 3))
print("R² Score:", round(r2, 3))

plt.figure(figsize=(8, 5))
sns.regplot(x=data['G2'], y=data['G3'], line_kws={"color": "red"})
plt.title('Зависимость итоговой оценки G3 от предыдущей G2')
plt.xlabel('G2 (оценка за предыдущий период)')
plt.ylabel('G3 (итоговая оценка)')
plt.grid(True)
plt.tight_layout()
plt.show()
