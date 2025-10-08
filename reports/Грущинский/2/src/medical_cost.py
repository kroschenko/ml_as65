import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('medical_cost_personal_dataset.csv')

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, R2: {r2}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['bmi'], y=data['charges'])
sns.lineplot(x=data['bmi'], y=model.predict(pd.get_dummies(data.drop('charges', axis=1), drop_first=True)), color='red')
plt.title('Зависимость Charges от BMI с линией регрессии')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()
##коммент для нового коммита
