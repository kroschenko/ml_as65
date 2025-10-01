from matplotlib import pyplot as plt
import pandas as pd
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

ds_fileName = "datasets/heart_disease_uci.csv"
df = pd.read_csv(ds_fileName)

# 1
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['id', 'dataset', 'num'], inplace=True)
df.replace({'TRUE': 1, 'FALSE': 0}, inplace=True)

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.dropna(inplace=True)

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# 3
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# 4
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
