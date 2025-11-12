import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("C:\\Users\\wlksm\\OneDrive\\Рабочий стол\\Уник\\PP\\Parser\\3\\src\\glass.csv")

X = data.drop('Type', axis=1)
y = data['Type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=16, stratify=y
)

knn_best = KNeighborsClassifier(n_neighbors=2)
knn_best.fit(X_train, y_train)
y_pred_knn = knn_best.predict(X_test)

dt = DecisionTreeClassifier(random_state=16)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=16)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


print("Отчёт по моделям\n")

print("k-NN:")
print(classification_report(y_test, y_pred_knn))
print("-" * 55)

print("Decision Tree:")
print(classification_report(y_test, y_pred_dt))
print("-" * 55)

print("SVM:")
print(classification_report(y_test, y_pred_svm))
print("-" * 55)

acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"Accuracy k-NN: {acc_knn:.3f}")
print(f"Accuracy Decision Tree: {acc_dt:.3f}")
print(f"Accuracy SVM: {acc_svm:.3f}")
