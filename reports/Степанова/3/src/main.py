import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



# 1. Загрузите данные, обработайте категориальные признаки и пропуски;
df = pd.read_csv("Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)

print(X.head())



# 2. Разделите данные на обучающую и тестовую выборки;
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)



# 3. Обучите модели k-NN, Decision Tree и SVM;
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# k-NN
f1_scores = []
k_values = range(1, 51)

print("Подбор k для k-NN:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    score = f1_score(y_test, preds)
    f1_scores.append(score)

    if k % 10 == 0:
        print(f"k={k}, F1={score:.4f}")

best_f1 = max(f1_scores)
best_k = k_values[f1_scores.index(best_f1)]
print(f"\nЛучший k: {best_k}")
print(f"F1-score при k={best_k}: {best_f1:.4f}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
pred_knn = knn_best.predict(X_test_scaled)
f1_knn = f1_score(y_test, pred_knn)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
f1_dt = f1_score(y_test, pred_dt)

# SVM
svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
f1_svm = f1_score(y_test, pred_svm)



# 4. Сравните модели по метрике F1-score для класса "отток";
print("\nСравнение моделей")
print("=======================")
print(f"F1-score k-NN (k={best_k}): {f1_knn:.4f}")
print(f"F1-score Decision Tree: {f1_dt:.4f}")
print(f"F1-score SVM: {f1_svm:.4f}")
