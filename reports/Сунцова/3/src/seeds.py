import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("seeds_dataset.txt", delim_whitespace=True , header=None)
print(df.head(5))

X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nЛучшая модель: {best_model_name} с точностью {accuracies[best_model_name]:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
y_pred_best = best_model.predict(X_scaled)

plt.figure(figsize=(8, 6))
for label in np.unique(y_pred_best):
    plt.scatter(
        X_pca[y_pred_best == label, 0],
        X_pca[y_pred_best == label, 1],
        label=f"Class {label}"
    )
plt.title(f"PCA Visualization - Predicted by {best_model_name}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
