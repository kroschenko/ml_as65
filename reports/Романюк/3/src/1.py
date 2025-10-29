
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score


df = pd.read_csv(r"C:\ОМО\3lab\mushrooms.csv")


print(df.head())


X = df.drop('class', axis=1)
y = df['class']

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42
)

models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred, pos_label='p')
    recall = recall_score(y_test, y_pred, pos_label='p')
    
    results[name] = {'Precision': precision, 'Recall': recall}

print("\n=== Результаты классификации ===")
for model_name, metrics in results.items():
    print(f"{model_name}: Precision = {metrics['Precision']:.4f}, Recall = {metrics['Recall']:.4f}")

print("\n=== Вывод ===")
best_model = max(results.items(), key=lambda x: x[1]['Recall'])
print(f"Лучший классификатор по полноте (важно при высокой цене ошибки): {best_model[0]}")
