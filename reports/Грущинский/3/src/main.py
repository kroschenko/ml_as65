import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score

data = pd.read_csv(r'ml_as65\reports\Грущинский\3\src\breast_cancer.csv')

label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

x, y = data.drop(['id', 'diagnosis'], axis=1, inplace=False), data['diagnosis']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn.fit(x_train, y_train)
dt.fit(x_train, y_train)
svm.fit(x_train, y_train)
k_range = range(1, 21)
recall_scores = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(x_train, y_train)
    y_pred = knn_k.predict(x_test)
    recall = recall_score(y_test, y_pred, pos_label=1)
    recall_scores.append(recall)

best_k = k_range[np.argmax(recall_scores)]
print(f'Лучшее k по recall: {best_k}')

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train, y_train)

models = {
    'k-NN': knn_best,
    'Decision Tree': dt,
    'SVM': svm
}

for name, model in models.items():
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1)
    print(f"\n{name}:\nМатрица ошибок:\n{cm}\n")
    print(f"Репорт классификации:\n{report}")
    print(f"Precision (malignant): {precision}")
    print(f"Recall (malignant): {recall}")
    print(f"F1-score (malignant): {f1}")


recall_scores_for_malignant = {}

for name, model in models.items():
    y_pred = model.predict(x_test)
    recall_malignant = recall_score(y_test, y_pred, pos_label=1)
    recall_scores_for_malignant[name] = recall_malignant


best_model_name = max(recall_scores_for_malignant, key=recall_scores_for_malignant.get)
best_recall = recall_scores_for_malignant[best_model_name]

print(f"Модель, максимально минимизирующая ложноотрицательные:\n{best_model_name}")
print(f"Recall (злокачественная опухоль) = {best_recall:.2f}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(x_test)

cm_best = confusion_matrix(y_test, y_pred_best)
print(f"\nМатрица ошибок для лучшей модели ({best_model_name}):\n{cm_best}")
