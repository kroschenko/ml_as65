import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score

from columns import file_path, columns



# 1 encode cat features
df = pd.read_csv(file_path, names=columns)

cat_cols = ['protocol_type', 'service', 'flag']

encoders = {col: LabelEncoder() for col in cat_cols}
for col in cat_cols:
    df[col] = encoders[col].fit_transform(df[col])

print(df.iloc[:, [1, 2, 3]].head())
print("\n")



# 2 target
df['target'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
df = df.drop('label', axis=1)

print(df.iloc[:, [-1]].head())
print("\n")



# 3 k-nn decisionTree svm
df_sample = df.sample(10000, random_state=42)
X = df_sample.drop('target', axis=1)
y = df_sample['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


results = []

# k-nn
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    start = time.time()
    knn.fit(X_train, y_train)
    end = time.time()
    y_pred = knn.predict(X_test)
    recall = recall_score(y_test, y_pred)
    results.append(("k-NN", f"k={k}", recall, end - start))

# decisionTree
tree = DecisionTreeClassifier(random_state=42)
start = time.time()
tree.fit(X_train, y_train)
end = time.time()
y_pred_tree = tree.predict(X_test)
recall_tree = recall_score(y_test, y_pred_tree)
results.append(("Decision Tree", "-", recall_tree, end - start))

# svm
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
start = time.time()
svm.fit(X_train, y_train)
end = time.time()
y_pred_svm = svm.predict(X_test)
recall_svm = recall_score(y_test, y_pred_svm)
results.append(("SVM", "-", recall_svm, end - start))



# 4
res_df = pd.DataFrame(results, columns=["Model", "Params", "Recall (Attack)", "Train Time (s)"])
print(res_df)



# 5 best model
best_model = res_df.loc[res_df['Recall (Attack)'].idxmax()]
print("\nBest model:")
print(best_model)
