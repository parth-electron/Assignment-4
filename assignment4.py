import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# ---- Task 1: Data Understanding ----
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
  
    print(f"'{DATA_PATH}' not found — using scikit-learn's built-in copy of "
          f"the same dataset instead. Place your Kaggle CSV at '{DATA_PATH}' to use it directly.")
    from sklearn.datasets import load_breast_cancer
    _raw = load_breast_cancer(as_frame=True)
    df = _raw.frame.copy()
    _rename = {}
    for col in _raw.feature_names:
        parts = col.split(" ")
        if parts[0] == "mean":
            _rename[col] = "_".join(parts[1:]) + "_mean"
        elif parts[0] == "worst":
            _rename[col] = "_".join(parts[1:]) + "_worst"
        else:
            _rename[col] = "_".join(parts[:-1]) + "_se"
    df = df.rename(columns=_rename)
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df = df.drop(columns=["target"])
    df.insert(0, "id", range(842302, 842302 + len(df)))

print("First five records:")
print(df.head())
print()

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical features:", numerical_features)
print("Target variable: diagnosis")
print()

print("Dataset Info:")
df.info()
print()
print("Summary Statistics:")
print(df.describe())

# ---- Task 2: Data Preprocessing ----
print()
print("Missing values per column:")
print(df.isnull().sum())

# Drop unnecessary columns: id (identifier, no predictive value) and
# any fully-empty stray column (Kaggle's raw CSV ships an empty "Unnamed: 32" column)
cols_to_drop = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
df = df.drop(columns=cols_to_drop)
print()
print(f"Dropped columns: {cols_to_drop}")

# Encode target variable: Malignant=1, Benign=0
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])  # B=0, M=1
print("Encoded diagnosis classes:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print()
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Feature scaling (fit on train only, apply to both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Task 3: Model Development ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print()
print("Predicted labels (first 15):", y_pred[:15])
print("Actual labels (first 15):   ", y_test.values[:15])

# ---- Task 4: Model Evaluation ----
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print()
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print()
print("Confusion Matrix:")
print(cm)
print()
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - KNN (k=5)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, le.classes_)
plt.yticks(tick_marks, le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=110)
plt.close()

# Accuracy vs K plot (bonus exploration, still useful for observations)
k_values = range(1, 21)
accuracies = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    accuracies.append(accuracy_score(y_test, knn_k.predict(X_test_scaled)))

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), accuracies, marker="o")
plt.title("KNN Test Accuracy vs. K")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=110)
plt.close()

print()
print("Plots saved: confusion_matrix.png, accuracy_vs_k.png")
