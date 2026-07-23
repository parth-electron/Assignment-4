# Assignment 4 — Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants a machine learning model that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements. This project develops a **K-Nearest Neighbors (KNN)** classifier to perform that classification.

## Dataset Link
Breast Cancer Wisconsin (Diagnostic) Data Set (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data



## Libraries Used
- `pandas` — data loading and exploration
- `numpy` — numerical operations
- `scikit-learn` — `train_test_split`, `StandardScaler`, `LabelEncoder`, `KNeighborsClassifier`, evaluation metrics
- `matplotlib` — visualization (confusion matrix, accuracy-vs-K plot)

## Methodology
1. **Data Understanding** — Loaded the dataset from `DATA_PATH`, inspected the first five records, identified the 30 numerical diagnostic measurements as input features and `diagnosis` as the target, and reviewed dataset info and summary statistics.
2. **Data Preprocessing**
   - Checked for missing values.
   - Dropped the `id` column (a non-predictive identifier) and the empty `Unnamed: 32` column that ships with the raw Kaggle CSV.
   - Encoded the target (`diagnosis`) as B=0, M=1 using `LabelEncoder`.
   - Standardized all numeric features with `StandardScaler` (fit on the training set only, then applied to both train and test).
   - Split the dataset into 80% training / 20% testing with stratification on the target.
3. **Model Development** — Trained a `KNeighborsClassifier` with **K = 5** on the scaled training data and predicted labels for the test set.
4. **Model Evaluation** — Evaluated the model with Accuracy, Precision, Recall, and F1-score, generated a confusion matrix, and additionally swept K from 1–20 to visualize how accuracy changes with K.
5. **Conclusion** — Summarized findings, explained the importance of feature scaling for KNN, and noted a limitation of the algorithm.

## Results

| Metric | Value |
|---|---|
| Accuracy | ≈ 0.9561 |
| Precision | ≈ 0.9744 |
| Recall | ≈ 0.9048 |
| F1-Score | ≈ 0.9383 |

**Confusion Matrix:**
```
              Predicted B   Predicted M
Actual B          71             1
Actual M           4            38
```

**Observations:**
- The model reaches ~95.6% accuracy with K=5, but 4 malignant tumors were misclassified as benign versus only 1 benign misclassified as malignant — in a clinical context, minimizing missed malignant cases (false negatives) matters more than overall accuracy.
- Precision on malignant cases (97.4%) is higher than recall (90.5%), meaning when the model predicts malignant it's usually right, but it misses some true malignant cases.
- Sweeping K from 1–20 shows accuracy is fairly stable across a range of values (roughly K=3–11), suggesting the classes are reasonably well separated once features are scaled.

## Conclusion
This project used a K-Nearest Neighbors (K=5) classifier to distinguish malignant from benign breast tumors using the Wisconsin Diagnostic Breast Cancer dataset. After dropping the identifier and empty stray column, encoding the diagnosis label, and standardizing all 30 numeric features, the model achieved roughly 95.6% accuracy, 97.4% precision, 90.5% recall, and a 93.8% F1-score on the held-out test set.

Feature scaling is essential for KNN because the algorithm classifies a point based on distance to its nearest neighbors, and features on larger numeric scales (like area) would otherwise dominate that distance calculation over features on smaller scales (like smoothness), regardless of true diagnostic importance. Standardizing puts every feature on a comparable scale.

A key limitation of KNN is that it is computationally expensive at prediction time on large datasets, since it must calculate the distance from a new point to every point in the training set rather than learning a compact set of parameters in advance, making it slow to scale compared to models like logistic regression.



> **Note:** `data.csv` is intentionally **not** included in this repository per the submission instructions (dataset license). Download it from the Kaggle link above before running the code.
