import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "iris.csv")
data = pd.read_csv(DATA_PATH)

# Encode categorical variables
# Encode target only
le = LabelEncoder()
data["Species"] = le.fit_transform(data["Species"])

# Normalize features only (exclude 'Species')
scaler = MinMaxScaler()
feature_cols = data.columns.drop("Species")
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Split features and target
x = data.drop(columns="Species")
y = data["Species"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Try different values of K
print("Experimenting with different K values:")
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k}, Accuracy={acc*100:.2f}")

# Use best K for visualization (e.g., K=3)
best_k = 3
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# Final evaluation
print("\nFinal Evaluation (K=3):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))

# Decision boundary visualization (using 2 features)
h = 0.02
x_min, x_max = x['SepalLengthCm'].min() - 1, x['SepalLengthCm'].max() + 1
y_min, y_max = x['SepalWidthCm'].min() - 1, x['SepalWidthCm'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Train on just 2 features for visualization
X_plot = x[['SepalLengthCm', 'SepalWidthCm']]
knn_2d = KNeighborsClassifier(n_neighbors=best_k)
knn_2d.fit(X_plot, y)

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
plt.scatter(X_plot['SepalLengthCm'], X_plot['SepalWidthCm'], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title(f'KNN Decision Boundaries (K={best_k})')
plt.show()