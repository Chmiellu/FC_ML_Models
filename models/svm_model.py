import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('../data/teams-stats-standard.csv')

# Prepare data - separate features and target variable
X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# SVM model creation
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test)

# Model evaluation
print("Dokładność modelu SVM:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))  # Increased width
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# SVM in 2D space using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train SVM on two PCA components
svm_model_pca = SVC(kernel='linear')
svm_model_pca.fit(X_pca[:len(X_train)], y_train)

# Predictions on the test set
y_pred_pca = svm_model_pca.predict(X_pca[len(X_train):])

# Visualization in 2D space
plt.figure(figsize=(12, 8))  # Increased width
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', style=y, s=100, alpha=0.7)
plt.title("SVM Decision Boundary in 2D Space (PCA)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Feature distribution plot
sns.pairplot(df, hue="GVB", vars=['IntReputation', 'Age', 'SkillMoves', 'Crossing', 'Finishing', 'HeadingAccuracy'])
plt.show()

# SVM coefficients plot (for linear kernel)
coefficients = svm_model.coef_.flatten()
features = X.columns

plt.figure(figsize=(14, 8))  # Increased width for better readability
plt.bar(features, coefficients)
plt.title("SVM Model Coefficients (Linear Kernel)")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=90)
plt.show()
