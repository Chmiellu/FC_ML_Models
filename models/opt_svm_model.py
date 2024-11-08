import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Najlepsze parametry: ", grid_search.best_params_)

# Model with best parameters
best_svm_model = grid_search.best_estimator_

# Predict on test set
y_pred_opt = best_svm_model.predict(X_test)

# Model evaluation
print("Dokładność modelu SVM (optymalizowanego):", accuracy_score(y_test, y_pred_opt))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_opt))

# Confusion matrix plot
cm_opt = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(12, 8))  # Increased width
sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Optimized SVM)')
plt.show()

# SVM in 2D space using PCA
pca_opt = PCA(n_components=2)
X_pca_opt = pca_opt.fit_transform(X_scaled)

# Train SVM on two PCA components
svm_model_pca_opt = best_svm_model
svm_model_pca_opt.fit(X_pca_opt[:len(X_train)], y_train)

# Predictions on the test set
y_pred_pca_opt = svm_model_pca_opt.predict(X_pca_opt[len(X_train):])

# Visualization in 2D space
plt.figure(figsize=(12, 8))  # Increased width
sns.scatterplot(x=X_pca_opt[:, 0], y=X_pca_opt[:, 1], hue=y, palette='coolwarm', style=y, s=100, alpha=0.7)
plt.title("SVM Decision Boundary in 2D Space (Optimized SVM)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Feature distribution plot
sns.pairplot(df, hue="GVB", vars=['IntReputation', 'Age', 'SkillMoves', 'Crossing', 'Finishing', 'HeadingAccuracy'])
plt.show()

# SVM coefficients plot (for linear kernel)
if grid_search.best_params_['kernel'] == 'linear':
    coefficients_opt = best_svm_model.coef_.flatten()
    features_opt = X.columns

    plt.figure(figsize=(14, 8))  # Increased width for better readability
    plt.bar(features_opt, coefficients_opt)
    plt.title("SVM Model Coefficients (Optimized Linear Kernel)")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=90)
    plt.show()
