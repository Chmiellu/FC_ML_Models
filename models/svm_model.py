import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Wczytanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')

# Przygotowanie danych - oddzielamy cechy od zmiennej celu
X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Tworzenie modelu SVM
svm_model = SVC(kernel='linear')

# Trening modelu
svm_model.fit(X_train, y_train)

# Przewidywanie na zestawie testowym
y_pred = svm_model.predict(X_test)

# Ocena modelu
print("Dokładność modelu SVM:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# Wykres: Macierz pomyłek
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Wykres: Wyświetlanie wyników klasyfikacji SVM w przestrzeni 2D (przy użyciu PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Trenowanie modelu SVM na dwóch pierwszych komponentach PCA
svm_model_pca = SVC(kernel='linear')
svm_model_pca.fit(X_pca[:len(X_train)], y_train)

# Predykcje na zestawie testowym
y_pred_pca = svm_model_pca.predict(X_pca[len(X_train):])

# Wizualizacja przestrzeni 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', style=y, s=100, alpha=0.7)
plt.title("SVM Decision Boundary in 2D Space (PCA)")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Dodatkowy wykres: Rozkład cech w stosunku do etykiet GVB
sns.pairplot(df, hue="GVB", vars=['IntReputation', 'Age', 'SkillMoves', 'Crossing', 'Finishing', 'HeadingAccuracy'])
plt.show()

# Wykres: Koeficjenty modelu SVM (jeśli kernel='linear')
coefficients = svm_model.coef_.flatten()
features = X.columns

plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.title("SVM Model Coefficients (Linear Kernel)")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=90)
plt.show()
