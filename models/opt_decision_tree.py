import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')

# Przygotowanie danych
X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Siatka parametrów do przeszukiwania
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Przeszukiwanie siatki parametrów
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Trening modelu z optymalizacją
grid_search.fit(X_train, y_train)

# Najlepsze parametry
best_params = grid_search.best_params_
print("Najlepsze parametry modelu drzewa decyzyjnego:", best_params)

# Predykcja z najlepszym modelem
y_pred_optimized = grid_search.predict(X_test)

# Ocena modelu
print("Dokładność modelu drzewa decyzyjnego (optymalizowane parametry):", accuracy_score(y_test, y_pred_optimized))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_optimized))

# Macierz pomyłek
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Drzewo decyzyjne (po optymalizacji)')
plt.show()
