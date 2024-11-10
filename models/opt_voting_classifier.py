import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie i przygotowanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')
X, y = df.drop(columns=['Club', 'GVB']), df['GVB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optymalizacja modeli bazowych
# 1. Random Forest
rf_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=0, n_iter=10)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# 2. Gradient Boosting
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, n_jobs=-1, verbose=0)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_

# 3. SVC
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, n_jobs=-1, verbose=0)
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_

# 4. K-Nearest Neighbors (kNN)
knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, n_jobs=-1, verbose=0)
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_

# Logistic Regression (bez optymalizacji)
lr_best = LogisticRegression(max_iter=1000, random_state=42)
lr_best.fit(X_train, y_train)

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[('rf', rf_best), ('gb', gb_best), ('svm', svm_best), ('knn', knn_best), ('lr', lr_best)],
    voting='soft',
    weights=[2, 2, 1, 1, 0.5]
)
voting_model.fit(X_train, y_train)

# Obliczanie dokładności dla każdego modelu
model_names = ["Random Forest", "Gradient Boosting", "SVM", "kNN", "Logistic Regression", "Voting Classifier"]
accuracies = [
    accuracy_score(y_test, rf_best.predict(X_test)),
    accuracy_score(y_test, gb_best.predict(X_test)),
    accuracy_score(y_test, svm_best.predict(X_test)),
    accuracy_score(y_test, knn_best.predict(X_test)),
    accuracy_score(y_test, lr_best.predict(X_test)),
    accuracy_score(y_test, voting_model.predict(X_test))
]

# Wykres dokładności
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies, palette="viridis")
plt.xlabel("Model")
plt.ylabel("Dokładność")
plt.title("Porównanie dokładności modeli")
plt.xticks(rotation=45)
plt.ylim(0.8, 1)  # Skala zaczyna się od 0.8 dla lepszej wizualizacji różnic
plt.tight_layout()
plt.show()

# Wyświetlenie dokładności i klasyfikacji w konsoli
for name, accuracy in zip(model_names, accuracies):
    print(f"{name} - Dokładność: {accuracy:.4f}")
