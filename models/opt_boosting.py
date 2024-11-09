import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie i przygotowanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')
X, y = df.drop(columns=['Club', 'GVB']), df['GVB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Parametry GridSearch
param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7],
              'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'subsample': [0.8, 0.9, 1.0]}

# Model i GridSearch
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Model, predykcja i ocena
best_gb_model = grid_search.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)
print(f"Dokładność: {accuracy_score(y_test, y_pred_gb)}\n{classification_report(y_test, y_pred_gb)}")

# Wykresy
sns.heatmap(confusion_matrix(y_test, y_pred_gb), annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title('Macierz pomyłek')
plt.savefig("gb_matrix_optimized.png")
plt.show()

plt.barh(X.columns, best_gb_model.feature_importances_, color='lightgreen')
plt.title('Ważność cech')
plt.tight_layout()
plt.savefig("gb_feature_importance_optimized.png")
plt.show()
