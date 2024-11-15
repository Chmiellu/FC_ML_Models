import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/teams-stats-standard.csv')

X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model with optimized hyperparameters
rf_model_optimized = RandomForestClassifier(
    n_estimators=10000,
    max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
)

rf_model_optimized.fit(X_train, y_train)

y_pred_optimized = rf_model_optimized.predict(X_test)

print("Dokładność modelu lasu losowego (optymalizacja):", accuracy_score(y_test, y_pred_optimized))
print("Raport klasyfikacji (optymalizacja):\n", classification_report(y_test, y_pred_optimized))

cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Las losowy (po optymalizacji)')
plt.savefig("rf_matrix_aggressive_optimized.png")
plt.show()

feature_importances = rf_model_optimized.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Ważność cech')
plt.ylabel('Cechy')
plt.title('Ważność cech w modelu lasu losowego')
plt.tight_layout()

plt.show()