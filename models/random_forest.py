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

rf_model_default = RandomForestClassifier(random_state=1)

rf_model_default.fit(X_train, y_train)

y_pred_default = rf_model_default.predict(X_test)

print("Dokładność modelu lasu losowego (domyślne parametry):", accuracy_score(y_test, y_pred_default))
print("Raport klasyfikacji (domyślne parametry):\n", classification_report(y_test, y_pred_default))

cm_default = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Las losowy')
plt.savefig("rf_matrix_default.png")  # Zapisanie wykresu do pliku
plt.show()
