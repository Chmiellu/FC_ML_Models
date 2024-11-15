import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



df = pd.read_csv('../data/teams-stats-standard.csv')

X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model_default = DecisionTreeClassifier(random_state=42)

dt_model_default.fit(X_train, y_train)

y_pred_default = dt_model_default.predict(X_test)

# evaluate the model
print("Dokładność modelu drzewa decyzyjnego (domyślne parametry):", accuracy_score(y_test, y_pred_default))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_default))

# confusion matrix
cm_default = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Drzewo decyzyjne')
plt.show()
