import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ładowanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')

# Przygotowanie danych
X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Drzewo decyzyjne - model z wartościami domyślnymi
dt_model_default = DecisionTreeClassifier(random_state=42)

# Trening modelu
dt_model_default.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred_default = dt_model_default.predict(X_test)

# Ocena modelu
print("Dokładność modelu drzewa decyzyjnego (domyślne parametry):", accuracy_score(y_test, y_pred_default))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_default))

# Macierz pomyłek
cm_default = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Drzewo decyzyjne')
plt.show()
