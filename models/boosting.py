# Importy
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ładowanie danych
df = pd.read_csv('../data/teams-stats-standard.csv')

# Przygotowanie danych
X = df.drop(columns=['Club', 'GVB'])  # Wszystkie kolumny oprócz 'Club' i 'GVB' to cechy
y = df['GVB']  # Kolumna 'GVB' to etykiety

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tworzenie i trenowanie modelu Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,        # Liczba iteracji (drzew), które będą dodawane
    learning_rate=0.1,       # Współczynnik uczenia (rozmiar kroku przy każdej iteracji)
    max_depth=3,             # Maksymalna głębokość drzewa
    random_state=42
)

# Trenowanie modelu na danych treningowych
gb_model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred_gb = gb_model.predict(X_test)

# Ocena modelu Gradient Boosting
print("Dokładność modelu Gradient Boosting:", accuracy_score(y_test, y_pred_gb))
print("Raport klasyfikacji (Gradient Boosting):\n", classification_report(y_test, y_pred_gb))

# Macierz pomyłek dla modelu Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Gradient Boosting')
plt.savefig("gb_matrix.png")  # Zapisanie wykresu do pliku
plt.show()

# Wykres ważności cech dla modelu Gradient Boosting
feature_importances_gb = gb_model.feature_importances_

# Tworzenie wykresu ważności cech
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances_gb, color='lightgreen')
plt.xlabel('Ważność cech')
plt.ylabel('Cechy')
plt.title('Ważność cech w modelu Gradient Boosting')
plt.tight_layout()

# Zapisanie wykresu
plt.savefig("gb_feature_importance.png")
plt.show()
