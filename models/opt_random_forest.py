import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Las losowy - model zoptymalizowany z agresywnymi parametrami
rf_model_optimized = RandomForestClassifier(
    n_estimators=10000,            # Większa liczba drzew, aby lepiej uśrednić wyniki
    max_depth=50,                # Głębsze drzewa dla uchwycenia złożoności danych
    min_samples_split=2,         # Minimalna liczba próbek do podziału, aby drzewa były bardziej szczegółowe
    min_samples_leaf=1,          # Minimalna liczba próbek na liść dla większej elastyczności modelu
    max_features='sqrt',         # Korzystanie z pierwiastka liczby cech w każdym podziale dla lepszej różnorodności
    bootstrap=True,              # Losowe próbkowanie z powtórzeniami dla lepszej stabilności modelu
)

# Trening zoptymalizowanego modelu
rf_model_optimized.fit(X_train, y_train)

# Predykcja na zbiorze testowym dla zoptymalizowanego modelu
y_pred_optimized = rf_model_optimized.predict(X_test)

# Ocena zoptymalizowanego modelu
print("Dokładność modelu lasu losowego (optymalizacja):", accuracy_score(y_test, y_pred_optimized))
print("Raport klasyfikacji (optymalizacja):\n", classification_report(y_test, y_pred_optimized))

# Macierz pomyłek dla zoptymalizowanego modelu
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimized, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Macierz pomyłek - Las losowy (po optymalizacji)')
plt.savefig("rf_matrix_aggressive_optimized.png")  # Zapisanie wykresu do pliku
plt.show()

feature_importances = rf_model_optimized.feature_importances_

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Ważność cech')
plt.ylabel('Cechy')
plt.title('Ważność cech w modelu lasu losowego')
plt.tight_layout()

plt.show()