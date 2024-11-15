import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/teams-stats.csv')


missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("Brakujące wartości w zbiorze danych: Brak brakujących wartości")
else:
    print("Brakujące wartości:\n", missing_values)


class_balance = df['GVB'].value_counts()
print("Rozkład klasy docelowej GVB:\n", class_balance)
X = df.drop(columns=['Club', 'GVB'])
y = df['GVB']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
new_class_balance = y_resampled.value_counts()
print("Rozkład klasy docelowej GVB po nadpróbkowaniu:\n", new_class_balance)


original_index = X.index
resampled_index = X_resampled.index

new_samples_index = set(resampled_index) - set(original_index)

new_samples = X_resampled.iloc[list(new_samples_index)]
print("Nowe próbki wygenerowane przez SMOTE:\n", new_samples)

new_clubs = [f"Club_{i}" for i in range(len(X), len(X_resampled) + len(X))]


scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

X_resampled_scaled_df = pd.DataFrame(X_resampled_scaled, columns=X.columns)

X_resampled_scaled_df['Club'] = new_clubs

X_resampled_scaled_df['GVB'] = y_resampled
X_resampled_scaled_df.to_csv('data/teams-stats-standard.csv', index=False)
print("Dane zostały zapisane do 'teams-stats-standard.csv'")