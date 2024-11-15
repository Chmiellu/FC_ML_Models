import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/teams-stats-standard.csv')
X, y = df.drop(columns=['Club', 'GVB']), df['GVB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
lr_model = LogisticRegression()


voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)],
    voting='hard'
)

voting_model.fit(X_train, y_train)

y_pred_voting = voting_model.predict(X_test)
print(f"Dokładność (Voting Classifier): {accuracy_score(y_test, y_pred_voting)}")
print("Raport klasyfikacji (Voting Classifier):\n", classification_report(y_test, y_pred_voting))

cm_voting = confusion_matrix(y_test, y_pred_voting)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_voting, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywiste')
plt.title('Macierz pomyłek - Voting Classifier')
plt.savefig("voting_matrix.png")
plt.show()
