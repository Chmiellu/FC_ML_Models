import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'data/teams-stats.ds.csv'
df = pd.read_csv(file_path)

champions_league_clubs = [
    'Real Madrid', 'Eintracht Frankfurt', 'Manchester City', 'Liverpool',
    'Chelsea', 'Tottenham Hotspur', 'Barcelona', 'Atlético Madrid', 'Sevilla',
    'Milan', 'Inter Milan', 'Napoli', 'Juventus', 'Bayern Munich',
    'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 'Paris Saint-Germain',
    'Marseille', 'Porto', 'Sporting CP', 'Ajax', 'Club Brugge', 'Red Bull Salzburg',
    'Celtic', 'Shakhtar Donetsk', 'Trabzonspor', 'Copenhagen'
]

df['ChampionsLeague'] = df['Club'].apply(lambda x: 1 if x in champions_league_clubs else 0)
df['GVB'] = df['Overall'].apply(lambda x: 1 if x >= 68 else 0)
skills = ['Dribbling', 'Crossing', 'Finishing', 'BallControl']
df_means = df.groupby('GVB')[skills].mean()

df_means.T.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
plt.title('Średnie umiejętności techniczne dla dobrych i słabszych klubów')
plt.xlabel('Umiejętności')
plt.ylabel('Średnia ocena')
plt.xticks(rotation=45)
plt.legend(['Słabszy klub (GVB=0)', 'Dobry klub (GVB=1)'])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='ChampionsLeague', hue='GVB', palette=['#1f77b4', '#ff7f0e'])
plt.title('Liczba klubów z i bez udziału w Lidze Mistrzów dla różnych klas GVB')
plt.xlabel('Udział w Lidze Mistrzów')
plt.ylabel('Liczba klubów')
plt.xticks([0, 1], ['Bez Ligi Mistrzów', 'Z Ligą Mistrzów'])
plt.legend(['Słabszy klub (GVB=0)', 'Dobry klub (GVB=1)'])
plt.show()
