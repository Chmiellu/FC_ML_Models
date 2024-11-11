import pandas as pd
from tabulate import tabulate

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

df_2022 = df[df['Year'] == 2022].copy()


df_2022['ChampionsLeague'] = df_2022['Club'].apply(lambda x: 1 if x in champions_league_clubs else 0)

df_2022['GVB'] = df_2022['Overall'].apply(lambda x: 1 if x >= 68 else 0)

selected_columns = [
    'Club', 'IntReputation', 'Age', 'SkillMoves', 'Crossing',
    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
    'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'ShotPower',
    'Jumping', 'LongShots', 'Aggression', 'Interceptions', 'Positioning',
    'Vision', 'Penalties', 'Composure', 'StandingTackle', 'SlidingTackle',
    'ChampionsLeague', 'GVB'
]

print(tabulate(df_2022[selected_columns], headers='keys', tablefmt='pretty'))

output_file_path = 'data/teams-stats.csv'
df_2022[selected_columns].to_csv(output_file_path, index=False)
