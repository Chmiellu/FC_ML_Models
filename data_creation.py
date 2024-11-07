import pandas as pd
from tabulate import tabulate

# Load the data
file_path = 'data/teams-stats.ds.csv'
df = pd.read_csv(file_path)

# Define the list of clubs that played in the Champions League in 2022
champions_league_clubs = [
    'Real Madrid', 'Eintracht Frankfurt', 'Manchester City', 'Liverpool',
    'Chelsea', 'Tottenham Hotspur', 'Barcelona', 'Atlético Madrid', 'Sevilla',
    'Milan', 'Inter Milan', 'Napoli', 'Juventus', 'Bayern Munich',
    'Borussia Dortmund', 'Bayer Leverkusen', 'RB Leipzig', 'Paris Saint-Germain',
    'Marseille', 'Porto', 'Sporting CP', 'Ajax', 'Club Brugge', 'Red Bull Salzburg',
    'Celtic', 'Shakhtar Donetsk', 'Trabzonspor', 'Copenhagen'
]

# Filter the DataFrame to include only clubs from the year 2022
df_2022 = df[df['Year'] == 2022].copy()

# Add a column to indicate Champions League participation
df_2022['ChampionsLeague'] = df_2022['Club'].apply(lambda x: 1 if x in champions_league_clubs else 0)

# Add the 'gvb' column based on the 'Overall' score
df_2022['GVB'] = df_2022['Overall'].apply(lambda x: 1 if x >= 68 else 0)

# Define the specific columns to keep in the new dataset
selected_columns = [
    'Club', 'IntReputation', 'Age', 'SkillMoves', 'Crossing',
    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
    'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'ShotPower',
    'Jumping', 'LongShots', 'Aggression', 'Interceptions', 'Positioning',
    'Vision', 'Penalties', 'Composure', 'StandingTackle', 'SlidingTackle',
    'ChampionsLeague', 'GVB'
]

# Display the table with selected columns
print(tabulate(df_2022[selected_columns], headers='keys', tablefmt='pretty'))

# Save the modified DataFrame to a new CSV file
output_file_path = 'data/teams-stats.csv'
df_2022[selected_columns].to_csv(output_file_path, index=False)
