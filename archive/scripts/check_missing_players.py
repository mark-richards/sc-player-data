import pandas as pd

# Load master data
df = pd.read_csv('data/processed/master_fanfooty_data.csv', low_memory=False)

# Check specific players
missing_players = ['Ash', 'Worrell', 'Himmelberg']

print("Checking missing players in master data:\n")
for surname in missing_players:
    player_data = df[df['Surname'].str.contains(surname, case=False, na=False)]

    if not player_data.empty:
        # Get player details
        first_name = player_data['First Name'].iloc[0]
        years = sorted(player_data['Year'].unique())
        max_year = years[-1]

        # Get latest round
        latest = player_data[player_data['Year'] == max_year].tail(1)
        latest_round = latest['Round'].iloc[0]

        print(f"{first_name} {surname}:")
        print(f"  Years played: {years[0]} - {max_year}")
        print(f"  Total games in data: {len(player_data)}")
        print(f"  Latest game: {max_year} {latest_round}")
        print(f"  Team: {latest['Team'].iloc[0]}")

        # Check if in 2025
        games_2025 = len(player_data[player_data['Year'] == 2025])
        print(f"  Games in 2025: {games_2025}")
        print()
    else:
        print(f"{surname}: NOT FOUND in master data\n")

# Check prediction criteria
print("\nPrediction criteria:")
print("- Players must have data in 2025 or 2026")
print("- Total players with 2025 data:", len(df[df['Year'] == 2025]['Player ID'].unique()))
print("- Total players with 2026 data:", len(df[df['Year'] == 2026]['Player ID'].unique()))
