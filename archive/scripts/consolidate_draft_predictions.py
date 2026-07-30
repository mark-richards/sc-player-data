import pandas as pd
import numpy as np
from pathlib import Path

# Load all 6 rounds of predictions
print("Loading predictions for rounds 1-6...")
rounds = []
for round_num in range(1, 7):
    file_path = Path(f'data/predictions/2026_round_{round_num}_predictions.csv')
    df = pd.read_csv(file_path)
    df['Round'] = round_num
    rounds.append(df)

# Combine all rounds
all_rounds = pd.concat(rounds, ignore_index=True)
print(f"Loaded {len(all_rounds)} player-round combinations ({all_rounds['Player ID'].nunique()} unique players)")

# Create summary statistics for each player
summary = all_rounds.groupby(['Player ID', 'First Name', 'Surname', 'Team', 'Position']).agg({
    'projected_score': ['mean', 'std', 'min', 'max'],
    'probability_gt_80': 'mean',
    'probability_gt_100': 'mean',
    'Round': 'count'  # Number of rounds (should be 6, or less if they have byes)
}).reset_index()

# Flatten column names
summary.columns = ['Player ID', 'First Name', 'Surname', 'Team', 'Position',
                   'Avg_Projected_Score', 'StdDev_Score', 'Min_Score', 'Max_Score',
                   'Avg_Prob_80+', 'Avg_Prob_100+', 'Rounds_Playing']

# Calculate number of byes in first 6 rounds
bye_counts = all_rounds[all_rounds['Opponent'] == 'BYE'].groupby('Player ID').size().reset_index(name='Byes_R1-6')
summary = summary.merge(bye_counts, on='Player ID', how='left')
summary['Byes_R1-6'] = summary['Byes_R1-6'].fillna(0).astype(int)

# Round the numeric columns for cleaner display
summary['Avg_Projected_Score'] = summary['Avg_Projected_Score'].round(1)
summary['StdDev_Score'] = summary['StdDev_Score'].round(1)
summary['Min_Score'] = summary['Min_Score'].round(1)
summary['Max_Score'] = summary['Max_Score'].round(1)
summary['Avg_Prob_80+'] = summary['Avg_Prob_80+'].round(1)
summary['Avg_Prob_100+'] = summary['Avg_Prob_100+'].round(1)

# Sort by average projected score (descending)
summary = summary.sort_values('Avg_Projected_Score', ascending=False)

# Save to CSV
output_path = Path('data/predictions/2026_draft_prep_summary_R1-6.csv')
summary.to_csv(output_path, index=False)
print(f"\nDraft prep summary saved to: {output_path}")

# Display top 50 players
print("\n" + "="*100)
print("TOP 50 PLAYERS FOR DRAFT (Based on Average Projected Score - Rounds 1-6)")
print("="*100)
print(summary.head(50).to_string(index=False))

# Display positional breakdowns
print("\n" + "="*100)
print("TOP 10 BY POSITION")
print("="*100)

for position in ['Midfielder', 'Forward', 'Back', 'Ruck']:
    print(f"\n--- {position.upper()} ---")
    pos_summary = summary[summary['Position'] == position].head(10)
    print(pos_summary[['First Name', 'Surname', 'Team', 'Avg_Projected_Score', 'Avg_Prob_80+', 'Byes_R1-6']].to_string(index=False))
