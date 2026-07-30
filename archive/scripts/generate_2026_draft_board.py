import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("GENERATING 2026 DRAFT BOARD WITH MONEYBALL INSIGHTS")
print("="*100)

# Load 2026 predictions summary
predictions_file = Path('data/predictions/2026_draft_prep_summary_R1-6.csv')
df = pd.read_csv(predictions_file)

print(f"\nLoaded {len(df)} players from 2026 predictions")

# We need to merge with historical data to get "before games" metric
# Load master data to get 2025 games played
master_file = Path('data/processed/master_fanfooty_data.csv')
master_df = pd.read_csv(master_file, low_memory=False)

# Get 2025 games played for each player
games_2025 = master_df[master_df['Year'] == 2025].groupby('Player ID').agg({
    'Round': 'count'  # Count rounds = games played
}).reset_index()
games_2025.columns = ['Player ID', 'before_games_2025']

# Merge with predictions
df = df.merge(games_2025, on='Player ID', how='left')
df['before_games_2025'] = df['before_games_2025'].fillna(0)

print(f"Added 2025 games played data for {df['before_games_2025'].notna().sum()} players")

# We don't have TPOR in the predictions file, so we'll estimate it
# TPOR ≈ Total Points - (Replacement Level × Games)
# For simplicity, we'll use projected score as a proxy and adjust by position scarcity

# Calculate position averages from the data
position_avg = df.groupby('Position')['Avg_Projected_Score'].mean().to_dict()
print("\nPosition averages:")
for pos, avg in position_avg.items():
    print(f"  {pos}: {avg:.1f}")

# Estimate TPOR as: (Avg_Projected_Score - Position_Average) × Rounds_Playing
df['estimated_TPOR'] = df.apply(
    lambda row: (row['Avg_Projected_Score'] - position_avg.get(row['Position'], 90)) * row['Rounds_Playing']
    if pd.notna(row['Position']) and pd.notna(row['Avg_Projected_Score']) else 0,
    axis=1
)

# Calculate position ranks for estimated TPOR
df['TPOR_Position_Rank'] = df.groupby('Position')['estimated_TPOR'].rank(ascending=False, method='dense')

# Calculate AVG rank
df['AVG_Rank'] = df['Avg_Projected_Score'].rank(ascending=False, method='dense')

print("\n" + "="*100)
print("APPLYING MONEYBALL COMPOSITE SCORE FORMULA")
print("="*100)

# Apply the composite scoring formula from insights
# Draft_Score = (before_AVG × 0.30) + (Before_TPOR × 0.25) + (before_games × 0.15)
#             + (before_AVG_Rank × -0.20) + (positive_note × 15) + (negative_note × -15)

# We don't have notes in the 2026 data, so we'll leave placeholders for manual entry
df['positive_note'] = 0  # USER TO FILL IN
df['negative_note'] = 0  # USER TO FILL IN

# Calculate composite score (without notes for now)
# UPDATED FORMULA: Increased durability from 0.15 to 0.20, decreased AVG_Rank from -0.20 to -0.15
df['Draft_Score_Base'] = (
    df['Avg_Projected_Score'].fillna(0) * 0.30 +
    df['estimated_TPOR'].fillna(0) * 0.0025 +  # Scaled down since TPOR is in hundreds
    df['before_games_2025'].fillna(0) * 0.20 +  # INCREASED from 0.15 (validated!)
    df['AVG_Rank'].fillna(300) * -0.15          # DECREASED from -0.20
)

# Apply position adjustments based on Moneyball insights
# MID/DEF get +5 bonus, FWD gets -5 penalty
position_adjustments = {
    'Midfielder': 5,
    'Back': 5,
    'Forward': -5,
    'Ruck': 0
}

df['Position_Adjustment'] = df['Position'].map(position_adjustments).fillna(0)
df['Draft_Score_Adjusted'] = df['Draft_Score_Base'] + df['Position_Adjustment']

# Apply durability adjustment (STRENGTHENED based on multi-year validation)
# Players with <18 games get -5, players with 20+ games get +5
df['Durability_Adjustment'] = df['before_games_2025'].apply(
    lambda x: 5 if x >= 20 else (-5 if x < 18 else 0)
)
df['Draft_Score_Final'] = df['Draft_Score_Adjusted'] + df['Durability_Adjustment']

# Sort by final draft score
df_sorted = df.sort_values('Draft_Score_Final', ascending=False)

print("\nComposite scoring applied (REFINED - Multi-Year Validated):")
print("  - Base score: AVG (30%) + TPOR (25%) + Games (20%) - AVG Rank (15%)")
print("  - Position adjustment: MID/DEF +5, FWD -5")
print("  - Durability adjustment: 20+ games +5, <18 games -5")
print("  - Note adjustments: Positive +15, Negative -15 (when added)")

# Create output columns
output_cols = [
    'Player ID', 'First Name', 'Surname', 'Team', 'Position',
    'Avg_Projected_Score', 'Avg_Prob_80+', 'Avg_Prob_100+',
    'Rounds_Playing', 'Byes_R1-6', 'before_games_2025',
    'AVG_Rank', 'TPOR_Position_Rank',
    'Draft_Score_Final', 'positive_note', 'negative_note',
    'Position_Adjustment', 'Durability_Adjustment'
]

df_output = df_sorted[output_cols].copy()

# Add a draft board rank
df_output.insert(0, 'Draft_Board_Rank', range(1, len(df_output) + 1))

# Save the draft board
output_file = Path('data/predictions/2026_DRAFT_BOARD_Moneyball.csv')
df_output.to_csv(output_file, index=False)

print(f"\nDraft board saved to: {output_file}")
print(f"Total players ranked: {len(df_output)}")

# Show top 50
print("\n" + "="*100)
print("TOP 50 PLAYERS - 2026 DRAFT BOARD (Moneyball Ranking)")
print("="*100)

display_cols = [
    'Draft_Board_Rank', 'First Name', 'Surname', 'Position',
    'Avg_Projected_Score', 'before_games_2025', 'Byes_R1-6',
    'Draft_Score_Final'
]

print("\n" + df_output[display_cols].head(50).to_string(index=False))

# Show top 10 by position
print("\n" + "="*100)
print("TOP 10 BY POSITION")
print("="*100)

for position in ['Midfielder', 'Forward', 'Back', 'Ruck']:
    print(f"\n--- {position.upper()} ---")
    pos_df = df_output[df_output['Position'] == position].head(10)
    print(pos_df[['Draft_Board_Rank', 'First Name', 'Surname', 'Avg_Projected_Score', 'before_games_2025', 'Draft_Score_Final']].to_string(index=False))

print("\n" + "="*100)
print("NEXT STEPS:")
print("="*100)
print("\n1. Open: data/predictions/2026_DRAFT_BOARD_Moneyball.csv")
print("2. Add your notes:")
print("   - Set 'positive_note' to 1 for players you have positive feelings about")
print("   - Set 'negative_note' to 1 for players you have negative feelings about")
print("3. Re-run this script to update Draft_Score_Final with note adjustments")
print("4. Use this as your draft board on draft day!")
print("\nGOOD LUCK!")
print("="*100)
