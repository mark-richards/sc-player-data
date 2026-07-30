import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("COMPREHENSIVE PLAYER LIST - DURABILITY & OUTPUT WEIGHTED RANKINGS")
print("="*100)

# Load the draft board
draft_board = pd.read_csv('data/predictions/2026_DRAFT_BOARD_Moneyball.csv')

print(f"\nLoaded {len(draft_board)} players")

# Calculate Durability Score (0-100 scale)
# Based on 2025 games played
def calculate_durability_score(games):
    """
    Durability Score (0-100):
    - 0-10 games: 0-40 (High Risk)
    - 11-17 games: 41-69 (Moderate Risk)
    - 18-19 games: 70-84 (Good)
    - 20-22 games: 85-94 (Very Good)
    - 23+ games: 95-100 (Excellent)
    """
    if pd.isna(games) or games == 0:
        return 0
    elif games <= 10:
        return int(games * 4)  # 0-40
    elif games <= 17:
        return int(40 + (games - 10) * 4)  # 41-68
    elif games <= 19:
        return int(70 + (games - 18) * 7)  # 70-84
    elif games <= 22:
        return int(85 + (games - 20) * 3)  # 85-93
    else:
        return min(100, int(95 + (games - 23) * 2))  # 95-100

draft_board['Durability_Score'] = draft_board['before_games_2025'].apply(calculate_durability_score)

# Classify durability risk
def classify_durability(score):
    if score >= 95:
        return 'Elite (23+ games)'
    elif score >= 85:
        return 'Very Good (20-22)'
    elif score >= 70:
        return 'Good (18-19)'
    elif score >= 41:
        return 'Moderate (11-17)'
    else:
        return 'High Risk (0-10)'

draft_board['Durability_Class'] = draft_board['Durability_Score'].apply(classify_durability)

# Calculate Output Score (0-100 scale based on projected AVG)
# Top player (137.6) = 100, scaling down from there
max_avg = draft_board['Avg_Projected_Score'].max()
min_avg = draft_board['Avg_Projected_Score'].min()

draft_board['Output_Score'] = draft_board['Avg_Projected_Score'].apply(
    lambda x: int(((x - min_avg) / (max_avg - min_avg)) * 100) if pd.notna(x) else 0
)

# Calculate Weighted Composite Score
# 60% Output + 40% Durability (based on multi-year validation showing durability matters)
draft_board['Weighted_Score'] = (
    draft_board['Output_Score'] * 0.60 +
    draft_board['Durability_Score'] * 0.40
).round(1)

# Create Overall Weighted Rank
draft_board['Weighted_Rank'] = draft_board['Weighted_Score'].rank(ascending=False, method='min').astype(int)

# Sort by weighted rank
df_sorted = draft_board.sort_values('Weighted_Rank')

print("\n" + "="*100)
print("SCORING METHODOLOGY")
print("="*100)

print("\nDurability Score (0-100):")
print("  95-100: Elite (23+ games in 2025)")
print("  85-94:  Very Good (20-22 games)")
print("  70-84:  Good (18-19 games)")
print("  41-69:  Moderate Risk (11-17 games)")
print("  0-40:   High Risk (0-10 games)")

print("\nOutput Score (0-100):")
print("  Based on projected average (scaled from min to max)")
print(f"  100 = {max_avg:.1f} avg (top projection)")
print(f"  0 = {min_avg:.1f} avg (lowest projection)")

print("\nWeighted Score:")
print("  60% Output + 40% Durability")
print("  Reflects multi-year finding: durability = 18 pt advantage")

# Summary statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

durability_dist = draft_board['Durability_Class'].value_counts().sort_index()
print("\nDurability Distribution:")
for category, count in durability_dist.items():
    pct = count / len(draft_board) * 100
    print(f"  {category}: {count} players ({pct:.1f}%)")

print(f"\nAverage Durability Score: {draft_board['Durability_Score'].mean():.1f}")
print(f"Average Output Score: {draft_board['Output_Score'].mean():.1f}")
print(f"Average Weighted Score: {draft_board['Weighted_Score'].mean():.1f}")

# Top 100 players
print("\n" + "="*100)
print("TOP 100 PLAYERS - WEIGHTED BY DURABILITY & OUTPUT")
print("="*100)

top_100 = df_sorted.head(100)

display_cols = [
    'Weighted_Rank', 'First Name', 'Surname', 'Position',
    'Avg_Projected_Score', 'before_games_2025',
    'Output_Score', 'Durability_Score', 'Weighted_Score',
    'Durability_Class', 'Byes_R1-6'
]

print("\n" + top_100[display_cols].to_string(index=False))

# Position-specific rankings
print("\n" + "="*100)
print("TOP 20 BY POSITION (Weighted Rankings)")
print("="*100)

for position in ['Midfielder', 'Back', 'Forward', 'Ruck']:
    print(f"\n--- {position.upper()} ---")
    pos_players = df_sorted[df_sorted['Position'] == position].head(20)

    pos_display = [
        'Weighted_Rank', 'First Name', 'Surname',
        'Avg_Projected_Score', 'before_games_2025',
        'Durability_Score', 'Weighted_Score'
    ]

    print(pos_players[pos_display].to_string(index=False))

# High Output + Low Durability (Risk Players)
print("\n" + "="*100)
print("HIGH RISK / HIGH REWARD PLAYERS")
print("="*100)
print("(High output score but low durability - injury concerns)")

risk_players = draft_board[
    (draft_board['Output_Score'] >= 60) &
    (draft_board['Durability_Score'] < 70)
].sort_values('Output_Score', ascending=False)

if len(risk_players) > 0:
    print(f"\nFound {len(risk_players)} high-risk players:")
    risk_display = [
        'Weighted_Rank', 'First Name', 'Surname', 'Position',
        'Avg_Projected_Score', 'before_games_2025',
        'Output_Score', 'Durability_Score', 'Durability_Class'
    ]
    print("\n" + risk_players[risk_display].head(20).to_string(index=False))
else:
    print("\nNo high-risk players found in top tier")

# Low Output + High Durability (Safe Floor Players)
print("\n" + "="*100)
print("SAFE FLOOR PLAYERS")
print("="*100)
print("(High durability but modest output - consistent but not elite)")

safe_players = draft_board[
    (draft_board['Durability_Score'] >= 85) &
    (draft_board['Output_Score'] < 50)
].sort_values('Durability_Score', ascending=False)

if len(safe_players) > 0:
    print(f"\nFound {len(safe_players)} safe floor players:")
    safe_display = [
        'Weighted_Rank', 'First Name', 'Surname', 'Position',
        'Avg_Projected_Score', 'before_games_2025',
        'Output_Score', 'Durability_Score', 'Weighted_Score'
    ]
    print("\n" + safe_players[safe_display].head(20).to_string(index=False))
else:
    print("\nNo safe floor players found")

# Elite Both (Dream Picks)
print("\n" + "="*100)
print("ELITE DURABILITY + ELITE OUTPUT (DREAM PICKS)")
print("="*100)
print("(Top 25% in both output and durability)")

elite_both = draft_board[
    (draft_board['Output_Score'] >= draft_board['Output_Score'].quantile(0.75)) &
    (draft_board['Durability_Score'] >= draft_board['Durability_Score'].quantile(0.75))
].sort_values('Weighted_Score', ascending=False)

print(f"\nFound {len(elite_both)} elite players (top 25% in both metrics):")
elite_display = [
    'Weighted_Rank', 'First Name', 'Surname', 'Position',
    'Avg_Projected_Score', 'before_games_2025',
    'Output_Score', 'Durability_Score', 'Weighted_Score', 'Byes_R1-6'
]
print("\n" + elite_both[elite_display].to_string(index=False))

# Save full list
output_file = Path('data/predictions/2026_DURABILITY_WEIGHTED_RANKINGS.csv')
output_cols = [
    'Weighted_Rank', 'Draft_Board_Rank', 'Player ID', 'First Name', 'Surname',
    'Team', 'Position', 'Avg_Projected_Score', 'before_games_2025',
    'Output_Score', 'Durability_Score', 'Weighted_Score', 'Durability_Class',
    'Byes_R1-6', 'TPOR_Position_Rank', 'Avg_Prob_80+', 'Avg_Prob_100+'
]

df_sorted[output_cols].to_csv(output_file, index=False)

print(f"\n\nComplete durability-weighted rankings saved to: {output_file}")
print(f"Total players ranked: {len(df_sorted)}")

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

print("\n1. BIGGEST MOVERS (Weighted Rank vs Draft Board Rank):")
df_sorted['Rank_Change'] = df_sorted['Draft_Board_Rank'] - df_sorted['Weighted_Rank']

biggest_risers = df_sorted.nlargest(10, 'Rank_Change')
print("\nTop 10 Risers (durability boosted them):")
for _, player in biggest_risers.iterrows():
    print(f"  {player['First Name']} {player['Surname']} ({player['Position']}): "
          f"Rank {player['Draft_Board_Rank']:.0f} -> {player['Weighted_Rank']} "
          f"(+{player['Rank_Change']:.0f}, {player['before_games_2025']:.0f} games)")

biggest_fallers = df_sorted.nsmallest(10, 'Rank_Change')
print("\nTop 10 Fallers (durability hurt them):")
for _, player in biggest_fallers.iterrows():
    print(f"  {player['First Name']} {player['Surname']} ({player['Position']}): "
          f"Rank {player['Draft_Board_Rank']:.0f} -> {player['Weighted_Rank']} "
          f"({player['Rank_Change']:.0f}, {player['before_games_2025']:.0f} games)")

print("\n2. POSITION AVERAGES:")
for position in ['Midfielder', 'Back', 'Forward', 'Ruck']:
    pos_avg = draft_board[draft_board['Position'] == position].agg({
        'Durability_Score': 'mean',
        'Output_Score': 'mean',
        'Weighted_Score': 'mean'
    })
    print(f"\n{position}:")
    print(f"  Avg Durability: {pos_avg['Durability_Score']:.1f}")
    print(f"  Avg Output: {pos_avg['Output_Score']:.1f}")
    print(f"  Avg Weighted: {pos_avg['Weighted_Score']:.1f}")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
