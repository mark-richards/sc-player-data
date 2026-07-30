import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("2026 LATE-ROUND GEM IDENTIFICATION")
print("Based on Multi-Year Validation (2022-2024)")
print("="*100)

# Load the updated draft board
draft_board = pd.read_csv('data/predictions/2026_DRAFT_BOARD_Moneyball.csv')

print(f"\nLoaded {len(draft_board)} players from draft board")

# Define "late round" as picks 50-150
# Based on validation, ~15 players per year drafted 50+ finish in top 50
late_round_candidates = draft_board[
    (draft_board['Draft_Board_Rank'] >= 50) &
    (draft_board['Draft_Board_Rank'] <= 150)
].copy()

print(f"Analyzing {len(late_round_candidates)} players in picks 50-150")

# Gem criteria based on multi-year validation:
# 1. High durability (20+ games in 2025)
# 2. Strong projected score (top tier for their draft range)
# 3. Position: MID or DEF preferred (57.9% and 36.8% of overperformers)
# 4. Strong TPOR rank (positional value)

print("\n" + "="*100)
print("GEM CRITERIA (Based on 2022-2024 Historical Gems)")
print("="*100)

criteria_results = {
    'high_durability': late_round_candidates['before_games_2025'] >= 20,
    'strong_projection': late_round_candidates['Avg_Projected_Score'] >= late_round_candidates['Avg_Projected_Score'].quantile(0.75),
    'preferred_position': late_round_candidates['Position'].isin(['Midfielder', 'Back']),
    'top_tpor_rank': late_round_candidates['TPOR_Position_Rank'] <= 30
}

# Create gem score (0-4 based on how many criteria met)
late_round_candidates['gem_score'] = sum(criteria_results.values())

print("\nCriteria breakdown:")
print(f"1. High Durability (20+ games in 2025): {criteria_results['high_durability'].sum()} players")
print(f"2. Strong Projection (top 25% of late-round): {criteria_results['strong_projection'].sum()} players")
print(f"3. Preferred Position (MID/DEF): {criteria_results['preferred_position'].sum()} players")
print(f"4. Top TPOR Rank (top 30 in position): {criteria_results['top_tpor_rank'].sum()} players")

# Identify top gems (meet 3+ criteria)
top_gems = late_round_candidates[late_round_candidates['gem_score'] >= 3].copy()
top_gems = top_gems.sort_values('Draft_Score_Final', ascending=False)

print("\n" + "="*100)
print(f"IDENTIFIED {len(top_gems)} HIGH-PROBABILITY LATE-ROUND GEMS (3+ criteria)")
print("="*100)

display_cols = [
    'Draft_Board_Rank', 'First Name', 'Surname', 'Position',
    'Avg_Projected_Score', 'before_games_2025', 'TPOR_Position_Rank',
    'Byes_R1-6', 'gem_score', 'Draft_Score_Final'
]

print("\nAll High-Probability Gems:")
print(top_gems[display_cols].to_string(index=False))

# Also identify "sleepers" (2 criteria, high upside)
sleepers = late_round_candidates[late_round_candidates['gem_score'] == 2].copy()
sleepers = sleepers.sort_values('Draft_Score_Final', ascending=False).head(15)

print("\n" + "="*100)
print(f"SLEEPER CANDIDATES (2 criteria, high upside) - Top 15")
print("="*100)
print(sleepers[display_cols].to_string(index=False))

# Position-specific gems
print("\n" + "="*100)
print("GEM CANDIDATES BY POSITION")
print("="*100)

for position in ['Midfielder', 'Back', 'Forward', 'Ruck']:
    pos_gems = late_round_candidates[
        (late_round_candidates['Position'] == position) &
        (late_round_candidates['gem_score'] >= 2)
    ].sort_values('Draft_Score_Final', ascending=False).head(5)

    if len(pos_gems) > 0:
        print(f"\n--- {position.upper()} ---")
        print(pos_gems[['Draft_Board_Rank', 'First Name', 'Surname', 'Avg_Projected_Score', 'before_games_2025', 'gem_score']].to_string(index=False))

# Historical comparison
print("\n" + "="*100)
print("HISTORICAL CONTEXT")
print("="*100)

print("\nHistorical late-round gems (2022-2024):")
print("2022: Rory Laird (pick 79 -> #2), Jack Sinclair (91 -> #8), George Hewett (66 -> #11)")
print("2023: Caleb Serong (69 -> #17), Luke Ryan (68 -> #20), Adam Treloar (98 -> #23)")
print("2024: Adam Treloar (67 -> #9), Dayne Zorko (103 -> #18), Patrick Cripps (66 -> #19)")

print("\nCommon traits of historical gems:")
print("- Average 20.4 games played in previous year")
print("- Strong positional value (TPOR)")
print("- 68% were MID/DEF")
print("- Often undervalued due to age, injury history, or team perception")

# Target recommendations
print("\n" + "="*100)
print("DRAFT DAY TARGETING STRATEGY")
print("="*100)

print("\nRounds 5-7 (picks 41-70):")
top_early_gems = top_gems[top_gems['Draft_Board_Rank'] <= 70].head(5)
if len(top_early_gems) > 0:
    print("TARGET THESE PLAYERS:")
    for _, player in top_early_gems.iterrows():
        print(f"  - {player['First Name']} {player['Surname']} ({player['Position']}) - Rank {player['Draft_Board_Rank']:.0f}, {player['before_games_2025']:.0f} games, Score {player['Avg_Projected_Score']:.1f}")

print("\nRounds 8-10 (picks 71-100):")
top_mid_gems = top_gems[(top_gems['Draft_Board_Rank'] > 70) & (top_gems['Draft_Board_Rank'] <= 100)].head(5)
if len(top_mid_gems) > 0:
    print("TARGET THESE PLAYERS:")
    for _, player in top_mid_gems.iterrows():
        print(f"  - {player['First Name']} {player['Surname']} ({player['Position']}) - Rank {player['Draft_Board_Rank']:.0f}, {player['before_games_2025']:.0f} games, Score {player['Avg_Projected_Score']:.1f}")

print("\nRounds 11-15 (picks 101-150):")
top_late_gems = top_gems[top_gems['Draft_Board_Rank'] > 100].head(5)
if len(top_late_gems) > 0:
    print("DEEP SLEEPERS:")
    for _, player in top_late_gems.iterrows():
        print(f"  - {player['First Name']} {player['Surname']} ({player['Position']}) - Rank {player['Draft_Board_Rank']:.0f}, {player['before_games_2025']:.0f} games, Score {player['Avg_Projected_Score']:.1f}")

# Save gems list
output_file = Path('data/predictions/2026_LATE_ROUND_GEMS.csv')
all_gems = late_round_candidates[late_round_candidates['gem_score'] >= 2].sort_values('Draft_Score_Final', ascending=False)
all_gems.to_csv(output_file, index=False)

print(f"\n\nComplete gems list saved to: {output_file}")
print(f"Total gem candidates: {len(all_gems)}")
print("="*100)
