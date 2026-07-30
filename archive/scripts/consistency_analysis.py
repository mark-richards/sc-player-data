import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("="*100)
print("CONSISTENCY ANALYSIS - Impact on Over/Under Performance")
print("="*100)

# Define consistency classification function
def classify_consistency(score):
    if score >= 80:
        return 'Elite (Very Consistent)'
    elif score >= 65:
        return 'Good (Reliable)'
    elif score >= 50:
        return 'Average (Moderate)'
    elif score >= 35:
        return 'Risky (Volatile)'
    else:
        return 'Very Risky (Boom/Bust)'

# First, analyze 2025 historical data to see if consistency predicted success
print("\n" + "="*100)
print("PHASE 1: HISTORICAL ANALYSIS (2025 Draft)")
print("="*100)

df_2025 = pd.read_csv('data/predictions/2025_draft_analysis.csv')

# Calculate consistency score (inverse of std dev - higher = more consistent)
# Normalize to 0-100 scale
if 'consistency_last_10' in df_2025.columns:
    max_std = df_2025['consistency_last_10'].max()
    min_std = df_2025['consistency_last_10'].min()

    # Inverse: low std = high consistency
    df_2025['Consistency_Score'] = df_2025['consistency_last_10'].apply(
        lambda x: 100 - ((x - min_std) / (max_std - min_std) * 100) if pd.notna(x) else 50
    ).round(1)

    df_2025['Consistency_Class'] = df_2025['Consistency_Score'].apply(classify_consistency)

    print(f"\nConsistency data available for {df_2025['consistency_last_10'].notna().sum()} players")
    print(f"Consistency std dev range: {min_std:.1f} - {max_std:.1f}")

    # Analyze consistency vs performance
    print("\n" + "="*100)
    print("CONSISTENCY vs PERFORMANCE IMPROVEMENT")
    print("="*100)

    consistency_tiers = pd.qcut(df_2025['Consistency_Score'],
                                  q=4,
                                  labels=['Q1 (Volatile)', 'Q2 (Below Avg)', 'Q3 (Above Avg)', 'Q4 (Consistent)'],
                                  duplicates='drop')

    df_2025['Consistency_Tier'] = consistency_tiers

    tier_performance = df_2025.groupby('Consistency_Tier').agg({
        'AVG_improvement': ['mean', 'median', 'std', 'count'],
        'Rank_improvement': ['mean', 'median'],
        'positional_success': ['mean', 'median'],
        'AVG_prediction_error': ['mean', 'median']
    }).round(2)

    print("\nPerformance by Consistency Tier:")
    print(tier_performance)

    # Statistical test: Does consistency predict improvement?
    consistent_players = df_2025[df_2025['Consistency_Score'] >= 65]
    volatile_players = df_2025[df_2025['Consistency_Score'] < 50]

    if len(consistent_players) > 0 and len(volatile_players) > 0:
        # AVG improvement
        consist_improvement = consistent_players['AVG_improvement'].dropna()
        volatile_improvement = volatile_players['AVG_improvement'].dropna()

        if len(consist_improvement) > 1 and len(volatile_improvement) > 1:
            t_stat, p_value = stats.ttest_ind(consist_improvement, volatile_improvement)

            print("\n" + "-"*100)
            print("STATISTICAL TEST: Consistent vs Volatile Players")
            print("-"*100)
            print(f"\nConsistent players (score 65+): {len(consistent_players)}")
            print(f"  Avg improvement: {consist_improvement.mean():.2f}")
            print(f"  Avg prediction error: {consistent_players['AVG_prediction_error'].mean():.2f}")
            print(f"\nVolatile players (score <50): {len(volatile_players)}")
            print(f"  Avg improvement: {volatile_improvement.mean():.2f}")
            print(f"  Avg prediction error: {volatile_players['AVG_prediction_error'].mean():.2f}")
            print(f"\nDifference: {consist_improvement.mean() - volatile_improvement.mean():.2f} points")
            print(f"P-value: {p_value:.4f} {'(STATISTICALLY SIGNIFICANT)' if p_value < 0.05 else '(NOT significant)'}")

        # Prediction accuracy
        print("\n" + "-"*100)
        print("PREDICTION ACCURACY")
        print("-"*100)

        print(f"\nConsistent players:")
        print(f"  Avg prediction error: {consistent_players['AVG_prediction_error'].mean():.2f}")
        print(f"  Prediction easier: {'YES' if consistent_players['AVG_prediction_error'].mean() < df_2025['AVG_prediction_error'].mean() else 'NO'}")

        print(f"\nVolatile players:")
        print(f"  Avg prediction error: {volatile_players['AVG_prediction_error'].mean():.2f}")
        print(f"  Prediction harder: {'YES' if volatile_players['AVG_prediction_error'].mean() > df_2025['AVG_prediction_error'].mean() else 'NO'}")

    # Over/under performers by consistency
    print("\n" + "="*100)
    print("OVER/UNDER PERFORMANCE by CONSISTENCY")
    print("="*100)

    # Overperformers (positive_success > 10)
    overperformers = df_2025[df_2025['positional_success'] > 10]
    underperformers = df_2025[df_2025['positional_success'] < -10]

    if len(overperformers) > 0:
        print(f"\nOverperformers (>10 position improvement): {len(overperformers)}")
        print(f"  Average consistency score: {overperformers['Consistency_Score'].mean():.1f}")

        over_consistency_dist = overperformers['Consistency_Class'].value_counts()
        print(f"  Consistency breakdown:")
        for category, count in over_consistency_dist.items():
            pct = count / len(overperformers) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")

    if len(underperformers) > 0:
        print(f"\nUnderperformers (<-10 position decline): {len(underperformers)}")
        print(f"  Average consistency score: {underperformers['Consistency_Score'].mean():.1f}")

        under_consistency_dist = underperformers['Consistency_Class'].value_counts()
        print(f"  Consistency breakdown:")
        for category, count in under_consistency_dist.items():
            pct = count / len(underperformers) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")

# Now calculate consistency for 2026 players
print("\n" + "="*100)
print("PHASE 2: 2026 PLAYER CONSISTENCY RATINGS")
print("="*100)

# Load master data to calculate 2025 consistency
master_df = pd.read_csv('data/processed/master_fanfooty_data.csv', low_memory=False)

# Calculate std dev for each player in 2025
player_2025_consistency = master_df[master_df['Year'] == 2025].groupby('Player ID').agg({
    'SC': ['mean', 'std', 'count'],
    'First Name': 'first',
    'Surname': 'first'
}).reset_index()

player_2025_consistency.columns = ['Player ID', 'AVG_2025', 'StdDev_2025', 'Games_2025', 'First Name', 'Surname']

# Calculate consistency score
max_std_2025 = player_2025_consistency['StdDev_2025'].max()
min_std_2025 = player_2025_consistency['StdDev_2025'].min()

player_2025_consistency['Consistency_Score'] = player_2025_consistency['StdDev_2025'].apply(
    lambda x: 100 - ((x - min_std_2025) / (max_std_2025 - min_std_2025) * 100) if pd.notna(x) else 50
).round(1)

player_2025_consistency['Consistency_Class'] = player_2025_consistency['Consistency_Score'].apply(classify_consistency)

# Coefficient of Variation (CV) = std/mean - more intuitive measure
player_2025_consistency['CV'] = (player_2025_consistency['StdDev_2025'] / player_2025_consistency['AVG_2025'] * 100).round(1)

print(f"\nCalculated consistency for {len(player_2025_consistency)} players from 2025 data")

# Load draft board and merge consistency
draft_board = pd.read_csv('data/predictions/2026_DRAFT_BOARD_Moneyball.csv')

draft_board = draft_board.merge(
    player_2025_consistency[['Player ID', 'StdDev_2025', 'Consistency_Score', 'Consistency_Class', 'CV']],
    on='Player ID',
    how='left'
)

# Fill missing with neutral values
draft_board['Consistency_Score'] = draft_board['Consistency_Score'].fillna(50)
draft_board['Consistency_Class'] = draft_board['Consistency_Class'].fillna('Average (Moderate)')
draft_board['CV'] = draft_board['CV'].fillna(draft_board['CV'].median())

print(f"\nMerged consistency data with draft board")

# Distribution
print("\n" + "="*100)
print("2026 CONSISTENCY DISTRIBUTION")
print("="*100)

consistency_dist = draft_board['Consistency_Class'].value_counts()
print("\nConsistency breakdown:")
for category, count in consistency_dist.items():
    pct = count / len(draft_board) * 100
    print(f"  {category}: {count} ({pct:.1f}%)")

print(f"\nAverage Consistency Score: {draft_board['Consistency_Score'].mean():.1f}")
print(f"Average Coefficient of Variation: {draft_board['CV'].mean():.1f}%")

# Top consistent players
print("\n" + "="*100)
print("TOP 20 MOST CONSISTENT PLAYERS (Elite Floor)")
print("="*100)

top_consistent = draft_board.nlargest(20, 'Consistency_Score')[
    ['Draft_Board_Rank', 'First Name', 'Surname', 'Position',
     'Avg_Projected_Score', 'StdDev_2025', 'Consistency_Score', 'CV', 'before_games_2025']
]

print("\n" + top_consistent.to_string(index=False))

# Most volatile players (boom/bust)
print("\n" + "="*100)
print("TOP 20 MOST VOLATILE PLAYERS (Boom/Bust)")
print("="*100)

most_volatile = draft_board.nsmallest(20, 'Consistency_Score')[
    ['Draft_Board_Rank', 'First Name', 'Surname', 'Position',
     'Avg_Projected_Score', 'StdDev_2025', 'Consistency_Score', 'CV', 'before_games_2025']
]

print("\n" + most_volatile.to_string(index=False))

# Create combined "Safety Score"
# Combines durability + consistency
print("\n" + "="*100)
print("SAFETY SCORE (Durability + Consistency)")
print("="*100)

# Load durability scores
durability_file = Path('data/predictions/2026_DURABILITY_WEIGHTED_RANKINGS.csv')
if durability_file.exists():
    durability_df = pd.read_csv(durability_file)

    draft_board = draft_board.merge(
        durability_df[['Player ID', 'Durability_Score']],
        on='Player ID',
        how='left',
        suffixes=('', '_dup')
    )

    # Safety Score = 50% Durability + 50% Consistency
    draft_board['Safety_Score'] = (
        draft_board['Durability_Score'].fillna(50) * 0.5 +
        draft_board['Consistency_Score'] * 0.5
    ).round(1)

    # Safety Class
    def classify_safety(score):
        if score >= 80:
            return 'Elite (Safe Pick)'
        elif score >= 65:
            return 'Good (Reliable)'
        elif score >= 50:
            return 'Average (Moderate Risk)'
        else:
            return 'High Risk (Volatile)'

    draft_board['Safety_Class'] = draft_board['Safety_Score'].apply(classify_safety)

    print("\nSafety Score = 50% Durability + 50% Consistency")
    print(f"Captures both availability AND predictability")

    # Top safe picks
    print("\n" + "="*100)
    print("SAFEST PICKS (Elite Durability + Elite Consistency)")
    print("="*100)

    safe_picks = draft_board[draft_board['Safety_Score'] >= 75].sort_values('Safety_Score', ascending=False)

    safe_display = [
        'Draft_Board_Rank', 'First Name', 'Surname', 'Position',
        'Avg_Projected_Score', 'Durability_Score', 'Consistency_Score',
        'Safety_Score', 'Safety_Class'
    ]

    print(f"\nFound {len(safe_picks)} players with Safety Score >= 75:")
    print("\n" + safe_picks[safe_display].to_string(index=False))

    # High risk picks
    print("\n" + "="*100)
    print("HIGHEST RISK PICKS (Low Durability OR Low Consistency)")
    print("="*100)

    risk_picks = draft_board[draft_board['Safety_Score'] < 50].sort_values('Avg_Projected_Score', ascending=False).head(20)

    print(f"\nTop 20 highest-risk players by projection:")
    print("\n" + risk_picks[safe_display].to_string(index=False))

# Save comprehensive file
output_file = Path('data/predictions/2026_CONSISTENCY_ANALYSIS.csv')
output_cols = [
    'Draft_Board_Rank', 'Player ID', 'First Name', 'Surname', 'Team', 'Position',
    'Avg_Projected_Score', 'before_games_2025',
    'StdDev_2025', 'CV', 'Consistency_Score', 'Consistency_Class',
    'Durability_Score', 'Safety_Score', 'Safety_Class',
    'Byes_R1-6', 'TPOR_Position_Rank'
]

cols_to_save = [col for col in output_cols if col in draft_board.columns]
draft_board[cols_to_save].to_csv(output_file, index=False)

print(f"\n\nComplete consistency analysis saved to: {output_file}")

# Key insights summary
print("\n" + "="*100)
print("KEY INSIGHTS - CONSISTENCY")
print("="*100)

if 'consistency_last_10' in df_2025.columns:
    print("\n1. DOES CONSISTENCY PREDICT SUCCESS?")
    print(f"   - Consistent players had {consistent_players['AVG_prediction_error'].mean():.1f} avg error")
    print(f"   - Volatile players had {volatile_players['AVG_prediction_error'].mean():.1f} avg error")
    print(f"   - Consistent = more predictable (easier to forecast)")

print("\n2. CONSISTENCY vs UPSIDE TRADE-OFF")
print("   - Elite consistent players: Safe floor, limited ceiling")
print("   - Volatile players: Boom/bust, can win or lose weeks")
print("   - Strategy: Balance your team with both")

print("\n3. SAFETY SCORE (Durability + Consistency)")
print(f"   - {len(safe_picks)} players with Safety Score 75+")
print("   - These are your 'sleep well at night' picks")
print("   - Target in early-mid rounds for stable core")

print("\n4. WHEN TO TARGET VOLATILE PLAYERS?")
print("   - Late rounds (picks 100+): High upside worth risk")
print("   - Need to catch up: Volatile players can make up ground")
print("   - Strong team core: Can afford 1-2 boom/bust picks")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
