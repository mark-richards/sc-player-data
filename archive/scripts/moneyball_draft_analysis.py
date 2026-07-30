import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

# Moneyball-style analysis of historical draft performance
# Goal: Identify which pre-draft features best predict final success

excel_file = Path('draft_prep/SC 2026/Previous drafts combined - 2026 draft prep.xlsx')
print("="*100)
print("MONEYBALL DRAFT ANALYSIS - AFL SuperCoach 2025")
print("="*100)

# Read the 2025 sheet with row 1 as the header (row 0 is section headers)
df = pd.read_excel(excel_file, sheet_name='2025', header=1)

print(f"\nLoaded {len(df)} draft picks from 2025 season")
print(f"Columns: {len(df.columns)}")

# Filter to only rows with actual data (where 'Name' is not null)
df_clean = df[df['Name'].notna()].copy()
print(f"Filtered to {len(df_clean)} picks with complete data")

# Convert numeric columns to proper types
numeric_cols = [
    'pick', 'before games', 'before AVG', 'before AVG Rank', 'Before Total', 'Before TPOR',
    'Before TPOR Rank by position', 'Final Games', 'Final average', 'Final AVG Rank',
    'Final Total', 'Final TPOR', 'Final TPOR Rank by position', 'Pre-Draft position Rank',
    'Before TPOR Rank result', 'my rank result'
]

for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Create derived features for analysis
print("\n" + "="*100)
print("CREATING DERIVED FEATURES")
print("="*100)

# 1. Calculate improvement metrics
df_clean['AVG_improvement'] = df_clean['Final average'] - df_clean['before AVG']
df_clean['Rank_improvement'] = df_clean['before AVG Rank'] - df_clean['Final AVG Rank']  # Positive = improved ranking
df_clean['TPOR_improvement'] = df_clean['Final TPOR'] - df_clean['Before TPOR']
df_clean['TPOR_Rank_improvement'] = df_clean['Before TPOR Rank by position'] - df_clean['Final TPOR Rank by position']

# 2. Calculate consistency (how close predictions were to reality)
df_clean['AVG_prediction_error'] = abs(df_clean['before AVG'] - df_clean['Final average'])
df_clean['TPOR_prediction_error'] = abs(df_clean['Before TPOR'] - df_clean['Final TPOR'])

# 3. Encode notes
df_clean['has_positive_note'] = (df_clean['note type'] == 'positive').astype(int)
df_clean['has_negative_note'] = (df_clean['note type'] == 'negative').astype(int)
df_clean['has_any_note'] = df_clean['note type'].notna().astype(int)

# 4. Calculate a "draft success" metric (did they outperform or underperform?)
# Success = Final TPOR Rank better than Pre-Draft position Rank
df_clean['positional_success'] = df_clean['Pre-Draft position Rank'] - df_clean['Final TPOR Rank by position']

print("\nDerived features created:")
print("- AVG_improvement: Change in average score from pre-draft to end-of-season")
print("- Rank_improvement: Change in AVG ranking (positive = moved up)")
print("- TPOR_improvement: Change in Total Points Over Replacement")
print("- AVG_prediction_error: How far off the pre-draft AVG prediction was")
print("- positional_success: Did player finish better than their pre-draft positional rank?")
print("- has_positive_note: Binary indicator for positive notes")
print("- has_negative_note: Binary indicator for negative notes")

# Show sample of the analysis dataframe
print("\n" + "="*100)
print("SAMPLE DATA (First 10 picks)")
print("="*100)

display_cols = ['Name', 'pick', 'before AVG', 'Final average', 'AVG_improvement',
                'before AVG Rank', 'Final AVG Rank', 'Rank_improvement', 'note type', 'positional_success']
print(df_clean[display_cols].head(10).to_string(index=False))

# Calculate summary statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

print(f"\nTotal picks analyzed: {len(df_clean)}")
print(f"Picks with positive notes: {df_clean['has_positive_note'].sum()}")
print(f"Picks with negative notes: {df_clean['has_negative_note'].sum()}")
print(f"Picks with no notes: {len(df_clean) - df_clean['has_any_note'].sum()}")

print("\nAverage improvements:")
print(f"  AVG score: {df_clean['AVG_improvement'].mean():.2f} points")
print(f"  Rank position: {df_clean['Rank_improvement'].mean():.2f} places")
print(f"  TPOR: {df_clean['TPOR_improvement'].mean():.2f} points")
print(f"  Positional success: {df_clean['positional_success'].mean():.2f} ranks")

print("\nAverage prediction errors:")
print(f"  AVG prediction error: {df_clean['AVG_prediction_error'].mean():.2f} points")
print(f"  TPOR prediction error: {df_clean['TPOR_prediction_error'].mean():.2f} points")

# KEY INSIGHT 1: Note type performance
print("\n" + "="*100)
print("KEY INSIGHT #1: HOW DID NOTES PERFORM?")
print("="*100)

note_groups = df_clean.groupby('note type', dropna=False).agg({
    'AVG_improvement': ['mean', 'median', 'std', 'count'],
    'Rank_improvement': ['mean', 'median'],
    'TPOR_improvement': ['mean', 'median'],
    'positional_success': ['mean', 'median'],
    'AVG_prediction_error': ['mean', 'median']
}).round(2)

print("\nPerformance by note type:")
print(note_groups)

# Statistical test: Do positive notes actually perform better?
positive_avg_improvement = df_clean[df_clean['has_positive_note'] == 1]['AVG_improvement'].dropna()
negative_avg_improvement = df_clean[df_clean['has_negative_note'] == 1]['AVG_improvement'].dropna()
no_note_avg_improvement = df_clean[df_clean['has_any_note'] == 0]['AVG_improvement'].dropna()

print("\n" + "-"*100)
print("STATISTICAL TESTS")
print("-"*100)

if len(positive_avg_improvement) > 0 and len(negative_avg_improvement) > 0:
    t_stat, p_value = stats.ttest_ind(positive_avg_improvement, negative_avg_improvement)
    print(f"\nT-test: Positive vs Negative notes (AVG improvement)")
    print(f"  Positive note average improvement: {positive_avg_improvement.mean():.2f}")
    print(f"  Negative note average improvement: {negative_avg_improvement.mean():.2f}")
    print(f"  Difference: {positive_avg_improvement.mean() - negative_avg_improvement.mean():.2f}")
    print(f"  P-value: {p_value:.4f} {'(STATISTICALLY SIGNIFICANT)' if p_value < 0.05 else '(NOT significant)'}")

if len(no_note_avg_improvement) > 0:
    print(f"\nPlayers with NO notes:")
    print(f"  Average improvement: {no_note_avg_improvement.mean():.2f}")
    print(f"  Sample size: {len(no_note_avg_improvement)}")

# KEY INSIGHT 2: Pre-draft metrics correlation with success
print("\n" + "="*100)
print("KEY INSIGHT #2: WHICH PRE-DRAFT METRICS PREDICT SUCCESS?")
print("="*100)

# Calculate correlations between pre-draft features and final performance
pre_draft_features = ['before games', 'before AVG', 'before AVG Rank', 'Before TPOR', 'Before TPOR Rank by position']
success_metrics = ['Final average', 'Final AVG Rank', 'Final TPOR', 'positional_success']

print("\nCorrelations between pre-draft features and final performance:")
print("(Higher absolute correlation = stronger predictive power)\n")

correlation_results = []
for pre_feat in pre_draft_features:
    for success_metric in success_metrics:
        data = df_clean[[pre_feat, success_metric]].dropna()
        if len(data) > 2:
            corr, p_val = pearsonr(data[pre_feat], data[success_metric])
            correlation_results.append({
                'Pre-Draft Feature': pre_feat,
                'Success Metric': success_metric,
                'Correlation': corr,
                'P-value': p_val,
                'Significant': 'YES' if p_val < 0.05 else 'NO'
            })

corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
print(corr_df.to_string(index=False))

# KEY INSIGHT 3: Best vs Worst picks
print("\n" + "="*100)
print("KEY INSIGHT #3: BEST AND WORST PICKS")
print("="*100)

# Sort by positional success
df_clean_sorted = df_clean.sort_values('positional_success', ascending=False)

print("\nTOP 10 BEST PICKS (Most positive positional success):")
best_cols = ['Name', 'pick', 'position', 'note type', 'Pre-Draft position Rank',
             'Final TPOR Rank by position', 'positional_success', 'Final average']
print(df_clean_sorted[best_cols].head(10).to_string(index=False))

print("\nTOP 10 WORST PICKS (Most negative positional success):")
print(df_clean_sorted[best_cols].tail(10).to_string(index=False))

# Save to CSV for further analysis
output_file = Path('data/predictions/2025_draft_analysis.csv')
df_clean.to_csv(output_file, index=False)
print(f"\n\nFull analysis data saved to: {output_file}")
print(f"Total columns in saved file: {len(df_clean.columns)}")
