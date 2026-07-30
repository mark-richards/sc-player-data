import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("="*100)
print("MULTI-YEAR VALIDATION ANALYSIS - Testing 2025 Insights on Historical Data")
print("="*100)

excel_file = Path('draft_prep/SC 2026/Previous drafts combined - 2026 draft prep.xlsx')

# First, check what data is available in each year's sheet
print("\nChecking available data in each year...")
xls = pd.ExcelFile(excel_file)

years_to_analyze = ['2020', '2022', '2023', '2024', '2025']
yearly_results = {}

for year in years_to_analyze:
    if year not in xls.sheet_names:
        print(f"\n{year}: Sheet not found")
        continue

    print(f"\n{'='*100}")
    print(f"ANALYZING {year} DRAFT DATA")
    print(f"{'='*100}")

    # Read the sheet
    df_raw = pd.read_excel(excel_file, sheet_name=year, header=None)

    # Check if it has the same structure as 2025 (with BEFORE/AFTER sections)
    # Row 0 should have section headers, Row 1 should have column names
    if year == '2025':
        # We know 2025 has the detailed structure
        df = pd.read_excel(excel_file, sheet_name=year, header=1)

        # Filter to only rows with actual data
        df_clean = df[df['Name'].notna()].copy()

        # Convert numeric columns
        numeric_cols = [
            'pick', 'before games', 'before AVG', 'before AVG Rank', 'Before Total', 'Before TPOR',
            'Before TPOR Rank by position', 'Final Games', 'Final average', 'Final AVG Rank',
            'Final Total', 'Final TPOR', 'Final TPOR Rank by position', 'Pre-Draft position Rank',
            'Before TPOR Rank result', 'my rank result'
        ]

        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Calculate derived metrics
        df_clean['AVG_improvement'] = df_clean['Final average'] - df_clean['before AVG']
        df_clean['Rank_improvement'] = df_clean['before AVG Rank'] - df_clean['Final AVG Rank']
        df_clean['TPOR_improvement'] = df_clean['Final TPOR'] - df_clean['Before TPOR']
        df_clean['positional_success'] = df_clean['Pre-Draft position Rank'] - df_clean['Final TPOR Rank by position']
        df_clean['AVG_prediction_error'] = abs(df_clean['before AVG'] - df_clean['Final average'])

        # Encode notes
        df_clean['has_positive_note'] = (df_clean['note type'] == 'positive').astype(int)
        df_clean['has_negative_note'] = (df_clean['note type'] == 'negative').astype(int)

        print(f"Loaded {len(df_clean)} draft picks")

        # Store results
        yearly_results[year] = {
            'data': df_clean,
            'total_picks': len(df_clean),
            'positive_notes': df_clean['has_positive_note'].sum(),
            'negative_notes': df_clean['has_negative_note'].sum()
        }

        # Key metrics
        positive_improvement = df_clean[df_clean['has_positive_note'] == 1]['AVG_improvement'].mean()
        negative_improvement = df_clean[df_clean['has_negative_note'] == 1]['AVG_improvement'].mean()

        yearly_results[year]['positive_avg_improvement'] = positive_improvement
        yearly_results[year]['negative_avg_improvement'] = negative_improvement
        yearly_results[year]['note_difference'] = positive_improvement - negative_improvement

        # Statistical test
        pos_data = df_clean[df_clean['has_positive_note'] == 1]['AVG_improvement'].dropna()
        neg_data = df_clean[df_clean['has_negative_note'] == 1]['AVG_improvement'].dropna()

        if len(pos_data) > 0 and len(neg_data) > 0:
            t_stat, p_value = stats.ttest_ind(pos_data, neg_data)
            yearly_results[year]['t_stat'] = t_stat
            yearly_results[year]['p_value'] = p_value

        print(f"\nNote Performance:")
        print(f"  Positive notes: {yearly_results[year]['positive_notes']} picks, {positive_improvement:.2f} avg improvement")
        print(f"  Negative notes: {yearly_results[year]['negative_notes']} picks, {negative_improvement:.2f} avg improvement")
        print(f"  Difference: {yearly_results[year]['note_difference']:.2f} points")
        if 'p_value' in yearly_results[year]:
            print(f"  P-value: {yearly_results[year]['p_value']:.4f} {'(SIGNIFICANT)' if yearly_results[year]['p_value'] < 0.05 else '(not significant)'}")

        # Position analysis
        position_performance = df_clean.groupby('position').agg({
            'AVG_improvement': 'mean',
            'positional_success': 'mean'
        }).round(2)

        print(f"\nPosition Performance:")
        print(position_performance)

        # Overperformers
        overperformers = df_clean[df_clean['positional_success'] > 20]
        if len(overperformers) > 0:
            yearly_results[year]['overperformers_count'] = len(overperformers)
            yearly_results[year]['overperformers_positions'] = overperformers['position'].value_counts().to_dict()

            print(f"\nOverperformers (>20 position improvement): {len(overperformers)}")
            print(f"  Position breakdown: {yearly_results[year]['overperformers_positions']}")

    else:
        # For other years, check the structure
        print(f"First 5 rows of data:")
        print(df_raw.head().to_string())

        # Check if columns include draft-relevant data
        if 'pick' in df_raw.iloc[0].values or 'pick' in df_raw.columns:
            print(f"\nYear {year} appears to have draft data but in a different format.")
            print(f"Columns: {df_raw.columns.tolist()}")
        else:
            print(f"\nYear {year} has different data structure - may need manual inspection")

# Summary across years
if len(yearly_results) > 0:
    print("\n" + "="*100)
    print("CROSS-YEAR VALIDATION SUMMARY")
    print("="*100)

    summary_data = []
    for year, results in yearly_results.items():
        if 'note_difference' in results:
            summary_data.append({
                'Year': year,
                'Total Picks': results['total_picks'],
                'Positive Notes': results['positive_notes'],
                'Negative Notes': results['negative_notes'],
                'Pos Improvement': results['positive_avg_improvement'],
                'Neg Improvement': results['negative_avg_improvement'],
                'Difference': results['note_difference'],
                'P-value': results.get('p_value', np.nan),
                'Significant': 'YES' if results.get('p_value', 1) < 0.05 else 'NO'
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\nNote Performance Across Years:")
        print(summary_df.to_string(index=False))

        # Calculate average difference
        avg_difference = summary_df['Difference'].mean()
        print(f"\nAverage difference across years: {avg_difference:.2f} points")
        print(f"Consistent pattern: {'YES - Positive notes consistently better' if avg_difference > 5 else 'MIXED RESULTS'}")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print("\nNOTE: Only 2025 has the detailed BEFORE/AFTER analysis structure.")
print("Other years contain raw draft pick data without pre/post comparison.")
print("\nTo validate across multiple years, you would need to:")
print("1. Add similar BEFORE/AFTER analysis to 2024, 2023, 2022 sheets")
print("2. Or create a new analysis comparing draft position to end-of-season performance")
