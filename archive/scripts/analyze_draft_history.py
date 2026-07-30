import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the historical draft workbook
excel_file = Path('draft_prep/SC 2026/Previous drafts combined - 2026 draft prep.xlsx')
print("Loading historical draft data...")

# Read 2025 sheet with proper header row
print("\n" + "="*100)
print("ANALYZING 2025 DRAFT DATA (DETAILED ANALYSIS)")
print("="*100)

# Read with first row as header
df_2025 = pd.read_excel(excel_file, sheet_name='2025', header=0)
print(f"\nShape: {df_2025.shape}")
print("\nAll columns:")
for i, col in enumerate(df_2025.columns):
    print(f"{i}: {col}")

print("\n\nFirst 15 rows (all columns):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
print(df_2025.head(15))

# Check for key columns we'd expect
print("\n" + "="*100)
print("IDENTIFYING KEY FEATURES")
print("="*100)

# Look for columns that might contain rankings, scores, consistency, notes
ranking_cols = [col for col in df_2025.columns if any(x in str(col).lower() for x in ['rank', 'order', 'pick'])]
score_cols = [col for col in df_2025.columns if any(x in str(col).lower() for x in ['score', 'avg', 'points', 'total'])]
consistency_cols = [col for col in df_2025.columns if any(x in str(col).lower() for x in ['consistency', 'std', 'variance', 'cv'])]
note_cols = [col for col in df_2025.columns if any(x in str(col).lower() for x in ['note', 'comment', 'positive', 'negative'])]

print(f"\nRanking-related columns: {ranking_cols}")
print(f"Score-related columns: {score_cols}")
print(f"Consistency-related columns: {consistency_cols}")
print(f"Note-related columns: {note_cols}")

# Check for BEFORE and AFTER sections
if 'BEFORE' in df_2025.columns and 'AFTER' in df_2025.columns:
    print("\nFound 'BEFORE' and 'AFTER' sections - likely pre-draft vs end-of-season comparison")

    # Get column positions
    before_idx = df_2025.columns.get_loc('BEFORE')
    after_idx = df_2025.columns.get_loc('AFTER')

    print(f"BEFORE section starts at column {before_idx}")
    print(f"AFTER section starts at column {after_idx}")

    print(f"\nColumns before BEFORE: {df_2025.columns[:before_idx].tolist()}")
    print(f"\nColumns between BEFORE and AFTER: {df_2025.columns[before_idx:after_idx].tolist()}")
    print(f"\nColumns after AFTER: {df_2025.columns[after_idx:].tolist()}")
