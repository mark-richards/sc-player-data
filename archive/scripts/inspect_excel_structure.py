import pandas as pd
from pathlib import Path

excel_file = Path('draft_prep/SC 2026/Previous drafts combined - 2026 draft prep.xlsx')

# Read first 5 rows without any header
df_raw = pd.read_excel(excel_file, sheet_name='2025', header=None, nrows=5)

print("First 5 rows of the Excel file:")
print("="*150)

# Show each row
for idx, row in df_raw.iterrows():
    print(f"\nRow {idx}:")
    for col_idx, value in enumerate(row):
        if pd.notna(value):
            print(f"  Col {col_idx}: {value}")
