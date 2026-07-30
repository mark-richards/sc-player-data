import pandas as pd
from pathlib import Path

# Load the 2026 SC player list with position data
sc_2026 = pd.read_csv('draft_prep/SC 2026/2026_SC_Player_list.csv')
print(f"Loaded {len(sc_2026)} players from 2026 SC player list")

# Create a mapping from feed_id to position
# The 'position' column has short codes like 'FWD', 'MID', 'DEF', 'RUC'
position_mapping = sc_2026[['feed_id', 'position']].copy()
position_mapping = position_mapping.dropna(subset=['position'])
position_mapping['position'] = position_mapping['position'].str.strip()

# Map SC position codes to FanFooty position names
position_map = {
    'DEF': 'Back',
    'MID': 'Midfielder',
    'FWD': 'Forward',
    'RUC': 'Ruck',
    # Handle dual positions - use first position
    'MID FWD': 'Midfielder',
    'DEF MID': 'Back',
    'FWD MID': 'Forward',
    'MID RUC': 'Midfielder',
    'DEF FWD': 'Back',
}

position_mapping['position_mapped'] = position_mapping['position'].map(lambda x: position_map.get(x.split()[0], x))

print(f"\nPosition mapping created for {len(position_mapping)} players")
print("\nSample mappings:")
print(position_mapping.head(10))

# Update predictions for rounds 1-6
rounds_updated = 0
players_updated = 0

for round_num in range(1, 7):
    pred_file = Path(f'data/predictions/2026_round_{round_num}_predictions.csv')

    if pred_file.exists():
        # Load predictions
        pred = pd.read_csv(pred_file)

        # Count missing positions before
        missing_before = pred['Position'].isna().sum()

        # Merge position data based on Player ID = feed_id
        pred = pred.merge(
            position_mapping[['feed_id', 'position_mapped']],
            left_on='Player ID',
            right_on='feed_id',
            how='left'
        )

        # Fill missing positions with mapped positions
        pred['Position'] = pred['Position'].fillna(pred['position_mapped'])

        # Drop temporary columns
        pred = pred.drop(columns=['feed_id', 'position_mapped'], errors='ignore')

        # Count missing positions after
        missing_after = pred['Position'].isna().sum()
        filled = missing_before - missing_after

        # Save updated predictions
        pred.to_csv(pred_file, index=False)

        rounds_updated += 1
        players_updated += filled

        print(f"Round {round_num}: Filled {filled} missing positions ({missing_after} still missing)")

print(f"\n✓ Updated {rounds_updated} round files")
print(f"✓ Filled positions for {players_updated} player-round combinations")
print("\nNow regenerate the draft summary to include these players in positional rankings.")
