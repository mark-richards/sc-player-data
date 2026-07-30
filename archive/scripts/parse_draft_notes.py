"""
Parse draft notes from 2021-2025 xlsx files into a unified format.
Classify notes into structured features and backtest against season outcomes.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

NOTES_DIR = Path('data/draft_notes')
OUTPUT_DIR = Path('data/draft')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 1: Parse notes from each year's xlsx
# ============================================================

def parse_2025_notes():
    """2025: Clean format with id, feed_id, name, notes in position tabs."""
    f = NOTES_DIR / 'draft_notes_2025.xlsx'
    rows = []
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        df = pd.read_excel(f, sheet_name=pos)
        for rank, (_, r) in enumerate(df.iterrows(), 1):
            rows.append({
                'feed_id': int(r['feed_id']),
                'name': r['name'],
                'position': pos,
                'year': 2025,
                'note': r.get('notes', np.nan),
                'user_rank': rank,
                'prev_avg': r.get('AV', np.nan),
                'prev_games': r.get('#', np.nan),
            })
    return pd.DataFrame(rows)


def parse_2024_notes():
    """2024: Headers in rows 0-1, Notes in col index 11, feed_id in col 4."""
    f = NOTES_DIR / 'draft_notes_2024.xlsx'
    rows = []
    for pos in ['DEF', 'MID', 'FWD']:
        df = pd.read_excel(f, sheet_name=pos)
        # Row 1 is the real header (id, first_name, ..., Notes, ...)
        # Data starts at row 2
        for rank, idx in enumerate(range(2, len(df)), 1):
            r = df.iloc[idx]
            fid = r.iloc[4]  # feed_id
            name = r.iloc[6]  # Name
            note = r.iloc[11]  # Notes
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                continue
            rows.append({
                'feed_id': fid,
                'name': name,
                'position': pos,
                'year': 2024,
                'note': note if pd.notna(note) else np.nan,
                'user_rank': rank,
            })
    # RUC has proper headers
    df = pd.read_excel(f, sheet_name='RUC')
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        rows.append({
            'feed_id': int(r['feed_id']),
            'name': r['Name'],
            'position': 'RUC',
            'year': 2024,
            'note': r.get('Notes', np.nan),
            'user_rank': rank,
        })
    return pd.DataFrame(rows)


def parse_2023_notes():
    """2023: Position tabs with header in row 0, notes in last col. Plus Watchlist."""
    f = NOTES_DIR / 'draft_notes_2023.xlsx'
    rows = []
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        df = pd.read_excel(f, sheet_name=pos)
        # Row 0 has column labels (id, feed_id, full_name, ..., Notes)
        for rank, idx in enumerate(range(1, len(df)), 1):
            r = df.iloc[idx]
            fid = r.iloc[1]  # feed_id
            name = r.iloc[2]  # full_name
            note = r.iloc[-1]  # last col = Notes
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                continue
            rows.append({
                'feed_id': fid,
                'name': name,
                'position': pos,
                'year': 2023,
                'note': note if pd.notna(note) else np.nan,
                'user_rank': rank,
            })

    # Also parse Watchlist tab (has notes for players across positions)
    wl = pd.read_excel(f, sheet_name='Watchlist')
    for _, r in wl.iterrows():
        if pd.notna(r.get('Player Name')) and pd.notna(r.get('Notes')):
            fid = r.get('Player ID')
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                fid = None
            rows.append({
                'feed_id': fid,
                'name': r['Player Name'],
                'position': None,
                'year': 2023,
                'note': r['Notes'],
                'user_rank': None,
                'source': 'watchlist',
            })
    return pd.DataFrame(rows)


def parse_2022_notes():
    """2022: Position tabs with headers in rows 0-1, Notes in col 5. Plus Watchlist."""
    f = NOTES_DIR / 'draft_notes_2022.xlsx'
    rows = []
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        df = pd.read_excel(f, sheet_name=pos)
        for rank, idx in enumerate(range(2, len(df)), 1):
            r = df.iloc[idx]
            fid = r.iloc[0]  # Player ID (= feed_id in this format)
            name = r.iloc[2]  # Full Name
            note = r.iloc[5]  # Notes
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                continue
            rows.append({
                'feed_id': fid,
                'name': name,
                'position': pos,
                'year': 2022,
                'note': note if pd.notna(note) else np.nan,
                'user_rank': rank,
            })

    # Watchlist
    wl = pd.read_excel(f, sheet_name='Watchlist')
    for _, r in wl.iterrows():
        if pd.notna(r.get('Player Name')) and pd.notna(r.get('Notes')):
            fid = r.get('Player ID')
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                fid = None
            rows.append({
                'feed_id': fid,
                'name': r['Player Name'],
                'position': None,
                'year': 2022,
                'note': r['Notes'],
                'user_rank': None,
                'source': 'watchlist',
            })
    return pd.DataFrame(rows)


def parse_2026_notes():
    """2026: Same format as 2025 but with explicit board_rank column for user ranking."""
    f = NOTES_DIR / 'draft_notes_2026.xlsx'
    rows = []
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        df = pd.read_excel(f, sheet_name=pos)
        for _, r in df.iterrows():
            fid = r.get('feed_id', None)
            if pd.isna(fid):
                continue
            board_rank = r.get('board_rank', np.nan)
            rows.append({
                'feed_id': int(fid),
                'name': r.get('name', ''),
                'position': pos,
                'year': 2026,
                'note': r.get('notes', np.nan),
                'user_rank': int(board_rank) if pd.notna(board_rank) else None,
            })
    return pd.DataFrame(rows)


def parse_2021_notes():
    """2021: Player Notes tab with Type, Player, Notes. Position tabs with notes."""
    f = NOTES_DIR / 'draft_notes_2021.xlsx'
    rows = []

    # Player Notes tab
    pn = pd.read_excel(f, sheet_name='Player Notes')
    for _, r in pn.iterrows():
        if pd.notna(r.get('Player')) and pd.notna(r.get('Notes')):
            fid = r.get('Feed ID')
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                fid = None
            note = r['Notes']
            note_type = r.get('Type', '')
            if pd.notna(note_type):
                note = f"[{note_type}] {note}"
            rows.append({
                'feed_id': fid,
                'name': r['Player'],
                'position': r.get('Position', None),
                'year': 2021,
                'note': note,
                'user_rank': None,
                'source': 'player_notes',
            })

    # Position tabs — similar structure to 2023
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        df = pd.read_excel(f, sheet_name=pos)
        # Row 0 has column labels
        for rank, idx in enumerate(range(1, len(df)), 1):
            r = df.iloc[idx]
            fid = r.iloc[1]  # feed_id
            name = r.iloc[2]  # name
            note = r.iloc[-1] if len(r) > 10 else np.nan  # last col = notes
            try:
                fid = int(float(fid))
            except (ValueError, TypeError):
                continue
            rows.append({
                'feed_id': fid,
                'name': name,
                'position': pos,
                'year': 2021,
                'note': note if pd.notna(note) else np.nan,
                'user_rank': rank,
            })
    return pd.DataFrame(rows)


# ============================================================
# SECTION 2: Classify notes into structured features
# ============================================================

def classify_note(note_text):
    """Parse a free-text note into structured boolean features."""
    if pd.isna(note_text) or not isinstance(note_text, str):
        return {}

    note = note_text.lower().strip()

    flags = {}

    # INJURY flags
    injury_words = ['hamstring', 'hammy', 'acl', 'mcl', 'ankle', 'knee', 'groin',
                     'calf', 'shoulder', 'back issue', 'back surg', 'plantar',
                     'concuss', 'broken', 'stress fracture', 'reco', 'surgery',
                     'pelvis', 'foot issue', 'foot stress', 'infection', 'sore',
                     'soreness', 'glandular']
    flags['note_injury'] = int(any(w in note for w in injury_words))

    # AVAILABILITY flags
    avail_words = ['in doubt', 'unlikely', 'out for', 'out rd', 'out round',
                   'missing', 'will miss', 'suspended', 'out indefinitely',
                   'out till', 'out until']
    flags['note_availability_concern'] = int(any(w in note for w in avail_words))

    # RETURNING from long-term injury
    flags['note_returning'] = int(any(w in note for w in ['returning', 'back from', 'back mid-year']))

    # EXPECTED IMPROVEMENT
    improve_words = ['should improve', 'should increase', 'big jump', 'could improve',
                     'should average more', 'can bounce back', 'might improve',
                     'expected to jump', 'great end', 'great finish', 'great pre-season',
                     'good praccy', 'dominant pre-season', 'should get more',
                     'training the house down', 'will improve']
    flags['note_expected_improve'] = int(any(w in note for w in improve_words))

    # EXPECTED DECLINE
    decline_words = ['unlikely to match', 'drop expected', 'will drop', 'may drop',
                     'overcoming', 'high risk']
    flags['note_expected_decline'] = int(any(w in note for w in decline_words))

    # ROLE CHANGE (positive - more midfield time / better role)
    role_pos_words = ['more mid', 'full time mid', 'midfield potential', 'mid time',
                      'mid role', 'pure mid', 'midfield at', 'potential mid',
                      'more opportunity', 'half back role', 'playing back',
                      'moving into midfield', 'moving to a half back',
                      'great role', 'better scoring', 'more kick-ins',
                      'fwd role']
    flags['note_role_change'] = int(any(w in note for w in role_pos_words))

    # LOVE / HIGH CONVICTION
    flags['note_love'] = int(any(w in note for w in ['love', 'great pick', 'value']))

    # BENCH / LATE PICK
    flags['note_bench_candidate'] = int(any(w in note for w in ['potential bench', 'bench', 'late pick', 'cheap']))

    # NEGATIVE SENTIMENT
    flags['note_avoid'] = int(any(w in note for w in ['avoid', 'won\'t play']))

    # HAS ANY NOTE AT ALL
    flags['has_note'] = 1

    return flags


def classify_all_notes(notes_df):
    """Apply classification to all notes."""
    classifications = notes_df['note'].apply(classify_note)
    class_df = pd.DataFrame(classifications.tolist()).fillna(0).astype(int)
    return pd.concat([notes_df, class_df], axis=1)


# ============================================================
# SECTION 3: Merge with season outcomes and backtest
# ============================================================

def backtest_notes(classified_notes):
    """Backtest: Do notes predict outcomes?"""
    master = pd.read_csv('data/processed/master_player_data.csv', low_memory=False)

    print("=" * 80)
    print("BACKTESTING DRAFT NOTES AGAINST SEASON OUTCOMES")
    print("=" * 80)

    # Get season outcomes for each year
    all_results = []
    for year in [2022, 2023, 2024, 2025]:
        # Season comparison file has before/after stats
        sc_file = OUTPUT_DIR / f'season_comparison_{year}.csv'
        if sc_file.exists():
            sc = pd.read_csv(sc_file)
            sc = sc[['feed_id', 'prev_avg', 'season_avg', 'season_games', 'avg_change']].copy()
            sc['Year'] = year
            all_results.append(sc)

    if not all_results:
        print("No season comparison files found!")
        return

    outcomes = pd.concat(all_results, ignore_index=True)

    # Merge notes with outcomes
    noted = classified_notes[classified_notes['note'].notna()].copy()
    noted = noted.drop_duplicates(subset=['feed_id', 'year'], keep='first')

    # Only include years where we have outcomes (2022-2025)
    noted = noted[noted['year'].isin([2022, 2023, 2024, 2025])]

    merged = noted.merge(outcomes, left_on=['feed_id', 'year'], right_on=['feed_id', 'Year'], how='inner')
    print(f"\nMatched {len(merged)} noted players to season outcomes")

    # Also get non-noted players for comparison
    all_ranked = classified_notes[classified_notes['user_rank'].notna()].copy()
    all_ranked = all_ranked.drop_duplicates(subset=['feed_id', 'year'], keep='first')
    all_ranked = all_ranked[all_ranked['year'].isin([2022, 2023, 2024, 2025])]
    all_merged = all_ranked.merge(outcomes, left_on=['feed_id', 'year'], right_on=['feed_id', 'Year'], how='inner')

    # ---- Analysis 1: Notes vs No Notes ----
    print("\n--- 1. Players WITH notes vs WITHOUT notes ---")
    with_notes = all_merged[all_merged['has_note'] == 1] if 'has_note' in all_merged.columns else all_merged[all_merged['note'].notna()]
    without_notes = all_merged[all_merged['note'].isna()]

    print(f"  With notes:    {len(with_notes):4d} players, avg change: {with_notes['avg_change'].mean():+.1f}, "
          f"avg season: {with_notes['season_avg'].mean():.1f}, avg games: {with_notes['season_games'].mean():.1f}")
    print(f"  Without notes: {len(without_notes):4d} players, avg change: {without_notes['avg_change'].mean():+.1f}, "
          f"avg season: {without_notes['season_avg'].mean():.1f}, avg games: {without_notes['season_games'].mean():.1f}")

    # ---- Analysis 2: Each note flag vs outcome ----
    print("\n--- 2. Note Classification Outcomes ---")
    flag_cols = [c for c in merged.columns if c.startswith('note_')]
    print(f"  {'Flag':<30} {'N':>4} {'Avg Change':>10} {'Avg Season':>10} {'Avg Games':>9} {'Signal'}")
    print(f"  {'-'*75}")

    baseline_change = all_merged['avg_change'].mean()
    for flag in flag_cols:
        flagged = merged[merged[flag] == 1]
        if len(flagged) >= 3:
            change = flagged['avg_change'].mean()
            season = flagged['season_avg'].mean()
            games = flagged['season_games'].mean()
            signal = change - baseline_change
            direction = "UP" if signal > 3 else "DOWN" if signal < -3 else "NEUTRAL"
            print(f"  {flag:<30} {len(flagged):>4} {change:>+10.1f} {season:>10.1f} {games:>9.1f} {signal:>+6.1f} {direction}")

    # ---- Analysis 3: Injury notes specifically ----
    print("\n--- 3. Injury Notes Deep Dive ---")
    injured_noted = merged[merged['note_injury'] == 1]
    avail_concern = merged[merged['note_availability_concern'] == 1]
    print(f"  Injury flagged:      {len(injured_noted)} players")
    if len(injured_noted) > 0:
        print(f"    Avg season games:  {injured_noted['season_games'].mean():.1f} (vs {all_merged['season_games'].mean():.1f} overall)")
        print(f"    Avg change:        {injured_noted['avg_change'].mean():+.1f}")
        missed_lots = injured_noted[injured_noted['season_games'] < 15]
        print(f"    Played <15 games:  {len(missed_lots)}/{len(injured_noted)} ({len(missed_lots)/len(injured_noted)*100:.0f}%)")

    print(f"\n  Availability concern: {len(avail_concern)} players")
    if len(avail_concern) > 0:
        print(f"    Avg season games:  {avail_concern['season_games'].mean():.1f}")
        print(f"    Avg change:        {avail_concern['avg_change'].mean():+.1f}")

    # ---- Analysis 4: "Should improve" accuracy ----
    print("\n--- 4. 'Should Improve' Accuracy ---")
    improvers = merged[merged['note_expected_improve'] == 1]
    if len(improvers) > 0:
        actually_improved = improvers[improvers['avg_change'] > 0]
        print(f"  Flagged as 'should improve': {len(improvers)}")
        print(f"  Actually improved:           {len(actually_improved)} ({len(actually_improved)/len(improvers)*100:.0f}%)")
        print(f"  Avg change:                  {improvers['avg_change'].mean():+.1f}")
        print(f"  vs baseline avg change:      {baseline_change:+.1f}")

        big_improvers = improvers[improvers['avg_change'] > 10]
        print(f"  Improved by 10+:             {len(big_improvers)} ({len(big_improvers)/len(improvers)*100:.0f}%)")

    # ---- Analysis 5: Role change accuracy ----
    print("\n--- 5. Role Change Accuracy ---")
    role_changers = merged[merged['note_role_change'] == 1]
    if len(role_changers) > 0:
        print(f"  Flagged role change: {len(role_changers)}")
        print(f"  Avg change:          {role_changers['avg_change'].mean():+.1f}")
        actually_improved = role_changers[role_changers['avg_change'] > 0]
        print(f"  Actually improved:   {len(actually_improved)} ({len(actually_improved)/len(role_changers)*100:.0f}%)")

    # ---- Analysis 6: User ranking quality ----
    print("\n--- 6. User Ranking Quality (Spearman) ---")
    from scipy.stats import spearmanr

    for year in [2022, 2023, 2024, 2025]:
        for pos in ['DEF', 'MID', 'FWD', 'RUC']:
            yr_pos = all_merged[(all_merged['year'] == year) & (all_merged['position'] == pos)
                                & all_merged['user_rank'].notna() & all_merged['season_avg'].notna()]
            if len(yr_pos) >= 10:
                s, _ = spearmanr(yr_pos['user_rank'], yr_pos['season_avg'])
                # Negative because lower rank = better player = higher season avg
                print(f"  {year} {pos}: Spearman r={-s:.3f} (n={len(yr_pos)})")

    # ---- Analysis 7: Per-note examples ----
    print("\n--- 7. Notable Predictions (Best & Worst) ---")
    if len(merged) > 0:
        merged['prediction_error'] = merged['avg_change']  # positive = improved, negative = declined

        # Best calls: flagged improvement & actually improved a lot
        best = merged[(merged['note_expected_improve'] == 1) & (merged['avg_change'] > 10)].sort_values('avg_change', ascending=False)
        print(f"\n  BEST 'should improve' calls:")
        for _, r in best.head(8).iterrows():
            print(f"    {r['name']:30s} {int(r['year'])} | Change: {r['avg_change']:+.1f} | Note: {r['note']}")

        # Worst injury calls that still played well
        worst_injury = merged[(merged['note_injury'] == 1) & (merged['avg_change'] > 5)].sort_values('avg_change', ascending=False)
        print(f"\n  INJURY flagged but IMPROVED:")
        for _, r in worst_injury.head(5).iterrows():
            print(f"    {r['name']:30s} {int(r['year'])} | Change: {r['avg_change']:+.1f} | Note: {r['note']}")

        # Players flagged injury who actually busted
        injury_busts = merged[(merged['note_injury'] == 1) & (merged['avg_change'] < -10)].sort_values('avg_change')
        print(f"\n  INJURY flagged and BUSTED:")
        for _, r in injury_busts.head(5).iterrows():
            print(f"    {r['name']:30s} {int(r['year'])} | Change: {r['avg_change']:+.1f} | Note: {r['note']}")

    return merged


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Parse all years
    print("Parsing draft notes...")
    frames = []
    for year, parser in [(2021, parse_2021_notes), (2022, parse_2022_notes),
                          (2023, parse_2023_notes), (2024, parse_2024_notes),
                          (2025, parse_2025_notes)]:
        df = parser()
        print(f"  {year}: {len(df)} entries, {df['note'].notna().sum()} with notes")
        frames.append(df)

    all_notes = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(all_notes)} entries, {all_notes['note'].notna().sum()} with notes")

    # Classify
    print("\nClassifying notes...")
    classified = classify_all_notes(all_notes)

    # Save unified notes
    classified.to_csv(OUTPUT_DIR / 'draft_notes_unified.csv', index=False)
    print(f"Saved {OUTPUT_DIR / 'draft_notes_unified.csv'}")

    # Save just the noted players for quick reference
    noted_only = classified[classified['note'].notna()].copy()
    noted_only.to_csv(OUTPUT_DIR / 'draft_notes_with_flags.csv', index=False)
    print(f"Saved {OUTPUT_DIR / 'draft_notes_with_flags.csv'} ({len(noted_only)} notes)")

    # Print note classification summary
    print("\n--- Note Classification Summary ---")
    flag_cols = [c for c in classified.columns if c.startswith('note_') or c == 'has_note']
    noted = classified[classified['note'].notna()]
    for flag in flag_cols:
        n = noted[flag].sum()
        pct = n / len(noted) * 100
        print(f"  {flag:<30}: {int(n):>4} ({pct:.0f}%)")

    # Backtest
    print()
    merged = backtest_notes(classified)
