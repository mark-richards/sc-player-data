"""
Backtest: Model improvements comparison
Compares:
  A. Base features (no notes, no projections, no team change)
  B. + All new features (projections + team change + notes + user rank)
  C. + Ensemble (XGBoost + Ridge blend)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from draft_features import build_draft_features, DRAFT_FEATURE_COLS, NOTE_FLAG_COLS, SC_RELEVANT_CAPS, filter_sc_relevant
from parse_draft_notes import (parse_2022_notes, parse_2023_notes,
                                parse_2024_notes, parse_2025_notes,
                                classify_all_notes)

# Load master data
master = pd.read_csv('data/processed/master_player_data.csv', low_memory=False)
master['Round_Num'] = pd.to_numeric(master['Round_Num'], errors='coerce')
master['Tag'] = master['Tag'].fillna('')
master['Tag 2'] = master['Tag 2'].fillna('')

# Parse notes for 2022-2025
print("Parsing notes...")
note_frames = []
for year, parser in [(2022, parse_2022_notes), (2023, parse_2023_notes),
                      (2024, parse_2024_notes), (2025, parse_2025_notes)]:
    df = parser()
    note_frames.append(df)
all_notes = pd.concat(note_frames, ignore_index=True)
classified = classify_all_notes(all_notes)
print(f"Total notes: {len(classified)}, with notes: {classified['note'].notna().sum()}")

# Build features + outcomes for each year (2016-2025)
print("\nBuilding features (2016-2025)...")
all_features = []
for y in range(2016, 2026):
    print(f"  {y}...")
    feats = build_draft_features(master, y)
    if feats.empty:
        continue

    # Season outcomes
    sy = master[(master['Year'] == y) & (master['played'] > 0)]
    so = sy.groupby('Player ID').agg(
        season_avg=('SC', 'mean'),
        season_games=('SC', 'count'),
    ).reset_index()
    so['Year'] = y

    feats = feats.merge(so, on=['Player ID', 'Year'], how='left')
    feats = feats.dropna(subset=['season_avg'])
    feats = feats[feats['season_games'] >= 5].copy()

    # Merge note flags (only available for 2022-2025)
    yr_notes = classified[classified['year'] == y].drop_duplicates(subset=['feed_id'], keep='first')
    if len(yr_notes) > 0:
        note_merge = yr_notes[['feed_id'] + NOTE_FLAG_COLS].copy()
        note_merge = note_merge.rename(columns={'feed_id': 'Player ID'})
        feats = feats.merge(note_merge, on='Player ID', how='left')

    # Fill missing note flags with 0
    for c in NOTE_FLAG_COLS:
        if c not in feats.columns:
            feats[c] = 999.0 if c == 'user_rank' else 0.0
        if c == 'user_rank':
            feats[c] = pd.to_numeric(feats[c], errors='coerce').fillna(999).astype(float)
        else:
            feats[c] = pd.to_numeric(feats[c], errors='coerce').fillna(0).astype(float)

    all_features.append(feats)

df = pd.concat(all_features, ignore_index=True)
pre_filter = len(df)
df = filter_sc_relevant(df)
print(f"Total: {pre_filter} -> {len(df)} SC-relevant player-seasons (caps: {SC_RELEVANT_CAPS})")
print(f"With notes: {(df['has_note'] == 1).sum()}")
print(f"With user rank: {(df['user_rank'] < 999).sum()}")

# New feature columns for comparison
SC_PROJ_COLS = ['sc_ownership_pct', 'sc_start_pct', 'sc_projected_avg', 'sc_projected_price',
                'sc_breakeven_avg', 'sc_is_hot', 'sc_is_cold']
TEAM_CHANGE_COLS = ['changed_team', 'new_team_strength', 'old_team_strength']
NEW_FEATURE_COLS = SC_PROJ_COLS + TEAM_CHANGE_COLS

# Build config sets
OLD_COLS = [c for c in DRAFT_FEATURE_COLS if c not in NEW_FEATURE_COLS]  # 137 features (pre-improvements)

# ---- Leave-one-year-out backtesting ----
print("\n" + "=" * 80)
print("LEAVE-ONE-YEAR-OUT BACKTESTING")
print("=" * 80)

years = list(range(2016, 2026))

# Ensure all columns exist
for c in DRAFT_FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0

# Config A: Old features (137), XGBoost only
# Config B: All features (147), XGBoost only
# Config C: All features (147), XGBoost + Ridge ensemble
configs = [
    ('A. Old features (137)', OLD_COLS, False),
    ('B. + Projections + Team change (147)', DRAFT_FEATURE_COLS, False),
    ('C. + Ensemble (XGB+Ridge)', DRAFT_FEATURE_COLS, True),
]

for name, feature_cols, use_ensemble in configs:
    print(f"\n--- {name} ({len(feature_cols)} features) ---")

    preds = np.zeros(len(df))

    for test_year in years:
        train_mask = df['Year'] != test_year
        test_mask = df['Year'] == test_year

        if test_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, 'season_avg']
        X_test = df.loc[test_mask, feature_cols].fillna(0)

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42,
            n_jobs=-1, tree_method='hist',
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, df.loc[test_mask, 'season_avg'])], verbose=False)

        if use_ensemble:
            # Ridge regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            ridge = Ridge(alpha=10.0, random_state=42)
            ridge.fit(X_train_scaled, y_train)
            preds[test_mask] = xgb.predict(X_test) * 0.70 + ridge.predict(X_test_scaled) * 0.30
        else:
            preds[test_mask] = xgb.predict(X_test)

    pred_col = f'pred_{name[:1]}'
    df[pred_col] = preds

    # Overall metrics
    valid = preds != 0  # exclude years with no test data
    overall_s, _ = spearmanr(preds[valid], df.loc[valid, 'season_avg'])
    overall_mae = mean_absolute_error(df.loc[valid, 'season_avg'], preds[valid])
    print(f"  OVERALL: Spearman={overall_s:.4f}, MAE={overall_mae:.2f}")

    # Per-year (show recent years in detail)
    for year in years:
        yr = df[df['Year'] == year]
        if len(yr) == 0:
            continue
        s, _ = spearmanr(yr[pred_col], yr['season_avg'])
        mae = mean_absolute_error(yr['season_avg'], yr[pred_col])
        o50 = set(yr.nlargest(50, pred_col)['Player ID'])
        a50 = set(yr.nlargest(50, 'season_avg')['Player ID'])
        overlap = len(o50 & a50)
        busts = yr.nlargest(30, pred_col)
        bust_count = (busts['season_avg'] < busts['prev_avg'] - 15).sum()
        print(f"  {year}: r={s:.3f}, MAE={mae:.1f}, Top50={overlap}/50, Busts={bust_count}/30")

    # Feature importance for last XGBoost model
    imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 15 features:")
    for feat, val in imp.head(15).items():
        print(f"    {feat:<35}: {val:.4f}")

    # Show new features specifically
    new_in_model = [c for c in NEW_FEATURE_COLS if c in feature_cols]
    if new_in_model:
        print(f"\n  New feature importances:")
        for feat in new_in_model:
            print(f"    {feat:<35}: {imp.get(feat, 0):.4f}")

# ---- Team change analysis ----
print("\n" + "=" * 80)
print("TEAM CHANGE IMPACT ANALYSIS")
print("=" * 80)
changed = df[df['changed_team'] == 1]
stayed = df[df['changed_team'] == 0]
print(f"\n  Changed team: {len(changed)} player-seasons")
print(f"  Stayed:       {len(stayed)} player-seasons")
if len(changed) > 10:
    print(f"  Changed team avg season_avg: {changed['season_avg'].mean():.1f}")
    print(f"  Stayed avg season_avg:       {stayed['season_avg'].mean():.1f}")
    ch_delta = changed['season_avg'] - changed['prev_avg']
    st_delta = stayed['season_avg'] - stayed['prev_avg']
    print(f"  Changed team avg change:     {ch_delta.mean():+.1f}")
    print(f"  Stayed avg change:           {st_delta.mean():+.1f}")

# ---- SC Projections analysis ----
print("\n" + "=" * 80)
print("SC PROJECTIONS ANALYSIS")
print("=" * 80)
has_proj = df[df['sc_projected_avg'] > 0]
no_proj = df[df['sc_projected_avg'] == 0]
print(f"\n  With SC projections: {len(has_proj)} player-seasons")
print(f"  Without (pre-2022):  {len(no_proj)} player-seasons")
if len(has_proj) > 10:
    proj_corr, _ = spearmanr(has_proj['sc_projected_avg'], has_proj['season_avg'])
    print(f"  SC projected_avg vs actual: Spearman={proj_corr:.3f}")
    own_corr, _ = spearmanr(has_proj['sc_ownership_pct'], has_proj['season_avg'])
    print(f"  SC ownership vs actual:     Spearman={own_corr:.3f}")
