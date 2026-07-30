"""
Backtest Draft Board v2 - Rich Features
=========================================
Uses master data to extract features the user correctly identified:
- Post-round 13 average (second half form)
- Injury tag counts
- Positive tags (star, hot, gun)
- Role tags
- Games played / durability
- Multi-year lookback from master data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

# Load master data
master = pd.read_csv('data/processed/master_player_data.csv', low_memory=False)
master['Round_Num'] = pd.to_numeric(master['Round_Num'], errors='coerce')
master = master[master['played'] > 0].copy()
master['Tag'] = master['Tag'].fillna('')
master['Tag 2'] = master['Tag 2'].fillna('')

print("Building rich pre-draft features from master data...")

def build_rich_features(master_df, target_year):
    """Build pre-draft features for target_year using only data available before the season."""
    prev_year = target_year - 1
    prev2_year = target_year - 2

    # Previous season data
    prev = master_df[master_df['Year'] == prev_year].copy()
    if prev.empty:
        return pd.DataFrame()

    # --- BASIC STATS ---
    basic = prev.groupby('Player ID').agg(
        prev_avg=('SC', 'mean'),
        prev_games=('SC', 'count'),
        prev_total=('SC', 'sum'),
        prev_std=('SC', 'std'),
        prev_ceiling=('SC', 'max'),
        prev_floor=('SC', 'min'),
        prev_median=('SC', 'median'),
    ).reset_index()
    basic['prev_cv'] = basic['prev_std'] / basic['prev_avg'].replace(0, np.nan)

    # --- POST-ROUND 13 AVERAGE (second half form) ---
    second_half = prev[prev['Round_Num'] > 13]
    second_half_stats = second_half.groupby('Player ID').agg(
        post_r13_avg=('SC', 'mean'),
        post_r13_games=('SC', 'count'),
        post_r13_std=('SC', 'std'),
        post_r13_ceiling=('SC', 'max'),
    ).reset_index()

    # First half for comparison
    first_half = prev[prev['Round_Num'] <= 13]
    first_half_stats = first_half.groupby('Player ID').agg(
        pre_r13_avg=('SC', 'mean'),
        pre_r13_games=('SC', 'count'),
    ).reset_index()

    # --- LAST 5 GAMES FORM ---
    last5 = prev.sort_values('Round_Num').groupby('Player ID').tail(5)
    last5_stats = last5.groupby('Player ID').agg(
        last5_avg=('SC', 'mean'),
        last5_std=('SC', 'std'),
    ).reset_index()

    # --- TAG FEATURES ---
    # Positive tags: star, hot, gun, x-factor
    positive_tags = ['star', 'hot', 'gun', 'x-factor']
    # Negative tags: injured, sore
    negative_tags = ['injured', 'sore']
    # Role tags of interest
    role_tags = ['ruck', 'tagger', 'tagged', 'sub']

    def count_tags(group, tag_list):
        tag_col = group['Tag'].str.lower() + ',' + group['Tag 2'].str.lower()
        count = 0
        for tag in tag_list:
            count += tag_col.str.contains(tag, na=False).sum()
        return count

    tag_features = prev.groupby('Player ID').apply(
        lambda g: pd.Series({
            'positive_tag_count': count_tags(g, positive_tags),
            'negative_tag_count': count_tags(g, negative_tags),
            'injury_tag_count': count_tags(g, ['injured', 'sore']),
            'star_tag_count': count_tags(g, ['star']),
            'hot_tag_count': count_tags(g, ['hot']),
            'tagged_count': count_tags(g, ['tagged']),
            'sub_count': count_tags(g, ['sub', 'subbed']),
        }),
        include_groups=False
    ).reset_index()

    # --- 2-YEAR LOOKBACK ---
    prev2 = master_df[(master_df['Year'] >= prev2_year) & (master_df['Year'] <= prev_year)]
    two_year = prev2.groupby('Player ID').agg(
        avg_2yr=('SC', 'mean'),
        games_2yr=('SC', 'count'),
        std_2yr=('SC', 'std'),
    ).reset_index()

    # --- CAREER STATS ---
    career = master_df[master_df['Year'] <= prev_year].groupby('Player ID').agg(
        career_games=('SC', 'count'),
        career_avg=('SC', 'mean'),
        first_year=('Year', 'min'),
    ).reset_index()
    career['age_proxy'] = target_year - career['first_year']

    # --- POSITION ---
    # Get most recent position
    pos = prev.groupby('Player ID')['sc_position'].last().reset_index()
    pos.columns = ['Player ID', 'position']

    # Names and team
    names = prev.groupby('Player ID').agg(
        first_name=('First Name', 'last'),
        last_name=('Last Name', 'last'),
        team=('Team', 'last'),
    ).reset_index()

    # --- MERGE ALL ---
    features = basic.copy()
    for df_merge in [second_half_stats, first_half_stats, last5_stats,
                     tag_features, two_year, career, pos, names]:
        features = features.merge(df_merge, on='Player ID', how='left')

    # --- DERIVED FEATURES ---
    features['second_half_form'] = features['post_r13_avg'].fillna(features['prev_avg']) - features['pre_r13_avg'].fillna(features['prev_avg'])
    features['last5_vs_avg'] = features['last5_avg'].fillna(features['prev_avg']) - features['prev_avg']
    features['games_pct'] = features['prev_games'] / 23
    features['injury_risk'] = features['injury_tag_count'] / features['prev_games'].replace(0, 1)
    features['avg_x_durability'] = features['prev_avg'] * features['games_pct']
    features['avg_x_consistency'] = features['prev_avg'] * (1 - features['prev_cv'].fillna(0.5))
    features['post_r13_premium'] = features['post_r13_avg'].fillna(0) - features['prev_avg']  # 2nd half > full season?
    features['two_yr_trend'] = features['prev_avg'] - features['avg_2yr'].fillna(features['prev_avg'])  # improving or declining?

    # Position dummies
    def simplify_pos(p):
        if pd.isna(p):
            return 'MID'
        p = str(p).upper()
        for pos in ['DEF', 'MID', 'FWD', 'RUC']:
            if pos in p:
                return pos
        return 'MID'

    features['primary_pos'] = features['position'].apply(simplify_pos)
    for p in ['DEF', 'MID', 'FWD', 'RUC']:
        features[f'pos_{p}'] = (features['primary_pos'] == p).astype(int)

    features['Year'] = target_year
    return features


# Build features for each year
all_features = []
for year in [2022, 2023, 2024, 2025]:
    feat = build_rich_features(master, year)
    all_features.append(feat)
    print(f"  {year}: {len(feat)} players")

df = pd.concat(all_features, ignore_index=True)

# Get actual season outcomes
season_frames = []
for year in [2022, 2023, 2024, 2025]:
    season = master[master['Year'] == year]
    season_stats = season.groupby('Player ID').agg(
        season_avg=('SC', 'mean'),
        season_games=('SC', 'count'),
        season_total=('SC', 'sum'),
    ).reset_index()
    season_stats['Year'] = year
    season_frames.append(season_stats)
season_df = pd.concat(season_frames, ignore_index=True)
df = df.merge(season_df, on=['Player ID', 'Year'], how='left')

# Filter to players who actually played 5+ games in the season
df = df.dropna(subset=['season_avg']).copy()
df = df[df['season_games'] >= 5].copy()
df['avg_change'] = df['season_avg'] - df['prev_avg']
df['big_decline'] = (df['avg_change'] < -15).astype(int)

print(f"\nTotal backtesting pool: {len(df)} player-seasons")
print(f"By year: {df.groupby('Year').size().to_dict()}")

# ====================================================================
# FEATURE COLUMNS
# ====================================================================
feature_cols_basic = ['prev_avg']

feature_cols_standard = [
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'career_games', 'age_proxy',
]

feature_cols_rich = [
    # Basic
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'career_games', 'age_proxy',
    # Second half / form
    'post_r13_avg', 'post_r13_games', 'post_r13_ceiling',
    'pre_r13_avg', 'last5_avg', 'last5_std',
    'second_half_form', 'last5_vs_avg', 'post_r13_premium',
    # Tags
    'positive_tag_count', 'negative_tag_count', 'injury_tag_count',
    'star_tag_count', 'hot_tag_count', 'tagged_count', 'sub_count',
    # Multi-year
    'avg_2yr', 'games_2yr', 'std_2yr', 'two_yr_trend',
    # Interactions
    'games_pct', 'injury_risk', 'avg_x_durability', 'avg_x_consistency',
    # Position
    'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_RUC',
]

# ====================================================================
# TEST EACH APPROACH
# ====================================================================
logo = LeaveOneGroupOut()
groups = df['Year']

results = {}

for name, feat_cols in [
    ("A. Baseline (prev_avg only)", feature_cols_basic),
    ("B. Standard features (9 features)", feature_cols_standard),
    ("C. Rich features (35 features)", feature_cols_rich),
]:
    X = df[feat_cols].fillna(0)
    y = df['season_avg']

    if len(feat_cols) == 1:
        # Just use the raw value as prediction for baseline
        df[f'pred_{name[:1]}'] = df['prev_avg']
    else:
        preds = np.zeros(len(df))
        for train_idx, test_idx in logo.split(X, y, groups):
            model = XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                reg_alpha=0.5, reg_lambda=2.0, random_state=42,
                n_jobs=-1, tree_method='hist',
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx],
                      eval_set=[(X.iloc[test_idx], y.iloc[test_idx])], verbose=False)
            preds[test_idx] = model.predict(X.iloc[test_idx])
        df[f'pred_{name[:1]}'] = preds

    pred_col = f'pred_{name[:1]}'
    overall_s, _ = spearmanr(df[pred_col], df['season_avg'])
    overall_mae = mean_absolute_error(df['season_avg'], df[pred_col])

    year_results = {}
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        s, _ = spearmanr(yr[pred_col], yr['season_avg'])
        mae = mean_absolute_error(yr['season_avg'], yr[pred_col])
        o50 = set(yr.nlargest(50, pred_col)['Player ID'])
        a50 = set(yr.nlargest(50, 'season_avg')['Player ID'])
        overlap = len(o50 & a50)
        busts = yr.nlargest(30, pred_col)['big_decline'].sum()
        year_results[year] = {'spearman': s, 'mae': mae, 'top50': overlap, 'busts': busts}

    results[name] = {
        'overall_spearman': overall_s,
        'overall_mae': overall_mae,
        'years': year_results,
    }

# Print results
print("\n" + "=" * 100)
print("RESULTS: COMPARING FEATURE SETS")
print("=" * 100)

for name, res in results.items():
    print(f"\n--- {name} ---")
    print(f"  Overall: Spearman r={res['overall_spearman']:.3f}, MAE={res['overall_mae']:.1f}")
    for year, yr_res in res['years'].items():
        print(f"  {year}: r={yr_res['spearman']:.3f}, MAE={yr_res['mae']:.1f}, "
              f"Top-50={yr_res['top50']}/50 ({yr_res['top50']*2}%), Busts in top-30={yr_res['busts']}")

# Summary table
print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"\n  {'Approach':<45} {'Spearman':>8} {'MAE':>6} {'Avg Top-50':>10} {'Avg Busts':>10}")
print(f"  {'-'*80}")
for name, res in results.items():
    avg_top50 = np.mean([v['top50'] for v in res['years'].values()])
    avg_busts = np.mean([v['busts'] for v in res['years'].values()])
    print(f"  {name:<45} {res['overall_spearman']:>8.3f} {res['overall_mae']:>6.1f} "
          f"{avg_top50:>8.1f}/50 {avg_busts:>8.1f}/30")

# Feature importance for the rich model
if len(feature_cols_rich) > 1:
    # Train one final model on all data to get importances
    X_all = df[feature_cols_rich].fillna(0)
    y_all = df['season_avg']
    final_model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42,
        n_jobs=-1, tree_method='hist',
    )
    final_model.fit(X_all, y_all, verbose=False)

    importances = pd.Series(final_model.feature_importances_, index=feature_cols_rich).sort_values(ascending=False)
    print("\n--- Feature Importance (Rich Model) ---")
    for feat, imp in importances.items():
        if imp > 0.005:
            print(f"  {feat:<30}: {imp:.3f}")

# ====================================================================
# SPECIFIC FEATURE VALUE ANALYSIS
# ====================================================================
print("\n" + "=" * 100)
print("FEATURE VALUE ANALYSIS")
print("=" * 100)

# 1. Post-R13 avg vs full-season avg
print("\n--- Post-R13 Average as Predictor ---")
valid = df.dropna(subset=['post_r13_avg'])
s_post, _ = spearmanr(valid['post_r13_avg'], valid['season_avg'])
s_full, _ = spearmanr(valid['prev_avg'], valid['season_avg'])
print(f"  Post-R13 avg -> next season avg: Spearman r={s_post:.3f}")
print(f"  Full season avg -> next season avg: Spearman r={s_full:.3f}")
print(f"  Improvement: {s_post - s_full:+.3f}")

# Weighted combo
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    valid['combo'] = valid['prev_avg'] * (1 - w) + valid['post_r13_avg'] * w
    s, _ = spearmanr(valid['combo'], valid['season_avg'])
    print(f"  {int((1-w)*100)}% full + {int(w*100)}% post-R13: r={s:.3f}")

# 2. Tags as predictors
print("\n--- Tag Impact on Season Performance ---")
for tag_col, tag_name in [
    ('star_tag_count', 'Star tags'),
    ('hot_tag_count', 'Hot tags'),
    ('positive_tag_count', 'Any positive tag'),
    ('injury_tag_count', 'Injury tags'),
]:
    has_tag = df[df[tag_col] > 0]
    no_tag = df[df[tag_col] == 0]
    if len(has_tag) > 10:
        # Compare: among similar-avg players, do tagged players do better?
        has_avg = has_tag['avg_change'].mean()
        no_avg = no_tag['avg_change'].mean()
        print(f"  {tag_name}: {len(has_tag)} players, avg change: {has_avg:+.1f} (vs {no_avg:+.1f} without), delta: {has_avg - no_avg:+.1f}")

# 3. Injury tag specifically
print("\n--- Injury Tag Impact (controlling for prev_avg) ---")
for avg_bucket in [(80, 100), (100, 115), (115, 200)]:
    bucket = df[(df['prev_avg'] >= avg_bucket[0]) & (df['prev_avg'] < avg_bucket[1])]
    inj = bucket[bucket['injury_tag_count'] > 0]
    no_inj = bucket[bucket['injury_tag_count'] == 0]
    if len(inj) >= 5 and len(no_inj) >= 5:
        print(f"  Avg {avg_bucket[0]}-{avg_bucket[1]}: "
              f"Injured ({len(inj)} players): season avg {inj['season_avg'].mean():.1f}, "
              f"Healthy ({len(no_inj)} players): season avg {no_inj['season_avg'].mean():.1f}, "
              f"Gap: {inj['season_avg'].mean() - no_inj['season_avg'].mean():+.1f}")

# 4. Second half form
print("\n--- Second Half Form as Predictor ---")
for form_bucket in [(-20, -5, 'Fading'), (-5, 5, 'Steady'), (5, 20, 'Finishing strong'), (20, 100, 'Surging')]:
    lo, hi, label = form_bucket
    bucket = df[(df['second_half_form'] >= lo) & (df['second_half_form'] < hi)]
    if len(bucket) >= 10:
        print(f"  {label:<20} ({len(bucket):>3} players): avg next season change: {bucket['avg_change'].mean():+.1f}, "
              f"bust rate: {bucket['big_decline'].mean()*100:.1f}%")

# 5. Games played
print("\n--- Durability Impact (among prev_avg > 85) ---")
quality = df[df['prev_avg'] > 85]
for games_bucket in [(0, 15), (15, 18), (18, 21), (21, 24)]:
    bucket = quality[(quality['prev_games'] >= games_bucket[0]) & (quality['prev_games'] < games_bucket[1])]
    if len(bucket) >= 10:
        print(f"  {games_bucket[0]}-{games_bucket[1]} games ({len(bucket):>3} players): "
              f"season avg {bucket['season_avg'].mean():.1f}, "
              f"avg change: {bucket['avg_change'].mean():+.1f}, "
              f"bust rate: {bucket['big_decline'].mean()*100:.1f}%")
