"""
Deep backtesting: Can we beat prev_avg as a draft ranking?
Focus on bust avoidance, regression to mean, and top-tier accuracy.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut

# Load data
frames = []
for y in [2022, 2023, 2024, 2025]:
    f = pd.read_csv(f'data/draft/season_comparison_{y}.csv')
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=['prev_avg', 'season_avg']).copy()
df = df[df['season_games'] >= 5].copy()
df['declined'] = (df['avg_change'] < -10).astype(int)
df['big_decline'] = (df['avg_change'] < -15).astype(int)

print("=" * 80)
print("DEEPER ANALYSIS: WHERE CAN WE BEAT PREV_AVG?")
print("=" * 80)

# ---- 1. TOP-TIER ACCURACY ----
print("\n--- 1. Top-100 Players Only (draft-relevant tier) ---")
for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    top100 = yr.nlargest(100, 'prev_avg')
    spear, _ = spearmanr(top100['prev_avg'], top100['season_avg'])
    top30_ours = set(top100.nlargest(30, 'prev_avg')['feed_id'])
    top30_actual = set(yr.nlargest(30, 'season_avg')['feed_id'])
    overlap = len(top30_ours & top30_actual)
    print(f"  {year}: Top-100 Spearman r={spear:.3f}, Top-30 overlap={overlap}/30 ({overlap/30*100:.0f}%)")

# ---- 2. BUST ANALYSIS ----
print("\n--- 2. Bust Analysis (top-100 by prev_avg) ---")
top_players = pd.DataFrame()
for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    top100 = yr.nlargest(100, 'prev_avg')
    top_players = pd.concat([top_players, top100])

print(f"  Total top-100 across 4 years: {len(top_players)}")
print(f"  Busts (>15 pt decline): {top_players['big_decline'].sum()} ({top_players['big_decline'].mean()*100:.1f}%)")

bust_features = ['prev_games', 'prev_cv', 'prev_std', 'career_games', 'age_proxy',
                 'prev_ceiling', 'prev_floor', 'prev_avg', 'prev_position_rank']
print(f"\n  {'Feature':<25} {'Non-bust':>10} {'Bust':>10} {'Signal':>10}")
print(f"  {'-'*55}")
for feat in bust_features:
    non_bust = top_players[top_players['big_decline'] == 0][feat].mean()
    bust = top_players[top_players['big_decline'] == 1][feat].mean()
    diff = bust - non_bust
    print(f"  {feat:<25} {non_bust:>10.1f} {bust:>10.1f} {diff:>+10.1f}")

# ---- 3. REGRESSION TO THE MEAN ----
print("\n--- 3. Regression-Adjusted Predictions ---")
pop_mean = df.groupby('Year')['prev_avg'].transform('mean')
for shrinkage in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
    df['regressed'] = df['prev_avg'] * shrinkage + pop_mean * (1 - shrinkage)
    s, _ = spearmanr(df['regressed'], df['season_avg'])
    overlaps = []
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        o50 = set(yr.nlargest(50, 'regressed')['feed_id'])
        a50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
        overlaps.append(len(o50 & a50))
    avg_o = np.mean(overlaps)
    marker = " <-- baseline" if shrinkage == 1.0 else ""
    print(f"  Shrinkage {shrinkage:.2f}: Spearman r={s:.3f}, Avg Top-50={avg_o:.1f}/50{marker}")

# ---- 4. DURABILITY-ADJUSTED ----
print("\n--- 4. Durability-Adjusted ---")
for dur_penalty in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    games_pct = df['prev_games'].fillna(0) / 23
    df['dur_adj'] = df['prev_avg'] - dur_penalty * (1 - games_pct) * 20
    s, _ = spearmanr(df['dur_adj'], df['season_avg'])
    overlaps = []
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        o50 = set(yr.nlargest(50, 'dur_adj')['feed_id'])
        a50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
        overlaps.append(len(o50 & a50))
    avg_o = np.mean(overlaps)
    print(f"  Penalty {dur_penalty:.1f}: Spearman r={s:.3f}, Avg Top-50={avg_o:.1f}/50")

# ---- 5. BEST COMBINED APPROACH ----
print("\n--- 5. Best Combined (regression + durability + consistency) ---")
best_s = -1
best_params = None

for shrink in [0.80, 0.85, 0.90, 0.95]:
    for dur_p in [0, 0.5, 1.0, 1.5]:
        for cv_p in [0, 5, 10, 15, 20]:
            pop_m = df.groupby('Year')['prev_avg'].transform('mean')
            regressed = df['prev_avg'] * shrink + pop_m * (1 - shrink)
            gp = df['prev_games'].fillna(0) / 23
            dur = dur_p * (1 - gp) * 20
            cv = cv_p * (df['prev_cv'].fillna(0.4) - 0.3)
            score = regressed - dur - cv
            s, _ = spearmanr(score, df['season_avg'])
            if s > best_s:
                best_s = s
                best_params = (shrink, dur_p, cv_p)

print(f"  Best params: shrinkage={best_params[0]}, dur_penalty={best_params[1]}, cv_penalty={best_params[2]}")
print(f"  Spearman: {best_s:.3f} (vs baseline 0.750)")

pop_m = df.groupby('Year')['prev_avg'].transform('mean')
regressed = df['prev_avg'] * best_params[0] + pop_m * (1 - best_params[0])
gp = df['prev_games'].fillna(0) / 23
dur = best_params[1] * (1 - gp) * 20
cv = best_params[2] * (df['prev_cv'].fillna(0.4) - 0.3)
df['best_combined'] = regressed - dur - cv

for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    s, _ = spearmanr(yr['best_combined'], yr['season_avg'])
    o50 = set(yr.nlargest(50, 'best_combined')['feed_id'])
    a50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
    overlap = len(o50 & a50)
    busts = yr.nlargest(30, 'best_combined')['big_decline'].sum()
    baseline_busts = yr.nlargest(30, 'prev_avg')['big_decline'].sum()
    print(f"  {year}: r={s:.3f}, Top-50={overlap}/50, Busts in top-30: {busts} (baseline: {baseline_busts})")

# ---- 6. XGBOOST WITH BUST-RISK FEATURES ----
print("\n--- 6. XGBoost v2 (enhanced features) ---")

feature_cols = [
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'career_games', 'age_proxy', 'prev_position_rank',
    'prev_total',
]

df['games_pct_f'] = df['prev_games'] / 23
df['avg_x_games_pct'] = df['prev_avg'] * df['games_pct_f']
df['avg_x_consistency'] = df['prev_avg'] * (1 - df['prev_cv'].fillna(0.5))
df['ceiling_floor_range'] = df['prev_ceiling'] - df['prev_floor']
df['avg_vs_median'] = df['prev_avg'] - df['prev_median']
df['avg_vs_ceiling'] = df['prev_avg'] / df['prev_ceiling'].replace(0, 1)
df['floor_ratio'] = df['prev_floor'] / df['prev_avg'].replace(0, 1)
df['regression_est'] = df['prev_avg'] * 0.85 + df.groupby('Year')['prev_avg'].transform('mean') * 0.15

for pos in ['DEF', 'MID', 'FWD', 'RUC']:
    df[f'pos_{pos}'] = (df['primary_pos'] == pos).astype(int)

all_features = feature_cols + [
    'games_pct_f', 'avg_x_games_pct', 'avg_x_consistency',
    'ceiling_floor_range', 'avg_vs_median', 'avg_vs_ceiling',
    'floor_ratio', 'regression_est',
    'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_RUC'
]

X = df[all_features].fillna(0)
y = df['season_avg']
groups = df['Year']

logo = LeaveOneGroupOut()
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

df['xgb_v2'] = preds

overall_s, _ = spearmanr(df['xgb_v2'], df['season_avg'])
print(f"  OVERALL Spearman: {overall_s:.3f}")

for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    s, _ = spearmanr(yr['xgb_v2'], yr['season_avg'])
    o50 = set(yr.nlargest(50, 'xgb_v2')['feed_id'])
    a50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
    overlap = len(o50 & a50)
    busts = yr.nlargest(30, 'xgb_v2')['big_decline'].sum()
    print(f"  {year}: r={s:.3f}, Top-50={overlap}/50, Busts in top-30: {busts}")

importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
print("\n  Top features:")
for feat, imp in importances.head(12).items():
    print(f"    {feat:<25}: {imp:.3f}")

# ---- FINAL SUMMARY ----
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print(f"\n  {'Approach':<40} {'Spearman':>8} {'Avg Top-50':>10} {'Avg Busts/30':>14}")
print(f"  {'-'*75}")

for name, col in [
    ("Baseline (prev_avg)", 'prev_avg'),
    ("Best Combined (regr+dur+cv)", 'best_combined'),
    ("XGBoost v2 (bust features)", 'xgb_v2'),
]:
    s, _ = spearmanr(df[col], df['season_avg'])
    overlaps = []
    busts_list = []
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        o50 = set(yr.nlargest(50, col)['feed_id'])
        a50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
        overlaps.append(len(o50 & a50))
        busts_list.append(yr.nlargest(30, col)['big_decline'].sum())
    print(f"  {name:<40} {s:>8.3f} {np.mean(overlaps):>8.1f}/50 {np.mean(busts_list):>12.1f}/30")

# ---- WHERE XGB ADDS VALUE: Head-to-head ----
print("\n--- Head-to-Head: Where XGB beats prev_avg ---")
for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]

    # Players XGB ranks higher than prev_avg AND who actually did better
    yr = yr.copy()
    yr['prev_rank'] = yr['prev_avg'].rank(ascending=False)
    yr['xgb_rank'] = yr['xgb_v2'].rank(ascending=False)
    yr['actual_rank'] = yr['season_avg'].rank(ascending=False)

    # XGB promoted (ranked higher than prev_avg) and correct
    promoted = yr[yr['xgb_rank'] < yr['prev_rank'] - 10]  # XGB ranked 10+ places higher
    promoted_correct = promoted[promoted['actual_rank'] < promoted['prev_rank']]  # actually improved

    # XGB demoted (ranked lower) and correct
    demoted = yr[yr['xgb_rank'] > yr['prev_rank'] + 10]
    demoted_correct = demoted[demoted['actual_rank'] > demoted['prev_rank']]

    print(f"  {year}: Promoted {len(promoted)} players ({len(promoted_correct)} correct, {len(promoted_correct)/max(len(promoted),1)*100:.0f}%)")
    print(f"         Demoted  {len(demoted)} players ({len(demoted_correct)} correct, {len(demoted_correct)/max(len(demoted),1)*100:.0f}%)")
