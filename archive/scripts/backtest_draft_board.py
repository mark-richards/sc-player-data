"""
Backtest Draft Board Accuracy
==============================
Compares multiple approaches for ranking players pre-draft against actual season outcomes.
Uses leave-one-year-out validation across 2022-2025.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor

# Load all season comparisons
frames = []
for y in [2022, 2023, 2024, 2025]:
    f = pd.read_csv(f'data/draft/season_comparison_{y}.csv')
    frames.append(f)
all_sc = pd.concat(frames, ignore_index=True)

# Focus on players with pre-draft data who actually played (>=5 games)
df = all_sc.dropna(subset=['prev_avg', 'season_avg']).copy()
df = df[df['season_games'] >= 5].copy()
print(f"Backtesting pool: {len(df)} player-seasons with 5+ games")
print(f"By year: {df.groupby('Year').size().to_dict()}\n")

# Helper to evaluate a ranking
def evaluate(predicted_col, actual_col='season_avg', data=None):
    if data is None:
        data = df
    results = {}
    for year in sorted(data['Year'].unique()):
        yr = data[data['Year'] == year].dropna(subset=[predicted_col, actual_col])
        if len(yr) < 10:
            continue
        spear, _ = spearmanr(yr[predicted_col], yr[actual_col])
        our_top50 = set(yr.nlargest(50, predicted_col)['feed_id'])
        actual_top50 = set(yr.nlargest(50, actual_col)['feed_id'])
        overlap = len(our_top50 & actual_top50)
        results[year] = {'spearman': spear, 'top50_overlap': overlap}
    overall = data.dropna(subset=[predicted_col, actual_col])
    spear_all, _ = spearmanr(overall[predicted_col], overall[actual_col])
    results['overall'] = spear_all
    return results


# ==========================================================================
# APPROACH 1: Simple baseline (rank by prev_avg)
# ==========================================================================
print("=" * 80)
print("APPROACH 1: SIMPLE BASELINE (rank by prev_avg)")
print("=" * 80)
r1 = evaluate('prev_avg')
for year, v in r1.items():
    if year == 'overall':
        print(f"\n  OVERALL Spearman: {v:.3f}")
    else:
        print(f"  {year}: Spearman r={v['spearman']:.3f}, Top-50 overlap={v['top50_overlap']}/50 ({v['top50_overlap']*2}%)")


# ==========================================================================
# APPROACH 2: Current Draft_Value formula (replicated)
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 2: CURRENT DRAFT_VALUE FORMULA")
print("=" * 80)

def compute_draft_value(group):
    g = group.copy()
    g['durability_score'] = np.clip(g['prev_games'].fillna(0) / 23 * 100, 0, 100)
    g['consistency_score'] = np.clip(100 - g['prev_cv'].fillna(0.5) * 200, 0, 100)

    mn, mx = g['prev_avg'].min(), g['prev_avg'].max()
    g['avg_norm'] = (g['prev_avg'] - mn) / (mx - mn) * 100 if mx > mn else 50

    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        pos_mask = g['primary_pos'] == pos
        pos_p = g[pos_mask].sort_values('prev_avg', ascending=False)
        n = {'DEF': 40, 'MID': 56, 'FWD': 40, 'RUC': 16}.get(pos, 40)
        if len(pos_p) > n + 5:
            repl = pos_p.iloc[n:n + 10]['prev_avg'].mean()
        else:
            repl = pos_p['prev_avg'].mean() * 0.7 if len(pos_p) > 0 else 70
        g.loc[pos_mask, 'replacement_level'] = repl

    g['tpor_est'] = (g['prev_avg'].fillna(0) - g['replacement_level'].fillna(70)) * g['prev_games'].fillna(0)
    mn2, mx2 = g['tpor_est'].min(), g['tpor_est'].max()
    g['tpor_norm'] = (g['tpor_est'] - mn2) / (mx2 - mn2) * 100 if mx2 > mn2 else 50

    g['age_bonus'] = g['age_proxy'].apply(lambda x: 5 if 4 <= x <= 8 else 0)
    pos_adj = {'MID': 3, 'DEF': 3, 'FWD': -3, 'RUC': 0}
    g['pos_adj'] = g['primary_pos'].map(pos_adj).fillna(0)

    g['draft_value_v1'] = (
        g['avg_norm'] * 0.35 +
        g['tpor_norm'] * 0.20 +
        g['durability_score'] * 0.20 +
        g['consistency_score'] * 0.15 +
        g['age_bonus'] +
        g['pos_adj']
    )
    return g

df = df.groupby('Year', group_keys=False).apply(compute_draft_value)
r2 = evaluate('draft_value_v1')
for year, v in r2.items():
    if year == 'overall':
        print(f"\n  OVERALL Spearman: {v:.3f}")
    else:
        print(f"  {year}: Spearman r={v['spearman']:.3f}, Top-50 overlap={v['top50_overlap']}/50 ({v['top50_overlap']*2}%)")


# ==========================================================================
# APPROACH 3: Optimized weights via grid search
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 3: GRID SEARCH OPTIMAL WEIGHTS")
print("=" * 80)

best_spear = -1
best_weights = None

for w_avg in np.arange(0.20, 0.55, 0.05):
    for w_tpor in np.arange(0.05, 0.30, 0.05):
        for w_dur in np.arange(0.05, 0.35, 0.05):
            w_con = round(1.0 - w_avg - w_tpor - w_dur, 2)
            if w_con < 0.0 or w_con > 0.30:
                continue

            df['test_value'] = (
                df['avg_norm'] * w_avg +
                df['tpor_norm'] * w_tpor +
                df['durability_score'] * w_dur +
                df['consistency_score'] * w_con +
                df['age_bonus'] +
                df['pos_adj']
            )
            s, _ = spearmanr(df['test_value'], df['season_avg'])
            if s > best_spear:
                best_spear = s
                best_weights = (round(w_avg, 2), round(w_tpor, 2), round(w_dur, 2), round(w_con, 2))

print(f"  Best weights: avg={best_weights[0]}, tpor={best_weights[1]}, dur={best_weights[2]}, con={best_weights[3]}")
print(f"  Spearman: {best_spear:.3f}")

df['opt_value'] = (
    df['avg_norm'] * best_weights[0] +
    df['tpor_norm'] * best_weights[1] +
    df['durability_score'] * best_weights[2] +
    df['consistency_score'] * best_weights[3] +
    df['age_bonus'] +
    df['pos_adj']
)
r3 = evaluate('opt_value')
for year, v in r3.items():
    if year == 'overall':
        print(f"\n  OVERALL Spearman: {v:.3f}")
    else:
        print(f"  {year}: Spearman r={v['spearman']:.3f}, Top-50 overlap={v['top50_overlap']}/50 ({v['top50_overlap']*2}%)")


# ==========================================================================
# APPROACH 4: XGBoost on pre-draft features (leave-one-year-out)
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 4: XGBOOST RANKING MODEL (leave-one-year-out)")
print("=" * 80)

feature_cols = [
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'career_games', 'age_proxy',
    'prev_position_rank', 'prev_total'
]

for pos in ['DEF', 'MID', 'FWD', 'RUC']:
    df[f'pos_{pos}'] = (df['primary_pos'] == pos).astype(int)
    feature_cols.append(f'pos_{pos}')

# Interaction features
df['avg_x_games'] = df['prev_avg'] * df['prev_games']
df['avg_x_consistency'] = df['prev_avg'] * (1 - df['prev_cv'].fillna(0.5))
df['ceiling_minus_floor'] = df['prev_ceiling'] - df['prev_floor']
df['games_pct'] = df['prev_games'] / 23
feature_cols += ['avg_x_games', 'avg_x_consistency', 'ceiling_minus_floor', 'games_pct']

X = df[feature_cols].fillna(0)
y = df['season_avg']
groups = df['Year']

logo = LeaveOneGroupOut()
all_preds = np.zeros(len(df))

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        n_jobs=-1, tree_method='hist',
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y.iloc[test_idx])], verbose=False)
    all_preds[test_idx] = model.predict(X_test)

df['xgb_predicted'] = all_preds
r4 = evaluate('xgb_predicted')
for year, v in r4.items():
    if year == 'overall':
        print(f"\n  OVERALL Spearman: {v:.3f}, MAE: {mean_absolute_error(y, all_preds):.1f}")
    else:
        yr = df[df['Year'] == year]
        mae = mean_absolute_error(yr['season_avg'], yr['xgb_predicted'])
        print(f"  {year}: Spearman r={v['spearman']:.3f}, MAE={mae:.1f}, Top-50 overlap={v['top50_overlap']}/50 ({v['top50_overlap']*2}%)")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n  Feature importance (last fold):")
for feat, imp in importances.head(10).items():
    print(f"    {feat:<25}: {imp:.3f}")


# ==========================================================================
# APPROACH 5: XGBoost predicting TPOR
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 5: XGBOOST predicting TPOR (leave-one-year-out)")
print("=" * 80)

y_tpor = df['TPOR'].fillna(0)
all_preds_tpor = np.zeros(len(df))

for train_idx, test_idx in logo.split(X, y_tpor, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y_tpor.iloc[train_idx]

    model_tpor = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        random_state=42, n_jobs=-1, tree_method='hist',
    )
    model_tpor.fit(X_train, y_train, verbose=False)
    all_preds_tpor[test_idx] = model_tpor.predict(X_test)

df['xgb_tpor_pred'] = all_preds_tpor

for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    spear_avg, _ = spearmanr(yr['xgb_tpor_pred'], yr['season_avg'])
    spear_tpor, _ = spearmanr(yr['xgb_tpor_pred'], yr['TPOR'].fillna(0))
    our_top50 = set(yr.nlargest(50, 'xgb_tpor_pred')['feed_id'])
    actual_top50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
    overlap = len(our_top50 & actual_top50)
    print(f"  {year}: vs avg r={spear_avg:.3f}, vs TPOR r={spear_tpor:.3f}, Top-50 overlap={overlap}/50 ({overlap*2}%)")

overall_tpor_r, _ = spearmanr(df['xgb_tpor_pred'], df['season_avg'])
print(f"\n  OVERALL vs season_avg: Spearman r={overall_tpor_r:.3f}")


# ==========================================================================
# APPROACH 6: Blended (XGB + durability + consistency adjustments)
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 6: BLENDED (XGB + durability + consistency)")
print("=" * 80)

# Normalize XGB predictions per year
df['xgb_norm'] = 0.0
for year in [2022, 2023, 2024, 2025]:
    mask = df['Year'] == year
    yr = df.loc[mask, 'xgb_predicted']
    mn, mx = yr.min(), yr.max()
    if mx > mn:
        df.loc[mask, 'xgb_norm'] = (yr - mn) / (mx - mn) * 100

df['dur_score'] = np.clip(df['prev_games'].fillna(0) / 23 * 100, 0, 100)
df['con_score'] = np.clip(100 - df['prev_cv'].fillna(0.5) * 200, 0, 100)

best_blend_spear = -1
best_blend = None
for w_xgb in np.arange(0.50, 0.90, 0.05):
    for w_dur in np.arange(0.0, 0.30, 0.05):
        w_con = round(1.0 - w_xgb - w_dur, 2)
        if w_con < 0 or w_con > 0.25:
            continue
        df['blend_v'] = df['xgb_norm'] * w_xgb + df['dur_score'] * w_dur + df['con_score'] * w_con
        s, _ = spearmanr(df['blend_v'], df['season_avg'])
        if s > best_blend_spear:
            best_blend_spear = s
            best_blend = (round(w_xgb, 2), round(w_dur, 2), round(w_con, 2))

print(f"  Best blend: xgb={best_blend[0]}, dur={best_blend[1]}, con={best_blend[2]}")
print(f"  Spearman: {best_blend_spear:.3f}")

df['blend_value'] = df['xgb_norm'] * best_blend[0] + df['dur_score'] * best_blend[1] + df['con_score'] * best_blend[2]
r6 = evaluate('blend_value')
for year, v in r6.items():
    if year == 'overall':
        print(f"\n  OVERALL Spearman: {v:.3f}")
    else:
        print(f"  {year}: Spearman r={v['spearman']:.3f}, Top-50 overlap={v['top50_overlap']}/50 ({v['top50_overlap']*2}%)")


# ==========================================================================
# APPROACH 7: XGB predicting COMPOSITE outcome (avg * games / 23)
# ==========================================================================
print("\n" + "=" * 80)
print("APPROACH 7: XGBOOST predicting COMPOSITE (avg * games_pct)")
print("=" * 80)

# Composite outcome: rewards both high scoring AND durability
df['composite_outcome'] = df['season_avg'] * (df['season_games'] / 23)
y_comp = df['composite_outcome']
all_preds_comp = np.zeros(len(df))

for train_idx, test_idx in logo.split(X, y_comp, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y_comp.iloc[train_idx]

    model_comp = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        random_state=42, n_jobs=-1, tree_method='hist',
    )
    model_comp.fit(X_train, y_train, verbose=False)
    all_preds_comp[test_idx] = model_comp.predict(X_test)

df['xgb_composite'] = all_preds_comp

for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year]
    spear_avg, _ = spearmanr(yr['xgb_composite'], yr['season_avg'])
    spear_comp, _ = spearmanr(yr['xgb_composite'], yr['composite_outcome'])
    our_top50 = set(yr.nlargest(50, 'xgb_composite')['feed_id'])
    actual_top50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
    overlap = len(our_top50 & actual_top50)
    print(f"  {year}: vs avg r={spear_avg:.3f}, vs composite r={spear_comp:.3f}, Top-50 overlap={overlap}/50 ({overlap*2}%)")

overall_comp, _ = spearmanr(df['xgb_composite'], df['season_avg'])
print(f"\n  OVERALL vs season_avg: Spearman r={overall_comp:.3f}")


# ==========================================================================
# SUMMARY COMPARISON TABLE
# ==========================================================================
print("\n" + "=" * 80)
print("SUMMARY: ALL APPROACHES COMPARED")
print("=" * 80)

approaches = [
    ("1. Baseline (prev_avg only)", r1['overall']),
    ("2. Current Draft_Value formula", r2['overall']),
    ("3. Optimized weights (grid search)", r3['overall']),
    ("4. XGBoost -> season_avg", r4['overall']),
    ("5. XGBoost -> TPOR", overall_tpor_r),
    ("6. Blended (XGB + dur + con)", r6['overall']),
    ("7. XGBoost -> composite (avg*games%)", overall_comp),
]

print(f"\n  {'Approach':<45} {'Spearman r':>10}")
print(f"  {'-' * 57}")
for name, val in sorted(approaches, key=lambda x: x[1], reverse=True):
    marker = " <-- BEST" if val == max(v for _, v in approaches) else ""
    print(f"  {name:<45} {val:>10.3f}{marker}")

# Top-50 overlap comparison
print(f"\n  {'Approach':<45} {'Avg Top-50 Overlap':>18}")
print(f"  {'-' * 65}")

for name, pred_col in [
    ("1. Baseline (prev_avg)", 'prev_avg'),
    ("2. Current Draft_Value", 'draft_value_v1'),
    ("3. Optimized weights", 'opt_value'),
    ("4. XGBoost -> avg", 'xgb_predicted'),
    ("6. Blended", 'blend_value'),
    ("7. XGBoost -> composite", 'xgb_composite'),
]:
    overlaps = []
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        our50 = set(yr.nlargest(50, pred_col)['feed_id'])
        act50 = set(yr.nlargest(50, 'season_avg')['feed_id'])
        overlaps.append(len(our50 & act50))
    avg_overlap = np.mean(overlaps)
    print(f"  {name:<45} {avg_overlap:>10.1f}/50 ({avg_overlap*2:.0f}%)")
