"""
Model Review Experiment Script
================================
Systematic comparison of multiple models, feature sets, and targets.
Temporal expanding-window CV (2021-2025) - identical to train_model.py baseline.

Results feed into reports/model_review_2026.md
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import lightgbm as lgb

from feature_engineering import engineer_features, get_feature_columns

# ── Config ──────────────────────────────────────────────────────────────────

PROCESSED_DATA_DIR = Path("data/processed")
MASTER_FILE = PROCESSED_DATA_DIR / "master_player_data.csv"
CV_YEARS = list(range(2021, 2026))

YEAR_WEIGHTS = {2025: 3.0, 2024: 2.5, 2023: 2.0, 2022: 1.5, 2021: 1.0}
DEFAULT_WEIGHT = 0.5

RESULTS = {}  # {experiment_name: {mae, rmse, r2, spearman, ...}}


def get_sample_weights(years):
    return np.array([YEAR_WEIGHTS.get(int(y), DEFAULT_WEIGHT) for y in years])


def cv_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    sp, _ = spearmanr(y_true, y_pred)
    return dict(mae=mae, rmse=rmse, r2=r2, spearman=sp)


def brier_improvement(y_pred, y_true, threshold):
    """Fit isotonic calibrator on OOF and return Brier improvement %."""
    iso = IsotonicRegression(out_of_bounds='clip')
    binary = (y_true >= threshold).astype(float)
    iso.fit(y_pred, binary)
    probs = iso.predict(y_pred)
    brier = float(np.mean((probs - binary) ** 2))
    naive = float(np.mean(binary) * (1 - np.mean(binary)))
    return (1 - brier / naive) * 100 if naive > 0 else 0.0, brier


def score_bucket_bias(y_true, y_pred):
    """Return mean bias per score bucket."""
    residuals = y_true - y_pred
    buckets = [(0, 60, '<60'), (60, 80, '60-80'), (80, 100, '80-100'),
               (100, 120, '100-120'), (120, 999, '120+')]
    result = {}
    for lo, hi, label in buckets:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            result[label] = {'bias': float(np.mean(residuals[mask])), 'n': int(mask.sum())}
    return result


def run_expanding_window_cv(model_factory, df, feature_cols, target='SC',
                             use_weights=True, transform=None, inverse=None):
    """
    Expanding window CV across CV_YEARS.
    model_factory(): callable that returns a fresh unfitted model.
    transform: optional function to apply to y_train before fitting (e.g. log)
    inverse: inverse transform to apply to predictions
    Returns: dict with aggregate metrics + OOF arrays
    """
    all_preds, all_actuals = [], []
    year_results = []

    for val_year in CV_YEARS:
        train_df = df[df['Year'] < val_year].dropna(subset=[target])
        val_df = df[df['Year'] == val_year].dropna(subset=[target])

        if val_df.empty or train_df.empty:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target].values
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target].values

        if transform is not None:
            y_fit = transform(y_train)
        else:
            y_fit = y_train

        model = model_factory()
        if use_weights:
            w = get_sample_weights(train_df['Year'])
            try:
                model.fit(X_train, y_fit, sample_weight=w)
            except TypeError:
                model.fit(X_train, y_fit)
        else:
            model.fit(X_train, y_fit)

        raw_preds = model.predict(X_val)
        if inverse is not None:
            preds = inverse(raw_preds)
        else:
            preds = raw_preds

        m = cv_metrics(y_val, preds)
        year_results.append({'year': val_year, **m})

        all_preds.extend(preds.tolist())
        all_actuals.extend(y_val.tolist())

    oof_preds = np.array(all_preds)
    oof_actuals = np.array(all_actuals)

    agg = cv_metrics(oof_actuals, oof_preds)
    agg['year_results'] = year_results
    agg['oof_preds'] = oof_preds
    agg['oof_actuals'] = oof_actuals
    agg['bucket_bias'] = score_bucket_bias(oof_actuals, oof_preds)

    # Probability calibration quality
    for thr in [80, 100, 120]:
        impr, brier = brier_improvement(oof_preds, oof_actuals, thr)
        agg[f'p{thr}_brier_improvement'] = round(impr, 1)
        agg[f'p{thr}_brier'] = round(brier, 4)

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading and engineering features...")
master_df = pd.read_csv(MASTER_FILE, low_memory=False)
featured_df = engineer_features(master_df, for_training=True)

# Filter: drop rows where all rolling features are 0 (first game)
rolling_cols = [c for c in get_feature_columns() if '_rolling_avg_' in c]
model_df = featured_df[featured_df[rolling_cols].abs().sum(axis=1) > 0].copy()
model_df = model_df.dropna(subset=['SC'])

BASE_FEATURES = get_feature_columns()
print(f"Model dataset: {len(model_df):,} rows | {len(BASE_FEATURES)} features | "
      f"SC mean={model_df['SC'].mean():.1f}, std={model_df['SC'].std():.1f}")

print("\n" + "="*70)
print("EXPERIMENT 1: XGBoost BASELINE (current model)")
print("="*70)

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'verbosity': 0,
}

def make_xgb(): return XGBRegressor(**XGB_PARAMS)

r_xgb = run_expanding_window_cv(make_xgb, model_df, BASE_FEATURES)
RESULTS['xgb_baseline'] = r_xgb
print(f"  MAE={r_xgb['mae']:.3f} | RMSE={r_xgb['rmse']:.3f} | R2={r_xgb['r2']:.4f} | Spearman={r_xgb['spearman']:.4f}")
print(f"  P(80+) Brier improvement: {r_xgb['p80_brier_improvement']:.1f}%")
print(f"  P(100+) Brier improvement: {r_xgb['p100_brier_improvement']:.1f}%")
print(f"  P(120+) Brier improvement: {r_xgb['p120_brier_improvement']:.1f}%")
print(f"  Bucket biases: {r_xgb['bucket_bias']}")

print("\n" + "="*70)
print("EXPERIMENT 2: LightGBM")
print("="*70)

LGB_PARAMS = {
    'objective': 'regression',
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

def make_lgb(): return lgb.LGBMRegressor(**LGB_PARAMS)

r_lgb = run_expanding_window_cv(make_lgb, model_df, BASE_FEATURES)
RESULTS['lightgbm'] = r_lgb
delta_mae = r_lgb['mae'] - r_xgb['mae']
print(f"  MAE={r_lgb['mae']:.3f} | RMSE={r_lgb['rmse']:.3f} | R2={r_lgb['r2']:.4f} | Spearman={r_lgb['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+) Brier improvement: {r_lgb['p80_brier_improvement']:.1f}%")
print(f"  Bucket biases: {r_lgb['bucket_bias']}")

# ── LightGBM feature importances ───────────────────────────────────────────
print("\n  Training LightGBM on full data for feature importance...")
full_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
full_lgb.fit(model_df[BASE_FEATURES].fillna(0), model_df['SC'],
             sample_weight=get_sample_weights(model_df['Year']))
lgb_imp = pd.Series(full_lgb.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)
print("  LightGBM Top 20:")
for fname, imp in lgb_imp.head(20).items():
    print(f"    {fname}: {imp}")

print("\n" + "="*70)
print("EXPERIMENT 3: Random Forest")
print("="*70)

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_leaf': 20,
    'max_features': 0.5,
    'random_state': 42,
    'n_jobs': -1,
}

def make_rf(): return RandomForestRegressor(**RF_PARAMS)

r_rf = run_expanding_window_cv(make_rf, model_df, BASE_FEATURES)
RESULTS['random_forest'] = r_rf
delta_mae = r_rf['mae'] - r_xgb['mae']
print(f"  MAE={r_rf['mae']:.3f} | RMSE={r_rf['rmse']:.3f} | R2={r_rf['r2']:.4f} | Spearman={r_rf['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+) Brier improvement: {r_rf['p80_brier_improvement']:.1f}%")

print("\n" + "="*70)
print("EXPERIMENT 4: Ridge Regression")
print("="*70)

def make_ridge():
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=100.0))])

r_ridge = run_expanding_window_cv(make_ridge, model_df, BASE_FEATURES, use_weights=False)
RESULTS['ridge'] = r_ridge
delta_mae = r_ridge['mae'] - r_xgb['mae']
print(f"  MAE={r_ridge['mae']:.3f} | RMSE={r_ridge['rmse']:.3f} | R2={r_ridge['r2']:.4f} | Spearman={r_ridge['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+) Brier improvement: {r_ridge['p80_brier_improvement']:.1f}%")

print("\n" + "="*70)
print("EXPERIMENT 5: ElasticNet")
print("="*70)

def make_elastic():
    return Pipeline([('scaler', StandardScaler()), ('en', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000))])

r_en = run_expanding_window_cv(make_elastic, model_df, BASE_FEATURES, use_weights=False)
RESULTS['elasticnet'] = r_en
delta_mae = r_en['mae'] - r_xgb['mae']
print(f"  MAE={r_en['mae']:.3f} | RMSE={r_en['rmse']:.3f} | R2={r_en['r2']:.4f} | Spearman={r_en['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")

print("\n" + "="*70)
print("EXPERIMENT 6: XGBoost with Log-Transformed Target")
print("="*70)

# Shift by +20 to handle negative SC scores before log
SCORE_SHIFT = 20

def log_transform(y):
    return np.log1p(np.maximum(y + SCORE_SHIFT, 0))

def log_inverse(yhat):
    return np.expm1(yhat) - SCORE_SHIFT

r_log = run_expanding_window_cv(make_xgb, model_df, BASE_FEATURES,
                                 transform=log_transform, inverse=log_inverse)
RESULTS['xgb_log_target'] = r_log
delta_mae = r_log['mae'] - r_xgb['mae']
print(f"  MAE={r_log['mae']:.3f} | RMSE={r_log['rmse']:.3f} | R2={r_log['r2']:.4f} | Spearman={r_log['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+) Brier improvement: {r_log['p80_brier_improvement']:.1f}%")
print(f"  P(120+) Brier improvement: {r_log['p120_brier_improvement']:.1f}%")
print(f"  Bucket biases (log model): {r_log['bucket_bias']}")

print("\n" + "="*70)
print("EXPERIMENT 7: XGBoost with Winsorised Target (cap at 155)")
print("="*70)

WINSOR_CAP = 155

def winsorise(y):
    return np.clip(y, None, WINSOR_CAP)

r_win = run_expanding_window_cv(make_xgb, model_df, BASE_FEATURES,
                                 transform=winsorise)
RESULTS['xgb_winsorised'] = r_win
delta_mae = r_win['mae'] - r_xgb['mae']
print(f"  MAE={r_win['mae']:.3f} | RMSE={r_win['rmse']:.3f} | R2={r_win['r2']:.4f} | Spearman={r_win['spearman']:.4f}")
print(f"  vs XGBoost: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+): {r_win['p80_brier_improvement']:.1f}%  P(120+): {r_win['p120_brier_improvement']:.1f}%")

print("\n" + "="*70)
print("EXPERIMENT 8: LightGBM with Extended Features")
print("="*70)

# Additional features not in current set
extra_features_to_add = []

def add_extended_features(df):
    """Add features identified as missing from the current set."""
    df = df.copy()

    grouped = df.groupby('Player ID')

    # EMA of SC (more weight to recent games)
    df['SC_ewm_3'] = grouped['SC'].transform(
        lambda x: x.shift(1).ewm(span=3, min_periods=1).mean()
    )
    df['SC_ewm_6'] = grouped['SC'].transform(
        lambda x: x.shift(1).ewm(span=6, min_periods=1).mean()
    )

    # EMA-based trend: short EMA - long EMA (MACD-style)
    df['SC_ema_trend'] = df['SC_ewm_3'] - df['SC_ewm_6']

    # Linear score trend over last 8 games (slope)
    def rolling_slope(x, n=8):
        def slope(vals):
            vals = vals.dropna()
            if len(vals) < 3:
                return 0.0
            idx = np.arange(len(vals))
            return float(np.polyfit(idx, vals, 1)[0])
        return x.shift(1).rolling(n, min_periods=3).apply(slope, raw=False)

    df['SC_trend_slope_8'] = grouped['SC'].transform(rolling_slope)

    # Skewness of last 10 SC scores (positive = occasional big scores)
    from scipy.stats import skew as scipy_skew
    def rolling_skew(x, n=10):
        def _skew(vals):
            vals = vals.dropna()
            if len(vals) < 4:
                return 0.0
            return float(scipy_skew(vals))
        return x.shift(1).rolling(n, min_periods=4).apply(_skew, raw=False)

    df['SC_skewness_10'] = grouped['SC'].transform(rolling_skew)
    df['SC_skewness_10'] = df['SC_skewness_10'].fillna(0)

    # Score variance/std of last 6 (different from consistency_last_10)
    df['SC_std_6'] = grouped['SC'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=2).std()
    ).fillna(0)

    # TOG% × SC avg (adjusted scoring rate per minute on ground)
    tog = 'Time on ground'
    if tog in df.columns:
        df['sc_x_tog_6'] = (
            df['SC_rolling_avg_6_games'] *
            df[tog].fillna(75) / 100.0  # TOG% → 0-1
        )

    # Price change momentum: cumulative price changes over 3 rounds
    df['price_change_sum3'] = grouped['price_change'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0)

    # Kick efficiency: kicks × disposal_eff (quality-adjusted kicking)
    disp_eff = 'Disposal efficiency'
    if disp_eff in df.columns:
        df['kick_quality'] = (
            df['Kicks_rolling_avg_6_games'] *
            df[disp_eff].fillna(70) / 100.0
        )

    # Hitout effectiveness: hitout * clearances (ruck dominance proxy)
    df['ruck_dominance'] = (
        df['Hitouts_rolling_avg_6_games'] *
        df.get('Clearances_rolling_avg_6_games', pd.Series(0, index=df.index))
    ).fillna(0)

    # Position-encoded as numeric (DEF=0, MID=1, RUC=2, FWD=3)
    pos_map = {'DEF': 0, 'MID': 1, 'RUCK': 2, 'FWD': 3, 'Unknown': 1}
    if 'simple_pos' not in df.columns:
        from feature_engineering import simplify_position
        df['simple_pos'] = df['ff_position'].apply(simplify_position)
    df['pos_numeric'] = df['simple_pos'].map(pos_map).fillna(1)

    return df

print("  Building extended features...")
extended_df = add_extended_features(model_df)

new_features = ['SC_ewm_3', 'SC_ewm_6', 'SC_ema_trend', 'SC_trend_slope_8',
                'SC_skewness_10', 'SC_std_6', 'sc_x_tog_6', 'price_change_sum3',
                'kick_quality', 'ruck_dominance', 'pos_numeric']
available_new = [f for f in new_features if f in extended_df.columns]
EXT_FEATURES = BASE_FEATURES + available_new
print(f"  Extended feature set: {len(EXT_FEATURES)} features (+{len(available_new)} new)")

def make_lgb_ext(): return lgb.LGBMRegressor(**LGB_PARAMS)

r_lgb_ext = run_expanding_window_cv(make_lgb_ext, extended_df, EXT_FEATURES)
RESULTS['lgb_extended_features'] = r_lgb_ext
delta_mae = r_lgb_ext['mae'] - r_xgb['mae']
print(f"  MAE={r_lgb_ext['mae']:.3f} | RMSE={r_lgb_ext['rmse']:.3f} | R2={r_lgb_ext['r2']:.4f} | Spearman={r_lgb_ext['spearman']:.4f}")
print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+): {r_lgb_ext['p80_brier_improvement']:.1f}%  P(120+): {r_lgb_ext['p120_brier_improvement']:.1f}%")

# Feature importance for extended model
full_lgb_ext = lgb.LGBMRegressor(**LGB_PARAMS)
full_lgb_ext.fit(extended_df[EXT_FEATURES].fillna(0), extended_df['SC'],
                  sample_weight=get_sample_weights(extended_df['Year']))
lgb_ext_imp = pd.Series(full_lgb_ext.feature_importances_, index=EXT_FEATURES).sort_values(ascending=False)
print("  Extended model Top 20:")
for fname, imp in lgb_ext_imp.head(20).items():
    print(f"    {fname}: {imp}")
print("  New features importance:")
for f in available_new:
    rank = list(lgb_ext_imp.index).index(f) + 1
    print(f"    {f}: {lgb_ext_imp[f]} (rank {rank}/{len(EXT_FEATURES)})")


print("\n" + "="*70)
print("EXPERIMENT 9: Separate Models Per Position (LightGBM)")
print("="*70)

from feature_engineering import simplify_position
if 'simple_pos' not in model_df.columns:
    model_df['simple_pos'] = model_df['ff_position'].apply(simplify_position)

all_pos_preds, all_pos_actuals = [], []
pos_results = {}

for pos in ['MID', 'DEF', 'FWD', 'RUCK']:
    pos_df = model_df[model_df['simple_pos'] == pos].copy()
    if len(pos_df) < 500:
        print(f"  {pos}: too few rows ({len(pos_df)}), skipping")
        continue

    yr_results = []
    pos_preds_all, pos_actuals_all = [], []

    for val_year in CV_YEARS:
        train_df = pos_df[pos_df['Year'] < val_year].dropna(subset=['SC'])
        val_df = pos_df[pos_df['Year'] == val_year].dropna(subset=['SC'])
        if val_df.empty or len(train_df) < 100:
            continue

        X_train = train_df[BASE_FEATURES].fillna(0)
        y_train = train_df['SC'].values
        X_val = val_df[BASE_FEATURES].fillna(0)
        y_val = val_df['SC'].values
        w = get_sample_weights(train_df['Year'])

        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X_train, y_train, sample_weight=w)
        preds = m.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        yr_results.append({'year': val_year, 'mae': mae, 'n': len(y_val)})
        pos_preds_all.extend(preds.tolist())
        pos_actuals_all.extend(y_val.tolist())

    if pos_preds_all:
        pos_mae = mean_absolute_error(pos_actuals_all, pos_preds_all)
        pos_sp, _ = spearmanr(pos_actuals_all, pos_preds_all)
        print(f"  {pos}: MAE={pos_mae:.3f}, Spearman={pos_sp:.4f}, n={len(pos_preds_all)}")
        pos_results[pos] = {'mae': pos_mae, 'spearman': pos_sp}
        all_pos_preds.extend(pos_preds_all)
        all_pos_actuals.extend(pos_actuals_all)

if all_pos_preds:
    combined_mae = mean_absolute_error(all_pos_actuals, all_pos_preds)
    combined_sp, _ = spearmanr(all_pos_actuals, all_pos_preds)
    RESULTS['per_position_lgb'] = {'mae': combined_mae, 'spearman': combined_sp,
                                    'per_pos': pos_results,
                                    'oof_preds': np.array(all_pos_preds),
                                    'oof_actuals': np.array(all_pos_actuals)}
    delta_mae = combined_mae - r_xgb['mae']
    print(f"\n  COMBINED: MAE={combined_mae:.3f}, Spearman={combined_sp:.4f}")
    print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")


print("\n" + "="*70)
print("EXPERIMENT 10: Stacked Ensemble (XGBoost + LightGBM + Ridge)")
print("="*70)

# Two-layer stacking with out-of-fold meta-learning
# Layer 1: XGBoost + LightGBM (temporal OOF)
# Layer 2: Ridge meta-learner on OOF predictions

all_meta_preds_l1_xgb, all_meta_preds_l1_lgb = [], []
all_meta_actuals = []
all_meta_years = []

for val_year in CV_YEARS:
    train_df = model_df[model_df['Year'] < val_year].dropna(subset=['SC'])
    val_df = model_df[model_df['Year'] == val_year].dropna(subset=['SC'])
    if val_df.empty or train_df.empty:
        continue

    X_train = train_df[BASE_FEATURES].fillna(0)
    y_train = train_df['SC'].values
    X_val = val_df[BASE_FEATURES].fillna(0)
    y_val = val_df['SC'].values
    w = get_sample_weights(train_df['Year'])

    m_xgb = make_xgb()
    m_xgb.fit(X_train, y_train, sample_weight=w)
    p_xgb = m_xgb.predict(X_val)

    m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
    m_lgb.fit(X_train, y_train, sample_weight=w)
    p_lgb = m_lgb.predict(X_val)

    all_meta_preds_l1_xgb.extend(p_xgb.tolist())
    all_meta_preds_l1_lgb.extend(p_lgb.tolist())
    all_meta_actuals.extend(y_val.tolist())
    all_meta_years.extend(val_df['Year'].tolist())

# Meta-features: XGBoost OOF + LightGBM OOF
meta_X = np.column_stack([all_meta_preds_l1_xgb, all_meta_preds_l1_lgb])
meta_y = np.array(all_meta_actuals)
meta_years = np.array(all_meta_years)

# Evaluate meta-learner with CV on OOF data (temporally nested)
stack_preds_oof = []
for val_year in CV_YEARS:
    train_mask = meta_years < val_year
    val_mask = meta_years == val_year
    if val_mask.sum() == 0 or train_mask.sum() < 50:
        continue

    meta_scaler = StandardScaler()
    meta_ridge = Ridge(alpha=10.0)
    X_meta_train = meta_scaler.fit_transform(meta_X[train_mask])
    X_meta_val = meta_scaler.transform(meta_X[val_mask])
    meta_ridge.fit(X_meta_train, meta_y[train_mask])
    stack_preds_oof.extend(meta_ridge.predict(X_meta_val).tolist())

if len(stack_preds_oof) > 0:
    # Align with actuals (skip year 2021 if meta has no train data)
    # Rebuild aligned actuals
    stack_actuals = []
    for val_year in CV_YEARS:
        val_mask = meta_years == val_year
        train_mask = meta_years < val_year
        if val_mask.sum() == 0 or train_mask.sum() < 50:
            continue
        stack_actuals.extend(meta_y[val_mask].tolist())

    stack_preds_oof = np.array(stack_preds_oof)
    stack_actuals = np.array(stack_actuals)
    stack_mae = mean_absolute_error(stack_actuals, stack_preds_oof)
    stack_sp, _ = spearmanr(stack_actuals, stack_preds_oof)
    RESULTS['stacked_ensemble'] = {'mae': stack_mae, 'spearman': stack_sp,
                                    'oof_preds': stack_preds_oof,
                                    'oof_actuals': stack_actuals}
    delta_mae = stack_mae - r_xgb['mae']
    print(f"  Stacked (XGB+LGB+Ridge): MAE={stack_mae:.3f}, Spearman={stack_sp:.4f}")
    print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")

    # Also check simple average ensemble
    simple_avg_preds = (np.array(all_meta_preds_l1_xgb) + np.array(all_meta_preds_l1_lgb)) / 2
    simple_mae = mean_absolute_error(np.array(all_meta_actuals), simple_avg_preds)
    simple_sp, _ = spearmanr(np.array(all_meta_actuals), simple_avg_preds)
    RESULTS['simple_avg_ensemble'] = {'mae': simple_mae, 'spearman': simple_sp}
    delta_simple = simple_mae - r_xgb['mae']
    print(f"  Simple avg (XGB+LGB): MAE={simple_mae:.3f}, Spearman={simple_sp:.4f}")
    print(f"  vs XGBoost baseline: dMAE={delta_simple:+.3f} ({delta_simple/r_xgb['mae']*100:+.1f}%)")


print("\n" + "="*70)
print("EXPERIMENT 11: Feature Ablation — Low-importance feature removal")
print("="*70)

# From XGBoost importance, identify bottom-30% features
full_xgb_final = XGBRegressor(**XGB_PARAMS)
full_xgb_final.fit(model_df[BASE_FEATURES].fillna(0), model_df['SC'],
                    sample_weight=get_sample_weights(model_df['Year']))
xgb_imp = pd.Series(full_xgb_final.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)

# Keep top 40 features (drop bottom ~25)
top40_features = list(xgb_imp.head(40).index)
dropped = [f for f in BASE_FEATURES if f not in top40_features]
print(f"  Dropping {len(dropped)} low-importance features: {dropped}")

def make_xgb_ablated(): return XGBRegressor(**XGB_PARAMS)

r_ablated = run_expanding_window_cv(make_xgb_ablated, model_df, top40_features)
RESULTS['xgb_ablated_top40'] = r_ablated
delta_mae = r_ablated['mae'] - r_xgb['mae']
print(f"  MAE={r_ablated['mae']:.3f} | Spearman={r_ablated['spearman']:.4f}")
print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")


print("\n" + "="*70)
print("EXPERIMENT 12: LightGBM with Quantile Huber objective")
print("="*70)

# Huber loss is more robust to outliers than MSE
LGB_HUBER = {**LGB_PARAMS, 'objective': 'huber', 'alpha': 0.9}

def make_lgb_huber(): return lgb.LGBMRegressor(**LGB_HUBER)

r_huber = run_expanding_window_cv(make_lgb_huber, model_df, BASE_FEATURES)
RESULTS['lgb_huber'] = r_huber
delta_mae = r_huber['mae'] - r_xgb['mae']
print(f"  MAE={r_huber['mae']:.3f} | RMSE={r_huber['rmse']:.3f} | R2={r_huber['r2']:.4f} | Spearman={r_huber['spearman']:.4f}")
print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+): {r_huber['p80_brier_improvement']:.1f}%  P(120+): {r_huber['p120_brier_improvement']:.1f}%")


print("\n" + "="*70)
print("EXPERIMENT 13: LightGBM + Extended Features + Tuned Hyperparameters")
print("="*70)

# Tune LightGBM hyperparameters for the extended feature set
LGB_TUNED = {
    'objective': 'regression',
    'n_estimators': 2000,
    'max_depth': 7,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_samples': 15,
    'num_leaves': 63,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

def make_lgb_tuned(): return lgb.LGBMRegressor(**LGB_TUNED)

r_lgb_tuned = run_expanding_window_cv(make_lgb_tuned, extended_df, EXT_FEATURES)
RESULTS['lgb_tuned_extended'] = r_lgb_tuned
delta_mae = r_lgb_tuned['mae'] - r_xgb['mae']
print(f"  MAE={r_lgb_tuned['mae']:.3f} | RMSE={r_lgb_tuned['rmse']:.3f} | R2={r_lgb_tuned['r2']:.4f} | Spearman={r_lgb_tuned['spearman']:.4f}")
print(f"  vs XGBoost baseline: dMAE={delta_mae:+.3f} ({delta_mae/r_xgb['mae']*100:+.1f}%)")
print(f"  P(80+): {r_lgb_tuned['p80_brier_improvement']:.1f}%  P(120+): {r_lgb_tuned['p120_brier_improvement']:.1f}%")
print(f"  Bucket biases: {r_lgb_tuned['bucket_bias']}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FULL RESULTS SUMMARY")
print("="*70)
print(f"{'Model':<35} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'Spearman':>10} {'dMAE%':>8}")
print("-"*70)

xgb_mae = RESULTS['xgb_baseline']['mae']
for name, r in RESULTS.items():
    mae = r.get('mae', float('nan'))
    rmse = r.get('rmse', float('nan'))
    r2 = r.get('r2', float('nan'))
    sp = r.get('spearman', float('nan'))
    delta = (mae - xgb_mae) / xgb_mae * 100
    marker = " <-- BEST" if mae == min(r.get('mae', 999) for r in RESULTS.values()) else ""
    print(f"{name:<35} {mae:>8.3f} {rmse:>8.3f} {r2:>7.4f} {sp:>10.4f} {delta:>+8.1f}%{marker}")

print("\nBrier improvements (OOF) for top models:")
for name in ['xgb_baseline', 'lightgbm', 'lgb_extended_features', 'lgb_tuned_extended', 'lgb_huber']:
    r = RESULTS.get(name, {})
    p80 = r.get('p80_brier_improvement', float('nan'))
    p100 = r.get('p100_brier_improvement', float('nan'))
    p120 = r.get('p120_brier_improvement', float('nan'))
    print(f"  {name:<35} P80={p80:.1f}%  P100={p100:.1f}%  P120={p120:.1f}%")

# ── Save results for report ─────────────────────────────────────────────────
save_results = {}
for name, r in RESULTS.items():
    sr = {k: v for k, v in r.items() if k not in ('oof_preds', 'oof_actuals')}
    save_results[name] = sr

Path("reports").mkdir(exist_ok=True)
with open("reports/experiment_results.json", "w") as f:
    json.dump(save_results, f, indent=2, default=str)
print("\nResults saved to reports/experiment_results.json")

# Save LightGBM extended feature importances
lgb_ext_imp.to_csv("reports/lgb_extended_feature_importance.csv")
print("Feature importances saved to reports/lgb_extended_feature_importance.csv")
print("\nDONE.")
