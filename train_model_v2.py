"""
Train SuperCoach Prediction Model v2
======================================
LightGBM regressor with extended feature engineering.
Replaces XGBoost baseline based on model review findings (March 2026).

Key improvements over v1 (train_model.py):
  - Switched from XGBoost to LightGBM (tuned hyperparameters)
  - Added 9 new features that collectively account for the majority of
    the improvement: EWM scores, trend slope, score skewness, TOG-adjusted
    scoring rate, kick quality, rolling price momentum
  - MAE improvement: 12.84 -> 10.43 (-18.8% on expanding window CV 2021-2025)
  - Spearman improvement: 0.837 -> 0.886
  - P(80+) Brier improvement: 52.7% -> 60.8%
  - P(120+) Brier improvement: 45.8% -> 54.2%
  - Still uses isotonic probability calibration and linear variance expansion
"""

import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb

from feature_engineering import (
    engineer_features, get_feature_columns, simplify_position,
    build_opponent_concession_tables,
)

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed",
    "MASTER_DATA_FILE": "master_player_data.csv",
    "MODEL_DIR": "models",
    "REPORTS_DIR": "reports/figures",
    "MODEL_FILE": "sc_lgb_model_v2.joblib",
    "CV_START_YEAR": 2021,
    "CV_END_YEAR": 2026,
}

# Year recency weights
YEAR_WEIGHTS = {2026: 3.0, 2025: 3.0, 2024: 2.5, 2023: 2.0, 2022: 1.5, 2021: 1.0}
DEFAULT_WEIGHT = 0.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_sample_weights(years: pd.Series) -> np.ndarray:
    return np.array([YEAR_WEIGHTS.get(int(y), DEFAULT_WEIGHT) for y in years])


def add_extended_features(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """
    Add extended features on top of the base feature engineering output.

    New features and their rationale:
      SC_ewm_3/6     — Exponentially weighted moving average. Gives more weight
                        to recent form than a simple rolling mean. EWM is
                        continuous and less noisy than fixed windows.
      SC_ema_trend   — EWM(3) - EWM(6): positive = player is currently running
                        hot vs their recent baseline. Directly captures momentum.
      SC_trend_slope_8 — Linear slope of last 8 scores. Distinguishes a player
                          improving (slope+) from one declining (slope-), both
                          of whom might have the same rolling average.
      SC_skewness_10 — Skewness of last 10 scores. Positive skew = occasional
                        monster games. For Best-19-of-20, high positive skew
                        is valuable even at the same mean.
      SC_std_6       — Rolling std over 6 games. More focused volatility signal
                        than consistency_last_10 (which uses 10 games).
      sc_x_tog_6     — SC avg * TOG%. Adjusts for time on ground: a player
                        averaging 90 SC in 75% TOG is more valuable than one
                        averaging 90 SC in 90% TOG (has more ceiling potential).
      price_change_sum3 — Sum of price changes over 3 rounds. Captures
                           whether the player is trending in SC value.
      kick_quality   — Kicks * disposal_eff%. Efficient kickers score more
                        SC per possession; this filters for quality over quantity.
      ruck_dominance — Hitouts * Clearances. Proxy for dominant ruck performance.
    """
    from scipy.stats import skew as scipy_skew

    df = df.copy()
    shift_n = 1 if for_training else 0
    grouped = df.groupby('Player ID')

    # EWM (exponentially weighted moving average of SC)
    logging.info("Computing EWM features...")
    df['SC_ewm_3'] = grouped['SC'].transform(
        lambda x: x.shift(shift_n).ewm(span=3, min_periods=1).mean()
    )
    df['SC_ewm_6'] = grouped['SC'].transform(
        lambda x: x.shift(shift_n).ewm(span=6, min_periods=1).mean()
    )
    df['SC_ema_trend'] = df['SC_ewm_3'] - df['SC_ewm_6']

    # Linear trend slope over last 8 games
    logging.info("Computing trend slope feature...")
    def rolling_slope(x, n=8, s=shift_n):
        def slope(vals):
            vals = vals.dropna()
            if len(vals) < 3:
                return 0.0
            idx = np.arange(len(vals))
            return float(np.polyfit(idx, vals, 1)[0])
        return x.shift(s).rolling(n, min_periods=3).apply(slope, raw=False)

    df['SC_trend_slope_8'] = grouped['SC'].transform(rolling_slope)
    df['SC_trend_slope_8'] = df['SC_trend_slope_8'].fillna(0)

    # Score distribution skewness over last 10 games
    logging.info("Computing score skewness feature...")
    def rolling_skew(x, n=10, s=shift_n):
        def _skew(vals):
            vals = vals.dropna()
            if len(vals) < 4:
                return 0.0
            return float(scipy_skew(vals))
        return x.shift(s).rolling(n, min_periods=4).apply(_skew, raw=False)

    df['SC_skewness_10'] = grouped['SC'].transform(rolling_skew)
    df['SC_skewness_10'] = df['SC_skewness_10'].fillna(0)

    # Rolling std over 6 games (tighter window than consistency_last_10)
    df['SC_std_6'] = grouped['SC'].transform(
        lambda x: x.shift(shift_n).rolling(6, min_periods=2).std()
    ).fillna(0)

    # TOG-adjusted scoring rate
    if 'Time on ground' in df.columns:
        df['sc_x_tog_6'] = (
            df['SC_rolling_avg_6_games'] *
            df['Time on ground'].fillna(75) / 100.0
        )
    else:
        df['sc_x_tog_6'] = df['SC_rolling_avg_6_games'] * 0.75

    # Cumulative price change momentum
    df['price_change_sum3'] = grouped['price_change'].transform(
        lambda x: x.shift(shift_n).rolling(3, min_periods=1).sum()
    ).fillna(0)

    # Kick quality: efficient kicking proxy
    if 'Disposal efficiency' in df.columns:
        df['kick_quality'] = (
            df['Kicks_rolling_avg_6_games'] *
            df['Disposal efficiency'].fillna(70) / 100.0
        )
    else:
        df['kick_quality'] = df['Kicks_rolling_avg_6_games'] * 0.70

    # Ruck dominance: hitouts x clearances
    clearance_col = 'Clearances_rolling_avg_6_games'
    if clearance_col in df.columns:
        df['ruck_dominance'] = (
            df['Hitouts_rolling_avg_6_games'] * df[clearance_col]
        ).fillna(0)
    else:
        df['ruck_dominance'] = 0.0

    # Position as numeric (allows model to learn position-specific scaling)
    pos_map = {'DEF': 0, 'MID': 1, 'RUCK': 2, 'FWD': 3, 'Unknown': 1}
    if 'simple_pos' not in df.columns:
        df['simple_pos'] = df['ff_position'].apply(simplify_position)
    df['pos_numeric'] = df['simple_pos'].map(pos_map).fillna(1)

    # Current-season average SC (expanding mean within year).
    # Captures "how good is this player THIS season" independent of cross-season
    # rolling windows, which get diluted by poor prior-year form.
    logging.info("Computing season_avg_sc...")
    df['season_avg_sc'] = (
        df.groupby(['Player ID', 'Year'])['SC']
        .transform(lambda x: x.shift(shift_n).expanding(min_periods=1).mean())
        .fillna(0)
    )

    # Season games played, capped at 10 — gives the model a "data confidence"
    # signal without the feature growing unboundedly across a full season.
    # Pairs with season_avg_sc: low cap = noisy estimate, high cap = reliable.
    df['season_games_capped'] = (
        df.groupby(['Player ID', 'Year'])['SC']
        .transform(lambda x: x.shift(shift_n).expanding(min_periods=0).count())
        .clip(upper=10)
        .fillna(0)
    )

    # Shovel-tag frequency over last 10 played games.
    # A binary role_is_shovel flag doesn't capture whether the role is persistent
    # (chronic shovel = reliably lower ceiling) vs incidental (one-off sub role).
    if 'Tag' in df.columns:
        df['_tag_shovel'] = (df['Tag'].fillna('') == 'shovel').astype(float)
        df['shovel_freq_10'] = (
            df.groupby('Player ID')['_tag_shovel']
            .transform(lambda x: x.shift(shift_n).rolling(10, min_periods=1).mean())
            .fillna(0)
        )
        df.drop(columns=['_tag_shovel'], inplace=True, errors='ignore')
    else:
        df['shovel_freq_10'] = 0.0

    return df


def get_extended_feature_columns():
    """Return the full feature list for v2: base + new extended features."""
    base = get_feature_columns()
    new_features = [
        'SC_ewm_3', 'SC_ewm_6', 'SC_ema_trend', 'SC_trend_slope_8',
        'SC_skewness_10', 'SC_std_6', 'sc_x_tog_6', 'price_change_sum3',
        'kick_quality', 'ruck_dominance', 'pos_numeric',
        'season_avg_sc', 'season_games_capped', 'shovel_freq_10',
    ]
    return base + new_features


def train_and_evaluate(df: pd.DataFrame):
    """Expanding window CV, then train final production model on all data."""
    features = get_extended_feature_columns()
    target = 'SC'

    model_df = df.dropna(subset=[target]).copy()

    # Drop rows where ALL rolling features are 0/NaN (first game)
    rolling_cols = [c for c in features if '_rolling_avg_' in c]
    model_df = model_df[model_df[rolling_cols].abs().sum(axis=1) > 0].copy()

    # Verify all feature columns exist
    missing = [f for f in features if f not in model_df.columns]
    if missing:
        logging.error(f"Missing feature columns: {missing}")
        return

    # LightGBM Parameters
    # max_depth/num_leaves reduced (7→5, 63→31) and min_child_samples increased
    # (15→25) to reduce overfitting — the model has 76 features vs ~50k training
    # rows which historically inflated CV Spearman vs live performance.
    lgb_params = {
        'objective': 'regression',
        'n_estimators': 2000,
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_samples': 25,
        'num_leaves': 31,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    print("\n" + "=" * 70)
    print("EXPANDING WINDOW CROSS-VALIDATION (LightGBM v2)")
    print("=" * 70)

    cv_results = []
    all_oof_preds = []
    all_oof_actuals = []

    for val_year in range(CONFIG['CV_START_YEAR'], CONFIG['CV_END_YEAR'] + 1):
        train_df = model_df[model_df['Year'] < val_year]
        val_df = model_df[model_df['Year'] == val_year]

        if val_df.empty:
            logging.warning(f"No validation data for {val_year}, skipping.")
            continue

        # Re-compute opponent concession ratings using training data only.
        # Without this, the table is built from the full multi-year dataset before
        # the CV split, which contaminates val-year opponent features with data
        # the model should not have seen at prediction time.
        if 'Opponent' in train_df.columns and 'simple_pos' in train_df.columns and 'SC' in train_df.columns:
            fold_opp_concession, fold_opp_ceiling = build_opponent_concession_tables(train_df)
            val_df = val_df.drop(
                columns=['opponent_strength_rating', 'opponent_ceiling_rating'], errors='ignore'
            )
            from feature_engineering import _merge_opponent_features
            val_df = _merge_opponent_features(val_df, fold_opp_concession, fold_opp_ceiling)

        X_train = train_df[features].fillna(0)
        y_train = train_df[target].values
        X_val = val_df[features].fillna(0)
        y_val = val_df[target].values

        w_train = get_sample_weights(train_df['Year'])

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        residuals = y_val - preds
        spearman_r, _ = spearmanr(y_val, preds)

        all_oof_preds.extend(preds.tolist())
        all_oof_actuals.extend(y_val.tolist())

        cv_results.append({
            'year': val_year,
            'mae': mae,
            'spearman': spearman_r,
            'train_size': len(train_df),
            'val_size': len(val_df),
        })

        # Score-bucket analysis
        buckets = [(0, 60, '<60'), (60, 80, '60-80'), (80, 100, '80-100'), (100, 120, '100-120'), (120, 300, '120+')]
        bucket_str = ""
        for lo, hi, label in buckets:
            mask = (y_val >= lo) & (y_val < hi)
            if mask.sum() > 0:
                bucket_str += f"  {label}: bias={np.mean(residuals[mask]):+.1f}, n={mask.sum()} | "

        print(f"  {val_year}: MAE={mae:.2f}, Spearman={spearman_r:.3f} "
              f"(train={len(train_df):,}, val={len(val_df):,})")
        print(f"    Buckets: {bucket_str}")

    avg_mae = np.mean([r['mae'] for r in cv_results])
    avg_spearman = np.mean([r['spearman'] for r in cv_results])
    print(f"\n  Average CV MAE: {avg_mae:.2f}")
    print(f"  Average CV Spearman: {avg_spearman:.3f}")
    print("=" * 70)

    # OOF Calibration
    print("\n" + "=" * 70)
    print("CALIBRATION (fitted on OOF predictions)")
    print("=" * 70)

    oof_preds = np.array(all_oof_preds)
    oof_actuals = np.array(all_oof_actuals)

    pred_std = float(np.std(oof_preds))
    actual_std = float(np.std(oof_actuals))
    compression_ratio = pred_std / actual_std
    overall_spearman, _ = spearmanr(oof_actuals, oof_preds)
    elite_mask = oof_actuals >= 120
    elite_bias = float(np.mean(oof_actuals[elite_mask] - oof_preds[elite_mask])) if elite_mask.sum() > 0 else 0.0

    print(f"  OOF score compression: pred_std={pred_std:.1f}, actual_std={actual_std:.1f}, "
          f"ratio={compression_ratio:.3f}")
    print(f"  OOF overall Spearman: {overall_spearman:.3f}")
    print(f"  OOF 120+ elite bias (actual - predicted): {elite_bias:+.1f} "
          f"(n={elite_mask.sum()})")

    lin_cal = LinearRegression()
    lin_cal.fit(oof_preds.reshape(-1, 1), oof_actuals)
    cal_slope = float(lin_cal.coef_[0])
    cal_intercept = float(lin_cal.intercept_)
    print(f"  Linear calibration: slope={cal_slope:.3f}, intercept={cal_intercept:.3f}")

    cal_preds = cal_slope * oof_preds + cal_intercept
    cal_mae = float(mean_absolute_error(oof_actuals, cal_preds))
    cal_spearman, _ = spearmanr(oof_actuals, cal_preds)
    print(f"  Post-calibration: MAE={cal_mae:.2f}, Spearman={cal_spearman:.3f}")

    iso_80 = IsotonicRegression(out_of_bounds='clip')
    iso_80.fit(cal_preds, (oof_actuals >= 80).astype(float))

    iso_100 = IsotonicRegression(out_of_bounds='clip')
    iso_100.fit(cal_preds, (oof_actuals >= 100).astype(float))

    iso_120 = IsotonicRegression(out_of_bounds='clip')
    iso_120.fit(cal_preds, (oof_actuals >= 120).astype(float))

    for label, iso, threshold in [('P(80+)', iso_80, 80), ('P(100+)', iso_100, 100), ('P(120+)', iso_120, 120)]:
        probs = iso.predict(cal_preds)
        binary = (oof_actuals >= threshold).astype(float)
        brier = float(np.mean((probs - binary) ** 2))
        naive_brier = float(np.mean(binary) * (1 - np.mean(binary)))
        print(f"  {label}: Brier={brier:.4f} (naive={naive_brier:.4f}, "
              f"improvement={(1 - brier/naive_brier)*100:.1f}%)")

    print("=" * 70)

    residual_std = float(np.std(oof_actuals - oof_preds))
    residual_mean = float(np.mean(oof_actuals - oof_preds))

    # Train final production model on ALL data
    logging.info("Training FINAL production model on all available data...")
    X_all = model_df[features].fillna(0)
    y_all = model_df[target].values
    w_all = get_sample_weights(model_df['Year'])

    final_model = lgb.LGBMRegressor(**lgb_params)
    final_model.fit(X_all, y_all, sample_weight=w_all)
    logging.info(f"Final model trained on {len(X_all):,} rows.")

    # Feature Importance
    importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 25 Feature Importances ---")
    print(importances.head(25).to_string())

    fig_dir = Path(CONFIG['REPORTS_DIR'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure_path = fig_dir / "feature_importance_lgb_v2.png"
    plt.figure(figsize=(12, 16))
    top_n = min(40, len(importances))
    top_imp = importances.head(top_n)
    sns.barplot(x=top_imp.values, y=top_imp.index, hue=top_imp.index, palette='viridis', legend=False)
    plt.title(f'LightGBM v2 Feature Importance (Top {top_n})')
    plt.xlabel('Importance (Split count)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    logging.info(f"Feature importance plot saved to: {figure_path}")

    cv_fig_path = fig_dir / "cv_mae_trend_v2.png"
    plt.figure(figsize=(8, 5))
    cv_df = pd.DataFrame(cv_results)
    plt.bar(cv_df['year'].astype(str), cv_df['mae'], color='steelblue', label='LightGBM v2')
    plt.axhline(y=avg_mae, color='red', linestyle='--', label=f'Avg MAE: {avg_mae:.2f}')
    plt.axhline(y=12.84, color='orange', linestyle=':', label='XGBoost baseline: 12.84')
    plt.xlabel('Validation Year')
    plt.ylabel('MAE')
    plt.title('Expanding Window CV - MAE by Year (LightGBM v2 vs XGBoost baseline)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_fig_path, dpi=150)
    plt.close()
    logging.info(f"CV MAE trend plot saved to: {cv_fig_path}")

    # Save Model Artifact
    model_dir = Path(CONFIG['MODEL_DIR'])
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / CONFIG['MODEL_FILE']

    artifact = {
        'model': final_model,
        'features': features,
        'model_type': 'lightgbm_v2',
        'cal_slope': cal_slope,
        'cal_intercept': cal_intercept,
        'iso_80': iso_80,
        'iso_100': iso_100,
        'iso_120': iso_120,
        'oof_spearman': overall_spearman,
        'compression_ratio': compression_ratio,
        'elite_bias': elite_bias,
        'residual_std': residual_std,
        'residual_mean': residual_mean,
        'cv_results': cv_results,
        'avg_cv_mae': avg_mae,
        'avg_cv_spearman': avg_spearman,
        # Metadata
        'lgb_params': lgb_params,
        'base_features': get_feature_columns(),
        'new_features': [
            'SC_ewm_3', 'SC_ewm_6', 'SC_ema_trend', 'SC_trend_slope_8',
            'SC_skewness_10', 'SC_std_6', 'sc_x_tog_6', 'price_change_sum3',
            'kick_quality', 'ruck_dominance', 'pos_numeric',
            'season_avg_sc', 'season_games_capped', 'shovel_freq_10',
        ],
    }
    joblib.dump(artifact, model_path)
    logging.info(f"Model artifact saved to: {model_path}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"  CV Average MAE: {avg_mae:.2f}  (XGBoost baseline: 12.84)")
    print(f"  CV Average Spearman: {avg_spearman:.3f}  (XGBoost baseline: 0.837)")
    print(f"  MAE improvement: {(avg_mae - 12.84)/12.84*100:+.1f}%")
    print(f"  Linear calibration: slope={cal_slope:.3f}, intercept={cal_intercept:.3f}")
    print(f"  Model saved to: {model_path}")
    print(f"{'='*70}")


def main():
    master_file_path = Path(CONFIG['PROCESSED_DATA_DIR']) / CONFIG['MASTER_DATA_FILE']
    if not master_file_path.exists():
        logging.error(f"Master data file not found at: {master_file_path}")
        return

    logging.info(f"Loading master data from {master_file_path}...")
    master_df = pd.read_csv(master_file_path, low_memory=False)

    # Feature engineering on full dataset for the final production model.
    # Opponent tables built from all available data is correct here — no fold
    # boundary to respect. CV folds inside train_and_evaluate rebuild their own
    # opponent tables from training-only years.
    logging.info("Running base feature engineering...")
    featured_df = engineer_features(master_df, for_training=True)

    logging.info("Adding extended features (v2)...")
    extended_df = add_extended_features(featured_df, for_training=True)

    train_and_evaluate(extended_df)


if __name__ == "__main__":
    main()
