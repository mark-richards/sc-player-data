"""
Train SuperCoach Prediction Model
===================================
XGBoost regressor with expanding window cross-validation.

Improvements over v1:
  - Year-recency sample weights (recent seasons weighted higher)
  - OOF predictions used to fit linear calibration (fixes score compression)
  - Isotonic regression calibrators for P(>80), P(>100), P(>120)
    replacing the naive Gaussian assumption
  - Spearman rank correlation reported alongside MAE
  - 120+ bucket analysis to monitor elite underestimation
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
from xgboost import XGBRegressor

from feature_engineering import engineer_features, get_feature_columns

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed",
    "MASTER_DATA_FILE": "master_player_data.csv",
    "MODEL_DIR": "models",
    "REPORTS_DIR": "reports/figures",
    "MODEL_FILE": "sc_xgb_model_v2.joblib",
    "CV_START_YEAR": 2021,
    "CV_END_YEAR": 2025,
}

# --- Year recency weights ---
# More recent seasons are weighted higher because game style and rules have changed.
YEAR_WEIGHTS = {2025: 3.0, 2024: 2.5, 2023: 2.0, 2022: 1.5, 2021: 1.0}
DEFAULT_WEIGHT = 0.5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_sample_weights(years: pd.Series) -> np.ndarray:
    """Returns per-sample weights based on year recency."""
    return np.array([YEAR_WEIGHTS.get(int(y), DEFAULT_WEIGHT) for y in years])


def train_and_evaluate(df: pd.DataFrame):
    """Expanding window CV, then train final production model on all data."""
    features = get_feature_columns()
    target = 'SC'

    model_df = df.dropna(subset=[target]).copy()

    # Verify all feature columns exist
    missing = [f for f in features if f not in model_df.columns]
    if missing:
        logging.error(f"Missing feature columns: {missing}")
        return

    # Drop rows where ALL rolling features are 0/NaN (first game for a player)
    rolling_cols = [c for c in features if '_rolling_avg_' in c]
    model_df = model_df[model_df[rolling_cols].abs().sum(axis=1) > 0].copy()

    # --- XGBoost Parameters ---
    xgb_params = {
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
    }

    # --- Expanding Window Cross-Validation ---
    print("\n" + "=" * 70)
    print("EXPANDING WINDOW CROSS-VALIDATION")
    print("=" * 70)

    cv_results = []
    all_residuals = []
    all_oof_preds = []
    all_oof_actuals = []

    for val_year in range(CONFIG['CV_START_YEAR'], CONFIG['CV_END_YEAR'] + 1):
        train_df = model_df[model_df['Year'] < val_year]
        val_df = model_df[model_df['Year'] == val_year]

        if val_df.empty:
            logging.warning(f"No validation data for {val_year}, skipping.")
            continue

        X_train = train_df[features]
        y_train = train_df[target]
        X_val = val_df[features]
        y_val = val_df[target]

        # Apply year-recency weights to training data
        w_train = get_sample_weights(train_df['Year'])

        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        residuals = y_val.values - preds
        spearman_r, _ = spearmanr(y_val.values, preds)

        # Collect OOF predictions for calibration fitting
        all_oof_preds.extend(preds.tolist())
        all_oof_actuals.extend(y_val.values.tolist())
        all_residuals.extend(residuals.tolist())

        cv_results.append({
            'year': val_year,
            'mae': mae,
            'spearman': spearman_r,
            'train_size': len(train_df),
            'val_size': len(val_df),
        })

        # Score-bucket analysis (including 120+)
        val_analysis = pd.DataFrame({'actual': y_val.values, 'predicted': preds, 'residual': residuals})
        buckets = [(0, 60, '<60'), (60, 80, '60-80'), (80, 100, '80-100'), (100, 120, '100-120'), (120, 300, '120+')]
        bucket_str = ""
        for lo, hi, label in buckets:
            bucket = val_analysis[(val_analysis['actual'] >= lo) & (val_analysis['actual'] < hi)]
            if len(bucket) > 0:
                bucket_str += f"  {label}: bias={bucket['residual'].mean():+.1f}, n={len(bucket)} | "

        print(f"  {val_year}: MAE={mae:.2f}, Spearman={spearman_r:.3f} "
              f"(train={len(train_df):,}, val={len(val_df):,})")
        print(f"    Buckets: {bucket_str}")

    avg_mae = np.mean([r['mae'] for r in cv_results])
    avg_spearman = np.mean([r['spearman'] for r in cv_results])
    print(f"\n  Average CV MAE: {avg_mae:.2f}")
    print(f"  Average CV Spearman: {avg_spearman:.3f}")
    print("=" * 70)

    # --- OOF Calibration ---
    print("\n" + "=" * 70)
    print("CALIBRATION (fitted on OOF predictions)")
    print("=" * 70)

    oof_preds = np.array(all_oof_preds)
    oof_actuals = np.array(all_oof_actuals)

    # Score compression diagnostics
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

    # Linear calibration: maps compressed raw predictions → actual score space
    # This is variance expansion: slope > 1 restores the compressed spread
    lin_cal = LinearRegression()
    lin_cal.fit(oof_preds.reshape(-1, 1), oof_actuals)
    cal_slope = float(lin_cal.coef_[0])
    cal_intercept = float(lin_cal.intercept_)
    print(f"  Linear calibration: slope={cal_slope:.3f}, intercept={cal_intercept:.3f}")
    print(f"  (Effective: projected = {cal_slope:.2f}×raw + {cal_intercept:.1f})")

    # Verify calibration quality on OOF
    cal_preds = cal_slope * oof_preds + cal_intercept
    cal_mae = float(mean_absolute_error(oof_actuals, cal_preds))
    cal_spearman, _ = spearmanr(oof_actuals, cal_preds)
    cal_std = float(np.std(cal_preds))
    print(f"  Post-calibration: MAE={cal_mae:.2f}, Spearman={cal_spearman:.3f}, std={cal_std:.1f}")

    # Isotonic regression calibrators for probability estimation
    # These are trained on OOF (calibrated) predictions vs binary targets
    # IsotonicRegression fits a monotone step function — no Gaussian assumption needed
    iso_80 = IsotonicRegression(out_of_bounds='clip')
    iso_80.fit(cal_preds, (oof_actuals >= 80).astype(float))

    iso_100 = IsotonicRegression(out_of_bounds='clip')
    iso_100.fit(cal_preds, (oof_actuals >= 100).astype(float))

    iso_120 = IsotonicRegression(out_of_bounds='clip')
    iso_120.fit(cal_preds, (oof_actuals >= 120).astype(float))

    # Validate calibration quality: perfectly calibrated → 50% of events occur above P50 threshold
    for label, iso, threshold in [('P(80+)', iso_80, 80), ('P(100+)', iso_100, 100), ('P(120+)', iso_120, 120)]:
        probs = iso.predict(cal_preds)
        # Brier score: mean((p - y)^2), lower is better; naive = class_mean*(1-class_mean)
        binary = (oof_actuals >= threshold).astype(float)
        brier = float(np.mean((probs - binary) ** 2))
        naive_brier = float(np.mean(binary) * (1 - np.mean(binary)))
        print(f"  {label}: Brier={brier:.4f} (naive={naive_brier:.4f}, "
              f"improvement={(1 - brier/naive_brier)*100:.1f}%)")

    print("=" * 70)

    # --- Residual calibration (kept for backward compat) ---
    residual_std = np.std(all_residuals)
    residual_mean = np.mean(all_residuals)
    logging.info(f"Legacy calibration: residual_mean={residual_mean:.2f}, residual_std={residual_std:.2f}")

    # --- Train final production model on ALL data ---
    logging.info("Training FINAL production model on all available data...")
    X_all = model_df[features]
    y_all = model_df[target]
    w_all = get_sample_weights(model_df['Year'])

    final_model = XGBRegressor(**xgb_params)
    final_model.fit(X_all, y_all, sample_weight=w_all, verbose=False)
    logging.info(f"Final model trained on {len(X_all):,} rows.")

    # --- Feature Importance ---
    importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 20 Feature Importances ---")
    print(importances.head(20).to_string())

    # Flag importance of new features
    new_features = ['SC_ceiling_rate_10', 'opponent_ceiling_rating',
                    'tag_notes_on_ball', 'tag_notes_managed', 'is_debut_season']
    print("\n--- New Feature Importances ---")
    for f in new_features:
        if f in importances.index:
            rank = list(importances.index).index(f) + 1
            print(f"  {f}: importance={importances[f]:.4f} (rank {rank}/{len(importances)})")

    fig_dir = Path(CONFIG['REPORTS_DIR'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Full importance plot
    figure_path = fig_dir / "feature_importance_xgb_v2.png"
    plt.figure(figsize=(12, 14))
    top_n = min(35, len(importances))
    top_imp = importances.head(top_n)
    sns.barplot(x=top_imp.values, y=top_imp.index, hue=top_imp.index, palette='viridis', legend=False)
    plt.title(f'XGBoost Feature Importance (Top {top_n})')
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    logging.info(f"Feature importance plot saved to: {figure_path}")

    # CV MAE trend plot
    cv_fig_path = fig_dir / "cv_mae_trend.png"
    plt.figure(figsize=(8, 5))
    cv_df = pd.DataFrame(cv_results)
    plt.bar(cv_df['year'].astype(str), cv_df['mae'], color='steelblue')
    plt.axhline(y=avg_mae, color='red', linestyle='--', label=f'Avg MAE: {avg_mae:.2f}')
    plt.xlabel('Validation Year')
    plt.ylabel('MAE')
    plt.title('Expanding Window CV - MAE by Year')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_fig_path, dpi=150)
    plt.close()
    logging.info(f"CV MAE trend plot saved to: {cv_fig_path}")

    # --- Save Model Artifact ---
    model_dir = Path(CONFIG['MODEL_DIR'])
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / CONFIG['MODEL_FILE']

    artifact = {
        'model': final_model,
        'features': features,
        # Linear calibration for variance expansion (raw XGBoost → calibrated scores)
        'cal_slope': cal_slope,
        'cal_intercept': cal_intercept,
        # Isotonic probability calibrators (calibrated_score → P(threshold))
        'iso_80': iso_80,
        'iso_100': iso_100,
        'iso_120': iso_120,
        # OOF diagnostics
        'oof_spearman': overall_spearman,
        'compression_ratio': compression_ratio,
        'elite_bias': elite_bias,
        # Legacy calibration (kept for backward compat)
        'residual_std': residual_std,
        'residual_mean': residual_mean,
        # CV summary
        'cv_results': cv_results,
        'avg_cv_mae': avg_mae,
        'avg_cv_spearman': avg_spearman,
    }
    joblib.dump(artifact, model_path)
    logging.info(f"Model artifact saved to: {model_path}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"  CV Average MAE: {avg_mae:.2f}")
    print(f"  CV Average Spearman: {avg_spearman:.3f}")
    print(f"  Linear calibration: slope={cal_slope:.3f}, intercept={cal_intercept:.3f}")
    print(f"  Model saved to: {model_path}")
    print(f"{'='*70}")


def main():
    """Main function to run the full training pipeline."""
    master_file_path = Path(CONFIG['PROCESSED_DATA_DIR']) / CONFIG['MASTER_DATA_FILE']
    if not master_file_path.exists():
        logging.error(f"Master data file not found at: {master_file_path}")
        return

    logging.info(f"Loading master data from {master_file_path}...")
    master_df = pd.read_csv(master_file_path, low_memory=False)

    featured_df = engineer_features(master_df, for_training=True)
    train_and_evaluate(featured_df)


if __name__ == "__main__":
    main()
