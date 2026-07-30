"""
generate_oof_predictions.py — Out-of-fold predictions for backtesting.

Runs the same expanding window CV as train_model_v2.py but saves per-player
predictions for each validation year. Each season's predictions are truly
out-of-sample (model trained on ALL prior seasons only).

CV folds:
  val_year=2021: train 2006-2020, predict 2021
  val_year=2022: train 2006-2021, predict 2022
  ...
  val_year=2025: train 2006-2024, predict 2025

After the CV loop, applies the saved isotonic calibrators (from the production
model artifact) to convert raw regression predictions to calibrated P(80+).

Output: data/predictions/oof_predictions_2021_2025.csv

Usage: py generate_oof_predictions.py
"""

import logging
import joblib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from feature_engineering import engineer_features, get_feature_columns
from train_model_v2 import add_extended_features, get_extended_feature_columns, get_sample_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed",
    "MASTER_DATA_FILE": "master_player_data.csv",
    "MODEL_DIR": "models",
    "PREDICTIONS_DIR": "data/predictions",
    "MODEL_FILE": "sc_lgb_model_v2.joblib",
    "CV_START_YEAR": 2021,
    "CV_END_YEAR": 2025,  # 2026 already has real prediction files
}

LGB_PARAMS = {
    "objective": "regression",
    "n_estimators": 2000,
    "max_depth": 7,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_samples": 15,
    "num_leaves": 63,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def load_and_engineer() -> pd.DataFrame:
    master_path = Path(CONFIG["PROCESSED_DATA_DIR"]) / CONFIG["MASTER_DATA_FILE"]
    log.info("Loading master data from %s...", master_path)
    df = pd.read_csv(master_path, low_memory=False)
    df["Round_Num"] = pd.to_numeric(df["Round_Num"], errors="coerce")
    df = df[df["played"] > 0].copy()
    log.info("Engineering features...")
    df = engineer_features(df, for_training=True)
    df = add_extended_features(df, for_training=True)
    return df


def main():
    df = load_and_engineer()
    features = get_extended_feature_columns()
    target = "SC"

    model_df = df.dropna(subset=[target]).copy()
    rolling_cols = [c for c in features if "_rolling_avg_" in c]
    model_df = model_df[model_df[rolling_cols].abs().sum(axis=1) > 0].copy()

    missing = [f for f in features if f not in model_df.columns]
    if missing:
        log.error("Missing feature columns: %s", missing)
        return

    # ── CV loop — collect OOF predictions with metadata ─────────────────────
    meta_cols = ["Player ID", "First Name", "Last Name", "Team", "Year", "Round_Num"]
    oof_records = []
    oof_raw_preds = []
    oof_actuals = []

    for val_year in range(CONFIG["CV_START_YEAR"], CONFIG["CV_END_YEAR"] + 1):
        train_df = model_df[model_df["Year"] < val_year]
        val_df = model_df[model_df["Year"] == val_year]

        if val_df.empty:
            log.warning("No validation data for %d, skipping.", val_year)
            continue
        if train_df.empty:
            log.warning("No training data before %d, skipping.", val_year)
            continue

        X_train = train_df[features].fillna(0)
        y_train = train_df[target].values
        X_val = val_df[features].fillna(0)
        y_val = val_df[target].values
        w_train = get_sample_weights(train_df["Year"])

        log.info(
            "Fold %d: training on %d rows (2006–%d), predicting %d rows...",
            val_year, len(train_df), val_year - 1, len(val_df),
        )
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=w_train)

        raw_preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, raw_preds)
        log.info("  MAE=%.2f", mae)

        # Collect metadata + predictions
        meta = val_df[meta_cols].copy()
        meta["actual_SC"] = y_val
        meta["raw_pred"] = raw_preds
        oof_records.append(meta)
        oof_raw_preds.extend(raw_preds.tolist())
        oof_actuals.extend(y_val.tolist())

    if not oof_records:
        log.error("No OOF predictions generated.")
        return

    oof_df = pd.concat(oof_records, ignore_index=True)
    oof_raw = np.array(oof_raw_preds)
    oof_act = np.array(oof_actuals)

    # ── Calibration: fit on OOF (same procedure as train_model_v2.py) ───────
    log.info("Fitting calibration on %d OOF predictions...", len(oof_raw))

    lin_cal = LinearRegression()
    lin_cal.fit(oof_raw.reshape(-1, 1), oof_act)
    cal_slope = float(lin_cal.coef_[0])
    cal_intercept = float(lin_cal.intercept_)
    log.info("Linear cal: slope=%.3f, intercept=%.3f", cal_slope, cal_intercept)

    cal_preds = cal_slope * oof_raw + cal_intercept

    iso_80 = IsotonicRegression(out_of_bounds="clip")
    iso_80.fit(cal_preds, (oof_act >= 80).astype(float))

    iso_100 = IsotonicRegression(out_of_bounds="clip")
    iso_100.fit(cal_preds, (oof_act >= 100).astype(float))

    iso_120 = IsotonicRegression(out_of_bounds="clip")
    iso_120.fit(cal_preds, (oof_act >= 120).astype(float))

    # Apply calibration to all OOF predictions
    oof_df["cal_pred"] = cal_slope * oof_df["raw_pred"].values + cal_intercept
    oof_df["prob_gt_80"] = iso_80.predict(oof_df["cal_pred"].values) * 100
    oof_df["prob_gt_100"] = iso_100.predict(oof_df["cal_pred"].values) * 100
    oof_df["prob_gt_120"] = iso_120.predict(oof_df["cal_pred"].values) * 100

    # Round outputs
    for col in ["cal_pred", "prob_gt_80", "prob_gt_100", "prob_gt_120"]:
        oof_df[col] = oof_df[col].round(1)

    out_path = Path(CONFIG["PREDICTIONS_DIR"]) / "oof_predictions_2021_2025.csv"
    oof_df.to_csv(out_path, index=False)
    log.info("Saved %d OOF predictions to %s", len(oof_df), out_path)

    # Quick calibration check
    print("\nCalibration check (OOF prob_gt_80 vs actual):")
    for lo in range(0, 100, 10):
        bucket = oof_df[(oof_df["prob_gt_80"] >= lo) & (oof_df["prob_gt_80"] < lo + 10)]
        if len(bucket) > 20:
            actual = (bucket["actual_SC"] >= 80).mean() * 100
            print(f"  Pred {lo:3d}-{lo+10}%: actual={actual:5.1f}%  n={len(bucket)}")

    overall_mae = mean_absolute_error(oof_df["actual_SC"], oof_df["cal_pred"])
    print(f"\nOverall calibrated MAE: {overall_mae:.2f}")
    print(f"Coverage: {len(oof_df['Year'].unique())} seasons "
          f"({oof_df['Year'].min()}-{oof_df['Year'].max()}), "
          f"{len(oof_df)} player-round pairs")


if __name__ == "__main__":
    main()
