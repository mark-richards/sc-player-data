"""
Generate SuperCoach Predictions
=================================
Loads the trained XGBoost model artifact and generates predictions for a specific round.
Probabilities derived from regression via Gaussian calibration (no separate classifiers).
"""

import pandas as pd
import logging
import joblib
from pathlib import Path
import argparse
import requests
import numpy as np
from scipy.stats import norm

from feature_engineering import (
    engineer_features, get_latest_player_features,
    build_opponent_concession_table, apply_fixture_opponent,
)
from train_model_v2 import add_extended_features

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed",
    "MASTER_DATA_FILE": "master_player_data.csv",
    "MODEL_DIR": "models",
    "PREDICTIONS_DIR": "data/predictions",
    "MODEL_FILE": "sc_lgb_model_v2.joblib",
    "FIXTURE_JSON_URL_TEMPLATE": "https://fixturedownload.com/feed/json/afl-{year}",
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_fixture_for_round(year: int, round_num: int) -> dict:
    """Fetches fixture from JSON feed and builds an opponent map for a specific round."""
    logging.info(f"Fetching fixture for {year} Round {round_num} from JSON feed...")
    url = CONFIG['FIXTURE_JSON_URL_TEMPLATE'].format(year=year)

    team_name_map = {
        "Adelaide Crows": "AD", "Brisbane Lions": "BL", "Carlton": "CA", "Collingwood": "CO",
        "Essendon": "ES", "Fremantle": "FR", "Geelong Cats": "GE", "Gold Coast SUNS": "GC",
        "GWS GIANTS": "WS", "Hawthorn": "HW", "Melbourne": "ME", "North Melbourne": "NM",
        "Port Adelaide": "PA", "Richmond": "RI", "St Kilda": "SK", "Sydney Swans": "SY",
        "West Coast Eagles": "WC", "Western Bulldogs": "WB"
    }

    try:
        response = requests.get(url, headers={'User-Agent': CONFIG['USER_AGENT']})
        response.raise_for_status()
        fixture_data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch fixture JSON: {e}")
        return {}
    except ValueError:
        logging.error("Failed to decode fixture JSON.")
        return {}

    fixture_map = {}
    for match in fixture_data:
        if match.get('RoundNumber') == round_num:
            home_team_full = match.get('HomeTeam')
            away_team_full = match.get('AwayTeam')

            if home_team_full and away_team_full:
                home_team_abbr = team_name_map.get(home_team_full)
                away_team_abbr = team_name_map.get(away_team_full)

                if home_team_abbr and away_team_abbr:
                    fixture_map[home_team_abbr] = away_team_abbr
                    fixture_map[away_team_abbr] = home_team_abbr

    if fixture_map:
        logging.info(f"Successfully built fixture map for {len(fixture_map)} teams.")
    else:
        logging.warning(f"Could not find any matches for Round {round_num} in the fixture feed.")

    return fixture_map


def main():
    """Load model artifact and generate predictions for a future round."""
    parser = argparse.ArgumentParser(description="Generate SuperCoach predictions for a specific round.")
    parser.add_argument("--year", type=int, required=True, help="The year of the AFL season.")
    parser.add_argument("--round", type=int, required=True, help="The upcoming round number to predict.")
    args = parser.parse_args()

    # Ensure latest data before generating predictions
    from waiver.fixture_schedule import round_is_complete
    if round_is_complete(args.round):
        logging.info("Refreshing live data before predictions...")
        try:
            from website.data.api_ingest import refresh_2026_data
            refresh_2026_data()
        except Exception as exc:
            logging.warning(f"SC API refresh failed (non-fatal): {exc}")
        try:
            from build_master_dataset import update_current_season
            update_current_season(year=args.year)
        except Exception as exc:
            logging.warning(f"master_player_data.csv update failed (non-fatal): {exc}")
    else:
        logging.info("Round %d still in progress — skipping data refresh.", args.round)

    # Load model artifact
    model_path = Path(CONFIG['MODEL_DIR']) / CONFIG['MODEL_FILE']
    logging.info(f"Loading model artifact from {model_path}...")
    artifact = joblib.load(model_path)

    model = artifact['model']
    features = artifact['features']
    residual_std = artifact['residual_std']  # kept for fallback

    # New calibration artifacts (available after model retrain)
    cal_slope = artifact.get('cal_slope', 1.0)
    cal_intercept = artifact.get('cal_intercept', 0.0)
    iso_80 = artifact.get('iso_80')
    iso_100 = artifact.get('iso_100')
    iso_120 = artifact.get('iso_120')

    has_new_cal = (cal_slope != 1.0 or cal_intercept != 0.0) and iso_100 is not None
    logging.info(
        f"Model loaded. {len(features)} features, residual_std={residual_std:.2f}, "
        f"cal_slope={cal_slope:.3f} ({'new isotonic cal' if has_new_cal else 'Gaussian fallback'})"
    )

    # Load master data
    master_file_path = Path(CONFIG['PROCESSED_DATA_DIR']) / CONFIG['MASTER_DATA_FILE']
    master_df = pd.read_csv(master_file_path, low_memory=False)
    master_df['Round_Num'] = pd.to_numeric(master_df['Round_Num'], errors='coerce')
    master_df = master_df[master_df['played'] > 0].copy()

    # Get fixture
    fixture = get_fixture_for_round(args.year, args.round)

    # Isolate historical data before the prediction round
    logging.info(f"Isolating all historical data before {args.year} Round {args.round}...")
    historical_df = master_df[
        (master_df['Year'] < args.year) |
        ((master_df['Year'] == args.year) & (master_df['Round_Num'] < args.round))
    ].copy()

    if historical_df.empty:
        logging.error("No historical data found before the target prediction round.")
        return

    # Engineer features in PREDICTION mode (no shift - includes latest game)
    df_with_features = engineer_features(historical_df, for_training=False)

    # Add extended features (EWM, trend slope, skewness, TOG-adjusted avg, etc.)
    df_with_features = add_extended_features(df_with_features, for_training=False)

    # Build opponent concession table from the featured historical data
    concession_table = build_opponent_concession_table(df_with_features)

    latest_features_df = get_latest_player_features(df_with_features)

    # Filter to recently active players
    logging.info(f"Filtering for players active in {args.year} or {args.year - 1}...")
    recent_years = [args.year, args.year - 1]
    prediction_data = latest_features_df[latest_features_df['Year'].isin(recent_years)].copy()

    # Override opponent_strength_rating with the actual fixture matchup
    if fixture:
        prediction_data = apply_fixture_opponent(prediction_data, concession_table, fixture)

    logging.info(f"Found {len(prediction_data)} recently active players to generate predictions for.")

    if prediction_data.empty:
        logging.error("Could not prepare any player data for prediction.")
        return

    # Generate predictions
    logging.info("Generating predictions...")
    X_predict = prediction_data[features]
    raw_scores = model.predict(X_predict)

    # Apply linear calibration (variance expansion: slope > 1 restores score spread)
    # Fall back to raw scores if model was trained without calibration
    projected_scores = cal_slope * raw_scores + cal_intercept

    # Derive probabilities
    if has_new_cal:
        # Isotonic regression calibrators: trained on OOF predictions, no Gaussian assumption
        prob_gt_80 = iso_80.predict(projected_scores) * 100
        prob_gt_100 = iso_100.predict(projected_scores) * 100
        prob_gt_120 = iso_120.predict(projected_scores) * 100
        logging.info("Using isotonic probability calibration.")
    else:
        # Gaussian fallback for old model artifacts
        prob_gt_80 = (1 - norm.cdf((80 - projected_scores) / residual_std)) * 100
        prob_gt_100 = (1 - norm.cdf((100 - projected_scores) / residual_std)) * 100
        prob_gt_120 = (1 - norm.cdf((120 - projected_scores) / residual_std)) * 100
        logging.info("Using Gaussian probability fallback (old model artifact).")

    # Build output
    output_df = prediction_data[['Player ID', 'First Name', 'Last Name', 'Team', 'sc_position']].copy()
    output_df['projected_score'] = np.round(projected_scores, 1)
    output_df['probability_gt_80'] = np.round(prob_gt_80, 1)
    output_df['probability_gt_100'] = np.round(prob_gt_100, 1)
    output_df['probability_gt_120'] = np.round(prob_gt_120, 1)
    output_df['pred_round'] = args.round

    output_df['Opponent'] = output_df['Team'].map(fixture)
    output_df['Opponent'] = output_df['Opponent'].fillna('BYE')

    output_df.sort_values(by='projected_score', ascending=False, inplace=True)

    final_cols = ['Player ID', 'First Name', 'Last Name', 'Team', 'sc_position', 'Opponent',
                  'projected_score', 'probability_gt_80', 'probability_gt_100',
                  'probability_gt_120', 'pred_round']
    output_df = output_df[final_cols]

    # Save
    output_dir = Path(CONFIG['PREDICTIONS_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{args.year}_round_{args.round}_predictions.csv"
    output_path = output_dir / output_filename

    output_df.to_csv(output_path, index=False)
    logging.info(f"Predictions successfully saved to: {output_path}")
    print(f"\n--- Top 15 Predicted Players for Round {args.round}, {args.year} ---")
    print(output_df.head(15).to_string())

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"  Total players predicted: {len(output_df)}")
    print(f"  Players with P(>120) >= 20%: {(output_df['probability_gt_120'] >= 20).sum()}")
    print(f"  Players with P(>100) >= 50%: {(output_df['probability_gt_100'] >= 50).sum()}")
    print(f"  Players with P(>80) >= 50%: {(output_df['probability_gt_80'] >= 50).sum()}")
    bye_count = (output_df['Opponent'] == 'BYE').sum()
    if bye_count > 0:
        print(f"  Players on BYE: {bye_count}")
    print(f"\n  Score distribution (calibrated):")
    print(f"    mean={output_df['projected_score'].mean():.1f}, "
          f"std={output_df['projected_score'].std():.1f}, "
          f"max={output_df['projected_score'].max():.1f}, "
          f"min={output_df['projected_score'].min():.1f}")


if __name__ == "__main__":
    main()
