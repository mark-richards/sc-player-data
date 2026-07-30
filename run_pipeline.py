import logging
import sys
import os
import pandas as pd
from pathlib import Path
import argparse
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Import the core logic from our component scripts ---
from download_2026_statspacks import download_statspacks
from ingest_fanfooty import get_match_ids_for_round, download_match_files
from build_features import process_round_data
from build_master_dataset import main as build_master_main
from train_model import main as train_main
from run_predictions import main as predict_main

# --- Configuration ---
CONFIG = {
    "PROCESSED_DATA_DIR": "data/processed",
    "MASTER_FANFOOTY_FILE": "master_fanfooty_data.csv",
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def append_to_master_fanfooty(new_round_df: pd.DataFrame):
    """Appends the new round's fanfooty data to the master fanfooty file."""
    master_path = Path(CONFIG['PROCESSED_DATA_DIR']) / CONFIG['MASTER_FANFOOTY_FILE']
    logging.info(f"Appending new data to {master_path}...")

    if master_path.exists():
        master_df = pd.read_csv(master_path, low_memory=False)
        # Ensure we don't add duplicate data if the script is re-run
        existing_games = master_df['Fanfooty Match ID'].unique()
        new_round_df = new_round_df[~new_round_df['Fanfooty Match ID'].isin(existing_games)]
        
        if not new_round_df.empty:
            updated_df = pd.concat([master_df, new_round_df], ignore_index=True)
            updated_df.to_csv(master_path, index=False)
            logging.info(f"Successfully appended {len(new_round_df)} new rows to the master fanfooty file.")
        else:
            logging.info("No new data to append. Master fanfooty file is already up-to-date for this round.")
    else:
        new_round_df.to_csv(master_path, index=False)
        logging.info("Master fanfooty file did not exist. Created a new one with the current round's data.")


def main():
    """Main orchestrator for the weekly SuperCoach pipeline."""
    parser = argparse.ArgumentParser(description="Run the weekly SuperCoach data and prediction pipeline.")
    parser.add_argument("--year", type=int, required=True, help="The year of the AFL season.")
    parser.add_argument("--round", type=int, required=True, help="The round number that has just COMPLETED.")
    parser.add_argument("--skip-statspacks", action="store_true", help="Skip statspack download (use existing files).")
    args = parser.parse_args()

    year = args.year
    completed_round = args.round
    prediction_round = completed_round + 1

    logging.info(f"--- STARTING WEEKLY PIPELINE FOR COMPLETED ROUND: {year} R{completed_round} ---")

    # --- Step 1: Download SC statspacks (force-refresh to pick up the new round's data) ---
    if args.skip_statspacks:
        logging.info(f"\n--- STEP 1: SKIPPING STATSPACK DOWNLOAD (--skip-statspacks) ---")
    else:
        logging.info(f"\n--- STEP 1: DOWNLOADING SC STATSPACKS (force refresh) ---")
        token = os.getenv("SC_API_TOKEN", "")
        if not token:
            logging.error("SC_API_TOKEN not set in .env. Cannot download statspacks. Halting pipeline.")
            return
        downloaded, _, errors = download_statspacks(year=year, token=token, force_refresh=True)
        if downloaded == 0 and errors > 0:
            logging.error("Statspack download failed entirely. Halting pipeline.")
            return

    # --- Step 2: Ingest fanfooty raw data for the completed round ---
    logging.info(f"\n--- STEP 2: INGESTING FANFOOTY RAW DATA ---")
    match_ids = get_match_ids_for_round(year, completed_round)
    if not match_ids:
        logging.error(f"Could not find any matches for {year} R{completed_round}. Halting pipeline.")
        return
    download_match_files(match_ids, year, completed_round)

    # --- Step 3: Process fanfooty raw data into a clean CSV ---
    logging.info(f"\n--- STEP 3: PROCESSING FANFOOTY RAW DATA ---")
    round_df = process_round_data(year, completed_round)
    if round_df is None or round_df.empty:
        logging.error(f"Failed to process fanfooty data for {year} R{completed_round}. Halting pipeline.")
        return

    # Save the per-round processed CSV (used by the newsletter's Role Watch)
    processed_dir = Path(CONFIG['PROCESSED_DATA_DIR'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    per_round_path = processed_dir / f"{year}_round_{completed_round}_fanfooty_data.csv"
    round_df.to_csv(per_round_path, index=False)
    logging.info(f"Saved per-round processed data to: {per_round_path}")

    # --- Step 4: Append to master fanfooty file ---
    logging.info(f"\n--- STEP 4: UPDATING MASTER FANFOOTY FILE ---")
    append_to_master_fanfooty(round_df)

    # --- Step 5: Rebuild master player dataset (SC statspacks + fanfooty enrichment) ---
    logging.info(f"\n--- STEP 5: REBUILDING MASTER PLAYER DATASET ---")
    build_master_main()

    # --- Step 6: Re-train the model ---
    logging.info(f"\n--- STEP 6: RE-TRAINING MODEL ---")
    train_main()

    # --- Step 7: Generate predictions for the next round ---
    logging.info(f"\n--- STEP 7: GENERATING PREDICTIONS FOR R{prediction_round} ---")
    sys.argv = [sys.argv[0], '--year', str(year), '--round', str(prediction_round)]
    predict_main()

    # --- Step 8: Generate waiver newsletter ---
    logging.info(f"\n--- STEP 8: GENERATING WAIVER NEWSLETTER (Round {completed_round}) ---")
    try:
        from run_waiver_newsletter import run_pipeline as waiver_pipeline
        md_path = waiver_pipeline(round_override=completed_round, dry_run=False)
        if md_path:
            logging.info(f"Newsletter saved to: {md_path}")
        else:
            logging.warning("Newsletter pipeline returned no output — check waiver logs.")
    except Exception as exc:
        logging.error(f"Waiver newsletter failed (non-fatal): {exc}", exc_info=True)

    logging.info(f"\n--- WEEKLY PIPELINE COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()
