import logging
import pandas as pd
from pathlib import Path
import time

# --- Import functions directly from our other scripts ---
# This is the key to the performance improvement.
from ingest_fanfooty import get_match_ids_for_round, download_match_files
from build_features import process_round_data

# --- Configuration ---
CONFIG = {
    "START_YEAR": 2011,
    "END_YEAR": 2025,
    "ROUNDS_PER_SEASON": 24, # A reasonable upper limit for home-and-away season
    "PROCESSED_DATA_DIR": "data/processed",
    "FINAL_OUTPUT_FILE": "master_fanfooty_data.csv"
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline_for_round(year: int, round_num: int):
    """
    Executes the ingest and build steps for a single round by calling functions directly.

    Args:
        year: The year to process.
        round_num: The round number to process.
    """
    logging.info(f"--- Processing: Year {year}, Round {round_num} ---")

    # --- Step 1: Ingest Raw Data ---
    match_ids = get_match_ids_for_round(year, round_num)
    if not match_ids:
        logging.warning(f"No match IDs found for {year} R{round_num}. Skipping.")
        return # Skip the rest of the steps for this round
    
    download_match_files(match_ids, year, round_num)

    # --- Step 2: Build Features ---
    round_df = process_round_data(year, round_num)
    
    if round_df is not None and not round_df.empty:
        output_dir = Path(CONFIG['PROCESSED_DATA_DIR'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{year}_round_{round_num}_fanfooty_data.csv"
        output_path = output_dir / output_filename
        round_df.to_csv(output_path, index=False)
        logging.info(f"Successfully processed and saved data for {year} R{round_num}.")
    else:
        logging.warning(f"No data was processed for {year} R{round_num}. No output file created.")


def combine_processed_files():
    """Finds all individual processed CSVs and combines them into one master file."""
    logging.info("--- Combining all processed CSV files into a master file ---")
    processed_dir = Path(CONFIG['PROCESSED_DATA_DIR'])
    all_csvs = list(processed_dir.glob("*_round_*_fanfooty_data.csv"))

    if not all_csvs:
        logging.error("No processed CSV files found to combine. Did the backfill run correctly?")
        return

    logging.info(f"Found {len(all_csvs)} files to combine.")
    
    df_list = [pd.read_csv(file) for file in all_csvs]
    master_df = pd.concat(df_list, ignore_index=True)
    
    output_path = processed_dir / CONFIG['FINAL_OUTPUT_FILE']
    master_df.to_csv(output_path, index=False)
    
    logging.info(f"Master file created with {len(master_df)} rows.")
    logging.info(f"Successfully saved master data to: {output_path}")


def main():
    """Main function to run the entire backfill and combination process."""
    
    # --- Part 1: Run the historical backfill ---
    for year in range(CONFIG['START_YEAR'], CONFIG['END_YEAR'] + 1):
        for round_num in range(1, CONFIG['ROUNDS_PER_SEASON'] + 1):
            output_filename = f"{year}_round_{round_num}_fanfooty_data.csv"
            output_path = Path(CONFIG['PROCESSED_DATA_DIR']) / output_filename
            if output_path.exists():
                logging.info(f"Output file for {year} R{round_num} already exists. Skipping.")
                continue

            run_pipeline_for_round(year, round_num)
            # We can remove the sleep timer as there's no process overhead anymore
            # The sleep inside download_match_files is still important for the web server

    # --- Part 2: Combine all the generated files ---
    combine_processed_files()

if __name__ == "__main__":
    main()
