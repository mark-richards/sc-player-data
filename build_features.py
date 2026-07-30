import pandas as pd
import logging
import argparse
import re
from pathlib import Path
import csv
import io

# --- Configuration ---
CONFIG = {
    "RAW_DATA_DIR": "data/raw/fanfooty",
    "PROCESSED_DATA_DIR": "data/processed",
    "COLUMN_HEADERS": [
        'Fanfooty Match ID', 'Fanfooty Match URL', 'Round', 'Year', 'Player ID',
        'First Name', 'Surname', 'Team', 'null', 'DT', 'SC', 'null2', 'null3',
        'null4', 'Kicks', 'Handballs', 'Marks', 'Tackles', 'Hitouts',
        'Frees for', 'Frees against', 'Goals', 'Behinds',
        'Match progress status', 'Tag', 'Tag Notes', 'Tag 2', 'Tag 2 Notes',
        'null5', 'null6', 'null7', 'null8', 'Position', 'Jumper Number',
        'null9', 'null10', 'null11', 'DT own %', 'SC own %', 'AF own %',
        'null12', 'AF Breakeven', 'null13', 'Contested Possessions',
        'Clearances', 'Clangers', 'Disposal efficiency', 'Time on ground',
        'Metres gained', 'Bench status'
    ],
    "AFL_TEAMS": ['AD', 'BL', 'CA', 'CO', 'ES', 'FR', 'GC', 'GE', 'HW', 'ME', 'NM', 'PA', 'RI', 'SK', 'SY', 'WB', 'WC', 'WS']
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_raw_match_file(file_path: Path) -> list[list]:
    """
    Parses a single raw .txt file from FanFooty using the robust csv module
    to handle complex cases like commas in text fields.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except IOError as e:
        logging.error(f"Could not read file {file_path}: {e}")
        return []

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 6:
        return []

    match_id = file_path.stem
    match_url = f"https://www.fanfooty.com.au/game/direct.php?id={match_id}"
    afl_round, afl_year = None, None
    data_start_index = -1

    try:
        line1_parts = [p.strip() for p in lines[0].split(',')]
        afl_round = line1_parts[4]
        line2_parts = [p.strip() for p in lines[1].split(',')]
        afl_year = line2_parts[1]
    except IndexError as e:
        logging.error(f"Could not parse metadata from {file_path.name}. Error: {e}")
        return []

    for i, line in enumerate(lines):
        if line.strip().startswith(','): continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) > 4:
            data_start_index = i
            break
            
    if data_start_index == -1:
        return []

    player_data_block = "\n".join(lines[data_start_index:])
    csv_file = io.StringIO(player_data_block)
    reader = csv.reader(csv_file)
    
    player_data_for_match = []
    for row in reader:
        line_data = [field.strip() for field in row]
        
        try:
            if len(line_data) > 3 and line_data[2] in CONFIG['AFL_TEAMS'] and line_data[3] not in CONFIG['AFL_TEAMS']:
                line_data.insert(2, None)
        except IndexError:
            pass

        full_player_row = [match_id, match_url, afl_round, afl_year] + line_data
        player_data_for_match.append(full_player_row)
        
    return player_data_for_match

def process_round_data(year: int, round_num: int) -> pd.DataFrame | None:
    """Finds all raw files for a given round, parses them, and returns a DataFrame."""
    input_dir = Path(CONFIG['RAW_DATA_DIR']) / str(year) / f"round_{round_num}"
    if not input_dir.exists():
        return None

    all_files = list(input_dir.glob("*.txt"))
    if not all_files:
        return None

    all_player_data = []
    for file_path in all_files:
        player_data = parse_raw_match_file(file_path)
        all_player_data.extend(player_data)

    if not all_player_data:
        return None

    max_cols = max(len(row) for row in all_player_data)
    padded_data = [row + [None] * (max_cols - len(row)) for row in all_player_data]
    df = pd.DataFrame(padded_data)
    
    num_data_cols = len(df.columns)
    dynamic_headers = CONFIG['COLUMN_HEADERS'].copy()
    if num_data_cols > len(dynamic_headers):
        extra_cols = [f"extra_col_{i+1}" for i in range(num_data_cols - len(dynamic_headers))]
        dynamic_headers.extend(extra_cols)
    
    df.columns = dynamic_headers[:num_data_cols]
    
    text_columns = [
        'Fanfooty Match URL', 'Round', 'First Name', 'Surname', 'Team',
        'Match progress status', 'Tag', 'Tag Notes', 'Tag 2', 'Tag 2 Notes',
        'Position'
    ]
    
    for col in df.columns:
        if col not in text_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- DEFINITIVE FIX: Data Validation Step ---
    # Remove any rows where the SuperCoach score is missing, as they are unusable.
    initial_rows = len(df)
    df.dropna(subset=['SC'], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows with missing SC scores.")

    return df

def main():
    """Main function to allow this script to be run standalone."""
    parser = argparse.ArgumentParser(description="Parse raw FanFooty data and build a clean feature set.")
    parser.add_argument("--year", type=int, required=True, help="The year of the AFL season.")
    parser.add_argument("--round", type=int, required=True, help="The round number to process.")
    args = parser.parse_args()

    logging.info(f"--- Running Standalone Feature Builder for Year: {args.year}, Round: {args.round} ---")
    round_df = process_round_data(args.year, args.round)
    
    if round_df is not None and not round_df.empty:
        output_dir = Path(CONFIG['PROCESSED_DATA_DIR'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{args.year}_round_{args.round}_fanfooty_data.csv"
        output_path = output_dir / output_filename
        round_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved processed data to: {output_path}")
    else:
        logging.warning("No data was processed. No output file created.")

    logging.info("--- Standalone Feature Builder finished ---")

if __name__ == "__main__":
    main()
