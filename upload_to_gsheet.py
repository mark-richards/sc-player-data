import gspread
import pandas as pd
import logging
import argparse
from pathlib import Path

# --- Configuration ---
CONFIG = {
    "PREDICTIONS_DIR": "data/predictions",
    "GSHEET_NAME": "SuperCoach Weekly Predictions", 
    "WORKSHEET_NAME": "Waiver Targets",
    # --- UPDATED: Point to the new service account key file ---
    "SERVICE_ACCOUNT_FILE": "service_account.json"
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_gspread() -> gspread.Client | None:
    """
    Handles Google authentication using a service account.
    This method is for automated, non-interactive execution.
    """
    try:
        # Use an absolute path to find the credentials file
        script_dir = Path(__file__).resolve().parent
        creds_path = script_dir / CONFIG['SERVICE_ACCOUNT_FILE']

        if not creds_path.exists():
            logging.error(f"Service account file not found at '{creds_path}'. Please follow setup instructions.")
            return None
            
        return gspread.service_account(filename=str(creds_path))
    except Exception as e:
        logging.error(f"Failed to authorize with Google Sheets using service account: {e}")
        return None

def main():
    """Main function to upload predictions to Google Sheets."""
    parser = argparse.ArgumentParser(description="Upload SuperCoach predictions to Google Sheets.")
    parser.add_argument("--year", type=int, required=True, help="The year of the predictions.")
    parser.add_argument("--round", type=int, required=True, help="The round number of the predictions.")
    args = parser.parse_args()

    prediction_filename = f"{args.year}_round_{args.round}_predictions.csv"
    prediction_path = Path(CONFIG['PREDICTIONS_DIR']) / prediction_filename
    
    if not prediction_path.exists():
        logging.error(f"Prediction file not found: {prediction_path}")
        return

    logging.info(f"Loading predictions from {prediction_path}...")
    df = pd.read_csv(prediction_path)
    df.fillna('', inplace=True)

    gc = authenticate_gspread()
    if not gc:
        return

    try:
        spreadsheet = gc.open(CONFIG['GSHEET_NAME'])
    except gspread.exceptions.SpreadsheetNotFound:
        logging.warning(f"Spreadsheet not found. Creating a new one named '{CONFIG['GSHEET_NAME']}'.")
        spreadsheet = gc.create(CONFIG['GSHEET_NAME'])
        # Share the sheet with your personal email so you can find it easily
        # NOTE: Replace with your actual email address
        spreadsheet.share('mark.richards0123@gmail.com', perm_type='user', role='writer')


    try:
        worksheet = spreadsheet.worksheet(CONFIG['WORKSHEET_NAME'])
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=CONFIG['WORKSHEET_NAME'], rows="1000", cols="20")

    logging.info("Clearing existing data and uploading new predictions...")
    worksheet.clear()
    data_to_upload = [df.columns.values.tolist()] + df.values.tolist()
    worksheet.update(data_to_upload, 'A1')

    logging.info("Successfully uploaded predictions to Google Sheets!")


if __name__ == "__main__":
    main()