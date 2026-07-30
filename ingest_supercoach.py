import pandas as pd
import requests
import logging
import argparse
from pathlib import Path

# --- Configuration ---
# IMPORTANT: Update these with your specific league details.
CONFIG = {
    "LEAGUE_ID": "13462", # Your specific league ID from the URL
    "ACCESS_TOKEN": "b849fa19223505f95e25c7ec130ebbb82c40fb10",
    "API_URL": "https://www.supercoach.com.au/2025/api/afl/draft/v1/leagues/{league_id}/playersStatus",
    "OUTPUT_DIR": "data/processed" # We save directly to processed as it's already clean data
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_player_availability(league_id: str, access_token: str) -> pd.DataFrame | None:
    """
    Fetches the ownership status for all players in a specific SuperCoach Draft league.

    Args:
        league_id: The ID of the draft league.
        access_token: The authorization token for the API.

    Returns:
        A DataFrame with player IDs and their availability status, or None on failure.
    """
    url = CONFIG['API_URL'].format(league_id=league_id)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "Mozilla/5.0"
    }
    
    logging.info(f"Fetching player status from SuperCoach API for league {league_id}...")
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        player_status_json = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from SuperCoach API: {e}")
        return None
    except ValueError: # Catches JSON decoding errors
        logging.error("Failed to decode JSON from SuperCoach API response.")
        return None

    # Convert the JSON dictionary to a DataFrame
    df = pd.DataFrame.from_dict(player_status_json, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'player_id', 'trade_status': 'availability'}, inplace=True)
    
    # We only need the player ID and their status (owner, free_agent, waiver)
    df = df[['player_id', 'availability']]
    # The API returns player_id as a string, convert to numeric to match our other data
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
    df.dropna(subset=['player_id'], inplace=True)
    df['player_id'] = df['player_id'].astype(int)

    logging.info(f"Successfully fetched availability for {len(df)} players.")
    return df

def main():
    """Main function to run the SuperCoach scraper and save the results."""
    parser = argparse.ArgumentParser(description="Scrape SuperCoach player availability for a league.")
    parser.add_argument("--year", type=int, required=True, help="The year of the season.")
    parser.add_argument("--round", type=int, required=True, help="The round number.")
    args = parser.parse_args()

    player_availability_df = get_player_availability(CONFIG['LEAGUE_ID'], CONFIG['ACCESS_TOKEN'])

    if player_availability_df is not None and not player_availability_df.empty:
        output_dir = Path(CONFIG['OUTPUT_DIR'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{args.year}_round_{args.round}_player_availability.csv"
        output_path = output_dir / output_filename
        
        player_availability_df.to_csv(output_path, index=False)
        logging.info(f"Successfully saved player availability data to: {output_path}")
    else:
        logging.warning("No player availability data was processed.")

if __name__ == "__main__":
    main()
