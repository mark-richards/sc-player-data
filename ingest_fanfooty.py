import requests
import time
import logging
import argparse
from pathlib import Path
from bs4 import BeautifulSoup

# --- Configuration ---
CONFIG = {
    "BASE_URL": "https://www.fanfooty.com.au",
    "LIVE_URL_TEMPLATE": "/live/{match_id}.txt",
    "FIXTURE_URL_TEMPLATE": "/game/archive.php?year={year}",
    "RAW_DATA_DIR": "data/raw/fanfooty",
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- Logging Setup ---
# Note: When this is imported, the root logger will be used.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_match_ids_for_round(year: int, round_num: int) -> list[str]:
    """Scrapes the FanFooty fixture page to find all match IDs for a specific round."""
    fixture_url = f"{CONFIG['BASE_URL']}{CONFIG['FIXTURE_URL_TEMPLATE'].format(year=year)}"
    logging.info(f"Fetching fixture data from: {fixture_url}")
    
    try:
        response = requests.get(fixture_url, headers={'User-Agent': CONFIG['USER_AGENT']})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch fixture page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    match_ids = []
    
    round_header_tag = None
    # Flexible search for "Round X" or "Opening Round" (relevant for 2024-2026)
    round_string_to_find = f"Round {round_num}"
    for h3_tag in soup.find_all(['h3', 'h2']): # Fanfooty sometimes swaps header levels
        if round_string_to_find in h3_tag.get_text(strip=True):
            round_header_tag = h3_tag
            break

    if not round_header_tag:
        logging.warning(f"Could not find header for '{round_string_to_find}' for {year}.")
        return []

    round_table = round_header_tag.find_next_sibling('table')
    if not round_table:
        logging.error(f"Found the header, but could not find the match table after it.")
        return []

    for link in round_table.find_all('a', href=True):
        # Improved: Look for any link with an ID parameter
        if 'id=' in link['href']:
            try:
                # Extract digits after 'id=', splitting off any extra params with '&'
                match_id = link['href'].split('id=')[-1].split('&')[0]
                if match_id.isdigit() and match_id not in match_ids:
                    match_ids.append(match_id)
            except (IndexError, ValueError):
                continue
    
    if not match_ids:
        logging.warning(f"Found header for '{round_string_to_find}', but no ID-based links found.")
    else:
        logging.info(f"Found {len(match_ids)} unique matches for Round {round_num}: {match_ids}")
        
    return match_ids


def download_match_files(match_ids: list[str], year: int, round_num: int):
    """Downloads the raw text file for each match ID and saves it."""
    if not match_ids:
        logging.warning("No match IDs provided, skipping download.")
        return

    output_dir = Path(CONFIG['RAW_DATA_DIR']) / str(year) / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for match_id in match_ids:
        file_path = output_dir / f"{match_id}.txt"
        if file_path.exists():
            continue

        url = f"{CONFIG['BASE_URL']}{CONFIG['LIVE_URL_TEMPLATE'].format(match_id=match_id)}"
        try:
            response = requests.get(url, headers={'User-Agent': CONFIG['USER_AGENT']})
            response.raise_for_status()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logging.info(f"Successfully saved: {file_path.name}")
            # time.sleep(1) 
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")

def main():
    """Main function to allow this script to be run standalone."""
    parser = argparse.ArgumentParser(description="Scrape FanFooty match data for a specific AFL round.")
    parser.add_argument("--year", type=int, required=True, help="The year of the AFL season.")
    parser.add_argument("--round", type=int, required=True, help="The round number to scrape.")
    args = parser.parse_args()

    logging.info(f"--- Running Standalone FanFooty Scraper for Year: {args.year}, Round: {args.round} ---")
    match_ids = get_match_ids_for_round(args.year, args.round)
    download_match_files(match_ids, args.year, args.round)
    logging.info("--- Standalone Scraper finished ---")

if __name__ == "__main__":
    main()
