"""
Build Master Player Dataset
============================
Merges SuperCoach completeStatspack data (backbone) with fanfooty enrichment data.

SC statspack provides: SC points, price, minutes, venue, raw stats (2008-2025)
Fanfooty provides: tags, contested possessions, clearances, clangers, disposal %, metres gained

Output: data/processed/master_player_data.csv
"""

import json
import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "STATSPACK_DIR": Path(f"data/raw/supercoach/{os.getenv('SEASON_YEAR', '2026')}"),
    "PLAYER_LIST": Path("draft_prep/SC 2026/2026_SC_Player_list.csv"),
    "FANFOOTY_MASTER": Path("data/processed/master_fanfooty_data.csv"),
    "OUTPUT_FILE": Path("data/processed/master_player_data.csv"),
}

# SC team abbrev -> fanfooty 2-letter code
SC_TO_FF_TEAM = {
    "ADE": "AD", "BRL": "BL", "CAR": "CA", "COL": "CO",
    "ESS": "ES", "FRE": "FR", "GCS": "GC", "GEE": "GE",
    "GWS": "WS", "HAW": "HW", "MEL": "ME", "NTH": "NM",
    "PTA": "PA", "RIC": "RI", "STK": "SK", "SYD": "SY",
    "WBD": "WB", "WCE": "WC",
}

# Fields to keep from each SC statspack record
SC_SCALAR_FIELDS = [
    "player_id", "season", "round", "points", "price", "price_change",
    "total_points", "total_games", "played", "games",
    "minutes_played", "position", "round_position",
    "kicks", "handballs", "marks", "tackles", "hitouts",
    "freekicks_for", "freekicks_against", "goals", "behinds",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_player_list() -> pd.DataFrame:
    """Load SC player list for ID mapping and metadata."""
    logging.info("Step 1: Loading SC player list...")
    df = pd.read_csv(CONFIG["PLAYER_LIST"])
    df = df[["id", "feed_id", "first_name", "last_name", "position", "long_team"]].copy()
    df.rename(columns={"id": "sc_player_id", "position": "sc_position"}, inplace=True)
    logging.info(f"  Loaded {len(df)} players with ID mapping")
    return df


def flatten_statspacks(player_list: pd.DataFrame) -> pd.DataFrame:
    """Flatten all statspack JSONs into a single DataFrame."""
    logging.info("Step 2: Flattening SC statspack JSON files...")

    id_to_meta = player_list.set_index("sc_player_id").to_dict("index")
    all_records = []

    json_files = sorted(CONFIG["STATSPACK_DIR"].glob("completeStatspack_player_id=*.json"))
    logging.info(f"  Found {len(json_files)} JSON files")

    for fpath in tqdm(json_files, desc="  Processing statspacks"):
        sc_pid = int(fpath.stem.split("=")[1])

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        player_stats = data.get("playerStats", [])
        if not player_stats:
            continue

        meta = id_to_meta.get(sc_pid, {})
        feed_id = meta.get("feed_id")
        first_name = meta.get("first_name", "")
        last_name = meta.get("last_name", "")
        sc_position = meta.get("sc_position", "")

        for rec in player_stats:
            row = {field: rec.get(field) for field in SC_SCALAR_FIELDS}

            # Flatten nested objects
            team = rec.get("team") or {}
            row["team_abbrev"] = team.get("abbrev", "")
            row["team_name"] = team.get("name", "")

            opp = rec.get("opp") or {}
            row["opp_abbrev"] = opp.get("abbrev", "")
            row["opp_name"] = opp.get("name", "")

            ven = rec.get("ven") or {}
            row["venue_name"] = ven.get("display_name", "")

            # Add player metadata
            row["feed_id"] = feed_id
            row["first_name"] = first_name
            row["last_name"] = last_name
            row["sc_position"] = sc_position

            all_records.append(row)

    df = pd.DataFrame(all_records)
    logging.info(f"  Flattened {len(df)} total records from {len(json_files)} files")
    return df


def prepare_fanfooty() -> pd.DataFrame:
    """Load and clean fanfooty master data, keeping only enrichment columns."""
    logging.info("Step 3: Preparing fanfooty enrichment data...")

    # Define columns we need (avoid loading the 40+ extra columns)
    usecols = [
        "Fanfooty Match ID", "Round", "Year", "Player ID",
        "DT", "SC",
        "Tag", "Tag Notes", "Tag 2", "Tag 2 Notes",
        "Position",
        "Kicks", "Handballs", "Marks", "Tackles", "Hitouts", "Goals", "Behinds",
        "Frees for", "Frees against",
        "Contested Possessions", "Clearances", "Clangers",
        "Disposal efficiency", "Time on ground", "Metres gained",
        "Bench status", "SC own %", "DT own %",
    ]

    df = pd.read_csv(CONFIG["FANFOOTY_MASTER"], usecols=usecols, low_memory=False)
    logging.info(f"  Loaded {len(df)} fanfooty records")

    # Clean types
    df = df.dropna(subset=["Player ID", "Year", "Round"])
    df["Year"] = df["Year"].astype(int)
    df["Player ID"] = df["Player ID"].astype(int)

    # Extract round number from "R1", "R2" etc.
    df["Round_Num"] = df["Round"].str.extract(r"(\d+)")
    df = df.dropna(subset=["Round_Num"])
    df["Round_Num"] = df["Round_Num"].astype(int)

    # Rename for merge
    df.rename(columns={
        "Player ID": "feed_id",
        "Year": "ff_year",
        "Position": "ff_position",
        "SC": "ff_SC",  # Keep as cross-reference, not primary
    }, inplace=True)

    # Keep only the enrichment columns
    ff_cols = [
        "feed_id", "ff_year", "Round_Num",
        "Fanfooty Match ID", "DT", "ff_SC",
        "Tag", "Tag Notes", "Tag 2", "Tag 2 Notes",
        "ff_position",
        "Kicks", "Handballs", "Marks", "Tackles", "Hitouts", "Goals", "Behinds",
        "Frees for", "Frees against",
        "Contested Possessions", "Clearances", "Clangers",
        "Disposal efficiency", "Time on ground", "Metres gained",
        "Bench status", "SC own %", "DT own %",
    ]
    df = df[ff_cols].copy()

    # De-duplicate: one record per player-year-round (take first if multiple)
    df = df.drop_duplicates(subset=["feed_id", "ff_year", "Round_Num"], keep="first")

    logging.info(f"  Prepared {len(df)} unique fanfooty enrichment records")
    return df


def load_dfs_data(years: list[int]) -> pd.DataFrame:
    """Load DFS Australia CBA, kick-in, and ruck contest CSVs into one DataFrame."""
    logging.info("Loading DFS Australia data...")
    dfs_dir = Path("data/raw/dfsaustralia")
    all_frames = []

    for year in years:
        year_dir = dfs_dir / str(year)
        cbas_path = year_dir / f"cbas_{year}.csv"
        kickins_path = year_dir / f"kickins_{year}.csv"
        ruck_path = year_dir / f"ruck_contests_{year}.csv"

        if not any(p.exists() for p in [cbas_path, kickins_path, ruck_path]):
            continue

        frames = []
        if cbas_path.exists():
            df = pd.read_csv(cbas_path)
            df["year"] = year
            frames.append(df)

        if frames:
            base = frames[0]
        else:
            continue

        if kickins_path.exists():
            ki = pd.read_csv(kickins_path)
            base = base.merge(ki, on=["feed_id", "round_num"], how="outer")
            base["year"] = year

        if ruck_path.exists():
            rc = pd.read_csv(ruck_path)
            base = base.merge(rc, on=["feed_id", "round_num"], how="outer")
            base["year"] = year

        all_frames.append(base)

    if not all_frames:
        logging.warning("  No DFS Australia data found.")
        return pd.DataFrame(columns=["feed_id", "year", "round_num", "cba_pct",
                                     "kickin_count", "kickin_po_pct", "ruck_contest_count"])

    combined = pd.concat(all_frames, ignore_index=True)
    combined["feed_id"] = pd.to_numeric(combined["feed_id"], errors="coerce").astype("Int64")
    combined["year"] = combined["year"].astype(int)
    combined["round_num"] = combined["round_num"].astype(int)
    logging.info(f"  Loaded {len(combined)} DFS rows across {combined['year'].nunique()} seasons")
    return combined


def merge_dfs_data(merged_df: pd.DataFrame, dfs_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join DFS Australia usage metrics into the SC+fanfooty merged dataset."""
    if dfs_df.empty:
        logging.warning("  DFS data is empty — skipping DFS merge.")
        return merged_df

    logging.info("Merging DFS Australia data...")
    merged_df["_pid"] = pd.to_numeric(merged_df["Player ID"], errors="coerce").astype("Int64")
    result = merged_df.merge(
        dfs_df,
        left_on=["_pid", "Year", "Round_Num"],
        right_on=["feed_id", "year", "round_num"],
        how="left",
        suffixes=("", "_dfs"),
    )
    result.drop(columns=["_pid", "feed_id", "year", "round_num"], inplace=True, errors="ignore")

    dfs_cols = ["cba_pct", "kickin_count", "kickin_po_pct", "ruck_contest_count"]
    matched = result["cba_pct"].notna().sum()
    logging.info(f"  DFS merge: {matched}/{len(result)} rows matched on CBA%")
    for col in dfs_cols:
        if col not in result.columns:
            result[col] = None
    return result


def merge_datasets(sc_df: pd.DataFrame, ff_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join SC data with fanfooty enrichment."""
    logging.info("Step 4: Merging SC backbone with fanfooty enrichment...")

    # Prepare SC join keys
    sc_df["season_int"] = sc_df["season"].astype(int)
    sc_df["round_int"] = sc_df["round"].astype(int)
    sc_df["feed_id"] = sc_df["feed_id"].astype("Int64")

    # Merge
    merged = sc_df.merge(
        ff_df,
        left_on=["feed_id", "season_int", "round_int"],
        right_on=["feed_id", "ff_year", "Round_Num"],
        how="left",
        suffixes=("", "_ff"),
    )

    # Flag fanfooty coverage
    merged["has_fanfooty_data"] = merged["Tag"].notna()

    # Drop temporary join columns
    merged.drop(columns=["ff_year", "Round_Num", "season_int", "round_int"], inplace=True, errors="ignore")

    matched = merged["has_fanfooty_data"].sum()
    total = len(merged)
    logging.info(f"  Merged: {total} total rows, {matched} with fanfooty data ({matched/total*100:.1f}%)")

    return merged


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, add derived fields, set final column order."""
    logging.info("Step 5-6: Finalizing columns and team mapping...")

    # Add Team column (fanfooty 2-letter code) for backward compatibility
    df["Team"] = df["team_abbrev"].map(SC_TO_FF_TEAM).fillna("")

    # Derived columns
    df["is_pre_season"] = (df["round"] == 0).astype(int)
    df["Round"] = df["round"].apply(lambda r: f"R{r}" if r > 0 else "R0")

    # Rename for compatibility with train_model.py
    df.rename(columns={
        "feed_id": "Player ID",
        "first_name": "First Name",
        "last_name": "Last Name",
        "season": "Year",
        "round": "Round_Num",
        "points": "SC",
        "kicks": "Kicks",
        "handballs": "Handballs",
        "marks": "Marks",
        "tackles": "Tackles",
        "hitouts": "Hitouts",
        "freekicks_for": "Frees for",
        "freekicks_against": "Frees against",
        "goals": "Goals",
        "behinds": "Behinds",
        "position": "position_rank",
        "round_position": "round_position_rank",
    }, inplace=True)

    # Convert Year to int
    df["Year"] = df["Year"].astype(int)

    # Final column order
    output_cols = [
        "Player ID", "First Name", "Last Name", "Team", "sc_position",
        "Year", "Round", "Round_Num", "is_pre_season",
        "SC", "price", "price_change", "minutes_played", "played",
        "position_rank", "round_position_rank",
        "Kicks", "Handballs", "Marks", "Tackles", "Hitouts",
        "Frees for", "Frees against", "Goals", "Behinds",
        "team_abbrev", "team_name", "opp_abbrev", "opp_name",
        "venue_name",
        "total_points", "total_games",
        "has_fanfooty_data",
        "DT", "ff_SC", "Tag", "Tag Notes", "Tag 2", "Tag 2 Notes",
        "ff_position", "Contested Possessions", "Clearances", "Clangers",
        "Disposal efficiency", "Time on ground", "Metres gained",
        "Bench status", "Fanfooty Match ID", "SC own %", "DT own %",
    ]

    # Only include columns that exist (handles edge cases)
    available = [c for c in output_cols if c in df.columns]
    df = df[available].copy()

    # Sort for usability
    df.sort_values(by=["Player ID", "Year", "Round_Num"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logging.info(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics for verification."""
    print("\n" + "=" * 70)
    print("MASTER PLAYER DATASET - SUMMARY")
    print("=" * 70)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Unique players: {df['Player ID'].nunique()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")

    print("\n--- Rows per Season ---")
    year_counts = df[df["Round_Num"] > 0].groupby("Year").agg(
        records=("SC", "size"),
        played=("played", "sum"),
        ff_matched=("has_fanfooty_data", "sum"),
    )
    for year, row in year_counts.iterrows():
        pct = row["ff_matched"] / row["records"] * 100 if row["records"] > 0 else 0
        print(f"  {year}: {int(row['records']):>5} records, {int(row['played']):>5} played, "
              f"{int(row['ff_matched']):>5} with fanfooty ({pct:.0f}%)")

    print("\n--- Column List ---")
    for i, col in enumerate(df.columns):
        non_null = int(df.iloc[:, i].notna().sum())
        print(f"  {i+1:>2}. {str(col):<30s} {non_null:>6,} non-null")

    # Verify a known player
    cripps = df[df["Player ID"] == 990704]
    if not cripps.empty:
        c2025 = cripps[(cripps["Year"] == 2025) & (cripps["played"] > 0)]
        print(f"\n--- Verification: Patrick Cripps (990704) ---")
        print(f"  Total records: {len(cripps)}")
        print(f"  2025 games played: {len(c2025)}")
        if not c2025.empty:
            print(f"  2025 total SC points: {c2025['SC'].sum()}")
            print(f"  2025 avg SC: {c2025['SC'].mean():.1f}")
            ff_matched = c2025["has_fanfooty_data"].sum()
            print(f"  2025 fanfooty coverage: {ff_matched}/{len(c2025)}")
            sample = c2025.iloc[0]
            print(f"  Sample R{int(sample['Round_Num'])}: SC={sample['SC']}, "
                  f"Tag={sample.get('Tag','')}, CP={sample.get('Contested Possessions','')}")

    print("\n" + "=" * 70)


def update_current_season(year: int, player_match_path: Path | None = None) -> int:
    """
    Merge current-season SC data into master_player_data.csv.

    Reads data/live/player_match_results.csv (written by api_ingest.refresh_2026_data),
    normalises to master format, replaces any existing rows for `year`, and saves.

    Returns the number of rows added for `year`.
    """
    from pathlib import Path as _Path

    # Default staging path — written by api_ingest.refresh_2026_data()
    project_root = _Path(__file__).resolve().parent
    if player_match_path is None:
        player_match_path = project_root / "data" / "live" / "player_match_results.csv"

    if not player_match_path.exists():
        logging.warning(f"update_current_season: {player_match_path} not found — skipping.")
        return 0

    # Load staging file
    pm = pd.read_csv(player_match_path, low_memory=False)
    if pm.empty:
        logging.warning("update_current_season: player_match_results.csv is empty — skipping.")
        return 0

    # Normalise to master column names
    pm = pm.rename(columns={
        "feed_id":    "Player ID",
        "first_name": "First Name",
        "last_name":  "Last Name",
        "round_x":    "Round_Num",
        "points_x":   "SC",
        "position":   "sc_position",
        "team":       "team_abbrev",
    })
    pm["Year"]         = year
    pm["Round"]        = pm["Round_Num"].apply(lambda r: f"R{int(r)}")
    pm["is_pre_season"] = (pm["Round_Num"] == 0).astype(int)
    pm["played"]       = (pm["SC"] > 0).astype(int)
    pm["Team"]         = pm["team_abbrev"].map(SC_TO_FF_TEAM).fillna("")
    pm["Player ID"]    = pd.to_numeric(pm["Player ID"], errors="coerce").astype("Int64")

    # Keep only columns that exist in master output
    master_cols = [
        "Player ID", "First Name", "Last Name", "Team", "sc_position",
        "Year", "Round", "Round_Num", "is_pre_season",
        "SC", "played", "team_abbrev",
    ]
    available = [c for c in master_cols if c in pm.columns]
    pm = pm[available].drop_duplicates(subset=["Player ID", "Round_Num"])

    # Enrich with fanfooty data (TOG, kicks, tags, etc.)
    try:
        ff_df = prepare_fanfooty()
        pm = pm.merge(
            ff_df,
            left_on=["Player ID", "Year", "Round_Num"],
            right_on=["feed_id", "ff_year", "Round_Num"],
            how="left",
        )
        pm["has_fanfooty_data"] = pm["Tag"].notna()
        pm.drop(columns=["feed_id", "ff_year"], inplace=True, errors="ignore")
        matched = pm["has_fanfooty_data"].sum()
        logging.info(f"update_current_season: fanfooty enrichment matched {matched}/{len(pm)} rows")
    except Exception as e:
        logging.warning(f"update_current_season: fanfooty merge failed — {e}")

    # Enrich with DFS Australia data (CBA%, kick-ins, ruck contests)
    try:
        dfs_df = load_dfs_data([year])
        pm = merge_dfs_data(pm, dfs_df)
    except Exception as e:
        logging.warning(f"update_current_season: DFS merge failed — {e}")

    # Load existing master, drop any rows for this year, and append fresh data
    master_path = CONFIG["OUTPUT_FILE"]
    if master_path.exists():
        master = pd.read_csv(master_path, low_memory=False)
        master = master[master["Year"] != year].copy()
    else:
        master = pd.DataFrame()

    updated = pd.concat([master, pm], ignore_index=True)
    updated.sort_values(by=["Player ID", "Year", "Round_Num"], inplace=True)
    updated.reset_index(drop=True, inplace=True)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(master_path, index=False)

    n = len(pm)
    logging.info(f"update_current_season: added {n} rows for {year} → {master_path}")
    return n


def main():
    # Step 1: Load player list
    player_list = load_player_list()

    # Step 2: Flatten statspacks
    sc_df = flatten_statspacks(player_list)

    # Step 3: Prepare fanfooty
    ff_df = prepare_fanfooty()

    # Step 4: Merge SC + fanfooty
    merged = merge_datasets(sc_df, ff_df)

    # Steps 5-6: Finalize (renames feed_id→Player ID, season→Year, round→Round_Num)
    master = finalize(merged)

    # Step 4b: Merge DFS Australia usage metrics (must run post-finalize for column names)
    years = sorted(master["Year"].dropna().astype(int).unique().tolist())
    dfs_df = load_dfs_data(years)
    master = merge_dfs_data(master, dfs_df)

    # Step 7: Save
    logging.info(f"Step 7: Saving to {CONFIG['OUTPUT_FILE']}...")
    CONFIG["OUTPUT_FILE"].parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(CONFIG["OUTPUT_FILE"], index=False)
    logging.info(f"  Saved {len(master):,} rows to {CONFIG['OUTPUT_FILE']}")

    # Print summary
    print_summary(master)


if __name__ == "__main__":
    main()
