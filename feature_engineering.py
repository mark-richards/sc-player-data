"""
Shared Feature Engineering Module
===================================
Single source of truth for all feature computation.
Both train_model.py and run_predictions.py import from this module.

Key design: rolling features use shift(1) during training to prevent data leakage.
For prediction, shift is omitted so rolling windows include the most recent game,
which is mathematically equivalent to shift(1) on a hypothetical next row.
"""

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Constants ---

SC_TO_FF_TEAM = {
    "ADE": "AD", "BRL": "BL", "CAR": "CA", "COL": "CO",
    "ESS": "ES", "FRE": "FR", "GCS": "GC", "GEE": "GE",
    "GWS": "WS", "HAW": "HW", "MEL": "ME", "NTH": "NM",
    "PTA": "PA", "RIC": "RI", "STK": "SK", "SYD": "SY",
    "WBD": "WB", "WCE": "WC",
}

IMPORTANT_TAGS = ['tagger', 'ruck', 'wing', 'job', 'star', 'hot', 'gun', 'shovel', 'guard', 'pocket']

# Tag Notes keyword patterns for role detection
ON_BALL_KEYWORDS = r'on\s+ball|starting\s+mid|central\s+mid|inside\s+mid|high\s+half\s+fwd|centr(?:al)?\s+mid|off\s+wing\s+mid'
MANAGED_KEYWORDS = r'manag(?:ed|ing)|limited\s+(?:min|time)|rested?\s+(?:after|this)|soreness|sore\s+\w+|medical\s+sub|test\s+to\s+play|quarter\s+time\s+sub'

ROLLING_WINDOWS = [3, 6, 10]

ROLLING_STATS = [
    'SC', 'Kicks', 'Handballs', 'Marks', 'Tackles', 'Hitouts',
    'Contested Possessions', 'Metres gained', 'Time on ground',
    'Goals', 'Clearances', 'Disposal efficiency', 'minutes_played',
    'cba_pct', 'kickin_count', 'kickin_po_pct', 'ruck_contest_count',
]

FF_COLS_TO_FILL = [
    'Tag', 'Tag 2', 'Tag Notes', 'Tag 2 Notes', 'ff_position',
    'Contested Possessions', 'Clearances', 'Clangers',
    'Disposal efficiency', 'Time on ground', 'Metres gained',
    'cba_pct', 'ruck_contest_count', 'kickin_count', 'kickin_po_pct',
]


def get_feature_columns():
    """Returns the canonical list of feature columns used by the model."""
    features = []

    # Multi-window rolling averages
    for stat in ROLLING_STATS:
        for w in ROLLING_WINDOWS:
            features.append(f'{stat}_rolling_avg_{w}_games')

    # Form trajectory
    features.append('SC_form_delta_3v10')
    features.append('minutes_form_delta_3v10')

    # Price-based
    features.append('price_per_point')
    features.append('price_momentum')

    # Experience
    features.append('career_games')
    features.append('season_games')

    # Scoring ceiling/floor
    features.append('SC_max_last_10')
    features.append('SC_min_last_10')

    # Opponent strength
    features.append('opponent_strength_rating')
    features.append('opponent_ceiling_rating')  # % of 100+ games opponent concedes to this position

    # Role tags
    for tag in IMPORTANT_TAGS:
        features.append(f'role_is_{tag}')

    # Tag Notes role signals
    features.append('tag_notes_on_ball')   # explicit on-ball role this game
    features.append('tag_notes_managed')   # load managed / sore / limited minutes

    # Volatility and durability
    features.append('consistency_last_10')
    features.append('games_played_last_season')

    # Ceiling potential
    features.append('SC_ceiling_rate_10')  # fraction of last 10 games scoring 100+

    # Debut / early career flag
    features.append('is_debut_season')     # first 3 career games or first 2 of season

    return features


def get_opponent(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an 'Opponent' column using SC opp_abbrev mapped to fanfooty 2-letter codes."""
    df['Opponent'] = df['opp_abbrev'].map(SC_TO_FF_TEAM).fillna(df['opp_abbrev'])
    return df


def simplify_position(position_str: str) -> str:
    """Simplifies complex position strings to one of four core positions."""
    if not isinstance(position_str, str):
        return 'Unknown'
    if 'Midfielder' in position_str:
        return 'MID'
    if 'Back' in position_str:
        return 'DEF'
    if 'Forward' in position_str:
        return 'FWD'
    if 'Ruck' in position_str:
        return 'RUCK'
    return 'Unknown'


def build_opponent_concession_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build opponent strength/ceiling tables from df (training data only).

    Returns (opp_concession, opp_ceiling) with the Year+1 lag already applied,
    ready to merge into any year's feature rows.

    Separating this from engineer_features lets CV folds build the table
    exclusively from training years, preventing cross-fold contamination.
    """
    opp_concession = df.groupby(['Year', 'Opponent', 'simple_pos'])['SC'].mean().reset_index()
    opp_concession.rename(columns={'SC': 'opponent_strength_rating'}, inplace=True)
    opp_concession['Year'] = opp_concession['Year'] + 1  # lag: use last year's data

    opp_ceiling = (
        df.groupby(['Year', 'Opponent', 'simple_pos'])['SC']
        .apply(lambda x: (x >= 100).mean())
        .reset_index(name='opponent_ceiling_rating')
    )
    opp_ceiling['Year'] = opp_ceiling['Year'] + 1  # lag

    return opp_concession, opp_ceiling


def _merge_opponent_features(
    df: pd.DataFrame,
    opp_concession: pd.DataFrame,
    opp_ceiling: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pre-built opponent tables into df, filling missing entries with means."""
    mean_sc = df['SC'].mean() if 'SC' in df.columns else opp_concession['opponent_strength_rating'].mean()
    df = pd.merge(df, opp_concession, on=['Year', 'Opponent', 'simple_pos'], how='left')
    df = pd.merge(df, opp_ceiling, on=['Year', 'Opponent', 'simple_pos'], how='left')
    df['opponent_strength_rating'] = df['opponent_strength_rating'].fillna(mean_sc)
    df['opponent_ceiling_rating'] = df['opponent_ceiling_rating'].fillna(
        df['opponent_ceiling_rating'].mean()
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    for_training: bool = True,
    opp_concession: pd.DataFrame | None = None,
    opp_ceiling: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Core feature engineering function. Single source of truth.

    Parameters:
        df: Master player data DataFrame (already filtered to played > 0).
        for_training: If True, uses shift(1) on rolling features to prevent data leakage.
                      If False (prediction), no shift - rolling windows include the most
                      recent game, correct for predicting the NEXT game.
        opp_concession: Pre-built opponent strength table (from training data only).
                        If None, built from df itself (backward-compatible default).
        opp_ceiling:    Pre-built opponent ceiling table (from training data only).
                        If None, built from df itself.
    """
    logging.info("Starting feature engineering...")

    df['Round_Num'] = pd.to_numeric(df['Round_Num'], errors='coerce')
    df.sort_values(by=['Player ID', 'Year', 'Round_Num'], inplace=True)

    # Filter to played games only
    df = df[df['played'] > 0].copy()
    logging.info(f"Filtered to {len(df)} played records")

    # Forward-fill fanfooty columns within each player to handle NaN gaps
    ff_cols_present = [c for c in FF_COLS_TO_FILL if c in df.columns]
    logging.info(f"Forward-filling {len(ff_cols_present)} fanfooty columns...")
    df[ff_cols_present] = df.groupby('Player ID')[ff_cols_present].ffill()

    # --- Opponent strength rating ---
    df = get_opponent(df)
    df['simple_pos'] = df['ff_position'].apply(simplify_position)

    if opp_concession is None or opp_ceiling is None:
        # Backward-compatible: build from df itself (used during prediction / full-data retrain)
        opp_concession, opp_ceiling = build_opponent_concession_tables(df)

    df = _merge_opponent_features(df, opp_concession, opp_ceiling)

    # --- Role tag features ---
    logging.info("Engineering player role features from tags...")
    df['Tag'] = df['Tag'].fillna('')
    df['Tag 2'] = df['Tag 2'].fillna('')
    df['all_tags'] = df['Tag'].str.lower() + ' ' + df['Tag 2'].str.lower()
    for tag in IMPORTANT_TAGS:
        df[f'role_is_{tag}'] = df['all_tags'].str.contains(tag, case=False, na=False).astype(int)
    df.drop(columns=['all_tags'], inplace=True)

    # --- Tag Notes role signals ---
    logging.info("Engineering Tag Notes role signals...")
    tag_notes = df['Tag Notes'].fillna('') if 'Tag Notes' in df.columns else pd.Series('', index=df.index)
    tag_2_notes = df['Tag 2 Notes'].fillna('') if 'Tag 2 Notes' in df.columns else pd.Series('', index=df.index)
    tag_notes_combined = tag_notes.str.lower() + ' ' + tag_2_notes.str.lower()
    df['tag_notes_on_ball'] = tag_notes_combined.str.contains(
        ON_BALL_KEYWORDS, na=False, regex=True
    ).astype(int)
    df['tag_notes_managed'] = tag_notes_combined.str.contains(
        MANAGED_KEYWORDS, na=False, regex=True
    ).astype(int)

    # --- Multi-window rolling averages ---
    shift_n = 1 if for_training else 0
    mode_label = "TRAINING (shift=1)" if for_training else "PREDICTION (shift=0)"
    logging.info(f"Computing rolling features in {mode_label} mode...")

    grouped = df.groupby('Player ID')

    for stat in tqdm(ROLLING_STATS, desc="Engineering rolling features"):
        if stat not in df.columns:
            logging.warning(f"Stat '{stat}' not found in DataFrame, skipping rolling features for it.")
            for w in ROLLING_WINDOWS:
                df[f'{stat}_rolling_avg_{w}_games'] = np.nan
            continue
        for w in ROLLING_WINDOWS:
            col = f'{stat}_rolling_avg_{w}_games'
            df[col] = grouped[stat].transform(
                lambda x, s=shift_n, win=w: x.shift(s).rolling(window=win, min_periods=1).mean()
            )

    # --- Form trajectory ---
    logging.info("Computing form trajectory features...")
    df['SC_form_delta_3v10'] = df['SC_rolling_avg_3_games'] - df['SC_rolling_avg_10_games']
    df['minutes_form_delta_3v10'] = df['minutes_played_rolling_avg_3_games'] - df['minutes_played_rolling_avg_10_games']

    # --- Price-based features ---
    logging.info("Computing price-based features...")
    sc_avg_10 = df['SC_rolling_avg_10_games']
    df['price_per_point'] = df['price'] / sc_avg_10.replace(0, np.nan)
    df['price_per_point'] = df['price_per_point'].fillna(0)

    df['price_momentum'] = df['price_change'] / df['price'].replace(0, np.nan)
    df['price_momentum'] = df['price_momentum'].fillna(0)

    # --- Experience features ---
    logging.info("Computing experience features...")
    df['career_games'] = grouped.cumcount() + 1
    df['season_games'] = df.groupby(['Player ID', 'Year']).cumcount() + 1

    # Debut / early-career flag: first 3 career games OR first 2 games of this season
    # These players have unreliable rolling features; model can discount accordingly
    df['is_debut_season'] = (
        (df['career_games'] <= 3) | (df['season_games'] <= 2)
    ).astype(int)

    # --- Scoring ceiling and floor ---
    logging.info("Computing scoring ceiling/floor features...")
    df['SC_max_last_10'] = grouped['SC'].transform(
        lambda x, s=shift_n: x.shift(s).rolling(window=10, min_periods=3).max()
    )
    df['SC_min_last_10'] = grouped['SC'].transform(
        lambda x, s=shift_n: x.shift(s).rolling(window=10, min_periods=3).min()
    )

    # Ceiling rate: fraction of last 10 games scoring 100+
    df['SC_ceiling_rate_10'] = grouped['SC'].transform(
        lambda x, s=shift_n: (x.shift(s) >= 100).rolling(window=10, min_periods=3).mean()
    )
    df['SC_ceiling_rate_10'] = df['SC_ceiling_rate_10'].fillna(0)

    # --- Volatility ---
    logging.info("Computing volatility features...")
    df['consistency_last_10'] = grouped['SC'].transform(
        lambda x, s=shift_n: x.shift(s).rolling(window=10, min_periods=3).std()
    )
    df['consistency_last_10'] = df['consistency_last_10'].fillna(df['consistency_last_10'].median())

    # --- Durability (games played last season) ---
    max_year = df['Year'].max()
    games_per_season = df[df['Year'] < max_year].groupby(['Player ID', 'Year']).size().reset_index(name='games_played_last_season')
    games_per_season['Year'] = games_per_season['Year'] + 1
    df = pd.merge(df, games_per_season, on=['Player ID', 'Year'], how='left')
    df['games_played_last_season'] = df['games_played_last_season'].fillna(0)

    # Fill any remaining NaN in feature columns
    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logging.info(f"Feature engineering complete. {len(df)} rows, {len(get_feature_columns())} features.")
    return df


def get_latest_player_features(df_with_features: pd.DataFrame) -> pd.DataFrame:
    """Returns the single most recent row of data for each player."""
    logging.info("Extracting the most recent feature set for each player...")
    latest = df_with_features.drop_duplicates(subset=['Player ID'], keep='last').copy()

    # Fill any remaining NaN in feature columns
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in latest.columns]
    latest[available] = latest[available].fillna(0)

    logging.info(f"Found latest data for {len(latest)} unique players.")
    return latest


def build_opponent_concession_table(df_with_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table: for each (opponent_team, position), how many SC points
    does that team concede on average, and at what rate does it concede 100+ scores?

    Uses the most recent completed year of data.
    Returns DataFrame with columns: [Opponent, simple_pos, opponent_strength_rating,
                                      opponent_ceiling_rating]
    """
    max_year = df_with_features['Year'].max()
    recent = df_with_features[df_with_features['Year'] == max_year].copy()

    table = recent.groupby(['Opponent', 'simple_pos']).agg(
        opponent_strength_rating=('SC', 'mean'),
        opponent_ceiling_rating=('SC', lambda x: (x >= 100).mean()),
    ).reset_index()

    logging.info(f"Built opponent concession table from {max_year} data: "
                 f"{len(table)} entries across {table['Opponent'].nunique()} teams")
    return table


def apply_fixture_opponent(prediction_df: pd.DataFrame,
                           concession_table: pd.DataFrame,
                           fixture_map: dict) -> pd.DataFrame:
    """
    Override the opponent_strength_rating in prediction data using the actual
    fixture matchup for the upcoming round.

    This makes predictions vary by round based on who the player is actually facing.
    """
    logging.info("Applying fixture-specific opponent ratings...")

    # Map each player's team to their fixture opponent
    prediction_df['fixture_opponent'] = prediction_df['Team'].map(fixture_map)
    prediction_df['fixture_opponent'] = prediction_df['fixture_opponent'].fillna('BYE')

    # Ensure simple_pos exists
    if 'simple_pos' not in prediction_df.columns:
        prediction_df['simple_pos'] = prediction_df['ff_position'].apply(simplify_position)

    # Drop stale opponent ratings (from the player's last historical game)
    prediction_df = prediction_df.drop(
        columns=['opponent_strength_rating', 'opponent_ceiling_rating'], errors='ignore'
    )

    # Merge the correct ratings for the fixture opponent
    prediction_df = pd.merge(
        prediction_df,
        concession_table.rename(columns={'Opponent': 'fixture_opponent'}),
        on=['fixture_opponent', 'simple_pos'],
        how='left'
    )

    # Fill BYE / missing data with the overall means
    overall_mean = concession_table['opponent_strength_rating'].mean()
    prediction_df['opponent_strength_rating'] = prediction_df['opponent_strength_rating'].fillna(overall_mean)
    if 'opponent_ceiling_rating' in prediction_df.columns:
        overall_ceiling = concession_table['opponent_ceiling_rating'].mean()
        prediction_df['opponent_ceiling_rating'] = prediction_df['opponent_ceiling_rating'].fillna(overall_ceiling)
    else:
        prediction_df['opponent_ceiling_rating'] = concession_table['opponent_ceiling_rating'].mean()

    matched = prediction_df['fixture_opponent'].ne('BYE').sum()
    logging.info(f"Fixture opponent ratings applied: {matched} matched, "
                 f"{len(prediction_df) - matched} on BYE (using mean={overall_mean:.1f})")

    return prediction_df
