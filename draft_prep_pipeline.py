"""
2026 SuperCoach Draft Preparation Pipeline
============================================
Generates:
  1. Season comparison files (2022-2025): BEFORE/AFTER stats per player
  2. Draft heat maps (2022-2025): pick effectiveness using position-weighted z-scores
  3. Draft success analysis: multi-year feature importance for identifying good picks
  4. 2026 Draft Board Excel: value-ordered player lists by position
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import zscore, rankdata
import logging
import warnings
from draft_features import build_draft_features, DRAFT_FEATURE_COLS, NOTE_FLAG_COLS, SC_RELEVANT_CAPS, filter_sc_relevant
from parse_draft_notes import (parse_2022_notes, parse_2023_notes,
                                parse_2024_notes, parse_2025_notes,
                                classify_all_notes, classify_note)

warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MASTER_FILE = Path('data/processed/master_player_data.csv')
COACH_LIST_FILE = Path('data/live/coach_list.csv')
PLAYER_LIST_DIR = Path('draft_prep/SC 2026')
OUTPUT_DIR = Path('data/draft')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Draft recap file paths per year
DRAFT_FILES = {
    2022: Path('archive/legacy/fantasy_banter_data/2022/draft_2022.csv'),
    2023: Path('archive/legacy/fantasy_banter_data/2023/draft_2023.csv'),
    2024: Path('heat_map/2024/2024_draft_leagues_1283_recap_combined.csv'),
    2025: Path('data/predictions/2025_draft_analysis.csv'),
}

# Position composition weights (from heat map methodology)
POS_WEIGHTS = {'DEF': 0.263, 'MID': 0.368, 'RUC': 0.105, 'FWD': 0.263}
POS_SLOTS = {'DEF': 5, 'MID': 7, 'RUC': 2, 'FWD': 5}
NUM_TEAMS = 8

# The user is RICHO (Mark, Sheez Nutz)
MY_COACH = 'RICHO'


# ====================================================================
# SECTION 1: Season Comparison Files
# ====================================================================

def simplify_position(pos_str):
    """Map SC position strings to simple DEF/MID/FWD/RUC."""
    if not isinstance(pos_str, str):
        return 'Unknown'
    pos_str = pos_str.upper()
    # Handle dual positions - return primary
    for p in ['MID', 'DEF', 'FWD', 'RUC']:
        if p in pos_str:
            return p
    if 'MIDFIELDER' in pos_str:
        return 'MID'
    if 'BACK' in pos_str or 'DEFENDER' in pos_str:
        return 'DEF'
    if 'FORWARD' in pos_str:
        return 'FWD'
    if 'RUCK' in pos_str:
        return 'RUC'
    return 'Unknown'


def get_all_positions(pos_str):
    """Return list of all eligible positions from a position string."""
    if not isinstance(pos_str, str):
        return []
    pos_str = pos_str.upper()
    positions = []
    for p in ['MID', 'DEF', 'FWD', 'RUC']:
        if p in pos_str:
            positions.append(p)
    if not positions:
        if 'MIDFIELDER' in pos_str:
            positions.append('MID')
        if 'BACK' in pos_str or 'DEFENDER' in pos_str:
            positions.append('DEF')
        if 'FORWARD' in pos_str:
            positions.append('FWD')
        if 'RUCK' in pos_str:
            positions.append('RUC')
    return positions if positions else ['Unknown']


def build_season_comparisons(master_df):
    """Build BEFORE/AFTER comparison for each player in each season (2022-2025)."""
    logging.info("=" * 80)
    logging.info("SECTION 1: Building season comparison files")
    logging.info("=" * 80)

    all_comparisons = []

    for year in [2022, 2023, 2024, 2025]:
        logging.info(f"Processing season {year}...")

        # Load pre-draft player list for this year
        pl_file = PLAYER_LIST_DIR / f'{year}_SC_Player_list.csv'
        if not pl_file.exists():
            logging.warning(f"Player list not found: {pl_file}")
            continue
        player_list = pd.read_csv(pl_file)

        # --- BEFORE data (from player list + master Y-1) ---
        before = player_list[['id', 'feed_id', 'first_name', 'last_name',
                              'team', 'position', 'previous_games',
                              'previous_average', 'previous_total']].copy()
        before.rename(columns={
            'previous_games': 'prev_games',
            'previous_average': 'prev_avg',
            'previous_total': 'prev_total',
        }, inplace=True)
        before['Year'] = year
        before['primary_pos'] = before['position'].apply(simplify_position)

        # Enrich with detailed Y-1 stats from master data
        prev_year = year - 1
        prev_data = master_df[(master_df['Year'] == prev_year) & (master_df['played'] > 0)].copy()

        if not prev_data.empty:
            prev_stats = prev_data.groupby('Player ID').agg(
                prev_std=('SC', 'std'),
                prev_ceiling=('SC', 'max'),
                prev_floor=('SC', 'min'),
                prev_median=('SC', 'median'),
            ).reset_index()
            before = before.merge(prev_stats, left_on='feed_id', right_on='Player ID', how='left')
            before.drop(columns=['Player ID'], errors='ignore', inplace=True)
        else:
            before['prev_std'] = np.nan
            before['prev_ceiling'] = np.nan
            before['prev_floor'] = np.nan
            before['prev_median'] = np.nan

        before['prev_cv'] = np.where(
            before['prev_avg'] > 0,
            before['prev_std'] / before['prev_avg'],
            np.nan
        )

        # Career games and age proxy
        career_data = master_df[
            (master_df['Year'] < year) & (master_df['played'] > 0)
        ].groupby('Player ID').agg(
            career_games=('SC', 'count'),
            first_year=('Year', 'min'),
        ).reset_index()
        before = before.merge(career_data, left_on='feed_id', right_on='Player ID', how='left')
        before.drop(columns=['Player ID'], errors='ignore', inplace=True)
        before['career_games'] = before['career_games'].fillna(0).astype(int)
        before['age_proxy'] = year - before['first_year'].fillna(year)

        # Position rank BEFORE (within position by prev_avg)
        before['prev_position_rank'] = before.groupby('primary_pos')['prev_avg'].rank(
            ascending=False, method='dense', na_option='bottom'
        )

        # --- AFTER data (season result from master data) ---
        season_data = master_df[(master_df['Year'] == year) & (master_df['played'] > 0)].copy()
        if season_data.empty:
            logging.warning(f"No master data for {year}")
            continue

        after = season_data.groupby('Player ID').agg(
            season_games=('SC', 'count'),
            season_avg=('SC', 'mean'),
            season_total=('SC', 'sum'),
            season_std=('SC', 'std'),
            season_ceiling=('SC', 'max'),
            season_floor=('SC', 'min'),
        ).reset_index()

        # Season position rank (need position from player list)
        after = after.merge(
            before[['feed_id', 'primary_pos']].drop_duplicates(),
            left_on='Player ID', right_on='feed_id', how='left'
        )
        after['season_position_rank'] = after.groupby('primary_pos')['season_avg'].rank(
            ascending=False, method='dense', na_option='bottom'
        )

        # TPOR: Total Points Over Replacement
        # Replacement level = average of players ranked (N_slots * NUM_TEAMS) to (N_slots * NUM_TEAMS + 10)
        for pos in ['DEF', 'MID', 'FWD', 'RUC']:
            pos_df = after[after['primary_pos'] == pos].sort_values('season_avg', ascending=False)
            n_starters = POS_SLOTS.get(pos, 5) * NUM_TEAMS
            if len(pos_df) > n_starters + 5:
                replacement_avg = pos_df.iloc[n_starters:n_starters + 10]['season_avg'].mean()
            elif len(pos_df) > n_starters:
                replacement_avg = pos_df.iloc[n_starters:]['season_avg'].mean()
            else:
                replacement_avg = pos_df['season_avg'].mean() * 0.7
            after.loc[after['primary_pos'] == pos, 'replacement_avg'] = replacement_avg

        after['TPOR'] = (after['season_avg'] - after['replacement_avg']) * after['season_games']

        # Merge BEFORE + AFTER
        comparison = before.merge(
            after[['Player ID', 'season_games', 'season_avg', 'season_total',
                   'season_std', 'season_ceiling', 'season_floor',
                   'season_position_rank', 'TPOR']],
            left_on='feed_id', right_on='Player ID', how='left'
        )
        comparison.drop(columns=['Player ID'], errors='ignore', inplace=True)

        # Delta
        comparison['avg_change'] = comparison['season_avg'] - comparison['prev_avg']
        comparison['rank_change'] = comparison['prev_position_rank'] - comparison['season_position_rank']
        comparison['games_change'] = comparison['season_games'] - comparison['prev_games']

        # Save per-year file
        out_file = OUTPUT_DIR / f'season_comparison_{year}.csv'
        comparison.to_csv(out_file, index=False)
        logging.info(f"  Saved {out_file} ({len(comparison)} players)")

        all_comparisons.append(comparison)

    all_comp_df = pd.concat(all_comparisons, ignore_index=True)
    all_comp_df.to_csv(OUTPUT_DIR / 'season_comparisons_all.csv', index=False)
    logging.info(f"Combined comparisons: {len(all_comp_df)} player-seasons")
    return all_comp_df


# ====================================================================
# SECTION 2: Draft Heat Maps
# ====================================================================

def load_draft_picks(year, player_list_df):
    """Load draft picks for a given year and standardize columns."""
    draft_file = DRAFT_FILES.get(year)
    if not draft_file or not draft_file.exists():
        logging.warning(f"Draft file not found for {year}")
        return pd.DataFrame()

    if year == 2025:
        # 2025 draft analysis already has coach names and feed_id as player_id
        raw = pd.read_csv(draft_file)
        picks = raw[['pick', 'player_id', 'position', 'coach']].copy()
        picks.rename(columns={'player_id': 'feed_id'}, inplace=True)
        picks['feed_id'] = picks['feed_id'].astype(int)
    else:
        # 2022-2024: player_id is SC internal id, need to map to feed_id
        raw = pd.read_csv(draft_file)
        pl = player_list_df[player_list_df['Year'] == year]
        if pl.empty:
            pl = pd.read_csv(PLAYER_LIST_DIR / f'{year}_SC_Player_list.csv')
            pl['Year'] = year

        picks = raw[['pick', 'player_id', 'position', 'user_team_id']].copy()
        picks = picks.merge(
            pl[['id', 'feed_id']].drop_duplicates(),
            left_on='player_id', right_on='id', how='left'
        )
        picks.drop(columns=['id'], errors='ignore', inplace=True)
        picks.rename(columns={'user_team_id': 'coach'}, inplace=True)

    picks['Year'] = year
    picks['draft_round'] = ((picks['pick'] - 1) // NUM_TEAMS) + 1
    return picks.dropna(subset=['feed_id']).copy()


def compute_heat_map(draft_picks, season_comp, year):
    """Compute position-weighted z-scores and nearest-7-rank for each draft pick."""
    logging.info(f"Computing heat map for {year}...")

    # Get season averages for all players (not just drafted)
    season = season_comp[season_comp['Year'] == year].copy()
    if season.empty:
        logging.warning(f"No season comparison data for {year}")
        return pd.DataFrame()

    # Only include players who actually played
    season = season[season['season_games'].notna() & (season['season_games'] > 0)].copy()

    # --- Position-specific z-scores for top N×8 players ---
    all_scores = []
    overall_pool = pd.DataFrame()

    for pos, n_slots in POS_SLOTS.items():
        n_top = n_slots * NUM_TEAMS
        pos_players = season[season['primary_pos'] == pos].sort_values('season_avg', ascending=False)
        top_pool = pos_players.head(n_top)

        if len(top_pool) < 3:
            continue

        pos_mean = top_pool['season_avg'].mean()
        pos_std = top_pool['season_avg'].std()

        # Compute z-score for ALL players of this position (not just top)
        pos_players = pos_players.copy()
        pos_players['pos_z'] = (pos_players['season_avg'] - pos_mean) / max(pos_std, 1)
        pos_players['scr_pos'] = pos
        pos_players['pos_mean'] = pos_mean
        pos_players['pos_std'] = pos_std

        all_scores.append(pos_players)
        overall_pool = pd.concat([overall_pool, top_pool])

    if not all_scores:
        return pd.DataFrame()

    scored_df = pd.concat(all_scores, ignore_index=True)

    # Overall z-score (across all positions)
    overall_mean = overall_pool['season_avg'].mean()
    overall_std = overall_pool['season_avg'].std()
    scored_df['overall_z'] = (scored_df['season_avg'] - overall_mean) / max(overall_std, 1)

    # Weighted score
    scored_df['weight'] = scored_df['scr_pos'].map(POS_WEIGHTS)
    scored_df['scr_wgt'] = scored_df['pos_z'] * scored_df['weight'] + scored_df['overall_z'] * (1 - scored_df['weight'])

    # Keep best score per player (for dual-position players)
    scored_df = scored_df.sort_values('scr_wgt', ascending=False).drop_duplicates(subset='feed_id')

    # Merge with draft picks
    heat = draft_picks[draft_picks['Year'] == year].merge(
        scored_df[['feed_id', 'scr_wgt', 'scr_pos', 'season_avg', 'season_games', 'first_name', 'last_name', 'team']],
        on='feed_id', how='left'
    )

    # Fill names for unmatched (players who didn't play)
    if 'first_name' not in heat.columns or heat['first_name'].isna().any():
        name_lookup = season[['feed_id', 'first_name', 'last_name', 'team']].drop_duplicates()
        for col in ['first_name', 'last_name', 'team']:
            heat[col] = heat[col].fillna(
                heat['feed_id'].map(name_lookup.set_index('feed_id')[col].to_dict())
            )

    heat['player_name'] = heat['first_name'].fillna('') + ' ' + heat['last_name'].fillna('')
    heat['scr_wgt'] = heat['scr_wgt'].fillna(heat['scr_wgt'].min() - 1)  # DNP gets worst score
    heat['season_avg'] = heat['season_avg'].fillna(0)
    heat['season_games'] = heat['season_games'].fillna(0)

    # --- Nearest-7-rank ---
    heat = heat.sort_values('pick').reset_index(drop=True)
    nearest_7_ranks = []

    for idx, row in heat.iterrows():
        current_pick = row['pick']
        # Get next 7 picks after this one
        next_picks = heat[(heat['pick'] > current_pick) & (heat['pick'] <= current_pick + 7)]

        if len(next_picks) == 0:
            # Last few picks - compare against remaining
            next_picks = heat[heat['pick'] > current_pick]

        if len(next_picks) > 0:
            combined = list(next_picks['scr_wgt']) + [row['scr_wgt']]
            ranks = rankdata(-np.array(combined), method='min')
            nearest_7_ranks.append(int(ranks[-1]))
        else:
            nearest_7_ranks.append(1)

    heat['nearest_7_rank'] = nearest_7_ranks

    # Output columns
    out_cols = ['pick', 'draft_round', 'coach', 'player_name', 'team', 'position',
                'season_avg', 'season_games', 'scr_wgt', 'nearest_7_rank', 'Year', 'feed_id']
    result = heat[[c for c in out_cols if c in heat.columns]].copy()
    result = result.round({'season_avg': 1, 'scr_wgt': 3})

    out_file = OUTPUT_DIR / f'draft_heat_map_{year}.csv'
    result.to_csv(out_file, index=False)
    logging.info(f"  Saved {out_file} ({len(result)} picks)")

    return result


def build_heat_maps(season_comp_df):
    """Build heat maps for all years."""
    logging.info("=" * 80)
    logging.info("SECTION 2: Building draft heat maps")
    logging.info("=" * 80)

    # Load player lists for ID mapping
    all_player_lists = []
    for year in [2022, 2023, 2024, 2025]:
        pl_file = PLAYER_LIST_DIR / f'{year}_SC_Player_list.csv'
        if pl_file.exists():
            pl = pd.read_csv(pl_file)
            pl['Year'] = year
            all_player_lists.append(pl)
    pl_combined = pd.concat(all_player_lists, ignore_index=True)

    all_heat_maps = []
    for year in [2022, 2023, 2024, 2025]:
        picks = load_draft_picks(year, pl_combined)
        if picks.empty:
            continue
        heat = compute_heat_map(picks, season_comp_df, year)
        if not heat.empty:
            all_heat_maps.append(heat)

    if all_heat_maps:
        combined = pd.concat(all_heat_maps, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / 'draft_heat_maps_all.csv', index=False)
        logging.info(f"Combined heat maps: {len(combined)} picks across {combined['Year'].nunique()} years")

        # Print coach summaries
        for year in combined['Year'].unique():
            yr_data = combined[combined['Year'] == year]
            print(f"\n--- {year} Draft Heat Map - Coach Summary ---")
            coach_smy = yr_data.groupby('coach').agg(
                picks=('pick', 'count'),
                avg_rank=('nearest_7_rank', 'mean'),
                rank_1=('nearest_7_rank', lambda x: (x == 1).sum()),
                rank_8=('nearest_7_rank', lambda x: (x == 8).sum()),
            ).sort_values('avg_rank')
            coach_smy['avg_rank'] = coach_smy['avg_rank'].round(2)
            print(coach_smy.to_string())

        return combined
    return pd.DataFrame()


# ====================================================================
# SECTION 3: Draft Success Analysis
# ====================================================================

def analyze_draft_success(season_comp_df, heat_map_df):
    """Multi-year analysis of what pre-draft features predict successful picks."""
    logging.info("=" * 80)
    logging.info("SECTION 3: Draft success analysis")
    logging.info("=" * 80)

    # Merge heat map picks with season comparison data
    # Drop columns from season_comp that also exist in heat_map to avoid _x/_y suffixes
    drop_cols = ['first_name', 'last_name', 'team', 'position', 'season_avg', 'season_games']
    sc_merge = season_comp_df.drop(columns=drop_cols, errors='ignore')
    analysis = heat_map_df.merge(sc_merge, on=['feed_id', 'Year'], how='left')

    # Add draft tier
    analysis['draft_tier'] = pd.cut(
        analysis['pick'],
        bins=[0, 20, 50, 100, 200],
        labels=['Elite (1-20)', 'Premium (21-50)', 'Mid (51-100)', 'Value (101+)']
    )

    analysis.to_csv(OUTPUT_DIR / 'draft_success_analysis.csv', index=False)
    logging.info(f"Saved draft_success_analysis.csv ({len(analysis)} picks)")

    # --- Print analysis report ---
    print("\n" + "=" * 100)
    print("MULTI-YEAR DRAFT SUCCESS ANALYSIS (2022-2025)")
    print("=" * 100)

    # 1. Draft tier performance
    print("\n--- 1. Performance by Draft Tier ---")
    tier_stats = analysis.groupby('draft_tier', observed=True).agg(
        n=('pick', 'count'),
        avg_season_avg=('season_avg', 'mean'),
        avg_prev_avg=('prev_avg', 'mean'),
        avg_change=('avg_change', 'mean'),
        avg_games=('season_games', 'mean'),
        avg_TPOR=('TPOR', 'mean'),
    ).round(1)
    print(tier_stats.to_string())

    # 2. Position safety
    print("\n--- 2. Position Safety ---")
    pos_stats = analysis.groupby('primary_pos').agg(
        n=('pick', 'count'),
        avg_season_avg=('season_avg', 'mean'),
        avg_games=('season_games', 'mean'),
        avg_change=('avg_change', 'mean'),
        avg_heat_rank=('nearest_7_rank', 'mean'),
        pct_improved=('avg_change', lambda x: (x > 0).mean() * 100),
    ).round(1)
    print(pos_stats.to_string())

    # 3. Durability impact
    print("\n--- 3. Durability Impact ---")
    analysis['durability_group'] = pd.cut(
        analysis['prev_games'].fillna(0),
        bins=[-1, 10, 17, 20, 30],
        labels=['0-10 games', '11-17 games', '18-20 games', '21+ games']
    )
    dur_stats = analysis.groupby('durability_group', observed=True).agg(
        n=('pick', 'count'),
        avg_season_avg=('season_avg', 'mean'),
        avg_games=('season_games', 'mean'),
        avg_TPOR=('TPOR', 'mean'),
    ).round(1)
    print(dur_stats.to_string())

    # 4. Consistency as predictor
    print("\n--- 4. Consistency (CV) as Predictor ---")
    analysis['cv_group'] = pd.cut(
        analysis['prev_cv'].fillna(1),
        bins=[-0.01, 0.2, 0.3, 0.4, 10],
        labels=['Very consistent (<0.2)', 'Consistent (0.2-0.3)', 'Variable (0.3-0.4)', 'Volatile (>0.4)']
    )
    cv_stats = analysis.groupby('cv_group', observed=True).agg(
        n=('pick', 'count'),
        avg_season_avg=('season_avg', 'mean'),
        avg_change=('avg_change', 'mean'),
        avg_games=('season_games', 'mean'),
    ).round(1)
    print(cv_stats.to_string())

    # 5. Career experience
    print("\n--- 5. Career Experience Impact ---")
    analysis['experience_group'] = pd.cut(
        analysis['career_games'].fillna(0),
        bins=[-1, 30, 80, 150, 500],
        labels=['Young (<30 games)', 'Developing (30-80)', 'Peak (80-150)', 'Veteran (150+)']
    )
    exp_stats = analysis.groupby('experience_group', observed=True).agg(
        n=('pick', 'count'),
        avg_season_avg=('season_avg', 'mean'),
        avg_change=('avg_change', 'mean'),
        avg_TPOR=('TPOR', 'mean'),
    ).round(1)
    print(exp_stats.to_string())

    # 6. Diamond profile (picked 80+, finished top 50 in position)
    print("\n--- 6. Diamond in the Rough Profile ---")
    diamonds = analysis[(analysis['pick'] >= 80) & (analysis['season_position_rank'] <= 50)].copy()
    print(f"Found {len(diamonds)} diamonds across {analysis['Year'].nunique()} years ({len(diamonds)/analysis['Year'].nunique():.0f}/year)")
    if len(diamonds) > 0:
        print(f"  Avg prev_games: {diamonds['prev_games'].mean():.1f}")
        print(f"  Avg prev_avg: {diamonds['prev_avg'].mean():.1f}")
        print(f"  Avg prev_cv: {diamonds['prev_cv'].mean():.2f}")
        print(f"  Avg career_games: {diamonds['career_games'].mean():.0f}")
        print(f"  Avg age_proxy: {diamonds['age_proxy'].mean():.1f}")
        print(f"  Position breakdown: {diamonds['primary_pos'].value_counts().to_dict()}")
        print(f"\n  Top diamonds:")
        for _, d in diamonds.sort_values('season_avg', ascending=False).head(10).iterrows():
            print(f"    {d['player_name']:25s} Pick {int(d['pick']):3d} | {d['primary_pos']} | "
                  f"Prev: {d['prev_avg']:.0f} avg, {int(d['prev_games'])} gm | "
                  f"Result: {d['season_avg']:.1f} avg, {int(d['season_games'])} gm | "
                  f"Change: {d['avg_change']:+.1f}")

    # 7. Bust profile (picked 1-40, bottom half of position)
    print("\n--- 7. Bust Profile ---")
    busts = analysis[(analysis['pick'] <= 40) & (analysis['season_position_rank'] > 30)].copy()
    print(f"Found {len(busts)} busts (picked 1-40, finished outside top 30 in position)")
    if len(busts) > 0:
        print(f"  Avg prev_games: {busts['prev_games'].mean():.1f}")
        print(f"  Avg prev_avg: {busts['prev_avg'].mean():.1f}")
        print(f"  Avg prev_cv: {busts['prev_cv'].mean():.2f}")
        print(f"  Position breakdown: {busts['primary_pos'].value_counts().to_dict()}")
        print(f"\n  Worst busts:")
        for _, b in busts.sort_values('avg_change').head(10).iterrows():
            print(f"    {b['player_name']:25s} Pick {int(b['pick']):3d} | {b['primary_pos']} | "
                  f"Prev: {b['prev_avg']:.0f} avg | Result: {b['season_avg']:.1f} avg | "
                  f"Change: {b['avg_change']:+.1f}")

    # 8. Feature importance (XGBoost)
    print("\n--- 8. Feature Importance for Predicting Season Average ---")
    feature_cols = ['prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
                    'career_games', 'age_proxy', 'prev_std']
    target = 'season_avg'

    model_df = analysis.dropna(subset=feature_cols + [target])
    if len(model_df) > 50:
        try:
            from xgboost import XGBRegressor
            from sklearn.metrics import mean_absolute_error

            X = model_df[feature_cols]
            y = model_df[target]

            xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbosity=0)
            xgb.fit(X, y)
            importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
            print("Feature importance for predicting season_avg from pre-draft stats:")
            for feat, imp in importances.items():
                print(f"  {feat:20s}: {imp:.3f}")

            preds = xgb.predict(X)
            print(f"\n  In-sample MAE: {mean_absolute_error(y, preds):.2f}")
        except Exception as e:
            logging.warning(f"Feature importance failed: {e}")

    # 9. Correlation matrix
    print("\n--- 9. Key Correlations with Season Average ---")
    corr_cols = feature_cols + [target, 'TPOR', 'season_games']
    corr_df = analysis[corr_cols].dropna()
    if len(corr_df) > 20:
        corrs = corr_df.corr()[target].drop(target).sort_values(ascending=False)
        for feat, corr in corrs.items():
            print(f"  {feat:20s}: {corr:+.3f}")

    # 10. Your drafting performance (RICHO)
    print(f"\n--- 10. {MY_COACH}'s Drafting Performance ---")
    my_picks = analysis[analysis['coach'] == MY_COACH]
    if not my_picks.empty:
        for year in sorted(my_picks['Year'].unique()):
            yr = my_picks[my_picks['Year'] == year]
            print(f"\n  {year}: {len(yr)} picks, avg nearest_7_rank: {yr['nearest_7_rank'].mean():.2f}")
            print(f"    Best pick: {yr.loc[yr['nearest_7_rank'].idxmin(), 'player_name']} "
                  f"(pick {int(yr.loc[yr['nearest_7_rank'].idxmin(), 'pick'])}, rank {int(yr.loc[yr['nearest_7_rank'].idxmin(), 'nearest_7_rank'])})")
    else:
        # Try matching by user_team_id pattern
        print("  (Coach identity could not be matched for 2022-2024 drafts)")

    return analysis


# ====================================================================
# SECTION 4: 2026 Draft Board (Excel)
# ====================================================================

def build_2026_draft_board(master_df, analysis_df):
    """Generate the 2026 draft board Excel with position tabs."""
    logging.info("=" * 80)
    logging.info("SECTION 4: Building 2026 draft board")
    logging.info("=" * 80)

    # Load 2026 player list
    pl_2026 = pd.read_csv(PLAYER_LIST_DIR / '2026_SC_Player_list.csv')

    # Load R1-6 predictions summary
    pred_file = Path('data/predictions/2026_rounds_1_to_6_summary.csv')
    if pred_file.exists():
        predictions = pd.read_csv(pred_file)
        # Normalize column names to match expected schema
        predictions.rename(columns={
            'avg_score': 'avg_projected',
            'score_range': 'matchup_variance',
        }, inplace=True)
    else:
        logging.warning("R1-6 summary not found, using individual round files")
        rounds = []
        for r in range(1, 7):
            rf = Path(f'data/predictions/2026_round_{r}_predictions.csv')
            if rf.exists():
                rdf = pd.read_csv(rf)
                rdf['Round'] = r
                rounds.append(rdf)
        if rounds:
            all_r = pd.concat(rounds)
            playing = all_r[all_r['Opponent'] != 'BYE']
            predictions = playing.groupby(['Player ID', 'First Name', 'Last Name', 'Team', 'sc_position']).agg(
                avg_projected=('projected_score', 'mean'),
                min_projected=('projected_score', 'min'),
                max_projected=('projected_score', 'max'),
                avg_p80=('probability_gt_80', 'mean'),
                avg_p100=('probability_gt_100', 'mean'),
                games_playing=('Round', 'count'),
            ).reset_index()
            predictions['matchup_variance'] = predictions['max_projected'] - predictions['min_projected']
        else:
            predictions = pd.DataFrame()

    # Load consistency ratings
    consistency_file = Path('data/predictions/2026_consistency_ratings.csv')
    if consistency_file.exists():
        consistency = pd.read_csv(consistency_file)
    else:
        consistency = pd.DataFrame()

    # 2025 season stats from master data
    season_2025 = master_df[(master_df['Year'] == 2025) & (master_df['played'] > 0)].copy()
    stats_2025 = season_2025.groupby('Player ID').agg(
        games_2025=('SC', 'count'),
        avg_2025=('SC', 'mean'),
        total_2025=('SC', 'sum'),
        std_2025=('SC', 'std'),
        ceiling_2025=('SC', 'max'),
        floor_2025=('SC', 'min'),
    ).reset_index()
    stats_2025['cv_2025'] = stats_2025['std_2025'] / stats_2025['avg_2025'].replace(0, np.nan)

    # --- POST-ROUND 13 AVERAGE (backtesting: 2nd-best feature, surging players have lowest bust rate) ---
    second_half = season_2025[season_2025['Round_Num'] > 13]
    post_r13 = second_half.groupby('Player ID').agg(
        post_r13_avg=('SC', 'mean'),
        post_r13_games=('SC', 'count'),
    ).reset_index()

    # --- 2-YEAR AVERAGE (backtesting: 2nd most important feature at 0.210 importance) ---
    prev_2yr = master_df[(master_df['Year'].isin([2024, 2025])) & (master_df['played'] > 0)]
    avg_2yr = prev_2yr.groupby('Player ID').agg(
        avg_2yr=('SC', 'mean'),
        games_2yr=('SC', 'count'),
    ).reset_index()

    # --- TAG FEATURES (backtesting: positive tags 0.025 importance, hot tags 0.019) ---
    master_df['Tag'] = master_df['Tag'].fillna('')
    master_df['Tag 2'] = master_df['Tag 2'].fillna('')

    def count_tags_for_player(group, tag_list):
        tag_col = group['Tag'].str.lower() + ',' + group['Tag 2'].str.lower()
        return sum(tag_col.str.contains(tag, na=False).sum() for tag in tag_list)

    # Tags from 2025 season
    tag_2025 = season_2025.groupby('Player ID').apply(
        lambda g: pd.Series({
            'positive_tags': count_tags_for_player(g, ['star', 'hot', 'gun', 'x-factor']),
            'injury_tags': count_tags_for_player(g, ['injured', 'sore']),
            'sub_tags': count_tags_for_player(g, ['sub', 'subbed']),
        }),
        include_groups=False
    ).reset_index()

    # Tags from 2024+2025 combined (2-year injury picture)
    recent_2yr = master_df[(master_df['Year'].isin([2024, 2025])) & (master_df['played'] > 0)]
    tag_2yr = recent_2yr.groupby('Player ID').apply(
        lambda g: pd.Series({
            'injury_tags_2yr': count_tags_for_player(g, ['injured', 'sore']),
        }),
        include_groups=False
    ).reset_index()

    # Career data
    career = master_df[master_df['played'] > 0].groupby('Player ID').agg(
        career_games=('SC', 'count'),
        first_year=('Year', 'min'),
    ).reset_index()
    career['age_proxy'] = 2026 - career['first_year']

    # Form trajectory (last 5 games vs season avg — backtesting: last5_avg at 0.036 importance)
    recent = season_2025.sort_values(['Player ID', 'Round_Num'])
    last_5 = recent.groupby('Player ID').tail(5).groupby('Player ID')['SC'].mean().reset_index()
    last_5.columns = ['Player ID', 'last5_avg']
    last_3 = recent.groupby('Player ID').tail(3).groupby('Player ID')['SC'].mean().reset_index()
    last_3.columns = ['Player ID', 'last_3_avg']
    last_10 = recent.groupby('Player ID').tail(10).groupby('Player ID')['SC'].mean().reset_index()
    last_10.columns = ['Player ID', 'last_10_avg']
    form = last_3.merge(last_10, on='Player ID').merge(last_5, on='Player ID')
    form['form_delta'] = form['last_3_avg'] - form['last_10_avg']

    # --- Build the board ---
    board = pl_2026[['id', 'feed_id', 'first_name', 'last_name', 'team', 'position',
                      'previous_games', 'previous_average', 'previous_total']].copy()
    board['primary_pos'] = board['position'].apply(simplify_position)
    board['all_positions'] = board['position'].apply(lambda x: ', '.join(get_all_positions(x)))

    # Merge all feature tables
    for feat_df in [stats_2025, post_r13, avg_2yr, tag_2025, tag_2yr, career, form]:
        board = board.merge(feat_df, left_on='feed_id', right_on='Player ID', how='left')
        board.drop(columns=['Player ID'], errors='ignore', inplace=True)

    # Merge predictions
    if not predictions.empty:
        pred_cols = ['Player ID', 'avg_projected', 'avg_p80', 'avg_p100']
        if 'matchup_variance' in predictions.columns:
            pred_cols.append('matchup_variance')
        if 'games_playing' in predictions.columns:
            pred_cols.append('games_playing')
        available = [c for c in pred_cols if c in predictions.columns]
        board = board.merge(predictions[available], left_on='feed_id', right_on='Player ID', how='left')
        board.drop(columns=['Player ID'], errors='ignore', inplace=True)

    # Merge consistency
    if not consistency.empty and 'std_score' in consistency.columns:
        board = board.merge(
            consistency[['Player ID', 'std_score', 'cv']].rename(columns={'std_score': 'matchup_std', 'cv': 'matchup_cv'}),
            left_on='feed_id', right_on='Player ID', how='left'
        )
        board.drop(columns=['Player ID'], errors='ignore', inplace=True)

    # --- TPOR estimate ---
    pos_replacement = {}
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        pos_players = board[board['primary_pos'] == pos].sort_values('avg_projected', ascending=False)
        n_starters = POS_SLOTS.get(pos, 5) * NUM_TEAMS
        if len(pos_players) > n_starters + 5:
            pos_replacement[pos] = pos_players.iloc[n_starters:n_starters + 10]['avg_projected'].mean()
        else:
            pos_replacement[pos] = pos_players['avg_projected'].mean() * 0.7 if len(pos_players) > 0 else 70

    board['replacement_level'] = board['primary_pos'].map(pos_replacement)
    board['TPOR_estimate'] = (board['avg_projected'].fillna(0) - board['replacement_level'].fillna(70)) * board.get('games_playing', pd.Series(5, index=board.index)).fillna(5)

    # --- Draft Value: Comprehensive XGBoost model (123 features) ---
    # Backtesting v3 over 2022-2025 (leave-one-year-out) showed:
    #   - 123 features, Spearman 0.771, MAE 9.6 (vs baseline 0.756, MAE 11.0)
    #   - 13% more accurate point predictions, 17% fewer busts
    #   - Key features: regression_est_85 (0.281), pct_above_80 (0.157),
    #     avg_2yr (0.067), avg_3yr (0.039), q75 (0.039)
    #   - Feature categories: Consistency 17.1%, Multi-year 11.3%, Basic 9.0%,
    #     Composition 4.1%, Form 4.0%
    #
    # Uses build_draft_features() from draft_features.py (shared module)

    from xgboost import XGBRegressor as DraftXGB
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Build training data: comprehensive features for 2022-2025 + season outcomes + notes
    logging.info("Building comprehensive training features for 2022-2025...")
    master_df['Tag'] = master_df['Tag'].fillna('')
    master_df['Tag 2'] = master_df['Tag 2'].fillna('')

    train_feature_frames = []
    for y in range(2016, 2026):  # 10 years of training data
        logging.info(f"  Building features for {y}...")
        tf = build_draft_features(master_df, y)
        if not tf.empty:
            train_feature_frames.append(tf)
    train_features = pd.concat(train_feature_frames, ignore_index=True)

    # Get season outcomes for training
    season_outcomes = []
    for y in range(2016, 2026):
        sy = master_df[(master_df['Year'] == y) & (master_df['played'] > 0)]
        so = sy.groupby('Player ID').agg(season_avg=('SC', 'mean'), season_games=('SC', 'count')).reset_index()
        so['Year'] = y
        season_outcomes.append(so)
    outcomes_df = pd.concat(season_outcomes, ignore_index=True)
    train_features = train_features.merge(outcomes_df, on=['Player ID', 'Year'], how='left')
    train_features = train_features.dropna(subset=['season_avg'])
    train_features = train_features[train_features['season_games'] >= 5].copy()

    # --- Filter to SC-relevant players only (top N per position per year) ---
    pre_filter = len(train_features)
    train_features = filter_sc_relevant(train_features)
    logging.info(f"  SC-relevant filter: {pre_filter} -> {len(train_features)} players "
                 f"(caps: {SC_RELEVANT_CAPS})")

    # --- Merge user draft notes for training years (2022-2025) ---
    logging.info("Loading user draft notes for 2022-2025...")
    note_parsers = {2022: parse_2022_notes, 2023: parse_2023_notes,
                    2024: parse_2024_notes, 2025: parse_2025_notes}
    note_frames = []
    for y, parser in note_parsers.items():
        try:
            nf = parser()
            note_frames.append(nf)
        except Exception as e:
            logging.warning(f"  Could not parse notes for {y}: {e}")

    if note_frames:
        all_notes = pd.concat(note_frames, ignore_index=True)
        classified_notes = classify_all_notes(all_notes)
        # Deduplicate per player per year, keep first (position tab takes priority)
        classified_notes = classified_notes.drop_duplicates(subset=['feed_id', 'year'], keep='first')
        # NOTE_FLAG_COLS includes 'user_rank', so no need to add it separately
        merge_cols = ['feed_id', 'year'] + [c for c in NOTE_FLAG_COLS if c in classified_notes.columns]
        note_merge = classified_notes[merge_cols].copy()
        note_merge = note_merge.rename(columns={'feed_id': 'Player ID', 'year': 'Year'})
        # Convert types before merge
        for c in NOTE_FLAG_COLS:
            if c in note_merge.columns:
                note_merge[c] = pd.to_numeric(note_merge[c], errors='coerce')
        train_features = train_features.merge(note_merge, on=['Player ID', 'Year'], how='left')
        logging.info(f"  Merged notes: {train_features.get('has_note', pd.Series()).notna().sum()} players with note data")
    else:
        logging.warning("  No notes loaded")

    # Fill missing note flags with 0, missing user_rank with 999
    for c in NOTE_FLAG_COLS:
        if c not in train_features.columns:
            train_features[c] = 0.0
        train_features[c] = pd.to_numeric(train_features[c], errors='coerce')
        if c == 'user_rank':
            train_features[c] = train_features[c].fillna(999.0)
        else:
            train_features[c] = train_features[c].fillna(0.0)

    # Ensure all feature columns exist
    for c in DRAFT_FEATURE_COLS:
        if c not in train_features.columns:
            train_features[c] = 0

    X_train = train_features[DRAFT_FEATURE_COLS].fillna(0)
    y_train = train_features['season_avg']

    # XGBoost model
    xgb_model = DraftXGB(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42,
        n_jobs=-1, tree_method='hist',
    )
    xgb_model.fit(X_train, y_train, verbose=False)

    # Ridge regression model (ensemble partner)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    logging.info(f"Ensemble trained on {len(X_train)} player-seasons with {len(DRAFT_FEATURE_COLS)} features (XGBoost + Ridge)")

    # Build 2026 features using the same comprehensive feature builder
    logging.info("Building 2026 comprehensive features...")
    features_2026 = build_draft_features(master_df, 2026)
    logging.info(f"  Built features for {len(features_2026)} players")

    # --- Load 2026 user notes if available ---
    notes_2026_file = Path('data/draft_notes/draft_notes_2026.xlsx')
    if not notes_2026_file.exists():
        notes_2026_file = Path('data/draft_notes/draft_notes_2026.csv')

    if notes_2026_file.exists():
        logging.info(f"Loading 2026 user notes from {notes_2026_file}...")
        try:
            if notes_2026_file.suffix == '.xlsx':
                # Parse position tabs with notes (same format as 2025)
                note_rows = []
                for pos in ['DEF', 'MID', 'FWD', 'RUC']:
                    try:
                        ndf = pd.read_excel(notes_2026_file, sheet_name=pos)
                        for _, r in ndf.iterrows():
                            fid = r.get('feed_id', None)
                            if pd.isna(fid):
                                continue
                            # Use explicit board_rank if set; else None (=unranked, will default to 999)
                            board_rank = r.get('board_rank', np.nan)
                            note_rows.append({
                                'feed_id': int(fid),
                                'note': r.get('notes', np.nan),
                                'user_rank': int(board_rank) if pd.notna(board_rank) else None,
                                'position': pos,
                                'year': 2026,
                            })
                    except Exception:
                        pass
                notes_2026 = pd.DataFrame(note_rows) if note_rows else pd.DataFrame()
            else:
                notes_2026 = pd.read_csv(notes_2026_file)

            if not notes_2026.empty:
                notes_2026_classified = classify_all_notes(notes_2026)
                notes_2026_classified = notes_2026_classified.drop_duplicates(subset=['feed_id'], keep='first')
                n_noted = notes_2026_classified['note'].notna().sum()
                logging.info(f"  Loaded {len(notes_2026_classified)} players, {n_noted} with notes")

                # Merge note flags into features_2026
                note_cols_to_merge = ['feed_id'] + NOTE_FLAG_COLS  # user_rank already in NOTE_FLAG_COLS
                note_cols_to_merge = list(dict.fromkeys(note_cols_to_merge))  # deduplicate
                note_cols_available = [c for c in note_cols_to_merge if c in notes_2026_classified.columns]
                notes_merge = notes_2026_classified[note_cols_available].copy()
                features_2026 = features_2026.merge(
                    notes_merge, left_on='Player ID', right_on='feed_id', how='left'
                )
                features_2026.drop(columns=['feed_id'], errors='ignore', inplace=True)
        except Exception as e:
            logging.warning(f"  Error loading 2026 notes: {e}")

    # Fill note columns with defaults (ensure float types for XGBoost)
    for c in NOTE_FLAG_COLS:
        if c not in features_2026.columns:
            features_2026[c] = 0.0
        if c == 'user_rank':
            features_2026[c] = pd.to_numeric(features_2026[c], errors='coerce').fillna(999).astype(float)
        else:
            features_2026[c] = pd.to_numeric(features_2026[c], errors='coerce').fillna(0).astype(float)
    if 'user_rank' not in features_2026.columns:
        features_2026['user_rank'] = 999.0
    features_2026['user_rank'] = pd.to_numeric(features_2026['user_rank'], errors='coerce').fillna(999).astype(float)

    # Merge comprehensive features into board
    feat_merge_cols = [c for c in DRAFT_FEATURE_COLS if c in features_2026.columns]
    feat_merge = features_2026[['Player ID'] + feat_merge_cols].copy()
    board = board.merge(feat_merge, left_on='feed_id', right_on='Player ID', how='left', suffixes=('', '_feat'))
    board.drop(columns=['Player ID'], errors='ignore', inplace=True)

    # Use comprehensive features for prediction
    x_cols = {}
    for feat in DRAFT_FEATURE_COLS:
        # Prefer the feature-engineered column, fall back to board column
        feat_col = f'{feat}_feat' if f'{feat}_feat' in board.columns else feat
        if feat_col in board.columns:
            x_cols[feat] = board[feat_col].fillna(0).values
        elif feat in board.columns:
            x_cols[feat] = board[feat].fillna(0).values
        else:
            x_cols[feat] = np.zeros(len(board))
    X_2026 = pd.DataFrame(x_cols, index=board.index)

    # Generate ensemble predictions (70% XGBoost + 30% Ridge)
    xgb_preds = xgb_model.predict(X_2026)
    X_2026_scaled = scaler.transform(X_2026)
    ridge_preds = ridge_model.predict(X_2026_scaled)
    board['draft_model_predicted'] = xgb_preds * 0.70 + ridge_preds * 0.30
    board['xgb_pred'] = xgb_preds
    board['ridge_pred'] = ridge_preds

    # Print XGBoost feature importance (top 20)
    importances = pd.Series(xgb_model.feature_importances_, index=DRAFT_FEATURE_COLS).sort_values(ascending=False)
    print("\n--- XGBoost Feature Importance (Top 20) - Ensemble ---")
    for feat, imp in importances.head(20).items():
        print(f"  {feat:<30}: {imp:.3f}")

    # Print Ridge top coefficients
    ridge_coefs = pd.Series(ridge_model.coef_, index=DRAFT_FEATURE_COLS).sort_values(ascending=False)
    print("\n--- Ridge Top 10 Positive Coefficients ---")
    for feat, coef in ridge_coefs.head(10).items():
        print(f"  {feat:<30}: {coef:+.3f}")
    print("\n--- Ridge Top 10 Negative Coefficients ---")
    for feat, coef in ridge_coefs.tail(10).items():
        print(f"  {feat:<30}: {coef:+.3f}")

    # Merge raw note text into board for display
    if notes_2026_file.exists() and 'notes_2026_classified' in dir() and not notes_2026_classified.empty:
        note_text = notes_2026_classified[['feed_id', 'note']].drop_duplicates(subset='feed_id')
        note_text = note_text.rename(columns={'note': 'user_notes'})
        board = board.merge(note_text, on='feed_id', how='left')

    # Enrich board with key features from comprehensive builder for display
    for col in ['post_r13_avg', 'avg_2yr', 'last5_avg', 'positive_tags',
                'injury_tags', 'injury_tags_2yr', 'second_half_form',
                'avg_3yr', 'pct_above_80', 'pct_above_100']:
        feat_col = f'{col}_feat' if f'{col}_feat' in board.columns else col
        if feat_col in board.columns and col not in board.columns:
            board[col] = board[feat_col]

    # --- Draft_Value: Blend round model + comprehensive draft model ---
    board['durability_score'] = np.clip(board['games_2025'].fillna(0) / 23 * 100, 0, 100)
    board['consistency_score'] = np.clip(100 - board['cv_2025'].fillna(0.5) * 200, 0, 100)

    # Normalize both predictions to 0-100
    for col in ['avg_projected', 'draft_model_predicted']:
        c_min = board[col].min()
        c_max = board[col].max()
        if c_max > c_min:
            board[f'{col}_norm'] = (board[col] - c_min) / (c_max - c_min) * 100
        else:
            board[f'{col}_norm'] = 50

    # Blend: 60% round model (matchup-aware) + 40% comprehensive draft model (123 features)
    board['Draft_Value'] = (
        board['avg_projected_norm'].fillna(0) * 0.60 +
        board['draft_model_predicted_norm'].fillna(0) * 0.40
    )

    # Small position tiebreaker
    pos_adj = {'MID': 1, 'DEF': 1, 'FWD': -1, 'RUC': 0}
    board['pos_adjustment'] = board['primary_pos'].map(pos_adj).fillna(0)
    board['Draft_Value'] += board['pos_adjustment']

    # Keep TPOR as an info column
    for col in ['TPOR_estimate']:
        c_min = board[col].min()
        c_max = board[col].max()
        if c_max > c_min:
            board[f'{col}_norm'] = (board[col] - c_min) / (c_max - c_min) * 100
        else:
            board[f'{col}_norm'] = 50

    # Diamond and Bust flags
    # Diamond: not top-50 by avg, but has strong durability + consistency + form
    board['is_diamond'] = (
        (board['avg_projected'].rank(ascending=False) > 50) &
        (board['games_2025'].fillna(0) >= 18) &
        (board['cv_2025'].fillna(1) < 0.3) &
        (board['form_delta'].fillna(0) > 0) &
        (board['avg_projected'].fillna(0) > 70)
    ).astype(int)

    # Bust risk: high projection but poor durability or high volatility
    board['bust_risk'] = (
        (board['avg_projected'].rank(ascending=False) <= 60) &
        ((board['games_2025'].fillna(0) < 15) | (board['cv_2025'].fillna(0) > 0.35))
    ).astype(int)

    # Position rank
    board['position_rank'] = board.groupby('primary_pos')['Draft_Value'].rank(ascending=False, method='dense')
    board['overall_rank'] = board['Draft_Value'].rank(ascending=False, method='dense')

    board.sort_values('Draft_Value', ascending=False, inplace=True)

    # --- Build Excel output ---
    output_cols = [
        'overall_rank', 'feed_id', 'first_name', 'last_name', 'team', 'all_positions', 'age_proxy',
        'games_2025', 'avg_2025', 'total_2025', 'std_2025', 'cv_2025', 'ceiling_2025', 'floor_2025',
        'post_r13_avg', 'post_r13_games', 'avg_2yr', 'avg_3yr', 'last5_avg',
        'pct_above_80', 'pct_above_100',
        'positive_tags', 'injury_tags', 'injury_tags_2yr',
        'avg_projected', 'draft_model_predicted', 'avg_p80', 'avg_p100', 'matchup_variance',
        'TPOR_estimate', 'durability_score', 'consistency_score', 'form_delta', 'second_half_form',
        'Draft_Value', 'position_rank',
        'is_diamond', 'bust_risk',
        'career_games', 'previous_games', 'previous_average',
        'user_notes',
    ]
    output_cols = [c for c in output_cols if c in board.columns]

    # Add notes column
    board['notes'] = ''

    excel_path = OUTPUT_DIR / '2026_DRAFT_BOARD.xlsx'

    # If file is locked (open in Excel), write to a timestamped copy
    import time
    try:
        with open(excel_path, 'a'):
            pass
    except PermissionError:
        ts = time.strftime('%Y%m%d_%H%M%S')
        excel_path = OUTPUT_DIR / f'2026_DRAFT_BOARD_{ts}.xlsx'
        logging.warning(f"  Original file locked, writing to {excel_path}")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Position tabs (capped to SC-relevant counts)
        sc_relevant_ids = set()
        for pos in ['DEF', 'MID', 'FWD', 'RUC']:
            cap = SC_RELEVANT_CAPS[pos]
            pos_df = board[board['all_positions'].str.contains(pos, na=False)].copy()
            pos_df['position_rank'] = pos_df['Draft_Value'].rank(ascending=False, method='dense')
            pos_df = pos_df.sort_values('Draft_Value', ascending=False).head(cap)
            pos_df[output_cols + ['notes']].to_excel(writer, sheet_name=pos, index=False)
            sc_relevant_ids.update(pos_df['feed_id'].tolist())

        # ALL tab: only SC-relevant players (union of position tabs)
        all_df = board[board['feed_id'].isin(sc_relevant_ids)].copy()
        all_df = all_df.sort_values('Draft_Value', ascending=False)
        all_df[output_cols + ['notes']].to_excel(writer, sheet_name='ALL', index=False)

        # Insights tab
        insights_data = {
            'Insight': [
                'DURABILITY IS KING',
                'CONSISTENCY PREDICTS RELIABILITY',
                'MID/DEF OVERPERFORM',
                'AVOID FWD FLYERS',
                'PEAK AGE WINDOW',
                'LATE ROUND VALUE EXISTS',
                'DIAMOND FLAG',
                'BUST RISK FLAG',
                'FORM TRAJECTORY',
                'MATCHUP VARIANCE',
            ],
            'Detail': [
                'Players with 20+ games last year average 15+ points more. Prioritize availability.',
                'Low CV (<0.3) players deliver more reliable season totals. Use consistency_score.',
                'Midfielders and Defenders have higher success rates. Add +3 to their value.',
                'Forwards are riskier. 7 of 10 worst picks historically were FWD. Subtract -3.',
                'Players in years 4-8 of career (age_proxy) tend to peak. Small bonus applied.',
                'Elite picks (1-20) regress to mean. Value picks (101+) often improve.',
                'is_diamond=1: Not a top-50 projection but has strong durability + consistency + form.',
                'bust_risk=1: High projection but poor durability (<15 games) or high volatility (CV>0.35).',
                'form_delta: last 3 games avg minus last 10 games avg. Positive = trending up.',
                'matchup_variance: difference between best and worst round projection. High = matchup dependent.',
            ]
        }
        pd.DataFrame(insights_data).to_excel(writer, sheet_name='Insights', index=False)

    logging.info(f"Draft board saved to: {excel_path}")

    # Print top 20
    print("\n" + "=" * 100)
    print("2026 DRAFT BOARD - TOP 25")
    print("=" * 100)
    top = board.head(25)
    print(f"{'#':>3} {'Player':<25} {'Pos':<8} {'Team':>4} {'2025 Gm':>7} {'2025 Avg':>8} {'Proj Avg':>8} "
          f"{'P>100':>5} {'Dur':>4} {'Con':>4} {'Form':>5} {'Value':>6} {'Flags'}")
    print("-" * 120)
    for _, row in top.iterrows():
        name = f"{row['first_name']} {row['last_name']}"
        flags = []
        if row.get('is_diamond', 0) == 1:
            flags.append('DIAMOND')
        if row.get('bust_risk', 0) == 1:
            flags.append('BUST RISK')
        print(f"{int(row['overall_rank']):>3} {name:<25} {row['all_positions']:<8} {row['team']:>4} "
              f"{int(row.get('games_2025', 0)):>7} {row.get('avg_2025', 0):>8.1f} "
              f"{row.get('avg_projected', 0):>8.1f} {row.get('avg_p100', 0):>5.1f} "
              f"{row.get('durability_score', 0):>4.0f} {row.get('consistency_score', 0):>4.0f} "
              f"{row.get('form_delta', 0):>+5.1f} {row['Draft_Value']:>6.1f} "
              f"{'  '.join(flags)}")

    # Print diamonds
    diamonds = board[board['is_diamond'] == 1].head(15)
    if len(diamonds) > 0:
        print(f"\n--- DIAMOND PICKS (Hidden Gems) ---")
        for _, row in diamonds.iterrows():
            name = f"{row['first_name']} {row['last_name']}"
            print(f"  {name:<25} {row['all_positions']:<8} Proj: {row.get('avg_projected', 0):.1f} | "
                  f"2025: {int(row.get('games_2025', 0))} gm, {row.get('avg_2025', 0):.1f} avg | "
                  f"Form: {row.get('form_delta', 0):+.1f}")

    # Print bust risks
    busts = board[board['bust_risk'] == 1].head(10)
    if len(busts) > 0:
        print(f"\n--- BUST RISKS (Proceed with Caution) ---")
        for _, row in busts.iterrows():
            name = f"{row['first_name']} {row['last_name']}"
            reason = 'low durability' if row.get('games_2025', 0) < 15 else 'high volatility'
            print(f"  {name:<25} {row['all_positions']:<8} Proj: {row.get('avg_projected', 0):.1f} | "
                  f"2025: {int(row.get('games_2025', 0))} gm, CV: {row.get('cv_2025', 0):.2f} | "
                  f"Risk: {reason}")

    return board


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 100)
    print("2026 SUPERCOACH DRAFT PREPARATION PIPELINE")
    print("=" * 100)

    # Load master data
    logging.info(f"Loading master data from {MASTER_FILE}...")
    master_df = pd.read_csv(MASTER_FILE, low_memory=False)
    master_df['Round_Num'] = pd.to_numeric(master_df['Round_Num'], errors='coerce')
    logging.info(f"Loaded {len(master_df)} rows, {master_df['Year'].nunique()} years")

    # Section 1: Season comparisons
    season_comp = build_season_comparisons(master_df)

    # Section 2: Heat maps
    heat_maps = build_heat_maps(season_comp)

    # Section 3: Success analysis
    if not heat_maps.empty:
        analysis = analyze_draft_success(season_comp, heat_maps)
    else:
        logging.warning("No heat map data - skipping success analysis")
        analysis = pd.DataFrame()

    # Section 4: Draft board
    board = build_2026_draft_board(master_df, analysis)

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print(f"  - season_comparison_{{2022-2025}}.csv")
    print(f"  - draft_heat_map_{{2022-2025}}.csv")
    print(f"  - draft_success_analysis.csv")
    print(f"  - 2026_DRAFT_BOARD.xlsx")


if __name__ == '__main__':
    main()
