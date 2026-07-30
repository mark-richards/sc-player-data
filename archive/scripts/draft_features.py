"""
Comprehensive Draft Feature Engineering
=========================================
Builds 120+ pre-draft features from master data for predicting next-season SC average.

Backtested over 2022-2025 with leave-one-year-out validation:
  - Spearman 0.771 (vs 0.756 baseline)
  - MAE 9.6 (vs 11.0 baseline) — 13% more accurate
  - Top-50 overlap 33.2/50 (vs 32.0)
  - Busts in top-30: 5.0 (vs 6.0) — 17% fewer

Key feature categories by importance:
  1. Regression estimate (28.1%) — shrink toward population mean
  2. Consistency (17.1%) — pct_above_80, q75, iqr
  3. Multi-year averages (11.3%) — avg_2yr, avg_3yr
  4. Basic scoring (9.0%) — prev_avg, prev_median
  5. Scoring composition (4.1%) — contested poss, sc per disposal
  6. Form & trajectory (4.0%) — post_r13_avg, last5/8 avg
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
import json
import logging
from pathlib import Path


# Feature columns used by the draft model (ordered for reference)
DRAFT_FEATURE_COLS = [
    # Basic
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'prev_total',
    # Form & Trajectory
    'post_r13_avg', 'post_r13_games', 'second_half_form', 'pre_r13_avg',
    'last3_avg', 'last5_avg', 'last8_avg',
    'last3_vs_season', 'last5_vs_season',
    'season_trend_slope', 'first3_avg', 'first5_avg',
    # Multi-year
    'avg_2yr', 'games_2yr', 'std_2yr', 'avg_3yr', 'games_3yr',
    'prev2_avg', 'prev2_games', 'yoy_change', 'two_yr_trend',
    # Scoring Composition
    'avg_kicks', 'avg_handballs', 'avg_disposals', 'avg_marks', 'avg_tackles',
    'avg_hitouts', 'avg_goals', 'avg_frees_for', 'avg_frees_against',
    'kick_ratio', 'avg_contested_poss', 'avg_clearances', 'avg_uncontested_poss',
    'contested_poss_ratio', 'avg_disposal_eff', 'avg_clangers', 'clanger_ratio',
    'avg_metres_gained', 'sc_per_disposal', 'avg_time_on_ground', 'sc_per_minute_tog',
    # Consistency & Reliability
    'score_skewness', 'pct_above_100', 'pct_above_80', 'pct_below_60', 'pct_below_40',
    'iqr', 'q25', 'q75', 'floor_10pct',
    'max_streak_80plus', 'max_streak_below60',
    # Team Context
    'team_avg_sc', 'team_top5_avg', 'team_total_players',
    'pos_teammate_avg', 'pos_teammate_count', 'pos_teammates_better', 'team_sc_share',
    'changed_team', 'new_team_strength', 'old_team_strength',
    # Opposition
    'avg_vs_strong_opp', 'avg_vs_weak_opp', 'opp_sensitivity',
    # Tags & Role
    'positive_tags', 'injury_tags', 'injury_tags_2yr', 'positive_tags_2yr',
    'tag_star', 'tag_hot', 'tag_gun', 'tag_injured', 'tag_sore',
    'tag_tagger', 'tag_tagged', 'tag_sub', 'tag_subbed',
    'tag_wing', 'tag_guard', 'tag_ruck', 'tag_spearhead',
    'tag_shovel', 'tag_job', 'tag_rookie', 'tag_cash',
    'is_tagger_or_tagged', 'is_sub_risk', 'is_wing_guard', 'is_inside_mid',
    'injury_rate',
    # Career & Age
    'career_games', 'age_proxy', 'career_avg', 'career_trend', 'pct_of_career_best',
    'games_minus_1', 'games_minus_2', 'games_minus_3',
    # Durability & Workload
    'games_pct', 'avg_minutes', 'avg_tog_pct', 'min_tog', 'bench_pct',
    'total_price_change', 'avg_price_change', 'price_growth_pct',
    # Interactions & Derived
    'avg_x_durability', 'avg_x_consistency', 'avg_x_age_sweet_spot',
    'ceiling_floor_range', 'avg_vs_median_gap', 'floor_ratio', 'ceiling_ratio',
    'regression_est_85', 'prev_position_rank',
    # SC Pre-Draft Projections (2022+ only)
    'sc_ownership_pct', 'sc_start_pct', 'sc_projected_avg', 'sc_projected_price',
    'sc_breakeven_avg', 'sc_is_hot', 'sc_is_cold',
    # Pre-season news signals (2024+ only; 0 for earlier years)
    'news_mention_count', 'news_injury_flag', 'news_positive_buzz',
    'news_role_change_flag', 'news_concern_flag',
    # Position dummies
    'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_RUC',
    # User draft notes (from pre-draft scouting)
    'note_injury', 'note_availability_concern', 'note_returning',
    'note_expected_improve', 'note_expected_decline', 'note_role_change',
    'note_love', 'note_bench_candidate', 'note_avoid', 'has_note',
    'user_rank',
]

# Note-specific feature columns (subset of DRAFT_FEATURE_COLS)
NOTE_FLAG_COLS = [
    'note_injury', 'note_availability_concern', 'note_returning',
    'note_expected_improve', 'note_expected_decline', 'note_role_change',
    'note_love', 'note_bench_candidate', 'note_avoid', 'has_note',
    'user_rank',
]

# SC-relevant player caps per position (8-team league squad sizes)
SC_RELEVANT_CAPS = {'FWD': 48, 'MID': 64, 'DEF': 48, 'RUC': 16}


def filter_sc_relevant(df, year_col='Year', pos_col='primary_pos', score_col='season_avg'):
    """Filter to only SC-relevant players: top N per position per year."""
    keep = []
    for y in df[year_col].unique():
        yr = df[df[year_col] == y]
        for pos, cap in SC_RELEVANT_CAPS.items():
            pos_players = yr[yr[pos_col] == pos].nlargest(cap, score_col)
            keep.append(pos_players)
    if keep:
        return pd.concat(keep, ignore_index=True)
    return df.iloc[0:0]


def load_predraft_projections(year):
    """Load pre-draft projection features from SuperCoach JSON files.
    Returns DataFrame with feed_id + 7 projection features. Available 2022-2026."""
    possible_paths = [
        Path(f'draft_prep/SC {year}/SC_{year}_player_data_pre_draft.json'),
        Path(f'draft_prep/SC {year}/SC_{year}_player_data.json'),
    ]
    json_file = None
    for p in possible_paths:
        if p.exists():
            json_file = p
            break
    if json_file is None:
        return pd.DataFrame()

    with open(json_file, 'r') as fh:
        data = json.load(fh)

    rows = []
    for player in data:
        ps_list = player.get('player_stats')
        if not ps_list:
            continue
        ps = ps_list[0]
        feed_id = player.get('feed_id')
        if not feed_id:
            continue

        # Ownership/start: normalize to 0-1 (some years use 0-100, others 0-1)
        own = ps.get('own', 0) or 0
        start = ps.get('start', 0) or 0
        if own > 1:
            own = own / 100.0
        if start > 1:
            start = start / 100.0

        # Projected avg: mean of non-zero ppts
        ppts = [ps.get('ppts1', 0) or 0, ps.get('ppts2', 0) or 0, ps.get('ppts3', 0) or 0]
        nonzero_ppts = [v for v in ppts if v > 0]
        proj_avg = np.mean(nonzero_ppts) if nonzero_ppts else 0

        rows.append({
            'feed_id': int(feed_id),
            'sc_ownership_pct': own,
            'sc_start_pct': start,
            'sc_projected_avg': proj_avg,
            'sc_projected_price': ps.get('pp1', 0) or 0,
            'sc_breakeven_avg': np.mean([ps.get('be1', 0) or 0, ps.get('be2', 0) or 0, ps.get('be3', 0) or 0]),
            'sc_is_hot': 1 if ps.get('hot') or ps.get('phot') else 0,
            'sc_is_cold': 1 if ps.get('cold') or ps.get('pcold') else 0,
        })
    return pd.DataFrame(rows)


def simplify_position(pos):
    if pd.isna(pos):
        return 'MID'
    pos = str(pos).upper()
    for p in ['DEF', 'MID', 'FWD', 'RUC']:
        if p in pos:
            return p
    return 'MID'


def _max_streak(arr):
    max_s = current = 0
    for v in arr:
        if v:
            current += 1
            max_s = max(max_s, current)
        else:
            current = 0
    return max_s


def build_draft_features(master_df, target_year):
    """
    Build comprehensive pre-draft features for target_year using only prior data.
    Returns a DataFrame with one row per player and 120+ feature columns.
    """
    prev_year = target_year - 1
    prev2_year = target_year - 2
    prev3_year = target_year - 3

    # Ensure tags are strings
    master_df = master_df.copy()
    master_df['Tag'] = master_df['Tag'].fillna('')
    master_df['Tag 2'] = master_df['Tag 2'].fillna('')

    # Derived columns
    if 'Disposals' not in master_df.columns:
        master_df['Disposals'] = master_df['Kicks'] + master_df['Handballs']
    if 'Uncontested_Poss' not in master_df.columns:
        master_df['Uncontested_Poss'] = master_df['Disposals'] - master_df['Contested Possessions'].fillna(0)

    prev = master_df[(master_df['Year'] == prev_year) & (master_df['played'] > 0)].copy()
    if prev.empty:
        return pd.DataFrame()

    all_prev = master_df[(master_df['Year'] <= prev_year) & (master_df['played'] > 0)].copy()
    two_yr = master_df[(master_df['Year'].isin([prev_year, prev2_year])) & (master_df['played'] > 0)].copy()
    three_yr = master_df[(master_df['Year'].isin([prev_year, prev2_year, prev3_year])) & (master_df['played'] > 0)].copy()

    features = {}
    players = prev['Player ID'].unique()

    for pid in players:
        p = prev[prev['Player ID'] == pid].sort_values('Round_Num')
        if len(p) == 0:
            continue

        f = {'Player ID': pid, 'Year': target_year}
        f['first_name'] = p['First Name'].iloc[-1]
        f['last_name'] = p['Last Name'].iloc[-1]
        f['team'] = p['Team'].iloc[-1]
        f['position'] = p['sc_position'].iloc[-1]

        sc = p['SC'].values
        n_games = len(sc)

        # 1. BASIC SCORING
        f['prev_avg'] = sc.mean()
        f['prev_games'] = n_games
        f['prev_total'] = sc.sum()
        f['prev_std'] = sc.std() if n_games > 1 else 0
        f['prev_cv'] = f['prev_std'] / f['prev_avg'] if f['prev_avg'] > 0 else 1
        f['prev_ceiling'] = sc.max()
        f['prev_floor'] = sc.min()
        f['prev_median'] = np.median(sc)

        # 2. SCORING COMPOSITION
        f['avg_kicks'] = p['Kicks'].mean()
        f['avg_handballs'] = p['Handballs'].mean()
        f['avg_disposals'] = p['Disposals'].mean()
        f['avg_marks'] = p['Marks'].mean()
        f['avg_tackles'] = p['Tackles'].mean()
        f['avg_hitouts'] = p['Hitouts'].mean()
        f['avg_goals'] = p['Goals'].mean()
        f['avg_frees_for'] = p['Frees for'].mean()
        f['avg_frees_against'] = p['Frees against'].mean()

        total_disp = p['Disposals'].sum()
        f['kick_ratio'] = p['Kicks'].sum() / total_disp if total_disp > 0 else 0.5

        for col, key in [('Contested Possessions', 'avg_contested_poss'),
                         ('Clearances', 'avg_clearances'),
                         ('Uncontested_Poss', 'avg_uncontested_poss'),
                         ('Disposal efficiency', 'avg_disposal_eff'),
                         ('Clangers', 'avg_clangers'),
                         ('Metres gained', 'avg_metres_gained'),
                         ('Time on ground', 'avg_time_on_ground')]:
            f[key] = p[col].mean() if p[col].notna().sum() > 0 else np.nan

        f['contested_poss_ratio'] = p['Contested Possessions'].sum() / total_disp if total_disp > 0 and p['Contested Possessions'].notna().sum() > 0 else np.nan
        f['clanger_ratio'] = p['Clangers'].sum() / total_disp if total_disp > 0 and p['Clangers'].notna().sum() > 0 else np.nan
        f['sc_per_disposal'] = f['prev_avg'] / f['avg_disposals'] if f['avg_disposals'] > 0 else 0

        avg_tog = f.get('avg_time_on_ground', np.nan)
        f['sc_per_minute_tog'] = f['prev_avg'] / (avg_tog * 1.2) if pd.notna(avg_tog) and avg_tog > 0 else np.nan

        # 3. FORM & TRAJECTORY
        second_half = p[p['Round_Num'] > 13]
        first_half = p[p['Round_Num'] <= 13]

        f['post_r13_avg'] = second_half['SC'].mean() if len(second_half) > 0 else f['prev_avg']
        f['post_r13_games'] = len(second_half)
        f['second_half_form'] = f['post_r13_avg'] - f['prev_avg']
        f['pre_r13_avg'] = first_half['SC'].mean() if len(first_half) > 0 else f['prev_avg']

        for n in [3, 5, 8]:
            f[f'last{n}_avg'] = sc[-n:].mean() if n_games >= n else f['prev_avg']
            f[f'last{n}_vs_season'] = f[f'last{n}_avg'] - f['prev_avg']
        for n in [3, 5]:
            f[f'first{n}_avg'] = sc[:n].mean() if n_games >= n else f['prev_avg']

        if n_games >= 5:
            coeffs = np.polyfit(np.arange(n_games), sc, 1)
            f['season_trend_slope'] = coeffs[0]
        else:
            f['season_trend_slope'] = 0

        # 4. MULTI-YEAR TRAJECTORY
        p_2yr = two_yr[two_yr['Player ID'] == pid]
        p_3yr = three_yr[three_yr['Player ID'] == pid]
        f['avg_2yr'] = p_2yr['SC'].mean() if len(p_2yr) > 0 else f['prev_avg']
        f['games_2yr'] = len(p_2yr)
        f['std_2yr'] = p_2yr['SC'].std() if len(p_2yr) > 1 else 0
        f['avg_3yr'] = p_3yr['SC'].mean() if len(p_3yr) > 0 else f['prev_avg']
        f['games_3yr'] = len(p_3yr)

        p_prev2 = master_df[(master_df['Player ID'] == pid) & (master_df['Year'] == prev2_year) & (master_df['played'] > 0)]
        f['prev2_avg'] = p_prev2['SC'].mean() if len(p_prev2) > 0 else np.nan
        f['yoy_change'] = f['prev_avg'] - f['prev2_avg'] if pd.notna(f['prev2_avg']) else 0
        f['prev2_games'] = len(p_prev2)
        f['two_yr_trend'] = f['prev_avg'] - f['avg_2yr']

        # 5. DURABILITY & WORKLOAD
        f['games_pct'] = n_games / 23
        f['avg_minutes'] = p['minutes_played'].mean()
        f['avg_tog_pct'] = p['Time on ground'].mean() if p['Time on ground'].notna().sum() > 0 else np.nan
        f['min_tog'] = p['Time on ground'].min() if p['Time on ground'].notna().sum() > 0 else np.nan
        f['bench_pct'] = p['Bench status'].mean() if p['Bench status'].notna().sum() > 0 else 0
        f['total_price_change'] = p['price_change'].sum() if p['price_change'].notna().sum() > 0 else 0
        f['avg_price_change'] = p['price_change'].mean() if p['price_change'].notna().sum() > 0 else 0
        if p['price'].notna().sum() > 0:
            f['price_growth_pct'] = (p['price'].iloc[-1] - p['price'].iloc[0]) / p['price'].iloc[0] if p['price'].iloc[0] > 0 else 0
        else:
            f['price_growth_pct'] = 0

        # 6. CONSISTENCY & RELIABILITY
        if n_games >= 3:
            f['score_skewness'] = skew(sc) if n_games >= 5 else 0
            f['pct_above_100'] = (sc >= 100).mean()
            f['pct_above_80'] = (sc >= 80).mean()
            f['pct_below_60'] = (sc < 60).mean()
            f['pct_below_40'] = (sc < 40).mean()
            f['iqr'] = np.percentile(sc, 75) - np.percentile(sc, 25)
            f['q25'] = np.percentile(sc, 25)
            f['q75'] = np.percentile(sc, 75)
            f['floor_10pct'] = np.percentile(sc, 10)
            f['max_streak_80plus'] = _max_streak(sc >= 80)
            f['max_streak_below60'] = _max_streak(sc < 60)
        else:
            for k in ['score_skewness', 'pct_above_100', 'pct_above_80', 'pct_below_60',
                       'pct_below_40', 'iqr', 'max_streak_80plus', 'max_streak_below60']:
                f[k] = 0
            f['q25'] = f['q75'] = f['floor_10pct'] = f['prev_avg']

        # 7. TEAM CONTEXT
        team = p['Team'].iloc[-1]
        primary_pos = simplify_position(f['position'])
        f['primary_pos'] = primary_pos

        teammates = prev[(prev['Team'] == team) & (prev['Player ID'] != pid)]
        if len(teammates) > 0:
            team_avgs = teammates.groupby('Player ID')['SC'].mean()
            f['team_avg_sc'] = team_avgs.mean()
            f['team_top5_avg'] = team_avgs.nlargest(5).mean()
            f['team_total_players'] = len(team_avgs)
        else:
            f['team_avg_sc'] = 75
            f['team_top5_avg'] = 100
            f['team_total_players'] = 20

        same_pos = teammates[teammates['sc_position'].apply(simplify_position) == primary_pos]
        if len(same_pos) > 0:
            pos_avgs = same_pos.groupby('Player ID')['SC'].mean()
            f['pos_teammate_avg'] = pos_avgs.mean()
            f['pos_teammate_count'] = len(pos_avgs)
            f['pos_teammates_better'] = (pos_avgs > f['prev_avg']).sum()
        else:
            f['pos_teammate_avg'] = 75
            f['pos_teammate_count'] = 0
            f['pos_teammates_better'] = 0

        team_total = prev[prev['Team'] == team]['SC'].sum()
        f['team_sc_share'] = p['SC'].sum() / team_total if team_total > 0 else 0

        # Team change detection
        p_prev2 = master_df[(master_df['Player ID'] == pid) &
                            (master_df['Year'] == prev2_year) &
                            (master_df['played'] > 0)]
        if len(p_prev2) > 0:
            team_prev2 = p_prev2['Team'].iloc[-1]
            if team != team_prev2:
                f['changed_team'] = 1
                old_tm = master_df[(master_df['Year'] == prev2_year) &
                                   (master_df['Team'] == team_prev2) &
                                   (master_df['Player ID'] != pid) &
                                   (master_df['played'] > 0)]
                if len(old_tm) > 0:
                    f['old_team_strength'] = old_tm.groupby('Player ID')['SC'].mean().mean()
                else:
                    f['old_team_strength'] = 75
                f['new_team_strength'] = f['team_avg_sc']
            else:
                f['changed_team'] = 0
                f['old_team_strength'] = f['team_avg_sc']
                f['new_team_strength'] = f['team_avg_sc']
        else:
            f['changed_team'] = 0
            f['old_team_strength'] = f['team_avg_sc']
            f['new_team_strength'] = f['team_avg_sc']

        # 8. OPPOSITION
        if 'opp_abbrev' in p.columns and p['opp_abbrev'].notna().sum() > 0:
            team_strengths = prev.groupby('opp_abbrev')['SC'].mean()
            median_opp = team_strengths.median()
            strong_opps = team_strengths[team_strengths > median_opp].index
            weak_opps = team_strengths[team_strengths <= median_opp].index
            vs_strong = p[p['opp_abbrev'].isin(strong_opps)]['SC']
            vs_weak = p[p['opp_abbrev'].isin(weak_opps)]['SC']
            f['avg_vs_strong_opp'] = vs_strong.mean() if len(vs_strong) > 0 else f['prev_avg']
            f['avg_vs_weak_opp'] = vs_weak.mean() if len(vs_weak) > 0 else f['prev_avg']
            f['opp_sensitivity'] = f['avg_vs_weak_opp'] - f['avg_vs_strong_opp']
        else:
            f['avg_vs_strong_opp'] = f['avg_vs_weak_opp'] = f['prev_avg']
            f['opp_sensitivity'] = 0

        # 9. TAGS
        all_tags = (p['Tag'].str.lower() + ',' + p['Tag 2'].str.lower()).str.cat(sep=',')
        for tag_name in ['star', 'hot', 'gun', 'x-factor', 'heart', 'injured', 'sore',
                         'tagger', 'tagged', 'sub', 'subbed', 'wing', 'guard', 'ruck',
                         'spearhead', 'pocket', 'job', 'shovel', 'switch', 'rookie', 'cash']:
            f[f'tag_{tag_name}'] = all_tags.count(tag_name)

        f['positive_tags'] = f['tag_star'] + f['tag_hot'] + f['tag_gun'] + f.get('tag_x-factor', 0) + f['tag_heart']
        f['injury_tags'] = f['tag_injured'] + f['tag_sore']
        f['is_tagger_or_tagged'] = f['tag_tagger'] + f['tag_tagged']
        f['is_sub_risk'] = f['tag_sub'] + f['tag_subbed']
        f['is_wing_guard'] = f['tag_wing'] + f['tag_guard']
        f['is_inside_mid'] = f['tag_shovel'] + f['tag_job']
        f['injury_rate'] = f['injury_tags'] / n_games if n_games > 0 else 0

        p_2yr_tags = two_yr[two_yr['Player ID'] == pid]
        if len(p_2yr_tags) > 0:
            t2 = (p_2yr_tags['Tag'].str.lower() + ',' + p_2yr_tags['Tag 2'].str.lower()).str.cat(sep=',')
            f['injury_tags_2yr'] = t2.count('injured') + t2.count('sore')
            f['positive_tags_2yr'] = sum(t2.count(t) for t in ['star', 'hot', 'gun'])
        else:
            f['injury_tags_2yr'] = f['positive_tags_2yr'] = 0

        # 10. CAREER & AGE
        career = all_prev[all_prev['Player ID'] == pid]
        f['career_games'] = len(career)
        f['first_year'] = career['Year'].min()
        f['age_proxy'] = target_year - f['first_year']
        f['career_avg'] = career['SC'].mean()

        yearly_avgs = career.groupby('Year')['SC'].mean()
        if len(yearly_avgs) >= 3:
            f['career_trend'] = np.polyfit(np.arange(len(yearly_avgs)), yearly_avgs.values, 1)[0]
        else:
            f['career_trend'] = 0
        f['pct_of_career_best'] = f['prev_avg'] / yearly_avgs.max() if len(yearly_avgs) > 0 and yearly_avgs.max() > 0 else 1

        for y_off in [1, 2, 3]:
            yr_d = master_df[(master_df['Player ID'] == pid) & (master_df['Year'] == target_year - y_off) & (master_df['played'] > 0)]
            f[f'games_minus_{y_off}'] = len(yr_d)

        # 11. INTERACTIONS
        f['avg_x_durability'] = f['prev_avg'] * f['games_pct']
        f['avg_x_consistency'] = f['prev_avg'] * (1 - f['prev_cv'])
        f['avg_x_age_sweet_spot'] = f['prev_avg'] * (1 if 4 <= f['age_proxy'] <= 8 else 0.9)
        f['ceiling_floor_range'] = f['prev_ceiling'] - f['prev_floor']
        f['avg_vs_median_gap'] = f['prev_avg'] - f['prev_median']
        f['floor_ratio'] = f['prev_floor'] / f['prev_avg'] if f['prev_avg'] > 0 else 0
        f['ceiling_ratio'] = f['prev_ceiling'] / f['prev_avg'] if f['prev_avg'] > 0 else 0
        f['regression_est_85'] = f['prev_avg'] * 0.85  # Will be adjusted per-year

        features[pid] = f

    result = pd.DataFrame(features.values())

    # Population-level adjustments
    pop_mean = result['prev_avg'].mean()
    result['regression_est_85'] = result['prev_avg'] * 0.85 + pop_mean * 0.15

    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        mask = result['primary_pos'] == pos
        result.loc[mask, 'prev_position_rank'] = result.loc[mask, 'prev_avg'].rank(ascending=False)
    result['prev_position_rank'] = result['prev_position_rank'].fillna(50)

    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        result[f'pos_{pos}'] = (result['primary_pos'] == pos).astype(int)

    # Merge pre-draft projections (available 2022-2026)
    projections = load_predraft_projections(target_year)
    if not projections.empty:
        result = result.merge(projections, left_on='Player ID', right_on='feed_id', how='left')
        result.drop(columns=['feed_id'], errors='ignore', inplace=True)

    # Fill missing projection features with 0
    for col in ['sc_ownership_pct', 'sc_start_pct', 'sc_projected_avg', 'sc_projected_price',
                'sc_breakeven_avg', 'sc_is_hot', 'sc_is_cold']:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0)

    # Merge pre-season news features (available for years with scraped data, e.g. 2024+)
    _news_path = Path('data/preseason_news_features.csv')
    if _news_path.exists():
        try:
            _news = pd.read_csv(_news_path)
            _yr_news = _news[_news['year'] == target_year][['feed_id'] + [
                'news_mention_count', 'news_injury_flag', 'news_positive_buzz',
                'news_role_change_flag', 'news_concern_flag',
            ]].copy()
            if not _yr_news.empty:
                _yr_news['feed_id'] = pd.to_numeric(_yr_news['feed_id'], errors='coerce')
                result = result.merge(_yr_news, left_on='Player ID', right_on='feed_id', how='left')
                result.drop(columns=['feed_id'], errors='ignore', inplace=True)
        except Exception:
            pass

    for col in ['news_mention_count', 'news_injury_flag', 'news_positive_buzz',
                'news_role_change_flag', 'news_concern_flag']:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0)

    return result
