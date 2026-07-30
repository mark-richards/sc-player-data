"""
Backtest Draft Board v3 - Comprehensive Feature Engineering
=============================================================
Systematically builds 100+ features across 10 categories to find what
genuinely predicts next-season SuperCoach average from pre-draft data.

Uses leave-one-year-out validation (2022-2025) to avoid overfitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, skew
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# LOAD DATA
# ====================================================================
print("Loading master data...")
master = pd.read_csv('data/processed/master_player_data.csv', low_memory=False)
master['Round_Num'] = pd.to_numeric(master['Round_Num'], errors='coerce')
master = master[master['played'] > 0].copy()
master['Tag'] = master['Tag'].fillna('')
master['Tag 2'] = master['Tag 2'].fillna('')

# Derived base columns
master['Disposals'] = master['Kicks'] + master['Handballs']
master['Kick_Ratio'] = master['Kicks'] / master['Disposals'].replace(0, np.nan)
master['Uncontested_Poss'] = master['Disposals'] - master['Contested Possessions'].fillna(0)

print(f"Loaded {len(master)} rows, {master['Year'].nunique()} years\n")


# ====================================================================
# FEATURE ENGINEERING FUNCTION
# ====================================================================

def build_comprehensive_features(master_df, target_year):
    """
    Build 100+ pre-draft features for target_year using only prior data.

    Feature categories:
    1. Basic scoring stats (prev season)
    2. Scoring composition (how they score, not just how much)
    3. Possession profile (contested vs uncontested, kicks vs handballs)
    4. Form & trajectory (second half, last 5, year-over-year)
    5. Durability & workload (games, minutes, bench, sub tags)
    6. Consistency & reliability (score distribution, streaks, floor)
    7. Team context (teammate competition, team strength)
    8. Opposition analysis (vs good teams, vs bad teams)
    9. Tags & role indicators (star, hot, injured, role tags)
    10. Career & age profile (experience curve, peak detection)
    """
    prev_year = target_year - 1
    prev = master_df[master_df['Year'] == prev_year].copy()
    if prev.empty:
        return pd.DataFrame()

    # Also need 2 years back for multi-year features
    prev2_year = target_year - 2
    prev3_year = target_year - 3

    all_prev = master_df[master_df['Year'] <= prev_year].copy()
    two_yr = master_df[master_df['Year'].isin([prev_year, prev2_year])].copy()
    three_yr = master_df[master_df['Year'].isin([prev_year, prev2_year, prev3_year])].copy()

    features = {}
    players = prev['Player ID'].unique()

    for pid in players:
        p = prev[prev['Player ID'] == pid].sort_values('Round_Num')
        if len(p) == 0:
            continue

        f = {'Player ID': pid, 'Year': target_year}

        # Names and identifiers
        f['first_name'] = p['First Name'].iloc[-1]
        f['last_name'] = p['Last Name'].iloc[-1]
        f['team'] = p['Team'].iloc[-1]
        f['position'] = p['sc_position'].iloc[-1]

        sc = p['SC'].values
        n_games = len(sc)

        # ==============================================================
        # 1. BASIC SCORING STATS
        # ==============================================================
        f['prev_avg'] = sc.mean()
        f['prev_games'] = n_games
        f['prev_total'] = sc.sum()
        f['prev_std'] = sc.std() if n_games > 1 else 0
        f['prev_cv'] = f['prev_std'] / f['prev_avg'] if f['prev_avg'] > 0 else 1
        f['prev_ceiling'] = sc.max()
        f['prev_floor'] = sc.min()
        f['prev_median'] = np.median(sc)

        # ==============================================================
        # 2. SCORING COMPOSITION (how they accumulate SC points)
        # ==============================================================
        f['avg_kicks'] = p['Kicks'].mean()
        f['avg_handballs'] = p['Handballs'].mean()
        f['avg_disposals'] = p['Disposals'].mean()
        f['avg_marks'] = p['Marks'].mean()
        f['avg_tackles'] = p['Tackles'].mean()
        f['avg_hitouts'] = p['Hitouts'].mean()
        f['avg_goals'] = p['Goals'].mean()
        f['avg_frees_for'] = p['Frees for'].mean()
        f['avg_frees_against'] = p['Frees against'].mean()

        # Kick-to-handball ratio (kicks score more in SC)
        total_disp = p['Disposals'].sum()
        f['kick_ratio'] = p['Kicks'].sum() / total_disp if total_disp > 0 else 0.5

        # Contested possession profile
        if p['Contested Possessions'].notna().sum() > 0:
            f['avg_contested_poss'] = p['Contested Possessions'].mean()
            f['avg_clearances'] = p['Clearances'].mean()
            f['contested_poss_ratio'] = p['Contested Possessions'].sum() / total_disp if total_disp > 0 else 0
        else:
            f['avg_contested_poss'] = np.nan
            f['avg_clearances'] = np.nan
            f['contested_poss_ratio'] = np.nan

        # Uncontested possessions
        if p['Uncontested_Poss'].notna().sum() > 0:
            f['avg_uncontested_poss'] = p['Uncontested_Poss'].mean()
        else:
            f['avg_uncontested_poss'] = np.nan

        # Disposal efficiency
        if p['Disposal efficiency'].notna().sum() > 0:
            f['avg_disposal_eff'] = p['Disposal efficiency'].mean()
        else:
            f['avg_disposal_eff'] = np.nan

        # Clangers (turnovers)
        if p['Clangers'].notna().sum() > 0:
            f['avg_clangers'] = p['Clangers'].mean()
            f['clanger_ratio'] = p['Clangers'].sum() / total_disp if total_disp > 0 else 0
        else:
            f['avg_clangers'] = np.nan
            f['clanger_ratio'] = np.nan

        # Metres gained
        if p['Metres gained'].notna().sum() > 0:
            f['avg_metres_gained'] = p['Metres gained'].mean()
        else:
            f['avg_metres_gained'] = np.nan

        # SC score per disposal (efficiency)
        f['sc_per_disposal'] = f['prev_avg'] / f['avg_disposals'] if f['avg_disposals'] > 0 else 0

        # SC score per minute (workrate)
        if p['Time on ground'].notna().sum() > 0:
            avg_tog = p['Time on ground'].mean()
            f['avg_time_on_ground'] = avg_tog
            f['sc_per_minute_tog'] = f['prev_avg'] / (avg_tog * 1.2) if avg_tog > 0 else 0  # ~120 min games, TOG is %
        else:
            f['avg_time_on_ground'] = np.nan
            f['sc_per_minute_tog'] = np.nan

        # ==============================================================
        # 3. FORM & TRAJECTORY
        # ==============================================================
        # Second half of season (post round 13)
        second_half = p[p['Round_Num'] > 13]
        first_half = p[p['Round_Num'] <= 13]

        if len(second_half) > 0:
            f['post_r13_avg'] = second_half['SC'].mean()
            f['post_r13_games'] = len(second_half)
            f['second_half_form'] = f['post_r13_avg'] - f['prev_avg']
        else:
            f['post_r13_avg'] = f['prev_avg']
            f['post_r13_games'] = 0
            f['second_half_form'] = 0

        if len(first_half) > 0:
            f['pre_r13_avg'] = first_half['SC'].mean()
        else:
            f['pre_r13_avg'] = f['prev_avg']

        # Last N games form
        for n in [3, 5, 8]:
            if n_games >= n:
                f[f'last{n}_avg'] = sc[-n:].mean()
                f[f'last{n}_vs_season'] = f[f'last{n}_avg'] - f['prev_avg']
            else:
                f[f'last{n}_avg'] = f['prev_avg']
                f[f'last{n}_vs_season'] = 0

        # First N games of season (did they start slow?)
        for n in [3, 5]:
            if n_games >= n:
                f[f'first{n}_avg'] = sc[:n].mean()
            else:
                f[f'first{n}_avg'] = f['prev_avg']

        # Linear trend through the season (slope of SC scores)
        if n_games >= 5:
            x = np.arange(n_games)
            coeffs = np.polyfit(x, sc, 1)
            f['season_trend_slope'] = coeffs[0]  # Points per game improvement
        else:
            f['season_trend_slope'] = 0

        # ==============================================================
        # 4. MULTI-YEAR TRAJECTORY
        # ==============================================================
        p_all = all_prev[all_prev['Player ID'] == pid]
        p_2yr = two_yr[two_yr['Player ID'] == pid]
        p_3yr = three_yr[three_yr['Player ID'] == pid]

        f['avg_2yr'] = p_2yr['SC'].mean() if len(p_2yr) > 0 else f['prev_avg']
        f['games_2yr'] = len(p_2yr)
        f['std_2yr'] = p_2yr['SC'].std() if len(p_2yr) > 1 else 0

        f['avg_3yr'] = p_3yr['SC'].mean() if len(p_3yr) > 0 else f['prev_avg']
        f['games_3yr'] = len(p_3yr)

        # Year-over-year trend
        p_prev2 = master_df[(master_df['Player ID'] == pid) & (master_df['Year'] == prev2_year) & (master_df['played'] > 0)]
        if len(p_prev2) > 0:
            f['prev2_avg'] = p_prev2['SC'].mean()
            f['yoy_change'] = f['prev_avg'] - f['prev2_avg']  # Improving or declining?
            f['prev2_games'] = len(p_prev2)
        else:
            f['prev2_avg'] = np.nan
            f['yoy_change'] = 0
            f['prev2_games'] = 0

        # 2-year trend
        f['two_yr_trend'] = f['prev_avg'] - f['avg_2yr']  # Positive = recent improvement

        # ==============================================================
        # 5. DURABILITY & WORKLOAD
        # ==============================================================
        f['games_pct'] = n_games / 23  # % of full season played
        f['avg_minutes'] = p['minutes_played'].mean()

        # Time on ground
        if p['Time on ground'].notna().sum() > 0:
            f['avg_tog_pct'] = p['Time on ground'].mean()
            f['min_tog'] = p['Time on ground'].min()  # Were they ever restricted?
        else:
            f['avg_tog_pct'] = np.nan
            f['min_tog'] = np.nan

        # Bench status frequency
        if p['Bench status'].notna().sum() > 0:
            f['bench_pct'] = p['Bench status'].mean()  # % of games starting on bench
        else:
            f['bench_pct'] = 0

        # Price trajectory (proxy for form/value changes during season)
        if p['price_change'].notna().sum() > 0:
            f['total_price_change'] = p['price_change'].sum()
            f['avg_price_change'] = p['price_change'].mean()
            f['price_end'] = p['price'].iloc[-1]
            f['price_start'] = p['price'].iloc[0]
            f['price_growth_pct'] = (f['price_end'] - f['price_start']) / f['price_start'] if f['price_start'] > 0 else 0
        else:
            f['total_price_change'] = 0
            f['avg_price_change'] = 0
            f['price_end'] = 0
            f['price_start'] = 0
            f['price_growth_pct'] = 0

        # ==============================================================
        # 6. CONSISTENCY & RELIABILITY
        # ==============================================================
        if n_games >= 3:
            f['score_skewness'] = skew(sc) if n_games >= 5 else 0
            f['pct_above_100'] = (sc >= 100).mean()
            f['pct_above_80'] = (sc >= 80).mean()
            f['pct_below_60'] = (sc < 60).mean()
            f['pct_below_40'] = (sc < 40).mean()

            # Interquartile range (middle 50% spread)
            f['iqr'] = np.percentile(sc, 75) - np.percentile(sc, 25)
            f['q25'] = np.percentile(sc, 25)
            f['q75'] = np.percentile(sc, 75)

            # "Bankable floor" - what can you rely on? (10th percentile)
            f['floor_10pct'] = np.percentile(sc, 10)

            # Max consecutive games above/below threshold
            above_80 = sc >= 80
            below_60 = sc < 60

            def max_streak(arr):
                max_s = 0
                current = 0
                for v in arr:
                    if v:
                        current += 1
                        max_s = max(max_s, current)
                    else:
                        current = 0
                return max_s

            f['max_streak_80plus'] = max_streak(above_80)
            f['max_streak_below60'] = max_streak(below_60)
        else:
            f['score_skewness'] = 0
            f['pct_above_100'] = 0
            f['pct_above_80'] = 0
            f['pct_below_60'] = 0
            f['pct_below_40'] = 0
            f['iqr'] = 0
            f['q25'] = f['prev_avg']
            f['q75'] = f['prev_avg']
            f['floor_10pct'] = f['prev_avg']
            f['max_streak_80plus'] = 0
            f['max_streak_below60'] = 0

        # ==============================================================
        # 7. TEAM CONTEXT
        # ==============================================================
        team = p['Team'].iloc[-1]
        pos = p['sc_position'].iloc[-1]

        # Simplify position
        def _sp(pp):
            if pd.isna(pp): return 'MID'
            pp = str(pp).upper()
            for x in ['DEF', 'MID', 'FWD', 'RUC']:
                if x in pp: return x
            return 'MID'

        primary_pos = _sp(pos)
        f['primary_pos'] = primary_pos

        # Team strength: average SC of all teammates
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

        # Position competition: average SC of same-position teammates
        same_pos_teammates = teammates[teammates['sc_position'].apply(_sp) == primary_pos]
        if len(same_pos_teammates) > 0:
            pos_avgs = same_pos_teammates.groupby('Player ID')['SC'].mean()
            f['pos_teammate_avg'] = pos_avgs.mean()
            f['pos_teammate_count'] = len(pos_avgs)
            # How many same-position teammates score more than this player?
            f['pos_teammates_better'] = (pos_avgs > f['prev_avg']).sum()
        else:
            f['pos_teammate_avg'] = 75
            f['pos_teammate_count'] = 0
            f['pos_teammates_better'] = 0

        # Player's share of team scoring
        team_total_sc = prev[prev['Team'] == team]['SC'].sum()
        f['team_sc_share'] = p['SC'].sum() / team_total_sc if team_total_sc > 0 else 0

        # ==============================================================
        # 8. OPPOSITION ANALYSIS
        # ==============================================================
        if 'opp_abbrev' in p.columns and p['opp_abbrev'].notna().sum() > 0:
            # Get overall team strengths from the full season
            team_strengths = prev.groupby('opp_abbrev')['SC'].mean()
            median_opp = team_strengths.median()

            # Split into strong and weak opponents
            strong_opps = team_strengths[team_strengths > median_opp].index
            weak_opps = team_strengths[team_strengths <= median_opp].index

            vs_strong = p[p['opp_abbrev'].isin(strong_opps)]['SC']
            vs_weak = p[p['opp_abbrev'].isin(weak_opps)]['SC']

            f['avg_vs_strong_opp'] = vs_strong.mean() if len(vs_strong) > 0 else f['prev_avg']
            f['avg_vs_weak_opp'] = vs_weak.mean() if len(vs_weak) > 0 else f['prev_avg']
            f['opp_sensitivity'] = f['avg_vs_weak_opp'] - f['avg_vs_strong_opp']  # Higher = matchup dependent
        else:
            f['avg_vs_strong_opp'] = f['prev_avg']
            f['avg_vs_weak_opp'] = f['prev_avg']
            f['opp_sensitivity'] = 0

        # ==============================================================
        # 9. TAGS & ROLE INDICATORS
        # ==============================================================
        all_tags = (p['Tag'].str.lower() + ',' + p['Tag 2'].str.lower()).str.cat(sep=',')

        # Positive tags
        for tag_name in ['star', 'hot', 'gun', 'x-factor', 'heart']:
            f[f'tag_{tag_name}'] = all_tags.count(tag_name)
        f['positive_tags'] = f['tag_star'] + f['tag_hot'] + f['tag_gun'] + f['tag_x-factor'] + f['tag_heart']

        # Negative/injury tags
        for tag_name in ['injured', 'sore']:
            f[f'tag_{tag_name}'] = all_tags.count(tag_name)
        f['injury_tags'] = f['tag_injured'] + f['tag_sore']

        # Role tags
        for tag_name in ['tagger', 'tagged', 'sub', 'subbed', 'wing', 'guard', 'ruck',
                         'spearhead', 'pocket', 'job', 'shovel', 'switch', 'rookie', 'cash']:
            f[f'tag_{tag_name}'] = all_tags.count(tag_name)

        # Key role indicators
        f['is_tagger_or_tagged'] = f['tag_tagger'] + f['tag_tagged']
        f['is_sub_risk'] = f['tag_sub'] + f['tag_subbed']
        f['is_wing_guard'] = f['tag_wing'] + f['tag_guard']
        f['is_inside_mid'] = f['tag_shovel'] + f['tag_job']  # Inside midfield tags
        f['is_key_position'] = f['tag_spearhead'] + f['tag_guard']  # Key position

        # Injury risk (tags per game)
        f['injury_rate'] = f['injury_tags'] / n_games if n_games > 0 else 0

        # 2-year injury picture
        p_2yr_player = two_yr[two_yr['Player ID'] == pid]
        if len(p_2yr_player) > 0:
            tags_2yr = (p_2yr_player['Tag'].str.lower() + ',' + p_2yr_player['Tag 2'].str.lower()).str.cat(sep=',')
            f['injury_tags_2yr'] = tags_2yr.count('injured') + tags_2yr.count('sore')
            f['positive_tags_2yr'] = sum(tags_2yr.count(t) for t in ['star', 'hot', 'gun'])
        else:
            f['injury_tags_2yr'] = 0
            f['positive_tags_2yr'] = 0

        # ==============================================================
        # 10. CAREER & AGE PROFILE
        # ==============================================================
        career = all_prev[all_prev['Player ID'] == pid]
        f['career_games'] = len(career)
        f['first_year'] = career['Year'].min()
        f['age_proxy'] = target_year - f['first_year']

        # Career average
        f['career_avg'] = career['SC'].mean()

        # Is player improving year over year? (career slope)
        yearly_avgs = career.groupby('Year')['SC'].mean()
        if len(yearly_avgs) >= 3:
            x = np.arange(len(yearly_avgs))
            coeffs = np.polyfit(x, yearly_avgs.values, 1)
            f['career_trend'] = coeffs[0]  # Points per year improvement
        else:
            f['career_trend'] = 0

        # Peak detection: is current avg above or below career best?
        yearly_maxavg = yearly_avgs.max() if len(yearly_avgs) > 0 else f['prev_avg']
        f['pct_of_career_best'] = f['prev_avg'] / yearly_maxavg if yearly_maxavg > 0 else 1

        # Games in each of last 3 seasons (durability pattern)
        for y_offset in [1, 2, 3]:
            yr_data = master_df[(master_df['Player ID'] == pid) & (master_df['Year'] == target_year - y_offset) & (master_df['played'] > 0)]
            f[f'games_minus_{y_offset}'] = len(yr_data)

        # ==============================================================
        # 11. INTERACTION FEATURES
        # ==============================================================
        f['avg_x_durability'] = f['prev_avg'] * f['games_pct']
        f['avg_x_consistency'] = f['prev_avg'] * (1 - f['prev_cv'])
        f['avg_x_age_sweet_spot'] = f['prev_avg'] * (1 if 4 <= f['age_proxy'] <= 8 else 0.9)
        f['ceiling_floor_range'] = f['prev_ceiling'] - f['prev_floor']
        f['avg_vs_median_gap'] = f['prev_avg'] - f['prev_median']  # Skewed by outlier games?
        f['floor_ratio'] = f['prev_floor'] / f['prev_avg'] if f['prev_avg'] > 0 else 0
        f['ceiling_ratio'] = f['prev_ceiling'] / f['prev_avg'] if f['prev_avg'] > 0 else 0

        # Regression estimate (shrink toward population mean)
        f['regression_est_85'] = f['prev_avg'] * 0.85  # Will be adjusted per-year

        features[pid] = f

    result = pd.DataFrame(features.values())

    # Population-level features (need all players)
    pop_mean = result['prev_avg'].mean()
    result['regression_est_85'] = result['prev_avg'] * 0.85 + pop_mean * 0.15

    # Position rank within year
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        mask = result['primary_pos'] == pos
        result.loc[mask, 'prev_position_rank'] = result.loc[mask, 'prev_avg'].rank(ascending=False)
    result['prev_position_rank'] = result['prev_position_rank'].fillna(50)

    # Position dummies
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        result[f'pos_{pos}'] = (result['primary_pos'] == pos).astype(int)

    return result


# ====================================================================
# BUILD FEATURES FOR ALL YEARS
# ====================================================================
print("Building comprehensive features for 2022-2025...")
all_features = []
for year in [2022, 2023, 2024, 2025]:
    feat = build_comprehensive_features(master, year)
    all_features.append(feat)
    print(f"  {year}: {len(feat)} players, {len(feat.columns)} features")

df = pd.concat(all_features, ignore_index=True)

# Get season outcomes
season_frames = []
for year in [2022, 2023, 2024, 2025]:
    season = master[master['Year'] == year]
    ss = season.groupby('Player ID').agg(
        season_avg=('SC', 'mean'),
        season_games=('SC', 'count'),
    ).reset_index()
    ss['Year'] = year
    season_frames.append(ss)
outcomes = pd.concat(season_frames, ignore_index=True)
df = df.merge(outcomes, on=['Player ID', 'Year'], how='left')

# Filter
df = df.dropna(subset=['season_avg']).copy()
df = df[df['season_games'] >= 5].copy()
df['avg_change'] = df['season_avg'] - df['prev_avg']
df['big_decline'] = (df['avg_change'] < -15).astype(int)

print(f"\nBacktesting pool: {len(df)} player-seasons")
print(f"By year: {df.groupby('Year').size().to_dict()}")
print(f"Total features available: {len([c for c in df.columns if c not in ['Player ID', 'Year', 'first_name', 'last_name', 'team', 'position', 'primary_pos', 'season_avg', 'season_games', 'avg_change', 'big_decline']])}")


# ====================================================================
# DEFINE FEATURE SETS TO TEST
# ====================================================================

# Baseline: just prev_avg
baseline_features = ['prev_avg']

# V1: Basic (what we had before)
basic_features = [
    'prev_avg', 'prev_games', 'prev_cv', 'prev_ceiling', 'prev_floor',
    'prev_std', 'prev_median', 'career_games', 'age_proxy',
]

# V2: + Form & Trajectory
form_features = basic_features + [
    'post_r13_avg', 'post_r13_games', 'second_half_form',
    'last5_avg', 'last5_vs_season', 'last3_avg', 'last8_avg',
    'season_trend_slope', 'avg_2yr', 'games_2yr', 'two_yr_trend',
    'yoy_change', 'prev2_avg',
]

# V3: + Scoring Composition
composition_features = form_features + [
    'avg_kicks', 'avg_handballs', 'avg_disposals', 'avg_marks', 'avg_tackles',
    'avg_hitouts', 'avg_goals', 'avg_frees_for', 'avg_frees_against',
    'kick_ratio', 'avg_contested_poss', 'avg_clearances', 'avg_uncontested_poss',
    'contested_poss_ratio', 'avg_disposal_eff', 'avg_clangers', 'clanger_ratio',
    'avg_metres_gained', 'sc_per_disposal', 'avg_time_on_ground', 'sc_per_minute_tog',
]

# V4: + Consistency & Reliability
consistency_features = composition_features + [
    'score_skewness', 'pct_above_100', 'pct_above_80', 'pct_below_60', 'pct_below_40',
    'iqr', 'q25', 'q75', 'floor_10pct',
    'max_streak_80plus', 'max_streak_below60',
]

# V5: + Team Context & Opposition
context_features = consistency_features + [
    'team_avg_sc', 'team_top5_avg', 'team_total_players',
    'pos_teammate_avg', 'pos_teammate_count', 'pos_teammates_better', 'team_sc_share',
    'avg_vs_strong_opp', 'avg_vs_weak_opp', 'opp_sensitivity',
]

# V6: + Tags & Role
tags_features = context_features + [
    'positive_tags', 'injury_tags', 'injury_tags_2yr', 'positive_tags_2yr',
    'tag_star', 'tag_hot', 'tag_gun', 'tag_injured', 'tag_sore',
    'tag_tagger', 'tag_tagged', 'tag_sub', 'tag_subbed',
    'tag_wing', 'tag_guard', 'tag_ruck', 'tag_spearhead',
    'tag_shovel', 'tag_job', 'tag_rookie', 'tag_cash',
    'is_tagger_or_tagged', 'is_sub_risk', 'is_wing_guard', 'is_inside_mid',
    'injury_rate',
]

# V7: FULL - Everything
full_features = tags_features + [
    'career_avg', 'career_trend', 'pct_of_career_best',
    'games_minus_1', 'games_minus_2', 'games_minus_3',
    'avg_x_durability', 'avg_x_consistency', 'avg_x_age_sweet_spot',
    'ceiling_floor_range', 'avg_vs_median_gap', 'floor_ratio', 'ceiling_ratio',
    'regression_est_85',
    'games_pct', 'avg_minutes', 'avg_tog_pct', 'min_tog', 'bench_pct',
    'total_price_change', 'avg_price_change', 'price_growth_pct',
    'prev_position_rank',
    'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_RUC',
    'avg_3yr', 'games_3yr', 'std_2yr', 'prev2_games',
    'first3_avg', 'first5_avg',
]

# Remove duplicates and non-existent columns
def clean_features(feat_list):
    seen = set()
    clean = []
    for f in feat_list:
        if f not in seen and f in df.columns:
            seen.add(f)
            clean.append(f)
    return clean

feature_sets = {
    'A. Baseline (prev_avg)': clean_features(baseline_features),
    'B. Basic (9 feat)': clean_features(basic_features),
    'C. + Form & Trajectory': clean_features(form_features),
    'D. + Scoring Composition': clean_features(composition_features),
    'E. + Consistency': clean_features(consistency_features),
    'F. + Team & Opposition': clean_features(context_features),
    'G. + Tags & Role': clean_features(tags_features),
    'H. FULL (all features)': clean_features(full_features),
}


# ====================================================================
# LEAVE-ONE-YEAR-OUT BACKTESTING
# ====================================================================
print("\n" + "=" * 100)
print("LEAVE-ONE-YEAR-OUT BACKTESTING")
print("=" * 100)

logo = LeaveOneGroupOut()
groups = df['Year']
results = {}

for name, feat_cols in feature_sets.items():
    X = df[feat_cols].fillna(0)
    y = df['season_avg']

    if len(feat_cols) == 1 and feat_cols[0] == 'prev_avg':
        # Baseline: just use raw prev_avg
        df[f'pred_{name[:1]}'] = df['prev_avg']
    else:
        preds = np.zeros(len(df))
        for train_idx, test_idx in logo.split(X, y, groups):
            model = XGBRegressor(
                n_estimators=800, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=15,
                reg_alpha=1.0, reg_lambda=3.0, random_state=42,
                n_jobs=-1, tree_method='hist',
                gamma=0.1,
            )
            model.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
                verbose=False,
            )
            preds[test_idx] = model.predict(X.iloc[test_idx])
        df[f'pred_{name[:1]}'] = preds

    pred_col = f'pred_{name[:1]}'
    overall_s, _ = spearmanr(df[pred_col], df['season_avg'])
    overall_mae = mean_absolute_error(df['season_avg'], df[pred_col])

    year_results = {}
    for year in [2022, 2023, 2024, 2025]:
        yr = df[df['Year'] == year]
        s, _ = spearmanr(yr[pred_col], yr['season_avg'])
        mae = mean_absolute_error(yr['season_avg'], yr[pred_col])
        o50 = set(yr.nlargest(50, pred_col)['Player ID'])
        a50 = set(yr.nlargest(50, 'season_avg')['Player ID'])
        overlap = len(o50 & a50)
        busts = yr.nlargest(30, pred_col)['big_decline'].sum()
        year_results[year] = {'spearman': s, 'mae': mae, 'top50': overlap, 'busts': busts}

    results[name] = {
        'n_features': len(feat_cols),
        'overall_spearman': overall_s,
        'overall_mae': overall_mae,
        'years': year_results,
    }

    avg_top50 = np.mean([v['top50'] for v in year_results.values()])
    avg_busts = np.mean([v['busts'] for v in year_results.values()])
    print(f"\n  {name} ({len(feat_cols)} features)")
    print(f"    Overall: Spearman r={overall_s:.3f}, MAE={overall_mae:.1f}")
    for year, yr_res in year_results.items():
        print(f"    {year}: r={yr_res['spearman']:.3f}, MAE={yr_res['mae']:.1f}, "
              f"Top-50={yr_res['top50']}/50, Busts={yr_res['busts']}/30")
    print(f"    Avg Top-50: {avg_top50:.1f}/50, Avg Busts: {avg_busts:.1f}/30")


# ====================================================================
# SUMMARY TABLE
# ====================================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"\n  {'Approach':<40} {'#Feat':>5} {'Spearman':>8} {'MAE':>6} {'Top-50':>8} {'Busts':>7}")
print(f"  {'-'*75}")

best_approach = max(results.items(), key=lambda x: x[1]['overall_spearman'])
for name, res in results.items():
    avg_top50 = np.mean([v['top50'] for v in res['years'].values()])
    avg_busts = np.mean([v['busts'] for v in res['years'].values()])
    marker = " ***" if name == best_approach[0] else ""
    print(f"  {name:<40} {res['n_features']:>5} {res['overall_spearman']:>8.3f} {res['overall_mae']:>6.1f} "
          f"{avg_top50:>6.1f}/50 {avg_busts:>5.1f}/30{marker}")


# ====================================================================
# FEATURE IMPORTANCE (train on all data with best feature set)
# ====================================================================
best_name = best_approach[0]
best_feats = feature_sets[best_name]

print(f"\n{'='*100}")
print(f"FEATURE IMPORTANCE - {best_name} ({len(best_feats)} features)")
print(f"{'='*100}")

X_all = df[best_feats].fillna(0)
y_all = df['season_avg']
final_model = XGBRegressor(
    n_estimators=800, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=15,
    reg_alpha=1.0, reg_lambda=3.0, random_state=42,
    n_jobs=-1, tree_method='hist', gamma=0.1,
)
final_model.fit(X_all, y_all, verbose=False)

importances = pd.Series(final_model.feature_importances_, index=best_feats).sort_values(ascending=False)
print(f"\n  {'Feature':<35} {'Importance':>10} {'Category':<25}")
print(f"  {'-'*75}")

# Categorize features
category_map = {
    'prev_avg': 'Basic', 'prev_games': 'Basic', 'prev_cv': 'Basic', 'prev_ceiling': 'Basic',
    'prev_floor': 'Basic', 'prev_std': 'Basic', 'prev_median': 'Basic', 'prev_total': 'Basic',
    'post_r13_avg': 'Form', 'second_half_form': 'Form', 'last5_avg': 'Form', 'last3_avg': 'Form',
    'last8_avg': 'Form', 'season_trend_slope': 'Form', 'yoy_change': 'Form',
    'avg_2yr': 'Multi-year', 'avg_3yr': 'Multi-year', 'two_yr_trend': 'Multi-year', 'prev2_avg': 'Multi-year',
    'avg_kicks': 'Composition', 'avg_handballs': 'Composition', 'avg_disposals': 'Composition',
    'avg_marks': 'Composition', 'avg_tackles': 'Composition', 'avg_hitouts': 'Composition',
    'avg_goals': 'Composition', 'kick_ratio': 'Composition', 'avg_contested_poss': 'Composition',
    'contested_poss_ratio': 'Composition', 'avg_clearances': 'Composition', 'sc_per_disposal': 'Composition',
    'avg_metres_gained': 'Composition', 'avg_disposal_eff': 'Composition', 'avg_clangers': 'Composition',
    'pct_above_100': 'Consistency', 'pct_above_80': 'Consistency', 'pct_below_60': 'Consistency',
    'iqr': 'Consistency', 'floor_10pct': 'Consistency', 'score_skewness': 'Consistency',
    'team_avg_sc': 'Team', 'team_top5_avg': 'Team', 'pos_teammate_avg': 'Team',
    'pos_teammates_better': 'Team', 'team_sc_share': 'Team',
    'avg_vs_strong_opp': 'Opposition', 'avg_vs_weak_opp': 'Opposition', 'opp_sensitivity': 'Opposition',
    'positive_tags': 'Tags', 'injury_tags': 'Tags', 'tag_star': 'Tags', 'tag_hot': 'Tags',
    'tag_injured': 'Tags', 'tag_sore': 'Tags', 'injury_rate': 'Tags',
    'career_games': 'Career', 'age_proxy': 'Career', 'career_avg': 'Career', 'career_trend': 'Career',
    'games_pct': 'Durability', 'avg_minutes': 'Durability', 'avg_tog_pct': 'Durability',
}

for feat, imp in importances.items():
    if imp >= 0.005:
        cat = category_map.get(feat, 'Other')
        print(f"  {feat:<35} {imp:>10.3f} {cat:<25}")

# Category totals
print(f"\n  Category importance totals:")
cat_importance = {}
for feat, imp in importances.items():
    cat = category_map.get(feat, 'Other')
    cat_importance[cat] = cat_importance.get(cat, 0) + imp
for cat, imp in sorted(cat_importance.items(), key=lambda x: -x[1]):
    print(f"    {cat:<25}: {imp:.3f} ({imp*100:.1f}%)")


# ====================================================================
# DETAILED COMPARISON: Where does the model beat baseline?
# ====================================================================
print(f"\n{'='*100}")
print("HEAD-TO-HEAD: FULL MODEL vs BASELINE (prev_avg)")
print(f"{'='*100}")

best_pred = f'pred_{best_name[:1]}'
for year in [2022, 2023, 2024, 2025]:
    yr = df[df['Year'] == year].copy()
    yr['baseline_rank'] = yr['prev_avg'].rank(ascending=False)
    yr['model_rank'] = yr[best_pred].rank(ascending=False)
    yr['actual_rank'] = yr['season_avg'].rank(ascending=False)

    # Where model promoted players and was correct
    promoted = yr[yr['model_rank'] < yr['baseline_rank'] - 5]
    promoted_correct = promoted[promoted['actual_rank'] < promoted['baseline_rank']]

    # Where model demoted and was correct
    demoted = yr[yr['model_rank'] > yr['baseline_rank'] + 5]
    demoted_correct = demoted[demoted['actual_rank'] > demoted['baseline_rank']]

    # Top 30 bust comparison
    model_top30 = yr.nlargest(30, best_pred)
    baseline_top30 = yr.nlargest(30, 'prev_avg')
    model_busts = model_top30['big_decline'].sum()
    baseline_busts = baseline_top30['big_decline'].sum()

    # Top 50 overlap
    model_o50 = len(set(yr.nlargest(50, best_pred)['Player ID']) & set(yr.nlargest(50, 'season_avg')['Player ID']))
    base_o50 = len(set(yr.nlargest(50, 'prev_avg')['Player ID']) & set(yr.nlargest(50, 'season_avg')['Player ID']))

    print(f"\n  {year}:")
    print(f"    Top-50 overlap: Model={model_o50}/50 vs Baseline={base_o50}/50")
    print(f"    Busts in top-30: Model={model_busts} vs Baseline={baseline_busts}")
    print(f"    Promotions: {len(promoted)} players raised 5+ spots ({len(promoted_correct)} correct, {len(promoted_correct)/max(len(promoted),1)*100:.0f}%)")
    print(f"    Demotions: {len(demoted)} players dropped 5+ spots ({len(demoted_correct)} correct, {len(demoted_correct)/max(len(demoted),1)*100:.0f}%)")

    # Show biggest model wins
    yr['rank_improvement'] = yr['baseline_rank'] - yr['actual_rank']  # Positive = model was right to promote
    model_wins = yr[yr['model_rank'] < yr['baseline_rank'] - 10].nlargest(5, 'rank_improvement')
    if len(model_wins) > 0:
        print(f"    Best model promotions (actual improvement):")
        for _, row in model_wins.iterrows():
            name = f"{row['first_name']} {row['last_name']}"
            print(f"      {name:<25} Base rank: {int(row['baseline_rank']):>3} -> Model rank: {int(row['model_rank']):>3} "
                  f"(Actual: {int(row['actual_rank']):>3}) prev_avg={row['prev_avg']:.0f} -> season={row['season_avg']:.0f}")
