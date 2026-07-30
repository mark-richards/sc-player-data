"""
Backtest: Newsletter recommendations vs avg3 for free agents.
Primary data source: data/raw/supercoach/2026/players_round_N.json
Free agent pool: current_free_agents_waiver.csv (current snapshot, used as proxy)

Metric: Among free agents who played in round N+1, did they score 80+?
Players who didn't play round N+1 are listed separately (may still have value
if they play within the next 3 rounds).
"""
import json, re, pandas as pd
from pathlib import Path

# ── Determine free agent pool ──
# current_free_agents_waiver.csv is stale/wrong-season; derive FA pool instead:
# FA = all players in the JSON minus currently-owned players (from current_teams.csv).
# We use round-14 JSON as the master player list (most complete).
with open('data/raw/supercoach/2026/players_round_14.json') as _f:
    _all_players = json.load(_f)
all_player_ids = {int(p['id']) for p in _all_players}
feed_to_pid = {str(p['feed_id']): int(p['id']) for p in _all_players}

teams_df = pd.read_csv('data/live/current_teams.csv')
owned_pids = set()
for _, row in teams_df.iterrows():
    fid = str(int(float(row['feed_id']))) if not pd.isna(row['feed_id']) else None
    pid = feed_to_pid.get(fid)
    if pid:
        owned_pids.add(pid)

fa_ids = all_player_ids - owned_pids
print(f"Total players in JSON: {len(all_player_ids)}")
print(f"Owned players identified: {len(owned_pids)}")
print(f"Free agent pool: {len(fa_ids)} players")

# ── Build name -> SC player_id lookup from player list ──
pl = pd.read_csv('draft_prep/SC 2026/2026_SC_Player_list.csv', low_memory=False)
pl['full_name'] = (pl['first_name'].str.strip() + ' ' + pl['last_name'].str.strip()).str.strip()
name_to_id = {row['full_name'].lower(): int(row['id']) for _, row in pl.iterrows() if not pd.isna(row['id'])}
id_to_name = {int(row['id']): row['full_name'] for _, row in pl.iterrows() if not pd.isna(row['id'])}


def load_round_json(rnd):
    path = f'data/raw/supercoach/2026/players_round_{rnd}.json'
    with open(path) as f:
        data = json.load(f)
    rows = []
    for p in data:
        if not p['player_stats']:
            continue
        stat = p['player_stats'][0]
        rows.append({
            'player_id': int(p['id']),
            'name': f"{p['first_name']} {p['last_name']}",
            'active': bool(p['active']),
            'round': int(stat.get('round', rnd)),
            'points': float(stat.get('points', 0)),
            'games': int(stat.get('games', 0)),
            'avg3': float(stat.get('avg3', 0) or 0),
            'avg5': float(stat.get('avg5', 0) or 0),
            'avg': float(stat.get('avg', 0) or 0),
            'total_games': int(stat.get('total_games', 0) or 0),
        })
    return pd.DataFrame(rows)


def parse_newsletter_picks(path, max_per_pos=5):
    """Extract top-N picks per position from newsletter Positional Targets section."""
    txt = Path(path).read_text(encoding='utf-8', errors='replace')
    sections = re.split(r'\n## ', txt)
    pt_section = next((s for s in sections if 'Positional Targets' in s.split('\n')[0]), '')
    picks = []
    pos = None
    pos_count = 0
    for line in pt_section.split('\n'):
        m_pos = re.match(r'^### (DEF|MID|FWD|RUC)\s*$', line.strip())
        if m_pos:
            pos = m_pos.group(1)
            pos_count = 0
            continue
        if pos and pos_count < max_per_pos and line.startswith('|') and '**' in line:
            cells = [c.strip() for c in line.split('|')]
            if len(cells) < 4:
                continue
            nc = cells[1]
            if '**Player**' in nc or '---' in nc:
                continue
            mn = re.search(r'\*\*(.+?)\*\*', nc)
            if not mn:
                continue
            raw_name = re.sub(r'[^\x00-\x7F]+', '', mn.group(1)).strip()
            raw_name = re.sub(r'\s+', ' ', raw_name).strip()
            if len(raw_name) < 3:
                continue
            # Match to SC player_id
            pid = name_to_id.get(raw_name.lower())
            if pid is None:
                parts = raw_name.lower().split()
                if parts:
                    last = parts[-1]
                    cands = [(n, v) for n, v in name_to_id.items() if n.split()[-1] == last]
                    if len(cands) == 1:
                        pid = cands[0][1]
                    elif len(cands) > 1 and len(parts) > 1:
                        first = parts[0]
                        exact = [c for c in cands if c[0].startswith(first)]
                        if len(exact) == 1:
                            pid = exact[0][1]
            picks.append({'pos': pos, 'name': raw_name, 'player_id': pid})
            pos_count += 1
    return picks


def evaluate_picks(player_ids, next_rd_df, label="", verbose=False):
    """
    Given a list of player_ids and the next-round dataframe,
    return hit stats and split by played/DNP.
    """
    played_hits = played_misses = 0
    dnp_count = 0
    details = []
    for pid in player_ids:
        row = next_rd_df[next_rd_df['player_id'] == pid]
        if row.empty:
            dnp_count += 1
            details.append({'pid': pid, 'name': id_to_name.get(pid, str(pid)), 'status': 'no data', 'score': None})
            continue
        row = row.iloc[0]
        score = row['points']
        played = row['games'] == 1
        if not played:
            dnp_count += 1
            details.append({'pid': pid, 'name': row['name'], 'status': 'DNP', 'score': 0})
        else:
            hit = score >= 80
            if hit:
                played_hits += 1
            else:
                played_misses += 1
            details.append({'pid': pid, 'name': row['name'], 'status': 'played', 'score': score, 'hit': hit})

    played_total = played_hits + played_misses
    return {
        'hits': played_hits,
        'played': played_total,
        'dnp': dnp_count,
        'pct': played_hits / played_total * 100 if played_total else 0,
        'details': details,
    }


# ── Newsletter paths ──
nl_paths = {
    8:  'reports/waiver_2026_rd08_2026-05-06.md',
    9:  'reports/waiver_2026_rd09_2026-05-11.md',
    10: 'reports/waiver_2026_rd10_2026-05-18.md',
    11: 'reports/waiver_2026_rd11_2026-05-26.md',
    12: 'reports/waiver_2026_rd12_2026-06-02.md',
    13: 'reports/waiver_2026_rd13_2026-06-10.md',
}

print("\n" + "=" * 80)
print("BACKTEST: Newsletter Positional Targets vs Avg3 Top-20 for Free Agents")
print("Source: players_round_N.json (SC API). Score source: players_round_N+1.json")
print("=" * 80)

all_nl = {'hits': 0, 'played': 0, 'dnp': 0}
all_a3 = {'hits': 0, 'played': 0, 'dnp': 0}
base_hits = base_played = 0

for rd, nl_path in nl_paths.items():
    nrd = rd + 1
    if nrd > 14:
        continue

    # Load round N data (for avg3) and round N+1 data (for actual scores)
    cur = load_round_json(rd)
    nxt = load_round_json(nrd)

    # Free agent pool at round N
    fa_cur = cur[cur['player_id'].isin(fa_ids)].copy()

    # Base rate: among all FAs who played in round N+1, how many scored 80+?
    fa_nxt = nxt[nxt['player_id'].isin(fa_ids)]
    fa_nxt_played = fa_nxt[fa_nxt['games'] == 1]
    br_h = int((fa_nxt_played['points'] >= 80).sum())
    br_t = len(fa_nxt_played)
    base_hits += br_h
    base_played += br_t

    # Players who actually played in round N+1 (among free agents)
    fa_nxt_played_ids = set(fa_nxt_played['player_id'].tolist())

    # Avg3 top-20: restrict to FAs who played in round N+1, ranked by avg3 at round N
    avg3_pool = fa_cur[
        (fa_cur['player_id'].isin(fa_nxt_played_ids)) &
        ((fa_cur['avg3'] > 0) | (fa_cur['total_games'] >= 1))
    ]
    avg3_top20_ids = avg3_pool.sort_values('avg3', ascending=False).head(20)['player_id'].tolist()
    a3_res = evaluate_picks(avg3_top20_ids, nxt)

    # Newsletter picks: restrict to those who played in round N+1
    nl_picks = parse_newsletter_picks(nl_path)
    nl_ids = [p['player_id'] for p in nl_picks
              if p['player_id'] is not None and p['player_id'] in fa_nxt_played_ids]
    nl_res = evaluate_picks(nl_ids, nxt, verbose=False)

    def fmt(h, t): return f"{h}/{t} ({h/t*100:.0f}%)" if t else 'n/a'

    print(f"\n--- Rd{rd} -> Rd{nrd} (only players who played Rd{nrd}) ---")
    print(f"  FA base rate (played):       {fmt(br_h, br_t)}")
    print(f"  Avg3 top-20 (played only):   {fmt(a3_res['hits'], a3_res['played'])}")
    print(f"  Newsletter ({len(nl_ids)} played): {fmt(nl_res['hits'], nl_res['played'])}")

    for r, label in [(all_nl, nl_res), (all_a3, a3_res)]:
        r['hits'] += label['hits']
        r['played'] += label['played']
        r['dnp'] += label['dnp']

    # Show details for manual verification
    print(f"\n  Avg3 top-20 detail (Rd{rd} avg3 -> Rd{nrd} score):")
    top20_detail = avg3_pool.sort_values('avg3', ascending=False).head(20)
    for _, row in top20_detail.iterrows():
        pid = row['player_id']
        nxt_row = nxt[nxt['player_id'] == pid]
        if nxt_row.empty:
            nxt_score = 'no data'; played_str = ''
        else:
            nxt_row = nxt_row.iloc[0]
            if nxt_row['games'] == 1:
                nxt_score = f"{nxt_row['points']:.0f}"
                played_str = 'HIT' if nxt_row['points'] >= 80 else 'miss'
            else:
                nxt_score = 'DNP'
                played_str = ''
        print(f"    {row['name']:<28} avg3={row['avg3']:>6.1f}  Rd{nrd}={nxt_score:<5} {played_str}")

    print(f"\n  Newsletter picks detail (Rd{rd} -> Rd{nrd}):")
    for pick in nl_picks:
        pid = pick['player_id']
        cur_row = cur[cur['player_id'] == pid] if pid else None
        avg3_val = f"{cur_row.iloc[0]['avg3']:.1f}" if (cur_row is not None and not cur_row.empty) else 'n/a'
        nxt_row = nxt[nxt['player_id'] == pid] if pid else None
        if nxt_row is None or nxt_row.empty:
            nxt_score = 'no data'; played_str = ''
        else:
            nxt_row = nxt_row.iloc[0]
            if nxt_row['games'] == 1:
                nxt_score = f"{nxt_row['points']:.0f}"
                played_str = 'HIT' if nxt_row['points'] >= 80 else 'miss'
            else:
                nxt_score = 'DNP'
                played_str = ''
        match_str = '' if pick['player_id'] else '(NO ID MATCH)'
        print(f"    [{pick['pos']}] {pick['name']:<26} avg3={avg3_val:<6}  Rd{nrd}={nxt_score:<5} {played_str} {match_str}")

print("\n" + "=" * 80)
print("OVERALL TOTALS")
print("=" * 80)
def fmt(h, t): return f"{h}/{t} ({h/t*100:.1f}%)" if t else 'n/a'
print(f"  FA base rate (played):          {fmt(base_hits, base_played)}")
print(f"  Avg3 top-20 (played FAs only):  {fmt(all_a3['hits'], all_a3['played'])}")
print(f"  Newsletter (played picks only): {fmt(all_nl['hits'], all_nl['played'])}")
print()
if all_a3['played']:
    print(f"  Avg3 vs base:       +{all_a3['hits']/all_a3['played']*100 - base_hits/base_played*100:.1f}pp")
if all_nl['played']:
    print(f"  Newsletter vs base: +{all_nl['hits']/all_nl['played']*100 - base_hits/base_played*100:.1f}pp")
    print(f"  Newsletter vs avg3: {all_nl['hits']/all_nl['played']*100 - all_a3['hits']/all_a3['played']*100:+.1f}pp")
