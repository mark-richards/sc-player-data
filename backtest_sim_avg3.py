"""
Simulate the new avg3-based position-constrained newsletter ranking.
Compares: base rate vs avg3 top-20 vs simulated newsletter (top-5 per position by avg3).
"""
import json
import pandas as pd

# Load player list with positions
pl = pd.read_csv('draft_prep/SC 2026/2026_SC_Player_list.csv', low_memory=False)

with open('data/raw/supercoach/2026/players_round_14.json') as f:
    all_players = json.load(f)
feed_to_pid = {str(p['feed_id']): int(p['id']) for p in all_players}
all_player_ids = {int(p['id']) for p in all_players}

pid_to_pos = {}
for _, row in pl.iterrows():
    if not pd.isna(row.get('feed_id')) and not pd.isna(row.get('position')):
        fid = str(int(float(row['feed_id'])))
        pid = feed_to_pid.get(fid)
        if pid:
            pos = str(row['position'])
            pid_to_pos[pid] = 'FWD' if pos == 'FLEX' else pos

teams_df = pd.read_csv('data/live/current_teams.csv')
owned_pids = set()
for _, row in teams_df.iterrows():
    fid = str(int(float(row['feed_id']))) if not pd.isna(row['feed_id']) else None
    pid = feed_to_pid.get(fid)
    if pid:
        owned_pids.add(pid)
fa_ids = all_player_ids - owned_pids


def load_round_json(rnd):
    with open(f'data/raw/supercoach/2026/players_round_{rnd}.json') as f:
        data = json.load(f)
    rows = []
    for p in data:
        if not p['player_stats']:
            continue
        stat = p['player_stats'][0]
        rows.append({
            'player_id': int(p['id']),
            'points': float(stat.get('points', 0)),
            'games': int(stat.get('games', 0)),
            'avg3': float(stat.get('avg3', 0) or 0),
        })
    return pd.DataFrame(rows)


def fmt(h, t):
    return f"{h}/{t} ({h/t*100:.1f}%)" if t else 'n/a'


all_sim = {'hits': 0, 'played': 0}
all_a3 = {'hits': 0, 'played': 0}
base_hits = base_played = 0

print("=" * 70)
print("Simulated new newsletter: top-5 per position by avg3 + DVP fixture bonus")
print("Baseline: avg3 top-20 (cross-position)")
print("=" * 70)

for rd in [8, 9, 10, 11, 12, 13]:
    nrd = rd + 1
    if nrd > 14:
        continue
    cur = load_round_json(rd)
    nxt = load_round_json(nrd)

    fa_cur = cur[cur['player_id'].isin(fa_ids)].copy()
    fa_nxt_played = nxt[nxt['player_id'].isin(fa_ids) & (nxt['games'] == 1)]
    played_ids = set(fa_nxt_played['player_id'].tolist())

    br_h = int((fa_nxt_played['points'] >= 80).sum())
    br_t = len(fa_nxt_played)
    base_hits += br_h
    base_played += br_t

    # Avg3 top-20 (cross-position, played only)
    avg3_pool = fa_cur[fa_cur['player_id'].isin(played_ids) & (fa_cur['avg3'] > 0)]
    a3_top20 = avg3_pool.sort_values('avg3', ascending=False).head(20)['player_id'].tolist()
    a3_plays = nxt[nxt['player_id'].isin(a3_top20) & (nxt['games'] == 1)]
    a3_h = int((a3_plays['points'] >= 80).sum())
    all_a3['hits'] += a3_h
    all_a3['played'] += len(a3_plays)

    # Simulated new newsletter: top 5 per position by avg3
    sim_ids = []
    for pos in ['DEF', 'MID', 'FWD', 'RUC']:
        n = 5 if pos != 'RUC' else 2
        pos_pool = fa_cur[
            fa_cur['player_id'].isin(played_ids) &
            (fa_cur['avg3'] > 0) &
            fa_cur['player_id'].apply(lambda pid: pid_to_pos.get(pid, '') == pos)
        ]
        top_n = pos_pool.sort_values('avg3', ascending=False).head(n)['player_id'].tolist()
        sim_ids.extend(top_n)

    sim_plays = nxt[nxt['player_id'].isin(sim_ids) & (nxt['games'] == 1)]
    sim_h = int((sim_plays['points'] >= 80).sum())
    all_sim['hits'] += sim_h
    all_sim['played'] += len(sim_plays)

    print(f"\nRd{rd}->Rd{nrd}:")
    print(f"  Base rate:      {fmt(br_h, br_t)}")
    print(f"  Avg3 top-20:    {fmt(a3_h, len(a3_plays))}")
    print(f"  Sim newsletter: {fmt(sim_h, len(sim_plays))}")

    # Show simulated picks with avg3 and actual scores
    print(f"  Sim picks (avg3 -> Rd{nrd} score):")
    for pid in sim_ids:
        pos = pid_to_pos.get(pid, '?')
        cur_row = fa_cur[fa_cur['player_id'] == pid]
        a3 = cur_row.iloc[0]['avg3'] if not cur_row.empty else 0
        nxt_row = nxt[nxt['player_id'] == pid]
        if nxt_row.empty:
            sc = 'no data'
            res = ''
        else:
            nr = nxt_row.iloc[0]
            if nr['games'] == 1:
                sc = f"{nr['points']:.0f}"
                res = 'HIT' if nr['points'] >= 80 else 'miss'
            else:
                sc = 'DNP'
                res = ''
        # Get name
        name_rows = pl[pl['feed_id'].apply(
            lambda fid: feed_to_pid.get(str(int(float(fid))), None) == pid if not pd.isna(fid) else False
        )].copy()
        if not name_rows.empty:
            r0 = name_rows.iloc[0]
            name = f"{r0['first_name']} {r0['last_name']}".strip()
        else:
            name = str(pid)
        print(f"    [{pos}] {name:<28} avg3={a3:>5.1f}  Rd{nrd}={sc:<5} {res}")

print("\n" + "=" * 70)
print("OVERALL TOTALS")
print("=" * 70)
print(f"  FA base rate:           {fmt(base_hits, base_played)}")
print(f"  Avg3 top-20:            {fmt(all_a3['hits'], all_a3['played'])}")
print(f"  Sim newsletter (avg3):  {fmt(all_sim['hits'], all_sim['played'])}")
print()
print(f"  Avg3 vs base:     +{all_a3['hits']/all_a3['played']*100 - base_hits/base_played*100:.1f}pp")
print(f"  Sim nl vs base:   +{all_sim['hits']/all_sim['played']*100 - base_hits/base_played*100:.1f}pp")
print(f"  Sim nl vs avg3:   {all_sim['hits']/all_sim['played']*100 - all_a3['hits']/all_a3['played']*100:+.1f}pp")
