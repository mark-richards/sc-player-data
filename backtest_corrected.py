import re, glob, warnings, pandas as pd, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

# Load fanfooty 2026 data (all players, all rounds)
ff_frames = []
for f in sorted(glob.glob('data/processed/2026_round_*_fanfooty_data.csv')):
    try:
        chunk = pd.read_csv(f, low_memory=False)
        chunk = chunk.rename(columns={c: c.strip() for c in chunk.columns})
        sc_col = next((c for c in chunk.columns if c.strip() in ('SC', 'sc')), None)
        pid_col = next((c for c in chunk.columns if 'Player ID' in c or c == 'feed_id'), None)
        if sc_col is None or pid_col is None:
            continue
        chunk = chunk[[pid_col, sc_col]].rename(columns={pid_col: 'feed_id', sc_col: 'sc'})
        chunk['feed_id'] = pd.to_numeric(chunk['feed_id'], errors='coerce')
        chunk['sc'] = pd.to_numeric(chunk['sc'], errors='coerce')
        m = re.search(r'2026_round_(\d+)', f)
        chunk['round'] = int(m.group(1)) if m else 0
        chunk = chunk.dropna()
        chunk = chunk[chunk['sc'] > 0]
        ff_frames.append(chunk)
    except Exception as e:
        print(f"  skip {f}: {e}")

ff_all = pd.concat(ff_frames)
print(f"Loaded fanfooty: {len(ff_all)} rows, {ff_all['feed_id'].nunique()} players, rounds {sorted(ff_all['round'].unique())}")

# Free agent pool = players in fanfooty but NOT currently owned
sc_df = pd.read_csv('data/live/player_stats_current.csv')
owned_ids = set(sc_df['feed_id'].unique())
fa_ids = set(ff_all['feed_id'].unique()) - owned_ids
print(f"Owned: {len(owned_ids)}, Free agents: {len(fa_ids)}")

# Name lookup from SC player list
try:
    pl = pd.read_csv('draft_prep/SC 2026/2026_SC_Player_list.csv', low_memory=False)
    pl['_full_name'] = (pl['first_name'].str.strip() + ' ' + pl['last_name'].str.strip()).str.strip()
    fid_to_name = {float(row['feed_id']): row['_full_name'] for _, row in pl.iterrows() if not pd.isna(row['feed_id'])}
    name_to_fid = {v.lower(): k for k, v in fid_to_name.items()}
except Exception as e:
    print(f"Player list error: {e}")
    fid_to_name = {}
    name_to_fid = {}


def ff_score(fid, rnd):
    row = ff_all[(ff_all['feed_id'] == fid) & (ff_all['round'] == rnd)]
    return float(row['sc'].values[0]) if len(row) else None


def parse_newsletter_picks(path, max_per_pos=5):
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
            clean = re.sub(r'[^\x00-\x7F]+', '', mn.group(1)).strip()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean) < 3:
                continue
            # Match name to feed_id
            name_lower = clean.lower()
            fid = name_to_fid.get(name_lower)
            if fid is None:
                parts = name_lower.split()
                if parts:
                    last = parts[-1]
                    cands = [(fid2, n) for fid2, n in fid_to_name.items()
                             if last in str(n).lower().split()[-1:]]
                    if len(cands) == 1:
                        fid = cands[0][0]
                    elif len(cands) > 1 and len(parts) > 1:
                        first = parts[0]
                        exact = [c for c in cands if str(c[1]).lower().startswith(first)]
                        if len(exact) == 1:
                            fid = exact[0][0]
            picks.append({'pos': pos, 'name': clean, 'fid': fid})
            pos_count += 1
    return picks


def avg3_fa_picks(round_num, top_n=20):
    ff_prev = ff_all[(ff_all['round'] <= round_num) & (ff_all['feed_id'].isin(fa_ids))]
    rows = []
    for fid, grp in ff_prev.groupby('feed_id'):
        recent = grp.sort_values('round').tail(3)
        if len(recent) < 1:
            continue
        rows.append({'fid': fid, 'avg3': recent['sc'].mean()})
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    return list(df.sort_values('avg3', ascending=False).head(top_n)['fid'])


def season_avg_fa_picks(round_num, top_n=20):
    ff_prev = ff_all[(ff_all['round'] <= round_num) & (ff_all['feed_id'].isin(fa_ids))]
    rows = []
    for fid, grp in ff_prev.groupby('feed_id'):
        n = len(grp)
        avg_s = grp['sc'].mean()
        avg_l5 = grp.tail(5)['sc'].mean()
        if n < 4:
            continue
        if n >= 5 and avg_l5 < 55:  # Fix 4: suppress underperformers
            continue
        sk = avg_s - (10.0 if avg_s < 70 and n >= 6 else 0.0)  # Fix 5: form decay
        rows.append({'fid': fid, 'sk': sk, 'avg_s': avg_s})
    if not rows:
        return []
    return list(pd.DataFrame(rows).sort_values('sk', ascending=False).head(top_n)['fid'])


def hit_rate_ff(fid_list, next_rd):
    h = t = 0
    for fid in fid_list:
        s = ff_score(fid, next_rd)
        if s is not None:
            t += 1
            h += (1 if s >= 80 else 0)
    return h, t


nls = {
    8: 'reports/waiver_2026_rd08_2026-05-06.md',
    9: 'reports/waiver_2026_rd09_2026-05-11.md',
    10: 'reports/waiver_2026_rd10_2026-05-18.md',
    11: 'reports/waiver_2026_rd11_2026-05-26.md',
    12: 'reports/waiver_2026_rd12_2026-06-02.md',
    13: 'reports/waiver_2026_rd13_2026-06-10.md',
}

# Base rate: FA pool 80+ rate per round
print("\n=== Base rate: % of all free agents scoring 80+ each round ===")
for rd in range(8, 15):
    rd_fa = ff_all[(ff_all['round'] == rd) & (ff_all['feed_id'].isin(fa_ids))]
    if len(rd_fa):
        pct = (rd_fa['sc'] >= 80).mean() * 100
        print(f"  Rd{rd}: {len(rd_fa)} FAs played, 80+: {int((rd_fa['sc']>=80).sum())} ({pct:.1f}%)")

print("\n=== CORRECTED BACKTEST: Free agents only, scores from fanfooty ===")
print(f"{'Round':<10} {'FA base rate':>14} {'Avg3 top-20':>14} {'SeasonAvg top-20':>18} {'Old NL':>12}")
print('-' * 72)

brt = brh = a3t = a3h = sat = sah = ont = onh = 0

for rd, path in nls.items():
    nrd = rd + 1
    if nrd > 14:
        continue

    rd_fa = ff_all[(ff_all['round'] == nrd) & (ff_all['feed_id'].isin(fa_ids))]
    br_h = int((rd_fa['sc'] >= 80).sum())
    br_t = len(rd_fa)

    a3_fids = avg3_fa_picks(rd)
    a3_h, a3_t = hit_rate_ff(a3_fids, nrd)

    sa_fids = season_avg_fa_picks(rd)
    sa_h, sa_t = hit_rate_ff(sa_fids, nrd)

    nl_picks = parse_newsletter_picks(path)
    on_fids = [p['fid'] for p in nl_picks if p['fid'] is not None]
    on_h, on_t = hit_rate_ff(on_fids, nrd)

    def fmt(h, t):
        return f"{h}/{t} ({h/t*100:.0f}%)" if t else 'n/a'

    print(f"Rd{rd}->Rd{nrd:<4} {fmt(br_h,br_t):>14} {fmt(a3_h,a3_t):>14} {fmt(sa_h,sa_t):>18} {fmt(on_h,on_t):>12}")
    brt += br_t; brh += br_h
    a3t += a3_t; a3h += a3_h
    sat += sa_t; sah += sa_h
    ont += on_t; onh += on_h

print('-' * 72)


def fmt(h, t):
    return f"{h}/{t} ({h/t*100:.1f}%)" if t else 'n/a'


print(f"{'TOTAL':<10} {fmt(brh,brt):>14} {fmt(a3h,a3t):>14} {fmt(sah,sat):>18} {fmt(onh,ont):>12}")
print()
print(f"Base FA 80+ rate: {brh/brt*100:.1f}%")
print(f"Avg3 top-20:      {a3h/a3t*100:.1f}%  (vs base: +{a3h/a3t*100 - brh/brt*100:.1f}pp)")
print(f"SeasonAvg top-20: {sah/sat*100:.1f}%  (vs base: +{sah/sat*100 - brh/brt*100:.1f}pp)")
if ont:
    print(f"Old newsletter:   {onh/ont*100:.1f}%  (vs base: +{onh/ont*100 - brh/brt*100:.1f}pp)")

# Drill into what the old newsletter was recommending
print("\n=== Old NL Rd13 pick resolution (to check name matching) ===")
for p in parse_newsletter_picks('reports/waiver_2026_rd13_2026-06-10.md'):
    sc14 = ff_score(p['fid'], 14) if p['fid'] else None
    hit = 'HIT' if (sc14 and sc14 >= 80) else ('miss' if sc14 is not None else 'no data')
    print(f"  [{p['pos']}] {p['name']:<30} fid={str(p['fid']):<10} Rd14={str(sc14):<6} {hit}")
