import re, glob, warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path

scores_df = pd.read_csv('data/live/player_stats_current.csv')
scores_df['full_name'] = (scores_df['first_name'] + ' ' + scores_df['last_name']).str.strip().str.lower()
score_lookup = {}
for _, row in scores_df.iterrows():
    fid = int(row['feed_id']); rnd = int(row['round'])
    if row['played'] > 0 and not pd.isna(row['points']):
        score_lookup.setdefault(fid, {})[rnd] = float(row['points'])
name_to_fid = {}
for _, row in scores_df[['feed_id','full_name']].drop_duplicates('feed_id').iterrows():
    name_to_fid[row['full_name']] = int(row['feed_id'])
fid_to_name = {v: k for k, v in name_to_fid.items()}

def lookup_score(name, rnd):
    key = name.strip().lower()
    fid = name_to_fid.get(key)
    if fid is None:
        last = key.split()[-1]
        cands = [n for n in name_to_fid if last in n.split()]
        if len(cands) == 1: fid = name_to_fid[cands[0]]
        elif len(cands) > 1:
            first = key.split()[0]
            exact = [n for n in cands if n.startswith(first)]
            if len(exact) == 1: fid = name_to_fid[exact[0]]
    if fid is None: return None
    return score_lookup.get(fid, {}).get(rnd)

def parse_newsletter_picks(path, max_per_pos=5):
    txt = Path(path).read_text(encoding='utf-8')
    sections = re.split(r'\n## ', txt)
    pt_section = next((s for s in sections if 'Positional Targets' in s.split('\n')[0]), '')
    picks = []; pos = None; pos_count = 0
    for line in pt_section.split('\n'):
        m_pos = re.match(r'^### (DEF|MID|FWD|RUC)\s*$', line.strip())
        if m_pos: pos = m_pos.group(1); pos_count = 0; continue
        if pos and pos_count < max_per_pos and line.startswith('|') and '**' in line:
            cells = [c.strip() for c in line.split('|')]
            if len(cells) < 4: continue
            nc = cells[1]
            if '**Player**' in nc or '---' in nc: continue
            mn = re.search(r'\*\*(.+?)\*\*', nc)
            if not mn: continue
            clean = re.sub(r'[^\x00-\x7F]+', '', mn.group(1)).strip()
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean) < 3: continue
            picks.append({'pos': pos, 'name': clean}); pos_count += 1
    return picks

def avg3_baseline(round_num, top_n=20):
    r = scores_df[(scores_df['round'] == round_num) & (scores_df['played'] > 0)]
    return [(row['full_name'], row['avg3']) for _, row in r.sort_values('avg3', ascending=False).head(top_n).iterrows()]

ff_frames = []
for f in sorted(glob.glob('data/processed/2026_round_*_fanfooty_data.csv')):
    try:
        chunk = pd.read_csv(f, usecols=['Player ID','SC'], low_memory=False)
        chunk.columns = ['feed_id','sc']
        chunk['feed_id'] = pd.to_numeric(chunk['feed_id'], errors='coerce')
        chunk['sc'] = pd.to_numeric(chunk['sc'], errors='coerce')
        m = re.search(r'2026_round_(\d+)', f)
        chunk['round'] = int(m.group(1)) if m else 0
        chunk = chunk.dropna(); chunk = chunk[chunk['sc'] > 0]
        ff_frames.append(chunk)
    except: pass
ff_all = pd.concat(ff_frames) if ff_frames else pd.DataFrame(columns=['feed_id','sc','round'])

def fixed_nl_picks(round_num, top_n=20):
    ff_to_rd = ff_all[ff_all['round'] <= round_num]
    if ff_to_rd.empty: return []
    rows = []
    for fid, grp in ff_to_rd.groupby('feed_id'):
        g = grp.sort_values('round'); n = len(g)
        if n < 4: continue
        avg_s = g['sc'].mean(); avg_l5 = g.tail(5)['sc'].mean()
        if n >= 5 and avg_l5 < 55: continue   # Fix 4: suppress underperformers
        sort_key = avg_s - (10.0 if (avg_s < 70 and n >= 6) else 0.0)  # Fix 5
        rows.append({'name': fid_to_name.get(int(fid),''), 'avg_s': avg_s, 'sk': sort_key})
    if not rows: return []
    df = pd.DataFrame(rows).sort_values('sk', ascending=False)
    return [(r['name'], r['avg_s']) for _, r in df.head(top_n).iterrows()]

def hit_rate(picks, next_rd, get_name):
    h = t = 0
    for p in picks:
        s = lookup_score(get_name(p), next_rd)
        if s is not None: t += 1; h += (1 if s >= 80 else 0)
    return h, t

nls = {8:'reports/waiver_2026_rd08_2026-05-06.md',9:'reports/waiver_2026_rd09_2026-05-11.md',
       10:'reports/waiver_2026_rd10_2026-05-18.md',11:'reports/waiver_2026_rd11_2026-05-26.md',
       12:'reports/waiver_2026_rd12_2026-06-02.md',13:'reports/waiver_2026_rd13_2026-06-10.md'}

print('\n=== BACKTEST: Old NL vs Avg3 top20 vs Fixed NL ===')
print('80+ hit rate in NEXT round\n')
print(f"{'Round':<10} {'Old NL':>14} {'Avg3 top20':>14} {'Fixed NL':>14}")
print('-'*54)
oh=ot=bh=bt=fh=ft=0
for rd, path in nls.items():
    nrd = rd+1
    if nrd > 14: continue
    op = parse_newsletter_picks(path); oh1,ot1 = hit_rate(op, nrd, lambda p: p['name'])
    bp = avg3_baseline(rd); bh1,bt1 = hit_rate(bp, nrd, lambda p: p[0])
    fp = fixed_nl_picks(rd); fh1,ft1 = hit_rate(fp, nrd, lambda p: p[0])
    def fmt(h,t): return f"{h}/{t} ({h/t*100:.0f}%)" if t else 'n/a'
    print(f"Rd{rd}->Rd{nrd:<4} {fmt(oh1,ot1):>14} {fmt(bh1,bt1):>14} {fmt(fh1,ft1):>14}")
    oh+=oh1;ot+=ot1;bh+=bh1;bt+=bt1;fh+=fh1;ft+=ft1

print('-'*54)
def fmt(h,t): return f"{h}/{t} ({h/t*100:.1f}%)" if t else 'n/a'
print(f"{'TOTAL':<10} {fmt(oh,ot):>14} {fmt(bh,bt):>14} {fmt(fh,ft):>14}")
print(f"\nOld NL: {oh/ot*100:.1f}%  |  Avg3: {bh/bt*100:.1f}%  |  Fixed NL: {fh/ft*100:.1f}%")

print('\n--- Fixed NL Rd13 picks (what it would have recommended) ---')
for name, avg in fixed_nl_picks(13, top_n=20):
    sc14 = lookup_score(name, 14)
    hit = 'HIT' if (sc14 and sc14 >= 80) else ('miss' if sc14 else 'no data')
    sc14_str = f"{sc14:.0f}" if sc14 else 'n/a'
    print(f"  {name:<32} avg={avg:.1f}  Rd14={sc14_str:<5} {hit}")

print('\n--- Old NL Rd13 picks (for comparison) ---')
for p in parse_newsletter_picks('reports/waiver_2026_rd13_2026-06-10.md', max_per_pos=5):
    sc14 = lookup_score(p['name'], 14)
    sc14_str = f"{sc14:.0f}" if sc14 else 'n/a'
    hit = 'HIT' if (sc14 and sc14 >= 80) else ('miss' if sc14 else 'no data')
    print(f"  [{p['pos']}] {p['name']:<30} Rd14={sc14_str:<5} {hit}")
