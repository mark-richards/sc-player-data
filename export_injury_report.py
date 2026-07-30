"""
Export 2026 pre-season injury report with feed_id, player name, injury type,
source articles and context snippets.

Outputs:
  data/draft/injury_report_2026.csv  — one row per player, injury summary
  data/draft/injury_mentions_2026.csv — one row per mention (more detail)
"""

import re
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH   = Path('data/afl_news.db')
PLAYER_LIST = Path('outputs/player_list.csv')

# Pre-season window for 2026
WINDOW_FROM = '2026-01-15'
WINDOW_TO   = '2026-02-26'

# ── Injury-type extraction ────────────────────────────────────────────────────

INJURY_PATTERNS = [
    # Specific body-part + type
    (r'\bACL\b',                              'ACL'),
    (r'\banterior cruciate\b',                'ACL'),
    (r'\bhamstring\b',                        'hamstring'),
    (r'\bshoulder\b',                         'shoulder'),
    (r'\bachilles\b',                         'achilles'),
    (r'\bknee\b',                             'knee'),
    (r'\bankle\b',                            'ankle'),
    (r'\bback\b',                             'back'),
    (r'\bfoot\b|\bfeet\b',                    'foot'),
    (r'\bcalf\b|\bcalves\b',                  'calf'),
    (r'\bgroin\b',                            'groin'),
    (r'\bquad\b|\bquadricep',                 'quad'),
    (r'\bhip\b',                              'hip'),
    (r'\brib\b',                              'ribs'),
    (r'\bcollarbone\b|\bclavicle\b',          'collarbone'),
    (r'\bwrist\b',                            'wrist'),
    (r'\belbow\b',                            'elbow'),
    (r'\bfractur',                            'fracture'),
    (r'\bconcussion\b',                       'concussion'),
    (r'\bsurgery\b|\boperation\b',            'surgery'),
    (r'\brecovery\b|\brehab\b',               'rehab'),
    (r'\bmanaged\b|\bmanagement\b',           'managed'),
    (r'\bsidelined\b|\bruled out\b',          'ruled out'),
    (r'\bpersonal\b',                         'personal leave'),
]

SEVERITY_PATTERNS = [
    (r'\bseasoning?\b|\bseason.ending\b',     'season-ending'),
    (r'\bmonths?\b',                          'months'),
    (r'\bweeks?\b',                           'weeks'),
    (r'\bround 1\b|\bopening round\b',        'may miss R1'),
    (r'\bmiss\b|\bmissing\b',                 'missing games'),
    (r'\bright to play\b|\bfit\b|\bclear',    'tracking to play'),
]


def extract_injury_type(ctx: str) -> str:
    """Extract primary injury type from context string."""
    ctx_l = ctx.lower()
    found = []
    for pattern, label in INJURY_PATTERNS:
        if re.search(pattern, ctx_l, re.IGNORECASE):
            found.append(label)
    return ', '.join(dict.fromkeys(found)) if found else 'injury'


def extract_severity(ctx: str) -> str:
    """Extract rough severity/timeline from context string."""
    ctx_l = ctx.lower()
    for pattern, label in SEVERITY_PATTERNS:
        if re.search(pattern, ctx_l, re.IGNORECASE):
            return label
    return ''


def load_player_names() -> dict:
    """Return {feed_id: name} from player list."""
    try:
        df = pd.read_csv(PLAYER_LIST)
        # Handle various column name conventions
        id_col   = next((c for c in df.columns if 'id' in c.lower()), None)
        name_col = next((c for c in df.columns if 'name' in c.lower()), None)
        if id_col and name_col:
            return dict(zip(df[id_col].astype(str), df[name_col]))
    except Exception:
        pass
    return {}


def main():
    if not DB_PATH.exists():
        print("DB not found:", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)

    mentions = pd.read_sql_query("""
        SELECT
            pm.feed_id,
            pm.raw_name,
            pm.keyword_tags,
            pm.context_window,
            a.headline,
            a.published_at,
            a.source,
            a.url
        FROM player_mentions pm
        JOIN articles a ON pm.article_id = a.id
        WHERE a.published_at >= ?
          AND a.published_at <= ?
          AND pm.keyword_tags LIKE '%injury%'
          AND pm.feed_id IS NOT NULL
        ORDER BY pm.feed_id, a.published_at DESC
    """, conn, params=[WINDOW_FROM, WINDOW_TO])

    conn.close()

    if mentions.empty:
        print("No injury mentions found.")
        return

    mentions['feed_id'] = pd.to_numeric(mentions['feed_id'], errors='coerce')
    mentions = mentions.dropna(subset=['feed_id'])
    mentions['feed_id'] = mentions['feed_id'].astype(int)

    # Enrich with injury type + severity from context
    mentions['injury_type'] = mentions['context_window'].apply(extract_injury_type)
    mentions['severity']    = mentions['context_window'].apply(extract_severity)

    # ── Summary: one row per player ───────────────────────────────────────────

    player_names = load_player_names()

    def agg_player(grp):
        # Combine all contexts, pick most informative injury type
        all_ctx = ' '.join(grp['context_window'].fillna(''))
        all_inj = extract_injury_type(all_ctx)
        all_sev = extract_severity(all_ctx)
        sources = ', '.join(grp['source'].unique())
        n_articles = len(grp)
        latest_date = grp['published_at'].max()
        best_headline = grp.sort_values('published_at', ascending=False)['headline'].iloc[0]
        best_ctx = grp.sort_values('published_at', ascending=False)['context_window'].iloc[0]
        raw_names = ' / '.join(grp['raw_name'].unique()[:3])
        return pd.Series({
            'raw_names':      raw_names,
            'injury_type':    all_inj,
            'severity':       all_sev,
            'article_count':  n_articles,
            'latest_date':    latest_date,
            'sources':        sources,
            'best_headline':  best_headline[:120],
            'best_context':   best_ctx[:300],
        })

    summary = mentions.groupby('feed_id').apply(agg_player, include_groups=False).reset_index()

    # Add canonical player names from player_list
    summary['player_name'] = summary['feed_id'].astype(str).map(player_names).fillna('')

    # Re-order columns
    summary = summary[[
        'feed_id', 'player_name', 'raw_names',
        'injury_type', 'severity', 'article_count', 'latest_date',
        'sources', 'best_headline', 'best_context',
    ]].sort_values(['article_count', 'latest_date'], ascending=[False, False])

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = Path('data/draft')
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path   = out_dir / 'injury_report_2026.csv'
    mentions_path  = out_dir / 'injury_mentions_2026.csv'

    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    mentions[['feed_id','raw_name','injury_type','severity','published_at',
              'source','headline','context_window']].to_csv(
        mentions_path, index=False, encoding='utf-8-sig'
    )

    print(f"\n2026 Pre-season Injury Report  ({WINDOW_FROM} to {WINDOW_TO})")
    print(f"{'='*70}")
    print(f"Players with injury flags: {len(summary)}")
    print(f"Total injury mentions:     {len(mentions)}")
    print(f"\nSaved:")
    print(f"  {summary_path}   ({len(summary)} players)")
    print(f"  {mentions_path}  ({len(mentions)} mentions)")
    print(f"\n{'Top injured players (by article count)':}")
    print(f"{'='*70}")
    print(f"{'feed_id':<10} {'player':<24} {'injury':<30} {'sev':<16} {'arts'}")
    print("-" * 90)
    for _, r in summary.head(50).iterrows():
        name = r['player_name'] or r['raw_names'].split('/')[0].strip()
        inj  = r['injury_type'][:28]
        sev  = r['severity'][:14]
        print(f"{r['feed_id']:<10} {name:<24} {inj:<30} {sev:<16} {int(r['article_count'])}")


if __name__ == '__main__':
    main()
