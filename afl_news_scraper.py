"""
AFL News Scraper — Proof of Concept
====================================
Scrapes afl.com.au/news/all-news (3 pages), extracts article text,
identifies bolded player names, and stores results in SQLite.

Requires:  playwright  rapidfuzz  httpx
Install:   pip install playwright rapidfuzz httpx
           playwright install chromium
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout
from rapidfuzz import process, fuzz

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH        = Path('data/afl_news.db')
PLAYER_LIST    = Path(f'draft_prep/SC 2026/2026_SC_Player_list.csv')
PAGE_URL       = 'https://www.afl.com.au/news/all-news'
PAGES_TO_FETCH = 3
FUZZY_THRESHOLD = 82     # rapidfuzz score cutoff (0-100)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT PRIMARY KEY,
    url             TEXT UNIQUE NOT NULL,
    source          TEXT NOT NULL,
    headline        TEXT,
    full_text       TEXT,
    author          TEXT,
    published_at    TEXT,
    scraped_at      TEXT NOT NULL,
    article_type    TEXT,
    wayback_url     TEXT,
    raw_html        TEXT
);

CREATE TABLE IF NOT EXISTS player_mentions (
    mention_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      TEXT NOT NULL REFERENCES articles(id),
    feed_id         TEXT,
    raw_name        TEXT NOT NULL,
    resolution_tier INTEGER,
    sentiment_score REAL,
    keyword_tags    TEXT,
    context_window  TEXT,
    is_bold         INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mentions_feed_id   ON player_mentions(feed_id);
CREATE INDEX IF NOT EXISTS idx_mentions_article   ON player_mentions(article_id);
CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);

CREATE VIEW IF NOT EXISTS player_news_timeline AS
SELECT
    pm.feed_id,
    a.published_at,
    a.source,
    a.headline,
    pm.keyword_tags,
    pm.sentiment_score,
    pm.context_window
FROM player_mentions pm
JOIN articles a ON pm.article_id = a.id
WHERE pm.feed_id IS NOT NULL
ORDER BY a.published_at DESC;
"""

# ── Player loader & normalisation ─────────────────────────────────────────────

def normalise_name(s: str) -> str:
    """Lowercase, strip accents, remove possessives and punctuation."""
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    s = re.sub(r"'s?\b", '', s)
    s = re.sub(r"[^a-z\s]", '', s.lower())
    return ' '.join(s.split())


@dataclass
class PlayerRegistry:
    """Three-tier player name → feed_id resolver."""
    exact:   dict = field(default_factory=dict)   # norm_fullname → feed_id
    surname: dict = field(default_factory=lambda: defaultdict(list))  # surname → [feed_id]
    meta:    dict = field(default_factory=dict)   # feed_id → {first, last, team}

    @classmethod
    def from_csv(cls, path: Path) -> 'PlayerRegistry':
        reg = cls()
        if not path.exists():
            log.warning(f"Player list not found at {path} — using dummy data")
            _load_dummy(reg)
            return reg

        df = pd.read_csv(path, low_memory=False)
        # Try various column names that might exist in the project's player_list.csv
        id_col    = next((c for c in ['feed_id', 'Player ID', 'id'] if c in df.columns), None)
        fname_col = next((c for c in ['first_name', 'First Name', 'firstName'] if c in df.columns), None)
        lname_col = next((c for c in ['last_name',  'Last Name',  'lastName']  if c in df.columns), None)
        team_col  = next((c for c in ['team', 'Team', 'team_abbrev']            if c in df.columns), None)

        if not all([id_col, fname_col, lname_col]):
            log.warning(f"Could not find required columns in {path}. Using dummy data.")
            _load_dummy(reg)
            return reg

        for _, row in df.iterrows():
            fid  = str(row[id_col])
            fn   = str(row[fname_col]).strip()
            ln   = str(row[lname_col]).strip()
            team = str(row[team_col]).strip() if team_col else ''

            full_norm = normalise_name(f'{fn} {ln}')
            reg.exact[full_norm] = fid
            reg.surname[ln.lower()].append(fid)
            reg.meta[fid] = {'first': fn, 'last': ln, 'team': team}

        log.info(f"Loaded {len(reg.exact)} players from {path}")
        return reg

    def resolve(self, name_str: str, context: str = '') -> tuple[Optional[str], int]:
        """
        Returns (feed_id | None, tier).
        tier: 1=exact, 2=surname, 3=fuzzy, 0=unresolved
        """
        norm = normalise_name(name_str)

        # Tier 1: exact full-name match
        if norm in self.exact:
            return self.exact[norm], 1

        # Tier 2: unambiguous surname match
        parts = norm.split()
        if parts:
            surname = parts[-1]
            candidates = self.surname.get(surname, [])
            if len(candidates) == 1:
                return candidates[0], 2
            if len(candidates) > 1:
                # Try to disambiguate by team name in surrounding context
                for fid in candidates:
                    team = self.meta[fid]['team'].lower()
                    if team and team in context.lower():
                        return fid, 2

        # Tier 3: fuzzy full-name search
        result = process.extractOne(
            norm,
            self.exact.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result:
            return self.exact[result[0]], 3

        return None, 0


def _load_dummy(reg: 'PlayerRegistry'):
    """Fallback dummy roster for testing without a real player list."""
    players = [
        ('11001', 'Marcus',  'Bontempelli', 'WBD'),
        ('11002', 'Max',     'Gawn',        'MEL'),
        ('11003', 'Zak',     'Butters',     'PTA'),
        ('11004', 'Patrick', 'Cripps',      'CAR'),
        ('11005', 'Lachie',  'Neale',       'BRL'),
        ('11006', 'Clayton', 'Oliver',      'MEL'),
        ('11007', 'Nick',    'Daicos',      'COL'),
        ('11008', 'Josh',    'Dunkley',     'BRL'),
        ('11009', 'Errol',   'Gulden',      'SYD'),
        ('11010', 'Isaac',   'Heeney',      'SYD'),
        ('11011', 'Tristan', 'Xerri',       'NTH'),
        ('11012', 'Brodie',  'Grundy',      'SYD'),
        ('11013', 'Tom',     'Green',       'GWS'),
        ('11014', 'Finn',    'Callaghan',   'GWS'),
        ('11015', 'Bailey',  'Smith',       'GEE'),
    ]
    for fid, fn, ln, team in players:
        full_norm = normalise_name(f'{fn} {ln}')
        reg.exact[full_norm] = fid
        reg.surname[ln.lower()].append(fid)
        reg.meta[fid] = {'first': fn, 'last': ln, 'team': team}
    log.info(f"Loaded {len(reg.exact)} dummy players")


# ── Keyword tagging ───────────────────────────────────────────────────────────

KEYWORD_PATTERNS = {
    'injury':           r'\b(injur|hamstring|acl|ankle|knee|shoulder|surgery|ruled out|unavailable)\b',
    'return':           r'\b(return|come back|back in|recover|fit)\b',
    'role_change':      r'\b(midfield|half.back|forward|ruck|role change|different role|named|named as)\b',
    'pre_season_buzz':  r'\b(pre.?season|training|practice|camp|trial|nab)\b',
    'form':             r'\b(form|career.best|best|dominant|outstanding|elite|struggled)\b',
    'contract':         r'\b(contract|sign|extension|trade|delist|rookie)\b',
    'selection':        r'\b(selected|named|squad|team|side|bench|emergency)\b',
}

# ── Sentiment keywords (rule-based override layer) ────────────────────────────
# These patterns catch fantasy-relevant injury/return language that VADER misses
# because it scores emotionally neutral medical language as neutral.
_INJURY_NEGATIVE = re.compile(
    r'\b(sidelined|ruled out|managed|unavailable|scans|concussion|surgery|hamstring|'
    r'knee surgery|ankle surgery|under an injury cloud|in doubt|test his|to be tested)\b',
    re.IGNORECASE,
)
_INJURY_POSITIVE = re.compile(
    r'\b(cleared|fit to play|passed fitness|returned to training|back in training|'
    r'ready to play|no concerns|fully fit|available to play)\b',
    re.IGNORECASE,
)


def score_sentiment(context_window: str) -> float:
    """
    Return a sentiment score in [-1.0, 1.0] for a player mention context window.

    Strategy: rule-based injury keywords override VADER when the context is
    clearly fantasy-negative (ruled out) or fantasy-positive (cleared to play).
    VADER handles nuanced form/role language well and is used as the fallback.
    """
    if not context_window:
        return 0.0

    # Check positive first — "cleared after his hamstring injury" should be +0.8,
    # not -0.8 (which would fire on "hamstring" if negative came first).
    if _INJURY_POSITIVE.search(context_window):
        return 0.8
    if _INJURY_NEGATIVE.search(context_window):
        return -0.8

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
        return _analyzer.polarity_scores(context_window)['compound']
    except ImportError:
        return 0.0


def extract_keyword_tags(text: str) -> list[str]:
    tags = []
    ltext = text.lower()
    for tag, pattern in KEYWORD_PATTERNS.items():
        if re.search(pattern, ltext, re.IGNORECASE):
            tags.append(tag)
    return tags


# ── Database ──────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    log.info(f"Database ready: {path}")
    return conn


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


_GENERIC_TITLES = {"more from telstra", "afl", "afl.com.au", ""}

def _headline_from_url(url: str) -> str:
    """Extract a readable headline from the URL slug as a last resort."""
    slug = url.rstrip("/").split("/")[-1]
    # Remove leading numeric ID if present (e.g. "1532695-medical-room-...")
    slug = re.sub(r"^\d+-?", "", slug)
    return slug.replace("-", " ").title()


def insert_article(conn: sqlite3.Connection, data: dict) -> bool:
    """Returns True if newly inserted, False if already existed."""
    try:
        cur = conn.execute("""
            INSERT OR IGNORE INTO articles
            (id, url, source, headline, full_text, author, published_at, scraped_at, article_type, raw_html)
            VALUES (:id, :url, :source, :headline, :full_text, :author,
                    :published_at, :scraped_at, :article_type, :raw_html)
        """, data)
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        log.error(f"DB insert error: {e}")
        return False


def insert_mention(conn: sqlite3.Connection, data: dict):
    conn.execute("""
        INSERT INTO player_mentions
        (article_id, feed_id, raw_name, resolution_tier, keyword_tags,
         sentiment_score, context_window, is_bold, created_at)
        VALUES (:article_id, :feed_id, :raw_name, :resolution_tier, :keyword_tags,
                :sentiment_score, :context_window, :is_bold, :created_at)
    """, data)
    conn.commit()


# ── Rate limiter ──────────────────────────────────────────────────────────────

class SimpleRateLimiter:
    """Async token-bucket style rate limiter per domain pool."""

    def __init__(self, min_delay: float = 3.0, jitter: float = 1.5):
        self.min_delay = min_delay
        self.jitter    = jitter
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, domain: str = 'default'):
        async with self._lock:
            now   = time.monotonic()
            last  = self._last.get(domain, 0.0)
            delay = self.min_delay + random.uniform(0, self.jitter)
            wait  = max(0.0, delay - (now - last))
            if wait:
                log.debug(f"Rate limit: waiting {wait:.1f}s for {domain}")
                await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()


# ── Playwright page scraper ───────────────────────────────────────────────────

async def fetch_news_listing(page: Page, url: str) -> list[dict]:
    """
    Load AFL news listing page and extract article card data.
    AFL.com.au is a React SPA — cards are injected by JS after load.
    Returns list of {url, headline, published_at, article_type}.
    """
    log.info(f"Fetching listing: {url}")
    try:
        await page.goto(url, wait_until='networkidle', timeout=30_000)
    except PWTimeout:
        log.warning("Timeout waiting for networkidle — proceeding with DOM as-is")

    # Wait for article cards to render
    try:
        await page.wait_for_selector('[class*="article-card"], [class*="news-list"] a, article a', timeout=10_000)
    except PWTimeout:
        log.warning("Article card selector timed out — page may have changed structure")

    html    = await page.content()
    soup    = BeautifulSoup(html, 'html.parser')
    articles = []

    # AFL.com.au uses several card structures — try multiple selectors
    selectors = [
        'article a[href*="/news/"]',
        'a[href*="/news/"][class*="card"]',
        '[data-testid*="article"] a',
        'a[href*="afl.com.au/news/"]',
    ]

    seen_hrefs = set()
    for sel in selectors:
        for el in soup.select(sel):
            href = el.get('href', '')
            if not href:
                continue
            if not href.startswith('http'):
                href = f'https://www.afl.com.au{href}'
            if href in seen_hrefs or '/news/' not in href:
                continue
            seen_hrefs.add(href)

            headline = (
                el.get_text(strip=True) or
                el.get('aria-label', '') or
                el.find('h2') and el.find('h2').get_text(strip=True) or
                ''
            )
            articles.append({
                'url':          href,
                'headline':     headline,
                'published_at': None,      # resolved when fetching full article
                'article_type': 'news',
            })

    log.info(f"  Found {len(articles)} article links")
    return articles


async def fetch_article_body(page: Page, url: str, rate: SimpleRateLimiter) -> Optional[dict]:
    """Fetch a single article and extract body text + bold player candidates."""
    await rate.wait('afl.com.au')
    log.info(f"  Fetching article: {url[:80]}…")

    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=25_000)
    except PWTimeout:
        log.warning(f"  Timeout on {url}")
        return None

    html = await page.content()
    soup = BeautifulSoup(html, 'html.parser')

    # ── Headline ──
    headline = ''
    for sel in ['h1', '[class*="article-title"]', '[class*="headline"]']:
        el = soup.select_one(sel)
        if el:
            headline = el.get_text(strip=True)
            break

    # ── Author ──
    author = ''
    for sel in ['[class*="author"]', '[rel="author"]', 'span[class*="byline"]']:
        el = soup.select_one(sel)
        if el:
            author = el.get_text(strip=True)
            break

    # ── Published date ──
    published_at = None
    for sel in ['time[datetime]', '[class*="date"]', 'meta[property="article:published_time"]']:
        el = soup.select_one(sel)
        if el:
            published_at = el.get('datetime') or el.get('content') or el.get_text(strip=True)
            break

    # ── Article body ──
    body_el = (
        soup.select_one('[class*="article-body"]') or
        soup.select_one('[class*="article-content"]') or
        soup.select_one('article') or
        soup.select_one('main')
    )
    if not body_el:
        log.warning(f"  Could not find article body on {url}")
        return None

    full_text = body_el.get_text(separator=' ', strip=True)

    # ── Bolded candidates ──
    bold_strings = []
    for tag in body_el.find_all(['b', 'strong']):
        text = tag.get_text(strip=True)
        # Filter: likely player names are 2–4 words, title case, no punctuation
        if 2 <= len(text.split()) <= 4 and re.match(r"^[A-Za-z\-' ]+$", text):
            bold_strings.append(text)

    return {
        'url':          url,
        'headline':     headline,
        'full_text':    full_text,
        'author':       author,
        'published_at': published_at,
        'bold_strings': bold_strings,
        'raw_html':     str(body_el)[:50_000],  # cap at 50k chars
    }


# ── "Load more" pagination ────────────────────────────────────────────────────

async def paginate_listing(page: Page, base_url: str, target_pages: int, rate: SimpleRateLimiter) -> list[dict]:
    """
    AFL.com.au uses a 'Load More' button rather than URL-based pagination.
    Click it (target_pages - 1) times to get multiple pages worth of articles.
    """
    await rate.wait('afl.com.au')
    articles = await fetch_news_listing(page, base_url)

    for page_num in range(2, target_pages + 1):
        await rate.wait('afl.com.au')
        log.info(f"Loading page {page_num} of {target_pages}…")

        # Try clicking a "Load more" / "Show more" button
        clicked = False
        for btn_sel in [
            'button:has-text("Load more")',
            'button:has-text("Show more")',
            '[class*="load-more"]',
            '[class*="show-more"]',
        ]:
            try:
                btn = page.locator(btn_sel).first
                if await btn.is_visible(timeout=3_000):
                    await btn.click()
                    await page.wait_for_load_state('networkidle', timeout=8_000)
                    clicked = True
                    log.info(f"  Clicked '{btn_sel}'")
                    break
            except Exception:
                continue

        if not clicked:
            log.warning(f"  No 'Load more' button found — trying URL param ?page={page_num}")
            # Fallback: try URL-param pagination
            paginated_url = f"{base_url}?page={page_num}"
            await rate.wait('afl.com.au')
            new_articles = await fetch_news_listing(page, paginated_url)
            articles.extend(new_articles)
        else:
            # Re-extract all cards (new ones appended to DOM)
            html = await page.content()
            soup = BeautifulSoup(html, 'html.parser')
            all_hrefs = {a['url'] for a in articles}
            for el in soup.select('article a[href*="/news/"], a[href*="/news/"][class*="card"]'):
                href = el.get('href', '')
                if not href.startswith('http'):
                    href = f'https://www.afl.com.au{href}'
                if href not in all_hrefs and '/news/' in href:
                    articles.append({
                        'url': href,
                        'headline': el.get_text(strip=True),
                        'published_at': None,
                        'article_type': 'news',
                    })
                    all_hrefs.add(href)
            log.info(f"  Total articles after page {page_num}: {len(articles)}")

    return articles


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn     = init_db(DB_PATH)
    registry = PlayerRegistry.from_csv(PLAYER_LIST)
    rate     = SimpleRateLimiter(min_delay=4.0, jitter=2.0)
    now_iso  = datetime.now(timezone.utc).isoformat()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={'width': 1280, 'height': 900},
            locale='en-AU',
            extra_http_headers={
                'Accept-Language': 'en-AU,en;q=0.9',
                'Referer':         'https://www.afl.com.au/',
            },
        )
        page = await context.new_page()

        # ── Step 1: Collect article links (3 pages) ──
        log.info(f"=== Scraping {PAGES_TO_FETCH} pages from AFL.com.au/news ===")
        article_links = await paginate_listing(page, PAGE_URL, PAGES_TO_FETCH, rate)
        log.info(f"Total article links collected: {len(article_links)}")

        # ── Step 2: Fetch each article ──
        total_articles = 0
        total_mentions = 0

        for link in article_links:
            url = link['url']
            aid = article_id(url)

            # Skip if already in DB
            existing = conn.execute('SELECT id FROM articles WHERE id=?', (aid,)).fetchone()
            if existing:
                log.info(f"  Already scraped: {url[:60]}…")
                continue

            article_data = await fetch_article_body(page, url, rate)
            if not article_data:
                continue

            # ── Store article ──
            raw_headline = article_data['headline'] or link.get('headline', '') or ''
            if raw_headline.lower().strip() in _GENERIC_TITLES:
                raw_headline = link.get('headline', '') or _headline_from_url(url)
            db_article = {
                'id':           aid,
                'url':          url,
                'source':       'afl.com.au',
                'headline':     raw_headline,
                'full_text':    article_data['full_text'],
                'author':       article_data['author'],
                'published_at': article_data['published_at'],
                'scraped_at':   now_iso,
                'article_type': link.get('article_type', 'news'),
                'raw_html':     article_data['raw_html'],
            }
            is_new = insert_article(conn, db_article)
            if is_new:
                total_articles += 1

            # ── Step 3: Resolve bold strings against player registry ──
            bold_strings  = article_data.get('bold_strings', [])
            full_text     = article_data.get('full_text', '')

            log.info(f"  Bold candidates: {bold_strings[:5]}{'…' if len(bold_strings) > 5 else ''}")

            seen_in_article = set()   # avoid duplicate mentions per article
            for name_str in bold_strings:
                if name_str in seen_in_article:
                    continue
                seen_in_article.add(name_str)

                # Extract surrounding context (±120 chars)
                idx = full_text.find(name_str)
                context_window = full_text[max(0, idx-120): idx+len(name_str)+120] if idx >= 0 else ''

                feed_id, tier = registry.resolve(name_str, context=context_window)

                keyword_tags = extract_keyword_tags(context_window)
                sentiment = score_sentiment(context_window)

                mention = {
                    'article_id':       aid,
                    'feed_id':          feed_id,
                    'raw_name':         name_str,
                    'resolution_tier':  tier if tier > 0 else None,
                    'keyword_tags':     json.dumps(keyword_tags),
                    'sentiment_score':  sentiment,
                    'context_window':   context_window[:400],
                    'is_bold':          1,
                    'created_at':       now_iso,
                }
                insert_mention(conn, mention)
                total_mentions += 1

                resolution_label = {1: 'EXACT', 2: 'SURNAME', 3: 'FUZZY', 0: 'UNRESOLVED'}[tier]
                player_label = f"→ {registry.meta[feed_id]['last']}" if feed_id else '→ UNRESOLVED'
                log.info(f"    {name_str:<25} [{resolution_label}] {player_label} | tags={keyword_tags}")

        await browser.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SCRAPE COMPLETE")
    print("=" * 70)
    print(f"  Articles stored:   {total_articles}")
    print(f"  Player mentions:   {total_mentions}")

    # Quick query of results
    rows = conn.execute("""
        SELECT
            pm.raw_name,
            pm.feed_id,
            pm.resolution_tier,
            pm.keyword_tags,
            a.headline,
            a.published_at
        FROM player_mentions pm
        JOIN articles a ON pm.article_id = a.id
        WHERE pm.is_bold = 1
        ORDER BY a.published_at DESC
        LIMIT 20
    """).fetchall()

    if rows:
        print(f"\n{'Name':<25} {'FeedID':<10} {'Tier':<6} {'Tags':<30} {'Headline'}")
        print("-" * 100)
        for raw_name, feed_id, tier, tags, headline, pub in rows:
            tier_lbl = {1: 'EXACT', 2: 'SNAME', 3: 'FUZZY', None: 'UNRES'}.get(tier, '?')
            tags_str = ', '.join(json.loads(tags or '[]'))[:28]
            print(f"  {raw_name:<23} {str(feed_id or 'NULL'):<10} {tier_lbl:<6} {tags_str:<30} {str(headline)[:45]}")

    conn.close()
    print(f"\nDatabase: {DB_PATH.resolve()}")


def run_if_stale(max_age_hours: float = 20.0) -> bool:
    """
    Run the scraper only if the most recent article in the DB is older than
    max_age_hours. Returns True if the scraper ran, False if data was fresh.
    Safe to call from the pipeline — catches all errors and never raises.
    """
    try:
        import sqlite3 as _sqlite3
        from datetime import datetime, timedelta, timezone as _tz

        if DB_PATH.exists():
            conn = _sqlite3.connect(DB_PATH)
            row = conn.execute(
                "SELECT MAX(published_at) FROM articles"
            ).fetchone()
            conn.close()
            if row and row[0]:
                latest = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                age = datetime.now(_tz.utc) - latest
                if age < timedelta(hours=max_age_hours):
                    log.info(
                        "News DB is fresh (latest article %.1fh ago) — skipping scrape.",
                        age.total_seconds() / 3600,
                    )
                    return False

        log.info("News DB is stale — running scraper...")
        asyncio.run(run())
        return True
    except Exception as exc:
        log.warning("News scraper failed (non-fatal): %s", exc)
        return False


if __name__ == '__main__':
    asyncio.run(run())
