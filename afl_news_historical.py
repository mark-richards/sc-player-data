"""
AFL Historical News Scraper — Phase 1 (Extended)
=================================================
Three sources for pre-season windows (8 weeks → 2 weeks before Round 1):

  1. The Roar (AFL)  — WordPress REST API, all years 2016-2025
  2. Fox Sports AFL  — daily sitemaps (2022-2025) + Wayback CDX (2016-2021)
  3. AFL.com.au      — Wayback CDX (2016-2021, static HTML era)
                       + direct Playwright (2022-2025)

All articles stored in data/afl_news.db (deduplication by URL hash means
the script is safe to stop and restart at any time).

Usage:
  python afl_news_historical.py                      # all sources, all years
  python afl_news_historical.py --years 2024 2025    # specific years
  python afl_news_historical.py --source theroar     # single source
  python afl_news_historical.py --features-only      # rebuild CSV from DB
"""

import argparse
import json
import logging
import random
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from afl_news_scraper import (
    DB_PATH, PLAYER_LIST,
    PlayerRegistry, article_id, extract_keyword_tags,
    init_db, insert_article, insert_mention,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURES_PATH = Path('data/preseason_news_features.csv')

AFL_SEASON_STARTS = {
    2016: date(2016, 3, 24), 2017: date(2017, 3, 23),
    2018: date(2018, 3, 22), 2019: date(2019, 3, 21),
    2020: date(2020, 3, 19), 2021: date(2021, 3, 18),
    2022: date(2022, 3, 17), 2023: date(2023, 3, 16),
    2024: date(2024, 3, 14), 2025: date(2025, 3, 13),
    2026: date(2026, 3, 12),
}

# AFL.com.au switched to React SPA around 2019; Wayback snapshots of
# React pages rarely capture article body content.
AFL_COMAU_STATIC_CUTOFF = 2021   # use Wayback httpx for ≤ this year
AFL_COMAU_PLAYWRIGHT_MIN = 2022  # use Playwright for ≥ this year

NEWS_FEATURE_COLS = [
    'news_mention_count',    # articles mentioning player in pre-season window
    'news_injury_flag',      # injury keyword in context
    'news_positive_buzz',    # pre_season_buzz keyword in context
    'news_role_change_flag', # role_change keyword in context
    'news_concern_flag',     # injury + return combo (being managed)
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def preseason_window(year: int) -> tuple[date, date]:
    """8 weeks to 2 weeks before AFL Round 1."""
    start = AFL_SEASON_STARTS[year]
    return start - timedelta(weeks=8), start - timedelta(weeks=2)


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        headers={
            'User-Agent': random.choice(USER_AGENTS),
            'Accept-Language': 'en-AU,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,*/*;q=0.8',
        },
        follow_redirects=True,
        timeout=timeout,
    )


def polite_sleep(lo: float = 2.0, hi: float = 4.0):
    time.sleep(random.uniform(lo, hi))


def extract_article_body(html: str, source: str) -> tuple[str, list[str]]:
    """
    Extract (full_text, bold_strings) from article HTML.
    Returns ('', []) if body can't be found or is too short.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Source-specific selectors (tried in order)
    SELECTORS = {
        'afl.com.au':         ['[class*="article-body"]', '[class*="article-content"]', 'article'],
        'foxsports.com.au':   ['[class*="article-content"]', '[class*="story-body"]',
                               '.story-content', '[class*="storyContent"]', 'article'],
        'theroar.com.au':     ['.article-content', '.post-content', '.entry-content', 'article'],
    }
    source_key = next((k for k in SELECTORS if k in source), None)
    candidates = list(SELECTORS.get(source_key, [])) + ['[class*="content"]', 'article', 'main']

    body = None
    for sel in candidates:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 150:
            body = el
            break

    if not body:
        return '', []

    full_text = body.get_text(' ', strip=True)
    bold_strings = [
        t.get_text(strip=True) for t in body.find_all(['b', 'strong'])
        if 2 <= len(t.get_text().split()) <= 4
        and re.match(r'^[A-Za-z\- ]+$', t.get_text())
    ]
    return full_text, bold_strings


# ── Player name extraction ─────────────────────────────────────────────────────

_NAME_RE = re.compile(r'\b([A-Z][a-z\-]+\s+[A-Z][a-z\-]+(?:\s+[A-Z][a-z\-]+)?)\b')
_SKIP = {
    'Pre Season', 'Trade Period', 'Round One', 'Round Two', 'Round Three',
    'Grand Final', 'Gold Coast', 'Port Adelaide', 'Greater Western',
    'West Coast', 'North Melbourne', 'St Kilda', 'Western Bulldogs',
    'Brisbane Lions', 'South Melbourne', 'Eden Park', 'Marvel Stadium',
    'Victoria Park', 'AFL Club', 'January February', 'March April',
    'New South', 'New Year',
}


def extract_player_mentions_from_text(text: str, registry: PlayerRegistry) -> list[dict]:
    """Regex Title-Case scan → PlayerRegistry resolver."""
    candidates = list(set(_NAME_RE.findall(text)))
    mentions, seen = [], set()
    for cand in candidates:
        if cand in _SKIP:
            continue
        feed_id, tier = registry.resolve(cand, context=text[:500])
        if not feed_id or feed_id in seen or tier == 0:
            continue
        seen.add(feed_id)
        idx = text.find(cand)
        ctx = text[max(0, idx - 120): idx + len(cand) + 120]
        mentions.append({
            'raw_name': cand, 'feed_id': feed_id, 'tier': tier,
            'context': ctx[:400], 'tags': extract_keyword_tags(ctx),
        })
    return mentions


# ── The Roar ──────────────────────────────────────────────────────────────────

def _theroar_afl_cat(client: httpx.Client) -> Optional[int]:
    try:
        r = client.get('https://www.theroar.com.au/wp-json/wp/v2/categories',
                       params={'slug': 'afl', 'per_page': 1})
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]['id']
    except Exception as e:
        log.warning(f"The Roar category lookup failed: {e}")
    return None


def scrape_theroar(date_from: date, date_to: date) -> list[dict]:
    """WordPress REST API — returns full article text inline, no per-article fetches."""
    articles = []
    with make_client() as client:
        cat_id = _theroar_afl_cat(client)
        if not cat_id:
            log.warning("The Roar: could not get AFL category ID")
            return []

        page = 1
        while True:
            polite_sleep(1.5, 2.5)
            try:
                resp = client.get(
                    'https://www.theroar.com.au/wp-json/wp/v2/posts',
                    params={
                        'categories': cat_id,
                        'after':      f'{date_from.isoformat()}T00:00:00',
                        'before':     f'{date_to.isoformat()}T23:59:59',
                        'per_page':   100,
                        'page':       page,
                        '_fields':    'id,link,title,date,content',
                    },
                )
                if resp.status_code in (400, 404):
                    break
                posts = resp.json()
                if not isinstance(posts, list) or not posts:
                    break
                for post in posts:
                    content_html = post.get('content', {}).get('rendered', '')
                    full_text = BeautifulSoup(content_html, 'html.parser').get_text(' ', strip=True)
                    headline = BeautifulSoup(
                        post.get('title', {}).get('rendered', ''), 'html.parser'
                    ).get_text()
                    articles.append({
                        'url':          post.get('link', ''),
                        'headline':     headline,
                        'published_at': post.get('date', '')[:10],
                        'full_text':    full_text,
                        'author':       '',
                        'source':       'theroar.com.au',
                        'bold_strings': [],
                    })
                log.info(f"  [Roar] page {page}: {len(posts)} posts (total {len(articles)})")
                if len(posts) < 100:
                    break
                page += 1
            except Exception as e:
                log.warning(f"  [Roar] REST API error p{page}: {e}")
                break

    # HTML fallback if REST API returned nothing
    if not articles:
        log.info("[Roar] REST API empty — trying HTML category pagination...")
        articles = _theroar_html_fallback(date_from, date_to)

    return articles


def _theroar_html_fallback(date_from: date, date_to: date) -> list[dict]:
    stubs, page = [], 1
    with make_client() as client:
        while page <= 400:
            polite_sleep(2.0, 3.5)
            try:
                r = client.get(f'https://www.theroar.com.au/afl/page/{page}/')
                if r.status_code != 200:
                    break
            except Exception:
                break
            soup = BeautifulSoup(r.text, 'html.parser')
            in_range = older = 0
            for a in soup.select('article h2 a, article h3 a, .entry-title a'):
                href = a.get('href', '')
                m = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                if not m:
                    continue
                ad = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if ad > date_to:
                    continue
                if ad < date_from:
                    older += 1
                    continue
                in_range += 1
                stubs.append({'url': href, 'headline': a.get_text(strip=True),
                               'published_at': ad.isoformat()})
            if older > 0 and in_range == 0:
                break
            page += 1

        articles = []
        for stub in stubs:
            polite_sleep(2.0, 4.0)
            try:
                r = client.get(stub['url'])
                soup = BeautifulSoup(r.text, 'html.parser')
                body = (soup.select_one('.article-content') or
                        soup.select_one('.post-content') or
                        soup.select_one('.entry-content') or
                        soup.select_one('article'))
                if not body:
                    continue
                articles.append({
                    **stub,
                    'full_text':    body.get_text(' ', strip=True),
                    'author':       '',
                    'source':       'theroar.com.au',
                    'bold_strings': [],
                })
            except Exception as e:
                log.warning(f"  [Roar HTML] fetch error: {e}")
    return articles


# ── Wayback Machine helpers ────────────────────────────────────────────────────

def wayback_cdx_urls(
    url_pattern: str,
    date_from: date,
    date_to: date,
    client: httpx.Client,
    max_results: int = 300,
) -> list[tuple[str, str]]:
    """
    Query Wayback CDX API for archived URLs matching pattern in date range.
    Returns [(timestamp, original_url), ...].
    Uses collapse=original to get one snapshot per unique URL.
    """
    polite_sleep(1.5, 2.5)
    try:
        resp = client.get(
            'https://web.archive.org/cdx/search/cdx',
            params={
                'url':      url_pattern,
                'output':   'json',
                'from':     date_from.strftime('%Y%m%d'),
                'to':       date_to.strftime('%Y%m%d'),
                'fl':       'timestamp,original',
                'filter':   'statuscode:200',
                'collapse': 'original',
                'limit':    max_results,
            },
            timeout=45.0,
        )
        data = resp.json()
        if not data or len(data) < 2:
            return []
        # First row is the header ['timestamp', 'original']
        return [(row[0], row[1]) for row in data[1:]]
    except Exception as e:
        log.warning(f"  Wayback CDX error for {url_pattern}: {e}")
        return []


def fetch_wayback_html(timestamp: str, url: str, client: httpx.Client) -> Optional[str]:
    """
    Fetch archived article HTML from Wayback Machine.
    Uses the `id_` modifier to get raw original HTML without the Wayback nav bar.
    """
    wb_url = f'https://web.archive.org/web/{timestamp}id_/{url}'
    polite_sleep(2.0, 4.0)
    try:
        resp = client.get(wb_url, timeout=35.0)
        if resp.status_code == 200 and len(resp.text) > 500:
            return resp.text
    except Exception as e:
        log.debug(f"  Wayback fetch failed ({url[:50]}): {e}")
    return None


# ── Fox Sports AFL ─────────────────────────────────────────────────────────────

def _foxsports_urls_from_sitemaps(
    date_from: date, date_to: date, client: httpx.Client
) -> list[tuple[str, str]]:
    """
    Iterate Fox Sports daily sitemaps for each day in the window.
    Returns [(published_date_str, url), ...] for AFL articles only.
    """
    results = []
    d = date_from
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    while d <= date_to:
        polite_sleep(1.5, 2.5)
        sm_url = (f'https://www.foxsports.com.au/sitemap.xml'
                  f'?yyyy={d.year}&mm={d.month:02d}&dd={d.day:02d}')
        try:
            resp = client.get(sm_url, timeout=20.0)
            if resp.status_code != 200:
                d += timedelta(days=1)
                continue
            root = ElementTree.fromstring(resp.text)
            for url_el in root.findall('.//sm:url', ns):
                loc = url_el.findtext('sm:loc', namespaces=ns) or ''
                lastmod = url_el.findtext('sm:lastmod', namespaces=ns) or d.isoformat()
                if '/afl/' in loc and 'foxsports.com.au' in loc:
                    results.append((lastmod[:10], loc))
        except Exception as e:
            log.debug(f"  Fox Sports sitemap {d}: {e}")
        d += timedelta(days=1)

    log.info(f"  [Fox sitemap] {len(results)} AFL URLs in window")
    return results


def _fetch_foxsports_article(url: str, client: httpx.Client) -> Optional[dict]:
    """Fetch a Fox Sports article via httpx (they server-render for SEO)."""
    polite_sleep(2.0, 3.5)
    try:
        resp = client.get(url, timeout=25.0)
        if resp.status_code != 200:
            return None
        full_text, bold_strings = extract_article_body(resp.text, 'foxsports.com.au')
        if not full_text or len(full_text) < 150:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        headline = ''
        for sel in ['h1', '[class*="headline"]', '[class*="article-title"]']:
            el = soup.select_one(sel)
            if el:
                headline = el.get_text(strip=True)
                break

        pub = soup.select_one('time[datetime]')
        published_at = pub.get('datetime', '')[:10] if pub else ''

        return {
            'url':          url,
            'headline':     headline,
            'published_at': published_at,
            'full_text':    full_text,
            'bold_strings': bold_strings,
            'author':       '',
            'source':       'foxsports.com.au',
        }
    except Exception as e:
        log.debug(f"  Fox Sports fetch error ({url[:50]}): {e}")
        return None


def scrape_foxsports(date_from: date, date_to: date, year: int) -> list[dict]:
    """
    2022-2025: Fox Sports daily sitemaps → direct httpx article fetch.
    2016-2021: Wayback CDX → Wayback article fetch (httpx).
    """
    articles = []
    with make_client() as client:
        if year >= 2022:
            log.info(f"[Fox] Year {year}: daily sitemaps")
            url_date_pairs = _foxsports_urls_from_sitemaps(date_from, date_to, client)
            for pub_date, url in url_date_pairs:
                aid = article_id(url)
                art = _fetch_foxsports_article(url, client)
                if art:
                    if not art.get('published_at'):
                        art['published_at'] = pub_date
                    articles.append(art)
                    log.debug(f"  [Fox] OK: {url[:70]}")
        else:
            log.info(f"[Fox] Year {year}: Wayback CDX")
            ts_urls = wayback_cdx_urls(
                'www.foxsports.com.au/afl/*', date_from, date_to, client, max_results=300
            )
            log.info(f"  [Fox Wayback] {len(ts_urls)} URLs from CDX")
            for ts, url in ts_urls:
                html = fetch_wayback_html(ts, url, client)
                if not html:
                    continue
                full_text, bold_strings = extract_article_body(html, 'foxsports.com.au')
                if not full_text or len(full_text) < 150:
                    continue
                soup = BeautifulSoup(html, 'html.parser')
                headline = ''
                for sel in ['h1', '[class*="headline"]', 'title']:
                    el = soup.select_one(sel)
                    if el:
                        headline = el.get_text(strip=True)
                        break
                articles.append({
                    'url':          url,
                    'headline':     headline,
                    'published_at': f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}',
                    'full_text':    full_text,
                    'bold_strings': bold_strings,
                    'author':       '',
                    'source':       'foxsports.com.au',
                })

    log.info(f"[Fox] Year {year}: {len(articles)} articles retrieved")
    return articles


# ── AFL.com.au ────────────────────────────────────────────────────────────────

def _fetch_afl_comau_playwright(urls: list[str]) -> list[dict]:
    """Fetch AFL.com.au articles via Playwright sync API (React SPA requires JS)."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    articles = []
    if not urls:
        return []

    log.info(f"  [AFL Playwright] fetching {len(urls)} articles...")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale='en-AU',
            extra_http_headers={'Accept-Language': 'en-AU,en;q=0.9'},
        )
        page = ctx.new_page()

        for url in urls:
            time.sleep(random.uniform(3.0, 5.0))
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=25_000)
                html = page.content()
                full_text, bold_strings = extract_article_body(html, 'afl.com.au')
                if not full_text or len(full_text) < 150:
                    continue

                soup = BeautifulSoup(html, 'html.parser')
                headline = ''
                for sel in ['h1', '[class*="headline"]', '[class*="article-title"]']:
                    el = soup.select_one(sel)
                    if el:
                        headline = el.get_text(strip=True)
                        break
                pub = soup.select_one('time[datetime]')
                published_at = pub.get('datetime', '')[:10] if pub else ''

                articles.append({
                    'url': url, 'headline': headline, 'published_at': published_at,
                    'full_text': full_text, 'bold_strings': bold_strings,
                    'author': '', 'source': 'afl.com.au',
                })
                log.debug(f"  [AFL PW] OK: {url[:70]}")
            except PWTimeout:
                log.debug(f"  [AFL PW] timeout: {url[:60]}")
            except Exception as e:
                log.debug(f"  [AFL PW] error: {e}")

        browser.close()
    return articles


def scrape_afl_comau(date_from: date, date_to: date, year: int) -> list[dict]:
    """
    AFL.com.au via Wayback CDX for all years.
    2016-2018 (static HTML): httpx fetch from Wayback.
    2019-2021 (early React): Wayback httpx, accept lower yield.
    2022-2025 (React SPA): Wayback CDX to get URLs, Playwright to fetch live.
    """
    articles = []
    with make_client() as client:
        ts_urls = wayback_cdx_urls(
            'www.afl.com.au/news/*', date_from, date_to, client, max_results=200
        )
        log.info(f"  [AFL CDX] {len(ts_urls)} URLs for {year}")

        if not ts_urls:
            return []

        if year <= AFL_COMAU_STATIC_CUTOFF:
            # Fetch archived HTML via Wayback (static era — body should be in HTML)
            for ts, url in ts_urls:
                html = fetch_wayback_html(ts, url, client)
                if not html:
                    continue
                full_text, bold_strings = extract_article_body(html, 'afl.com.au')
                if not full_text or len(full_text) < 150:
                    continue
                soup = BeautifulSoup(html, 'html.parser')
                headline = ''
                for sel in ['h1', '[class*="headline"]', '[class*="article-title"]', 'title']:
                    el = soup.select_one(sel)
                    if el:
                        headline = el.get_text(strip=True)[:200]
                        break
                articles.append({
                    'url': url, 'headline': headline,
                    'published_at': f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}',
                    'full_text': full_text, 'bold_strings': bold_strings,
                    'author': '', 'source': 'afl.com.au',
                })
        else:
            # React SPA era: fetch live URLs via Playwright
            live_urls = [url for _, url in ts_urls]
            articles = _fetch_afl_comau_playwright(live_urls[:80])  # cap per year

    log.info(f"[AFL.com.au] Year {year}: {len(articles)} articles retrieved")
    return articles


# ── Store articles + extract mentions ─────────────────────────────────────────

def process_and_store(
    raw_articles: list[dict],
    registry: PlayerRegistry,
    conn: sqlite3.Connection,
    year: int,
    source: str,
) -> tuple[int, int]:
    """Insert articles and extract/store player mentions. Returns (new_arts, new_ments)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    new_arts = new_ments = skipped = 0

    for art in raw_articles:
        url = art.get('url', '')
        if not url:
            continue
        aid = article_id(url)

        if conn.execute('SELECT 1 FROM articles WHERE id=?', (aid,)).fetchone():
            skipped += 1
            continue

        if insert_article(conn, {
            'id': aid, 'url': url, 'source': art.get('source', source),
            'headline': art.get('headline', ''), 'full_text': art.get('full_text', ''),
            'author': art.get('author', ''), 'published_at': art.get('published_at', ''),
            'scraped_at': now_iso, 'article_type': 'news', 'raw_html': None,
        }):
            new_arts += 1

        text = art.get('full_text', '')
        bold = art.get('bold_strings', [])
        seen: set[str] = set()
        mention_list = []

        # Bold strings first (higher precision)
        for name_str in bold:
            fid, tier = registry.resolve(name_str, context=text[:500])
            if fid and fid not in seen:
                seen.add(fid)
                idx = text.find(name_str)
                ctx = text[max(0, idx - 120): idx + len(name_str) + 120]
                mention_list.append({
                    'raw_name': name_str, 'feed_id': fid, 'tier': tier,
                    'context': ctx[:400], 'tags': extract_keyword_tags(ctx), 'is_bold': 1,
                })

        # Full-text scan for articles without reliable bold tags
        if not mention_list:
            for m in extract_player_mentions_from_text(text, registry):
                if m['feed_id'] not in seen:
                    seen.add(m['feed_id'])
                    m['is_bold'] = 0
                    mention_list.append(m)

        for m in mention_list:
            insert_mention(conn, {
                'article_id': aid, 'feed_id': m['feed_id'], 'raw_name': m['raw_name'],
                'resolution_tier': m['tier'], 'keyword_tags': json.dumps(m['tags']),
                'context_window': m['context'], 'is_bold': m.get('is_bold', 0),
                'created_at': now_iso,
            })
            new_ments += 1

    if skipped:
        log.debug(f"  {skipped} articles already in DB (skipped)")
    return new_arts, new_ments


# ── Feature aggregation ────────────────────────────────────────────────────────

def build_preseason_news_features(year: int, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Aggregate player_mentions for year's pre-season window into feature columns.
    Returns DataFrame: feed_id (int) + NEWS_FEATURE_COLS.
    """
    if not db_path.exists():
        return pd.DataFrame(columns=['feed_id'] + NEWS_FEATURE_COLS)

    date_from, date_to = preseason_window(year)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("""
            SELECT
                pm.feed_id,
                COUNT(DISTINCT a.id)                                     AS news_mention_count,
                MAX(CASE WHEN pm.keyword_tags LIKE '%injury%'
                    THEN 1 ELSE 0 END)                                   AS news_injury_flag,
                MAX(CASE WHEN pm.keyword_tags LIKE '%pre_season_buzz%'
                    THEN 1 ELSE 0 END)                                   AS news_positive_buzz,
                MAX(CASE WHEN pm.keyword_tags LIKE '%role_change%'
                    THEN 1 ELSE 0 END)                                   AS news_role_change_flag,
                MAX(CASE WHEN pm.keyword_tags LIKE '%injury%'
                         AND pm.keyword_tags LIKE '%return%'
                    THEN 1 ELSE 0 END)                                   AS news_concern_flag
            FROM player_mentions pm
            JOIN articles a ON pm.article_id = a.id
            WHERE pm.feed_id IS NOT NULL
              AND a.published_at >= ?
              AND a.published_at <= ?
            GROUP BY pm.feed_id
        """, conn, params=[date_from.isoformat(), date_to.isoformat()])
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=['feed_id'] + NEWS_FEATURE_COLS)

    df['feed_id'] = pd.to_numeric(df['feed_id'], errors='coerce')
    df = df.dropna(subset=['feed_id'])
    df['feed_id'] = df['feed_id'].astype(int)
    return df


def rebuild_features_csv(years: list[int] | None = None) -> pd.DataFrame:
    """Rebuild data/preseason_news_features.csv from current DB state."""
    if years is None:
        years = sorted(AFL_SEASON_STARTS.keys())

    all_dfs = []
    log.info("Pre-season news feature summary:")
    for year in years:
        feats = build_preseason_news_features(year)
        if not feats.empty:
            feats = feats.copy()
            feats['year'] = year
            all_dfs.append(feats)
            log.info(
                f"  {year}: {len(feats):3d} players | "
                f"injury={int(feats['news_injury_flag'].sum())}  "
                f"buzz={int(feats['news_positive_buzz'].sum())}  "
                f"role={int(feats['news_role_change_flag'].sum())}  "
                f"concern={int(feats['news_concern_flag'].sum())}"
            )
        else:
            log.info(f"  {year}: no data")

    if not all_dfs:
        return pd.DataFrame(columns=['feed_id', 'year'] + NEWS_FEATURE_COLS)

    combined = pd.concat(all_dfs, ignore_index=True)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FEATURES_PATH, index=False)
    log.info(f"Saved {FEATURES_PATH} ({len(combined)} rows, {len(all_dfs)} years)")
    return combined


# ── Per-year orchestrator ──────────────────────────────────────────────────────

def run_year(
    year: int,
    registry: PlayerRegistry,
    conn: sqlite3.Connection,
    sources: list[str],
):
    date_from, date_to = preseason_window(year)
    log.info(f"\n{'='*65}")
    log.info(f"Year {year}  |  window: {date_from} → {date_to}")
    log.info(f"{'='*65}")

    total_arts = total_ments = 0

    if 'theroar' in sources:
        articles = scrape_theroar(date_from, date_to)
        a, m = process_and_store(articles, registry, conn, year, 'theroar.com.au')
        log.info(f"  [Roar]   stored {a} new articles, {m} mentions")
        total_arts += a; total_ments += m

    if 'foxsports' in sources:
        articles = scrape_foxsports(date_from, date_to, year)
        a, m = process_and_store(articles, registry, conn, year, 'foxsports.com.au')
        log.info(f"  [Fox]    stored {a} new articles, {m} mentions")
        total_arts += a; total_ments += m

    if 'afl' in sources:
        articles = scrape_afl_comau(date_from, date_to, year)
        a, m = process_and_store(articles, registry, conn, year, 'afl.com.au')
        log.info(f"  [AFL]    stored {a} new articles, {m} mentions")
        total_arts += a; total_ments += m

    log.info(f"  Year {year} total: {total_arts} new articles, {total_ments} new mentions")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AFL Historical News Scraper')
    parser.add_argument(
        '--years', nargs='+', type=int,
        default=list(range(2016, 2026)),
        help='Years to scrape (default: 2016-2025)',
    )
    parser.add_argument(
        '--source', choices=['theroar', 'foxsports', 'afl', 'all'], default='all',
        help='Which source to scrape (default: all)',
    )
    parser.add_argument(
        '--features-only', action='store_true',
        help='Skip scraping; just rebuild features CSV from existing DB',
    )
    args = parser.parse_args()

    sources = ['theroar', 'foxsports', 'afl'] if args.source == 'all' else [args.source]

    conn     = init_db(DB_PATH)
    registry = PlayerRegistry.from_csv(PLAYER_LIST)

    if not args.features_only:
        log.info(f"Sources: {sources}  |  Years: {args.years}")
        for year in args.years:
            run_year(year, registry, conn, sources)
        log.info("\nAll years complete.")

    conn.close()

    log.info("\nRebuilding pre-season news features CSV...")
    features = rebuild_features_csv()

    if not features.empty:
        years_with_data = features[features['news_mention_count'] > 0]['year'].nunique()
        flagged = features[
            (features['news_injury_flag']      == 1) |
            (features['news_positive_buzz']    == 1) |
            (features['news_role_change_flag'] == 1)
        ].sort_values(['year', 'news_mention_count'], ascending=[True, False])

        print(f"\n{'='*65}")
        print(f"FLAGGED PLAYERS  ({len(flagged)} across {years_with_data} years with data)")
        print(f"{'='*65}")
        print(f"{'feed_id':<12} {'year':<6} {'mentions':<10} {'inj':<5} {'buzz':<6} {'role':<6} concern")
        print("-" * 60)
        for _, row in flagged.head(50).iterrows():
            print(
                f"{int(row['feed_id']):<12} {int(row['year']):<6} "
                f"{int(row['news_mention_count']):<10} "
                f"{int(row['news_injury_flag']):<5} "
                f"{int(row['news_positive_buzz']):<6} "
                f"{int(row['news_role_change_flag']):<6} "
                f"{int(row['news_concern_flag'])}"
            )


if __name__ == '__main__':
    main()
