"""
ingest_afl_news_historical.py — Historical AFL news ingestion for 2021-2026

Fetches two types of articles:
  1. Medical Room (weekly AFL injury lists) — known URLs via Playwright
  2. General in-season news (fantasy, selection, form) — AFL content API
     (https://aflapi.afl.com.au/content/afl/text/EN/) — no Playwright needed,
     articles include full body HTML, pagination covers back to 2020+

Season offset anchors (API returns articles in reverse-chronological order):
  2021 season: offsets 27530 → 23925
  2022 season: offsets 22216 → 18541
  2023 season: offsets 17004 → 13568
  2024 season: offsets 11842 → 8271
  2025 season: offsets  6596 → 3058
  2026 season: offsets  1500 → 0  (estimated, current)

Usage:
    py ingest_afl_news_historical.py                    # all modes
    py ingest_afl_news_historical.py --mode medical     # injury lists only
    py ingest_afl_news_historical.py --mode general     # API-based general news
    py ingest_afl_news_historical.py --year 2023        # specific year (general mode)
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from afl_news_scraper import (
    DB_PATH, PLAYER_LIST, USER_AGENTS,
    PlayerRegistry, SimpleRateLimiter,
    init_db, article_id, insert_article, insert_mention,
    score_sentiment, extract_keyword_tags,
    _GENERIC_TITLES, _headline_from_url,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ── Known Medical Room article URLs (seed list from web search) ──────────────
# Article IDs are AFL.com.au sequential integers; slugs confirm content.

MEDICAL_ROOM_URLS = [
    # 2021
    "https://www.afl.com.au/news/563083/medical-room-the-full-afl-injury-list-r1",
    "https://www.afl.com.au/news/579121/medical-room-the-full-afl-injury-list-r4",
    "https://www.afl.com.au/news/598652/medical-room-the-full-afl-injury-list-r6",
    "https://www.afl.com.au/news/612200/medical-room-the-full-afl-injury-list-r9",
    "https://www.afl.com.au/news/616472/medical-room-the-full-afl-injury-list-r10",
    "https://www.afl.com.au/news/620731/medical-room-the-full-afl-injury-list-r11",
    "https://www.afl.com.au/news/624808/medical-room-the-full-afl-injury-list-r12",
    "https://www.afl.com.au/news/631969/medical-room-the-full-afl-injury-list-r14",
    "https://www.afl.com.au/news/659070/medical-room-the-full-afl-injury-list-r21",
    # 2022
    "https://www.afl.com.au/news/724977/medical-room-the-full-afl-injury-list-r2",
    "https://www.afl.com.au/news/745262/medical-room-the-full-afl-injury-list-r6",
    "https://www.afl.com.au/news/749948/medical-room-the-full-afl-injury-list-r7",
    "https://www.afl.com.au/news/772282/medical-room-the-full-afl-injury-list-r12",
    "https://www.afl.com.au/news/794258/medical-room-the-full-afl-injury-list-r17",
    "https://www.afl.com.au/news/798615/medical-room-the-full-afl-injury-list-r18",
    "https://www.afl.com.au/news/817362/medical-room-the-full-afl-injury-list-r22",
    # 2023
    "https://www.afl.com.au/news/881690/medical-room-the-full-afl-injury-list-r1",
    "https://www.afl.com.au/news/907509/medical-room-the-full-afl-injury-list-r6",
    "https://www.afl.com.au/news/917788/medical-room-the-full-afl-injury-list-r8",
    "https://www.afl.com.au/news/927432/medical-room-the-full-afl-injury-list-r10",
    "https://www.afl.com.au/news/940746/medical-room-the-full-afl-injury-list-r12",
    "https://www.afl.com.au/news/953771/medical-room-the-full-afl-injury-list-r15",
    "https://www.afl.com.au/news/992483/medical-room-the-full-afl-injury-list-r21",
    "https://www.afl.com.au/news/1009341/medical-room-the-full-afl-injury-list-r23",
    # 2024
    "https://www.afl.com.au/news/1110215/medical-room-the-full-afl-injury-list-r6",
    "https://www.afl.com.au/news/1130293/medical-room-the-full-afl-injury-list-r10",
    "https://www.afl.com.au/news/1179016/medical-room-the-full-afl-injury-list-r20",
    "https://www.afl.com.au/news/1184362/medical-room-the-full-afl-injury-list-r21",
    "https://www.afl.com.au/news/1195285/medical-room-the-full-afl-injury-list-r23",
    "https://www.afl.com.au/news/1229384/medical-room-the-full-afl-injury-list-gf",
    # 2025
    "https://www.afl.com.au/news/1310849/medical-room-the-full-afl-injury-list-r8",
    "https://www.afl.com.au/news/1371273/medical-room-the-full-afl-injury-list-r20",
    "https://www.afl.com.au/news/1403956/medical-room-the-full-afl-injury-list-fw1",
    "https://www.afl.com.au/news/1415911/medical-room-the-full-afl-injury-list-fw2",
    "https://www.afl.com.au/news/1428357/medical-room-the-full-afl-injury-list-gf",
    # 2026
    "https://www.afl.com.au/news/1471556/medical-room-the-full-afl-injury-list-or",
    "https://www.afl.com.au/news/1474771/medical-room-the-full-afl-injury-list-r1",
    "https://www.afl.com.au/news/1484108/medical-room-the-full-afl-injury-list-r3",
    "https://www.afl.com.au/news/1491910/medical-room-the-full-afl-injury-list-r5",
    "https://www.afl.com.au/news/1497939/medical-room-the-full-afl-injury-list-r6",
    "https://www.afl.com.au/news/1508056/medical-room-the-full-afl-injury-list-r8",
    "https://www.afl.com.au/news/1513224/medical-room-the-full-afl-injury-list-r9",
    "https://www.afl.com.au/news/1518239/medical-room-the-full-afl-injury-list-r10",
    "https://www.afl.com.au/news/1522891/medical-room-the-full-afl-injury-list-r11",
    "https://www.afl.com.au/news/1532695/medical-room-the-full-afl-injury-list-r13",
]

# ── AFL content API — season offset windows ──────────────────────────────────
# Offsets for AFL season (March-September) in the content API.
# API returns articles newest-first; higher offset = older articles.
CONTENT_API_BASE = "https://aflapi.afl.com.au/content/afl/text/EN/"

SEASON_OFFSETS = {
    2021: (27530, 23925),
    2022: (22216, 18541),
    2023: (17004, 13568),
    2024: (11842,  8271),
    2025: ( 6596,  3058),
    2026: ( 1800,     0),  # current season (estimated)
}

# Keywords that make an article relevant for fantasy/injury analysis
RELEVANT_TITLE_KEYWORDS = [
    "injury", "medical room", "team", "selection", "named", "whisper",
    "fantasy", "supercoach", "waiver", "trade", "form", "role",
    "managed", "ruled out", "test", "fitness", "return", "cleared",
    "midfielder", "forward", "defender", "ruck", "position",
    "stocktake", "buy", "sell", "upgrade", "guns",
]

def _is_relevant(article: dict) -> bool:
    """Return True if the article is likely useful for player analysis."""
    title = (article.get("title") or "").lower()
    desc  = (article.get("description") or "").lower()
    combined = title + " " + desc
    return any(kw in combined for kw in RELEVANT_TITLE_KEYWORDS)


# ── Article fetcher (reuses Playwright logic from afl_news_scraper) ───────────

async def fetch_article(page, url: str, rate: SimpleRateLimiter) -> dict | None:
    """Fetch a single AFL.com.au article. Returns None on failure."""
    await rate.wait("afl.com.au")
    log.info("  Fetching: %s", url[:90])
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
        if resp and resp.status == 404:
            log.debug("  404: %s", url)
            return None
    except PWTimeout:
        log.warning("  Timeout: %s", url[:80])
        return None
    except Exception as exc:
        log.warning("  Error fetching %s: %s", url[:80], exc)
        return None

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")

    # Headline
    headline = ""
    for sel in ["h1", '[class*="article-title"]', '[class*="headline"]']:
        el = soup.select_one(sel)
        if el:
            headline = el.get_text(strip=True)
            break
    if not headline or headline.lower().strip() in _GENERIC_TITLES:
        headline = _headline_from_url(url)

    # Published date
    published_at = None
    for sel in ['time[datetime]', '[class*="date"]', 'meta[property="article:published_time"]']:
        el = soup.select_one(sel)
        if el:
            published_at = el.get("datetime") or el.get("content") or el.get_text(strip=True)
            break

    # Body
    body_el = (
        soup.select_one('[class*="article-body"]')
        or soup.select_one('[class*="article-content"]')
        or soup.select_one("article")
        or soup.select_one("main")
    )
    if not body_el:
        return None

    full_text = body_el.get_text(separator=" ", strip=True)

    bold_strings = []
    for tag in body_el.find_all(["b", "strong"]):
        text = tag.get_text(strip=True)
        if 2 <= len(text.split()) <= 4 and re.match(r"^[A-Za-z\-' ]+$", text):
            bold_strings.append(text)

    return {
        "url": url,
        "headline": headline,
        "full_text": full_text,
        "published_at": published_at,
        "bold_strings": bold_strings,
        "raw_html": str(body_el)[:50_000],
        "author": "",
    }


def fetch_general_news_from_api(year: int, sample_every: int = 20) -> list[dict]:
    """
    Fetch relevant general news articles from the AFL content API for a given year.

    Uses pre-computed season offset anchors to target the AFL season window
    (March-September). Samples every `sample_every` articles for efficiency,
    then filters by title/description relevance keywords.

    Returns list of article dicts with keys: url, headline, full_text,
    published_at, article_type, author, raw_html, bold_strings.
    """
    import requests
    from bs4 import BeautifulSoup as BS4

    if year not in SEASON_OFFSETS:
        log.warning("No offset anchor for year %d", year)
        return []

    start_offset, end_offset = SEASON_OFFSETS[year]
    # Step equals batch_size to avoid overlapping fetches
    batch_size = sample_every
    log.info("[api] Fetching general news for %d (offsets %d → %d, batch=%d)",
             year, start_offset, end_offset, batch_size)

    results = []

    for offset in range(end_offset, start_offset, batch_size):
        try:
            r = requests.get(CONTENT_API_BASE, params={
                "tagExpression": '("News")',
                "sort": "publishDate:desc",
                "offset": offset,
                "limit": batch_size,
            }, timeout=20)
            items = r.json().get("content", [])
        except Exception as exc:
            log.warning("[api] Request failed at offset %d: %s", offset, exc)
            continue

        for item in items:
            if not _is_relevant(item):
                continue

            aid = item.get("id")
            slug = item.get("titleUrlSegment", "")
            if not aid or not slug:
                continue

            url = f"https://www.afl.com.au/news/{aid}/{slug}"
            title = item.get("title") or _headline_from_url(url)
            published_at = item.get("date")

            # Extract body text and bold strings from the HTML body
            body_html = item.get("body") or ""
            if not body_html:
                continue
            soup = BS4(body_html, "html.parser")
            full_text = soup.get_text(separator=" ", strip=True)
            bold_strings = []
            for tag in soup.find_all(["b", "strong"]):
                text = tag.get_text(strip=True)
                if 2 <= len(text.split()) <= 4 and re.match(r"^[A-Za-z\-' ]+$", text):
                    bold_strings.append(text)

            results.append({
                "url":          url,
                "headline":     title,
                "full_text":    full_text,
                "published_at": published_at,
                "article_type": "general",
                "author":       (item.get("author") or ""),
                "raw_html":     body_html[:50_000],
                "bold_strings": bold_strings,
            })

        # Advance by batch to avoid refetching same articles
        # (we already stepped by sample_every, just skip the rest of this batch)

    log.info("[api] Year %d: found %d relevant articles from %d offsets sampled",
             year, len(results), (start_offset - end_offset) // sample_every)
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _process_mentions(conn, aid: str, full_text: str, bold_strings: list,
                      registry, now_iso: str):
    """Extract and store player mentions from article text."""
    for name_str in set(bold_strings):
        idx_pos = full_text.find(name_str)
        context = full_text[max(0, idx_pos-120): idx_pos+len(name_str)+120] if idx_pos >= 0 else ""
        feed_id, tier = registry.resolve(name_str, context=context)
        insert_mention(conn, {
            "article_id":      aid,
            "feed_id":         feed_id,
            "raw_name":        name_str,
            "resolution_tier": tier if tier > 0 else None,
            "keyword_tags":    json.dumps(extract_keyword_tags(context)),
            "sentiment_score": score_sentiment(context),
            "context_window":  context[:400],
            "is_bold":         1,
            "created_at":      now_iso,
        })


async def run(mode: str = "all", year_filter: int | None = None):
    conn = init_db(DB_PATH)
    registry = PlayerRegistry.from_csv(PLAYER_LIST)
    rate = SimpleRateLimiter(min_delay=3.5, jitter=1.5)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Already-scraped URL set
    existing_urls = {
        row[0] for row in conn.execute("SELECT url FROM articles").fetchall()
    }
    log.info("DB already has %d articles — will skip duplicates.", len(existing_urls))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 900},
            locale="en-AU",
        )
        page = await ctx.new_page()

        urls_to_fetch: list[tuple[str, str]] = []  # (url, article_type)
        api_articles: list[dict] = []

        # ── Phase 1: Collect Medical Room URLs ───────────────────────────────
        if mode in ("all", "medical"):
            for url in MEDICAL_ROOM_URLS:
                if url not in existing_urls:
                    urls_to_fetch.append((url, "medical_room"))
            log.info("[medical] %d Medical Room URLs to fetch.", len([u for u, t in urls_to_fetch if t == "medical_room"]))

        # ── Phase 2: General news via AFL content API (no Playwright needed) ──
        if mode in ("all", "general"):
            years = [year_filter] if year_filter else list(range(2021, 2027))
            api_articles: list[dict] = []
            for yr in years:
                yr_articles = fetch_general_news_from_api(yr, sample_every=50)
                api_articles.extend(yr_articles)
                log.info("[api] %d: %d relevant articles found", yr, len(yr_articles))
            log.info("[api] Total general articles to process: %d", len(api_articles))

        # ── Phase 3a: Fetch Medical Room articles via Playwright ──────────────
        total_articles = 0
        total_mentions = 0
        skipped = 0

        log.info("=== Fetching %d Medical Room articles via Playwright ===", len(urls_to_fetch))

        for idx, (url, article_type) in enumerate(urls_to_fetch, 1):
            if url in existing_urls:
                skipped += 1
                continue

            aid = article_id(url)
            if conn.execute("SELECT id FROM articles WHERE id=?", (aid,)).fetchone():
                existing_urls.add(url)
                skipped += 1
                continue

            data = await fetch_article(page, url, rate)
            if not data:
                continue

            db_article = {
                "id":           aid,
                "url":          url,
                "source":       "afl.com.au",
                "headline":     data["headline"],
                "full_text":    data["full_text"],
                "author":       data["author"],
                "published_at": data["published_at"],
                "scraped_at":   now_iso,
                "article_type": article_type,
                "raw_html":     data["raw_html"],
            }
            is_new = insert_article(conn, db_article)
            if is_new:
                total_articles += 1
                existing_urls.add(url)

            _process_mentions(conn, aid, data.get("full_text", ""),
                              data.get("bold_strings", []), registry, now_iso)
            total_mentions += sum(1 for _ in data.get("bold_strings", []))

            if idx % 10 == 0:
                log.info("[%d/%d] Medical Room: stored=%d mentions=%d skipped=%d",
                         idx, len(urls_to_fetch), total_articles, total_mentions, skipped)

        # ── Phase 3b: Process API-fetched general articles (no extra fetch needed) ──
        if mode in ("all", "general"):
            log.info("=== Processing %d general articles from API ===", len(api_articles))
            for idx, data in enumerate(api_articles, 1):
                url = data["url"]
                if url in existing_urls:
                    skipped += 1
                    continue
                aid = article_id(url)
                if conn.execute("SELECT id FROM articles WHERE id=?", (aid,)).fetchone():
                    existing_urls.add(url)
                    skipped += 1
                    continue

                db_article = {
                    "id":           aid,
                    "url":          url,
                    "source":       "afl.com.au",
                    "headline":     data["headline"],
                    "full_text":    data["full_text"],
                    "author":       data["author"],
                    "published_at": data["published_at"],
                    "scraped_at":   now_iso,
                    "article_type": data["article_type"],
                    "raw_html":     data["raw_html"],
                }
                is_new = insert_article(conn, db_article)
                if is_new:
                    total_articles += 1
                    existing_urls.add(url)

                _process_mentions(conn, aid, data.get("full_text", ""),
                                  data.get("bold_strings", []), registry, now_iso)
                total_mentions += len(data.get("bold_strings", []))

                if idx % 50 == 0:
                    log.info("[%d/%d] General: stored=%d mentions=%d skipped=%d",
                             idx, len(api_articles), total_articles, total_mentions, skipped)

        await browser.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("HISTORICAL INGEST COMPLETE")
    print("=" * 65)
    print(f"  New articles stored:  {total_articles}")
    print(f"  New player mentions:  {total_mentions}")
    print(f"  Skipped (duplicate):  {skipped}")

    # Coverage by year
    rows = conn.execute("""
        SELECT strftime('%Y', published_at) as yr,
               article_type,
               COUNT(*) as n
        FROM articles
        WHERE article_type IN ('medical_room', 'general', 'fantasy', 'selection', 'injury', 'form', 'role_change', 'news')
        GROUP BY yr, article_type
        ORDER BY yr, article_type
    """).fetchall()
    if rows:
        print("\n  Coverage by year/type:")
        for yr, atype, n in rows:
            print(f"    {yr}  {atype:<15} {n:>4} articles")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "medical", "general", "search"],
                        default="all", help="What to scrape")
    parser.add_argument("--year", type=int, default=None,
                        help="Limit general search to a specific year")
    args = parser.parse_args()
    asyncio.run(run(mode=args.mode, year_filter=args.year))
