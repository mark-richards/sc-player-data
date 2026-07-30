"""
Scrape AFL.com.au news directly via Playwright for 2026 pre-season window.
CDX/Wayback won't work for the current year, so we browse the live site.

Targets:
  - https://www.afl.com.au/news  (general news, filtered by date)
  - https://www.afl.com.au/news?tagNames=injury-updates  (injury tag)

Results stored in data/afl_news.db, then features CSV rebuilt.
"""

import json
import logging
import random
import sqlite3
import time
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from afl_news_scraper import (
    DB_PATH, PLAYER_LIST,
    PlayerRegistry, article_id, extract_keyword_tags,
    init_db, insert_article, insert_mention,
)
from afl_news_historical import (
    extract_article_body, extract_player_mentions_from_text,
    process_and_store, rebuild_features_csv, preseason_window,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

WINDOW_FROM, WINDOW_TO = preseason_window(2026)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15',
]

AFL_NEWS_URLS = [
    # General news listing — will filter by date
    'https://www.afl.com.au/news',
    # Injury-specific tag page
    'https://www.afl.com.au/news?tagNames=injury-updates',
    # Club injury reports (pre-season is peak injury report time)
    'https://www.afl.com.au/news?tagNames=injury-report',
    'https://www.afl.com.au/news?tagNames=pre-season',
]


def scrape_afl_news_2026() -> list[dict]:
    """Browse AFL.com.au news listing via Playwright and collect pre-season articles."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    articles = []
    seen_urls = set()

    log.info(f"AFL.com.au Playwright scrape | window: {WINDOW_FROM} to {WINDOW_TO}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale='en-AU',
            extra_http_headers={'Accept-Language': 'en-AU,en;q=0.9'},
        )
        page = ctx.new_page()

        for listing_url in AFL_NEWS_URLS:
            log.info(f"  Navigating to: {listing_url}")
            try:
                page.goto(listing_url, wait_until='domcontentloaded', timeout=30_000)
                time.sleep(3)
            except Exception as e:
                log.warning(f"  Failed to load listing: {e}")
                continue

            # Scroll and collect article links — stop when we hit articles older than window
            article_links = []
            prev_count = -1
            scroll_attempts = 0

            while scroll_attempts < 15:
                # Collect all news article links on current page
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')

                new_links = []
                # AFL.com.au article link patterns
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if '/news/' in href and 'afl.com.au' not in href:
                        full = f'https://www.afl.com.au{href}' if href.startswith('/') else href
                        if full not in seen_urls and '/news/' in full:
                            new_links.append(full)

                article_links.extend(new_links)
                article_links = list(dict.fromkeys(article_links))  # deduplicate

                if len(article_links) == prev_count:
                    break  # no new links found after scroll
                prev_count = len(article_links)

                # Try to scroll down or click "load more"
                try:
                    load_more = page.query_selector('button:has-text("Load more"), button:has-text("Show more"), [class*="load-more"]')
                    if load_more:
                        load_more.click()
                        time.sleep(2)
                    else:
                        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                        time.sleep(2)
                except Exception:
                    break
                scroll_attempts += 1

            log.info(f"  Found {len(article_links)} article links from {listing_url}")

            # Fetch each article
            for url in article_links:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                time.sleep(random.uniform(2.0, 4.0))

                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=25_000)
                    html = page.content()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Check publish date
                    pub_el = soup.select_one('time[datetime], [class*="date"], [class*="published"]')
                    pub_date = ''
                    if pub_el:
                        dt = pub_el.get('datetime', '') or pub_el.get_text(strip=True)
                        pub_date = dt[:10] if dt else ''

                    # Filter by window (skip if date known and out of range)
                    if pub_date and (pub_date < str(WINDOW_FROM) or pub_date > str(WINDOW_TO)):
                        log.debug(f"  Out of window ({pub_date}): {url[:60]}")
                        continue

                    full_text, bold_strings = extract_article_body(html, 'afl.com.au')
                    if not full_text or len(full_text) < 150:
                        continue

                    headline = ''
                    for sel in ['h1', '[class*="headline"]', '[class*="article-title"]']:
                        el = soup.select_one(sel)
                        if el:
                            headline = el.get_text(strip=True)[:200]
                            break

                    articles.append({
                        'url': url,
                        'headline': headline,
                        'published_at': pub_date,
                        'full_text': full_text,
                        'bold_strings': bold_strings,
                        'author': '',
                        'source': 'afl.com.au',
                    })
                    log.info(f"  OK [{pub_date}]: {headline[:70]}")

                except PWTimeout:
                    log.debug(f"  Timeout: {url[:60]}")
                except Exception as e:
                    log.debug(f"  Error ({url[:50]}): {e}")

        browser.close()

    log.info(f"AFL.com.au: {len(articles)} articles fetched")
    return articles


def main():
    conn     = init_db(DB_PATH)
    registry = PlayerRegistry.from_csv(PLAYER_LIST)

    articles = scrape_afl_news_2026()
    new_arts, new_ments = process_and_store(articles, registry, conn, 2026, 'afl.com.au')
    log.info(f"Stored: {new_arts} new articles, {new_ments} new mentions")

    conn.close()

    log.info("Rebuilding features CSV...")
    rebuild_features_csv(years=[2026])


if __name__ == '__main__':
    main()
