"""Financial news ingestion pipeline.

Fetches news from NewsAPI and RSS feeds, normalizes with Polars,
and persists to PostgreSQL for later RAG embedding.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import feedparser
import polars as pl
import requests  # type: ignore[import-untyped]
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import Engine, text

from src.utils.db import get_postgres_engine

NEWS_TABLE: str = "financial_news"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {NEWS_TABLE} (
    id               SERIAL PRIMARY KEY,
    date             DATE         NOT NULL,
    ticker           VARCHAR(10),
    title            TEXT         NOT NULL,
    source           VARCHAR(100) NOT NULL,
    url              TEXT,
    summary          TEXT,
    embedding_status VARCHAR(20)  NOT NULL DEFAULT 'pending',
    UNIQUE (url)
);
"""

RSS_FEEDS: dict[str, str] = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Google Finance": (
        "https://news.google.com/rss/search"
        "?q=stock+market+finance&hl=en-US&gl=US&ceid=US:en"
    ),
    "CNBC": (
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=100003114"
    ),
}

# Individual stocks FIRST so _match_ticker tags the specific company,
# not a broad index.  Indices/ETFs at the end.
TICKER_KEYWORDS: dict[str, list[str]] = {
    # ── Technology ─────────────────────────────────────────────
    "AAPL": ["AAPL", "Apple", "iPhone", "Tim Cook"],
    "MSFT": ["MSFT", "Microsoft", "Azure", "Satya Nadella"],
    "GOOG": ["GOOG", "Alphabet", "Google", "Sundar Pichai"],
    "AMZN": ["AMZN", "Amazon", "AWS", "Andy Jassy"],
    "META": ["META", "Meta Platforms", "Facebook", "Mark Zuckerberg", "Instagram"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA", "Jensen Huang"],
    "TSLA": ["TSLA", "Tesla", "Elon Musk"],
    "AVGO": ["AVGO", "Broadcom"],
    "CRM": ["Salesforce", "Marc Benioff"],
    "AMD": ["AMD", "Advanced Micro Devices", "Lisa Su"],
    # ── Financials ─────────────────────────────────────────────
    "JPM": ["JPM", "JPMorgan", "JP Morgan", "Jamie Dimon"],
    "BAC": ["Bank of America", "BofA", "BAC"],
    "GS": ["Goldman Sachs", "David Solomon", "Goldman"],
    "MS": ["Morgan Stanley", "James Gorman"],
    "WFC": ["WFC", "Wells Fargo"],
    "BLK": ["BlackRock", "Larry Fink"],
    "AXP": ["AXP", "American Express", "Amex"],
    "C": ["Citigroup", "Citibank", "Jane Fraser", "Citi"],
    # ── Healthcare ─────────────────────────────────────────────
    "UNH": ["UNH", "UnitedHealth"],
    "JNJ": ["JNJ", "Johnson & Johnson", "Johnson and Johnson", "J&J"],
    "LLY": ["Eli Lilly", "Lilly"],
    "PFE": ["PFE", "Pfizer"],
    "ABBV": ["ABBV", "AbbVie"],
    "MRK": ["MRK", "Merck"],
    "TMO": ["TMO", "Thermo Fisher"],
    # ── Consumer ───────────────────────────────────────────────
    "PG": ["Procter & Gamble", "Procter and Gamble", "P&G"],
    "KO": ["Coca-Cola", "Coca Cola"],
    "PEP": ["PEP", "PepsiCo", "Pepsi"],
    "COST": ["Costco"],
    "WMT": ["WMT", "Walmart"],
    "HD": ["Home Depot"],
    "MCD": ["MCD", "McDonald's", "McDonalds"],
    # ── Energy ─────────────────────────────────────────────────
    "XOM": ["XOM", "ExxonMobil", "Exxon Mobil", "Exxon"],
    "CVX": ["CVX", "Chevron"],
    "COP": ["ConocoPhillips", "Conoco Phillips"],
    "SLB": ["SLB", "Schlumberger"],
    # ── Industrials ────────────────────────────────────────────
    "CAT": ["Caterpillar"],
    "HON": ["Honeywell"],
    "UPS": ["United Parcel"],
    "BA": ["Boeing"],
    "GE": ["GE Aerospace", "General Electric"],
    "RTX": ["RTX", "Raytheon"],
    # ── Utilities & REITs ──────────────────────────────────────
    "NEE": ["NextEra Energy", "NextEra"],
    "DUK": ["Duke Energy"],
    "AMT": ["AMT", "American Tower"],
    "PLD": ["PLD", "Prologis"],
    # ── Communication ──────────────────────────────────────────
    "DIS": ["Disney", "Walt Disney"],
    "NFLX": ["NFLX", "Netflix"],
    "CMCSA": ["CMCSA", "Comcast"],
    # ── Materials ──────────────────────────────────────────────
    "LIN": ["Linde"],
    "APD": ["APD", "Air Products"],
    "NEM": ["NEM", "Newmont"],
    # ── Indices / ETFs (last — so specific stocks match first) ─
    "SPY": ["S&P 500", "S&P500", "SPY"],
}

# Primary company name for per-ticker GDELT queries.
_TICKER_COMPANY: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "CRM": "Salesforce",
    "AMD": "AMD",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "WFC": "Wells Fargo",
    "BLK": "BlackRock",
    "AXP": "American Express",
    "C": "Citigroup",
    "UNH": "UnitedHealth",
    "JNJ": "Johnson Johnson",
    "LLY": "Eli Lilly",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "TMO": "Thermo Fisher",
    "PG": "Procter Gamble",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "COST": "Costco",
    "WMT": "Walmart",
    "HD": "Home Depot",
    "MCD": "McDonald",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "SLB": "Schlumberger",
    "CAT": "Caterpillar",
    "HON": "Honeywell",
    "UPS": "UPS",
    "BA": "Boeing",
    "GE": "GE Aerospace",
    "RTX": "Raytheon",
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "AMT": "American Tower",
    "PLD": "Prologis",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "CMCSA": "Comcast",
    "LIN": "Linde",
    "APD": "Air Products",
    "NEM": "Newmont",
}

_GDELT_BASE_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_DELAY: float = 6.0  # seconds between requests (API limit: 1 req/5s)

# Sector-based queries for NewsAPI (covers all 52 tickers)
_NEWSAPI_QUERIES: list[str] = [
    "stock market Wall Street",
    "S&P 500 SPY",
    "Apple Microsoft Google Alphabet",
    "Amazon Meta Facebook Nvidia",
    "Tesla AMD Broadcom Salesforce",
    "JPMorgan Goldman Sachs Morgan Stanley",
    "Bank of America Wells Fargo BlackRock Citigroup",
    "UnitedHealth Pfizer Eli Lilly AbbVie Merck",
    "Walmart Costco Home Depot McDonald's PepsiCo",
    "ExxonMobil Chevron Boeing Caterpillar",
    "Disney Netflix Comcast",
    "NextEra Duke Energy Prologis Newmont",
]


def _match_ticker(title: str, summary: str) -> str | None:
    """Match a news article to a ticker based on keyword presence.

    Uses keyword-hit counting so that the ticker with the most
    evidence wins, reducing positional bias from dict iteration
    order.  Ties are broken by ``TICKER_KEYWORDS`` insertion order
    (individual stocks before indices).

    Args:
        title: Article title.
        summary: Article summary text.

    Returns:
        Matched ticker symbol or None if no match.
    """
    combined = f"{title} {summary}".upper()
    best_ticker: str | None = None
    best_count = 0
    for ticker, keywords in TICKER_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.upper() in combined)
        if count > best_count:
            best_count = count
            best_ticker = ticker
    return best_ticker


def _clean_html(raw: str) -> str:
    """Strip HTML tags and return plain text.

    Args:
        raw: Raw HTML string.

    Returns:
        Cleaned plain text.
    """
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _truncate(text_str: str, max_len: int = 500) -> str:
    """Truncate text to max_len characters.

    Args:
        text_str: Input text.
        max_len: Maximum character length.

    Returns:
        Truncated text.
    """
    if len(text_str) <= max_len:
        return text_str
    return text_str[:max_len].rsplit(" ", 1)[0] + "..."


class NewsIngester:
    """Fetches financial news from NewsAPI and RSS feeds.

    Args:
        engine: SQLAlchemy engine. If None, creates one from env vars.
        max_retries: Max attempts per source on network failure.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        max_retries: int = 3,
    ) -> None:
        load_dotenv()
        self.engine = engine or get_postgres_engine()
        self.max_retries = max_retries
        self._newsapi_key = os.getenv("NEWSAPI_KEY", "")

    def _ensure_table(self) -> None:
        """Create the news table if it doesn't exist."""
        with self.engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
        logger.info("Table '{}' ready", NEWS_TABLE)

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------

    def _fetch_newsapi(self) -> list[dict[str, str | None]]:
        """Fetch articles from NewsAPI for tracked tickers.

        Returns:
            List of normalized article dicts.
        """
        if not self._newsapi_key:
            logger.warning("NEWSAPI_KEY not set, skipping NewsAPI source")
            return []

        articles: list[dict[str, str | None]] = []

        for query in _NEWSAPI_QUERIES:
            for attempt in range(1, self.max_retries + 1):
                try:
                    newsapi_params: dict[str, str | int] = {
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                        "apiKey": self._newsapi_key,
                    }
                    resp = requests.get(
                        "https://newsapi.org/v2/everything",
                        params=newsapi_params,
                        timeout=15,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    if data.get("status") != "ok":
                        logger.warning(
                            "NewsAPI returned status={} for query '{}'",
                            data.get("status"),
                            query,
                        )
                        break

                    for art in data.get("articles", []):
                        raw_date = art.get("publishedAt", "")
                        parsed_date = _parse_date(raw_date)
                        title = art.get("title", "") or ""
                        description = art.get("description", "") or ""
                        content = art.get("content", "") or ""
                        summary_text = _truncate(
                            _clean_html(description or content)
                        )

                        articles.append({
                            "date": str(parsed_date) if parsed_date else None,
                            "ticker": _match_ticker(title, summary_text),
                            "title": title,
                            "source": (
                            f"NewsAPI/"
                            f"{art.get('source', {}).get('name', 'Unknown')}"
                        ),
                            "url": art.get("url"),
                            "summary": summary_text,
                        })

                    logger.info(
                        "NewsAPI query '{}' returned {} articles",
                        query,
                        len(data.get("articles", [])),
                    )
                    break  # success

                except requests.RequestException as exc:
                    logger.warning(
                        "NewsAPI attempt {}/{} failed for '{}': {}",
                        attempt,
                        self.max_retries,
                        query,
                        exc,
                    )
                    if attempt == self.max_retries:
                        logger.error("NewsAPI exhausted retries for '{}'", query)
                    else:
                        time.sleep(2**attempt)

        return articles

    # ------------------------------------------------------------------
    # RSS Feeds
    # ------------------------------------------------------------------

    def _fetch_rss(self) -> list[dict[str, str | None]]:
        """Fetch articles from configured RSS feeds.

        Returns:
            List of normalized article dicts.
        """
        articles: list[dict[str, str | None]] = []

        for source_name, feed_url in RSS_FEEDS.items():
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(
                        "Fetching RSS '{}' | attempt {}/{}",
                        source_name,
                        attempt,
                        self.max_retries,
                    )
                    feed = feedparser.parse(feed_url)

                    if feed.bozo and not feed.entries:
                        raise ConnectionError(
                            f"RSS parse error: {feed.bozo_exception}"
                        )

                    for entry in feed.entries:
                        title = entry.get("title", "") or ""
                        raw_summary = (
                            entry.get("summary", "")
                            or entry.get("description", "")
                            or ""
                        )
                        summary_text = _truncate(_clean_html(raw_summary))
                        raw_date = (
                            entry.get("published", "")
                            or entry.get("updated", "")
                        )
                        parsed_date = _parse_date(raw_date)

                        articles.append({
                            "date": str(parsed_date) if parsed_date else None,
                            "ticker": _match_ticker(title, summary_text),
                            "title": title,
                            "source": source_name,
                            "url": entry.get("link"),
                            "summary": summary_text,
                        })

                    logger.info(
                        "RSS '{}' returned {} entries",
                        source_name,
                        len(feed.entries),
                    )
                    break  # success

                except Exception as exc:
                    logger.warning(
                        "RSS attempt {}/{} failed for '{}': {}",
                        attempt,
                        self.max_retries,
                        source_name,
                        exc,
                    )
                    if attempt == self.max_retries:
                        logger.error("RSS exhausted retries for '{}'", source_name)
                    else:
                        time.sleep(2**attempt)

        return articles

    # ------------------------------------------------------------------
    # GDELT Historical
    # ------------------------------------------------------------------

    def _fetch_gdelt(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        max_records: int = 250,
    ) -> list[dict[str, str | None]]:
        """Fetch articles from GDELT DOC API for a specific ticker.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start of date range (inclusive).
            end_date: End of date range (exclusive).
            max_records: Maximum articles per request (GDELT caps at 250).

        Returns:
            List of normalized article dicts with ticker pre-assigned.
        """
        company = _TICKER_COMPANY.get(ticker, ticker)
        query = f'("{company}" OR "{ticker}") (stock OR shares OR earnings)'

        for attempt in range(1, self.max_retries + 1):
            try:
                gdelt_params: dict[str, str | int] = {
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": max_records,
                    "format": "json",
                    "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
                    "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
                    "sourcelang": "english",
                }
                resp = requests.get(
                    _GDELT_BASE_URL,
                    params=gdelt_params,
                    headers={"User-Agent": "TitaniumAlpha/1.0"},
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = _GDELT_DELAY * (2 ** attempt)
                    logger.warning(
                        "GDELT rate-limited for {} (attempt {}), waiting {:.0f}s",
                        ticker, attempt, wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                articles: list[dict[str, str | None]] = []

                for art in data.get("articles", []):
                    title = art.get("title", "") or ""
                    url = art.get("url", "") or ""
                    seen_date = art.get("seendate", "") or ""
                    domain = art.get("domain", "") or ""
                    parsed_date = _parse_date(seen_date)

                    if not title or not url:
                        continue

                    articles.append({
                        "date": str(parsed_date) if parsed_date else None,
                        "ticker": ticker,
                        "title": _truncate(title, 500),
                        "source": f"GDELT/{domain}",
                        "url": url,
                        "summary": "",
                    })

                logger.debug(
                    "GDELT {} [{} → {}]: {} articles",
                    ticker,
                    start_date,
                    end_date,
                    len(articles),
                )
                return articles

            except requests.RequestException as exc:
                logger.warning(
                    "GDELT attempt {}/{} failed for {}: {}",
                    attempt, self.max_retries, ticker, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(_GDELT_DELAY * attempt)

        return []

    def _fetch_google_news(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, str | None]]:
        """Fetch articles from Google News RSS for a specific ticker.

        Uses date-filtered RSS search. No API key required.
        Rate limit is lenient (~1 req/2s recommended).

        Args:
            ticker: Stock ticker symbol.
            start_date: Start of date range (inclusive).
            end_date: End of date range (exclusive).

        Returns:
            List of normalized article dicts with ticker pre-assigned.
        """
        company = _TICKER_COMPANY.get(ticker, ticker)
        query = (
            f"{ticker} {company} stock "
            f"after:{start_date.isoformat()} "
            f"before:{end_date.isoformat()}"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    "https://news.google.com/rss/search",
                    params={
                        "q": query,
                        "hl": "en-US",
                        "gl": "US",
                        "ceid": "US:en",
                    },
                    headers={"User-Agent": "Mozilla/5.0 TitaniumAlpha/1.0"},
                    timeout=30,
                )
                resp.raise_for_status()

                import xml.etree.ElementTree as ET

                root = ET.fromstring(resp.content)
                items = root.findall(".//item")

                articles: list[dict[str, str | None]] = []
                for item in items:
                    title_el = item.find("title")
                    link_el = item.find("link")
                    pub_date_el = item.find("pubDate")

                    title = title_el.text if title_el is not None else ""
                    url = link_el.text if link_el is not None else ""
                    pub_raw = pub_date_el.text if pub_date_el is not None else ""

                    if not title or not url:
                        continue

                    parsed_date = _parse_date(pub_raw or "")

                    # Extract source from title (Google News format: "Title - Source")
                    source = "Google News"
                    if " - " in title:
                        parts = title.rsplit(" - ", 1)
                        source = f"GNews/{parts[1].strip()}"
                        title = parts[0].strip()

                    articles.append({
                        "date": str(parsed_date) if parsed_date else None,
                        "ticker": ticker,
                        "title": _truncate(title, 500),
                        "source": source,
                        "url": url,
                        "summary": "",
                    })

                logger.debug(
                    "Google News {} [{} → {}]: {} articles",
                    ticker, start_date, end_date, len(articles),
                )
                return articles

            except (requests.RequestException, ET.ParseError) as exc:
                logger.warning(
                    "Google News attempt {}/{} failed for {}: {}",
                    attempt, self.max_retries, ticker, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(3 * attempt)

        return []

    def backfill_historical(
        self,
        start_year: int = 2017,
        end_year: int | None = None,
        tickers: list[str] | None = None,
        chunk_months: int = 6,
        source: str = "google",
        delay: float = 2.0,
    ) -> int:
        """Backfill historical news for all tickers.

        Iterates tickers x date chunks, fetching articles per chunk.
        Supports Google News RSS (default, no API key) or GDELT.

        Args:
            start_year: First year to fetch.
            end_year: Last year to fetch (inclusive). Defaults to current year.
            tickers: Ticker list override. Defaults to config/tickers.json.
            chunk_months: Size of each date chunk in months (default 6).
            source: ``"google"`` or ``"gdelt"``.
            delay: Seconds between requests (default 2.0 for Google, 6.0 for GDELT).

        Returns:
            Total number of new rows inserted.
        """
        self._ensure_table()

        if end_year is None:
            end_year = date.today().year

        if source == "gdelt":
            delay = max(delay, _GDELT_DELAY)

        if tickers is None:
            tickers_path = Path(__file__).resolve().parents[2] / "config" / "tickers.json"
            with open(tickers_path) as f:
                cfg = json.load(f)
            tickers = cfg.get("tickers", [])
            if cfg.get("benchmark"):
                tickers = [cfg["benchmark"]] + tickers

        # Build date chunks
        chunks: list[tuple[date, date]] = []
        current = date(start_year, 1, 1)
        final = min(date(end_year + 1, 1, 1), date.today())
        while current < final:
            chunk_end = current + timedelta(days=chunk_months * 30)
            if chunk_end > final:
                chunk_end = final
            chunks.append((current, chunk_end))
            current = chunk_end

        total_inserted = 0
        total_tickers = len(tickers)
        total_requests = total_tickers * len(chunks)
        fetch_fn = (
            self._fetch_google_news if source == "google" else self._fetch_gdelt
        )

        logger.info(
            "Backfill plan: {} tickers × {} chunks = {} requests "
            "(~{:.0f} min at {:.1f}s/req) via {}",
            total_tickers,
            len(chunks),
            total_requests,
            total_requests * delay / 60,
            delay,
            source,
        )

        req_num = 0
        for t_idx, ticker in enumerate(tickers, 1):
            for chunk_start, chunk_end in chunks:
                req_num += 1
                if req_num % 20 == 1 or req_num == total_requests:
                    logger.info(
                        "[{}/{}] {} | {} → {}",
                        req_num, total_requests, ticker, chunk_start, chunk_end,
                    )

                articles = fetch_fn(ticker, chunk_start, chunk_end)
                if articles:
                    df = pl.DataFrame(articles)
                    df = df.with_columns(
                        pl.col("date").str.to_date("%Y-%m-%d", strict=False)
                    )
                    df = df.filter(pl.col("date").is_not_null())
                    df = df.unique(subset=["url"], keep="first")
                    df = df.filter(pl.col("title").str.len_chars() > 0)

                    if df.height > 0:
                        inserted = self._save_to_postgres(df)
                        total_inserted += inserted

                time.sleep(delay)

        logger.info(
            "Backfill complete: {} new articles inserted across {} tickers",
            total_inserted, total_tickers,
        )
        return total_inserted

    # ------------------------------------------------------------------
    # Re-match NULL tickers
    # ------------------------------------------------------------------

    def rematch_null_tickers(self) -> int:
        """Re-run ticker matching on existing articles with NULL ticker.

        Reads articles from PostgreSQL where ticker IS NULL, re-applies
        ``_match_ticker``, and updates rows that now have a match.

        Returns:
            Number of rows updated.
        """
        updated = 0
        with self.engine.begin() as conn:
            rows = conn.execute(text(
                f"SELECT id, title, COALESCE(summary, '') as summary "
                f"FROM {NEWS_TABLE} WHERE ticker IS NULL"
            )).fetchall()

            for row in rows:
                matched = _match_ticker(row[1], row[2])
                if matched:
                    conn.execute(
                        text(
                            f"UPDATE {NEWS_TABLE} SET ticker = :ticker "
                            f"WHERE id = :id"
                        ),
                        {"ticker": matched, "id": row[0]},
                    )
                    updated += 1

        logger.info("Re-matched {} articles with previously NULL ticker", updated)
        return updated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_to_postgres(self, df: pl.DataFrame) -> int:
        """Insert news articles into PostgreSQL, skipping duplicates.

        Args:
            df: DataFrame matching the financial_news schema.

        Returns:
            Number of rows inserted.
        """
        rows = df.to_dicts()
        if not rows:
            return 0

        insert_sql = text(f"""
            INSERT INTO {NEWS_TABLE}
                (date, ticker, title, source, url, summary, embedding_status)
            VALUES
                (:date, :ticker, :title, :source, :url, :summary, 'pending')
            ON CONFLICT (url) DO NOTHING
        """)

        with self.engine.begin() as conn:
            result = conn.execute(insert_sql, rows)
            inserted = result.rowcount

        logger.info(
            "Inserted {} new articles into '{}' ({} duplicates skipped)",
            inserted,
            NEWS_TABLE,
            len(rows) - inserted,
        )
        return inserted

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> pl.DataFrame:
        """Execute the full news ingestion pipeline.

        Fetches from all sources, deduplicates, saves to PostgreSQL,
        and returns the combined DataFrame.

        Returns:
            Polars DataFrame with all fetched articles.
        """
        self._ensure_table()

        all_articles: list[dict[str, str | None]] = []
        all_articles.extend(self._fetch_newsapi())
        all_articles.extend(self._fetch_rss())

        if not all_articles:
            logger.warning("No articles fetched from any source")
            return pl.DataFrame(
                schema={
                    "date": pl.Date,
                    "ticker": pl.Utf8,
                    "title": pl.Utf8,
                    "source": pl.Utf8,
                    "url": pl.Utf8,
                    "summary": pl.Utf8,
                }
            )

        df = pl.DataFrame(all_articles)

        # Cast date strings to pl.Date, dropping rows with unparseable dates
        df = df.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d", strict=False)
        )
        df = df.filter(pl.col("date").is_not_null())

        # Deduplicate by URL within this batch
        df = df.unique(subset=["url"], keep="first")

        # Filter out articles with empty titles
        df = df.filter(pl.col("title").str.len_chars() > 0)

        logger.info(
            "Fetched {} unique articles | {} with ticker match",
            df.height,
            df.filter(pl.col("ticker").is_not_null()).height,
        )

        self._save_to_postgres(df)
        return df


def _parse_date(raw: str) -> date | None:
    """Parse various date formats into a date object.

    Tries ``datetime.fromisoformat`` first (handles fractional seconds
    common in NewsAPI payloads, e.g. ``2024-01-15T10:30:00.123Z``),
    then falls back to ``strptime`` for RSS-style RFC 822 dates.

    Args:
        raw: Raw date string from API or RSS.

    Returns:
        Parsed date or None if parsing fails.
    """
    if not raw:
        return None

    raw = raw.strip()

    # ISO 8601 — handles fractional seconds that strptime can't
    # (Python 3.10 fromisoformat doesn't accept "Z"; normalize to +00:00)
    try:
        normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except (ValueError, TypeError):
        pass

    # GDELT compact ISO: 20240104T184500Z
    gdelt_formats = [
        "%Y%m%dT%H%M%SZ",
        "%Y%m%dT%H%M%S",
    ]
    for fmt in gdelt_formats:
        try:
            dt = datetime.strptime(raw, fmt)
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.date()
        except ValueError:
            continue

    # RSS-style RFC 822 dates
    rss_formats = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
    ]
    for fmt in rss_formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.date()
        except ValueError:
            continue

    logger.debug("Could not parse date: '{}'", raw)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial news ingestion pipeline")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill historical news from GDELT (2017-present)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2017,
        help="Start year for backfill (default: 2017)",
    )
    parser.add_argument(
        "--source",
        choices=["google", "gdelt"],
        default="google",
        help="Backfill source: google (default) or gdelt",
    )
    parser.add_argument(
        "--rematch",
        action="store_true",
        help="Re-run ticker matching on NULL-ticker articles",
    )
    args = parser.parse_args()

    ingester = NewsIngester()

    if args.rematch:
        ingester.rematch_null_tickers()

    if args.backfill:
        ingester.backfill_historical(
            start_year=args.start_year, source=args.source,
        )
    else:
        result = ingester.run()
        logger.info("Total articles: {}", result.height)
        if result.height > 0:
            ticker_counts = (
                result.group_by("source").len().sort("len", descending=True)
            )
            for row in ticker_counts.iter_rows(named=True):
                logger.info("  {}: {} articles", row["source"], row["len"])
