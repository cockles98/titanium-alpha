"""Financial news ingestion pipeline.

Fetches news from NewsAPI and RSS feeds, normalizes with Polars,
and persists to PostgreSQL for later RAG embedding.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone

import feedparser
import polars as pl
import requests
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

TICKER_KEYWORDS: dict[str, list[str]] = {
    "SPY": ["S&P 500", "S&P500", "SPY", "index fund"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA", "Jensen Huang"],
    "AAPL": ["AAPL", "Apple", "iPhone", "Tim Cook"],
    "QQQ": ["QQQ", "Nasdaq", "NASDAQ", "Invesco"],
}


def _match_ticker(title: str, summary: str) -> str | None:
    """Match a news article to a ticker based on keyword presence.

    Args:
        title: Article title.
        summary: Article summary text.

    Returns:
        Matched ticker symbol or None if no match.
    """
    combined = f"{title} {summary}".upper()
    for ticker, keywords in TICKER_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in combined:
                return ticker
    return None


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
        queries = [
            "stock market",
            "NVDA NVIDIA",
            "AAPL Apple",
            "SPY S&P 500",
            "QQQ Nasdaq",
        ]

        for query in queries:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = requests.get(
                        "https://newsapi.org/v2/everything",
                        params={
                            "q": query,
                            "language": "en",
                            "sortBy": "publishedAt",
                            "pageSize": 20,
                            "apiKey": self._newsapi_key,
                        },
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

    Args:
        raw: Raw date string from API or RSS.

    Returns:
        Parsed date or None if parsing fails.
    """
    if not raw:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.date()
        except ValueError:
            continue

    logger.debug("Could not parse date: '{}'", raw)
    return None


if __name__ == "__main__":
    ingester = NewsIngester()
    result = ingester.run()
    logger.info("Total articles: {}", result.height)
    if result.height > 0:
        ticker_counts = result.group_by("source").len().sort("len", descending=True)
        for row in ticker_counts.iter_rows(named=True):
            logger.info("  {}: {} articles", row["source"], row["len"])
