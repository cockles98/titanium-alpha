"""Tests for src.data.news_ingestion module.

Covers standalone helpers (_match_ticker, _clean_html, _truncate, _parse_date)
and the NewsIngester class (fetch, persist, orchestration) with full mocking
of external dependencies (requests, feedparser, PostgreSQL).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import polars as pl
import pytest

from src.data.news_ingestion import (
    NewsIngester,
    _clean_html,
    _match_ticker,
    _parse_date,
    _truncate,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def ingester(mock_engine: MagicMock) -> NewsIngester:
    """NewsIngester with mocked engine and empty API key."""
    with patch("src.data.news_ingestion.load_dotenv"), \
         patch("src.data.news_ingestion.os.getenv", return_value="test-key"):
        ing = NewsIngester(engine=mock_engine, max_retries=3)
    return ing


@pytest.fixture()
def ingester_no_key(mock_engine: MagicMock) -> NewsIngester:
    """NewsIngester with mocked engine and no API key."""
    with patch("src.data.news_ingestion.load_dotenv"), \
         patch("src.data.news_ingestion.os.getenv", return_value=""):
        ing = NewsIngester(engine=mock_engine, max_retries=3)
    return ing


def _make_rss_feed(entries: list[dict], bozo: bool = False) -> SimpleNamespace:
    """Build a fake feedparser result object."""
    feed = SimpleNamespace()
    feed.bozo = bozo
    feed.bozo_exception = Exception("parse error") if bozo else None
    feed.entries = [SimpleNamespace(**e) for e in entries]
    # feedparser entries support .get()
    for entry in feed.entries:
        entry.get = lambda key, default="", _e=entry: getattr(_e, key, default)
    return feed


def _make_newsapi_response(articles: list[dict]) -> MagicMock:
    """Build a fake requests.Response for NewsAPI."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"status": "ok", "articles": articles}
    return resp


# ======================================================================
# 1. _match_ticker
# ======================================================================


class TestMatchTicker:
    """Tests for _match_ticker helper."""

    def test_match_spy(self) -> None:
        assert _match_ticker("S&P 500 hits record high", "") == "SPY"

    def test_match_nvda(self) -> None:
        assert _match_ticker("Earnings report", "NVIDIA beats expectations") == "NVDA"

    def test_match_aapl(self) -> None:
        assert _match_ticker("Tim Cook announces new iPhone", "") == "AAPL"

    def test_match_qqq(self) -> None:
        assert _match_ticker("", "Nasdaq composite rises") == "QQQ"

    def test_no_match(self) -> None:
        assert _match_ticker("Weather forecast today", "Rain expected") is None

    def test_empty_strings(self) -> None:
        assert _match_ticker("", "") is None

    def test_case_insensitive(self) -> None:
        assert _match_ticker("nvidia stock soars", "") == "NVDA"

    def test_first_match_wins(self) -> None:
        # SPY keywords checked before NVDA in dict order
        result = _match_ticker("S&P 500 and NVIDIA news", "")
        assert result == "SPY"


# ======================================================================
# 2. _clean_html
# ======================================================================


class TestCleanHtml:
    """Tests for _clean_html helper."""

    def test_strips_tags(self) -> None:
        assert _clean_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_string(self) -> None:
        assert _clean_html("") == ""

    def test_none_like_empty(self) -> None:
        # falsy input
        assert _clean_html("") == ""

    def test_plain_text_unchanged(self) -> None:
        assert _clean_html("No HTML here") == "No HTML here"

    def test_nested_tags(self) -> None:
        html = "<div><ul><li>Item 1</li><li>Item 2</li></ul></div>"
        result = _clean_html(html)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "<" not in result


# ======================================================================
# 3. _truncate
# ======================================================================


class TestTruncate:
    """Tests for _truncate helper."""

    def test_short_text_unchanged(self) -> None:
        text = "Short text"
        assert _truncate(text) == text

    def test_exact_500_unchanged(self) -> None:
        text = "a" * 500
        assert _truncate(text) == text

    def test_long_text_truncated(self) -> None:
        text = " ".join(["word"] * 200)  # well over 500 chars
        result = _truncate(text)
        assert len(result) <= 503  # 500 + "..."
        assert result.endswith("...")

    def test_custom_max_len(self) -> None:
        text = "Hello beautiful world of finance"
        result = _truncate(text, max_len=15)
        assert result.endswith("...")
        assert len(result) <= 18

    def test_empty_string(self) -> None:
        assert _truncate("") == ""

    def test_single_long_word(self) -> None:
        text = "a" * 600
        result = _truncate(text, max_len=500)
        assert result.endswith("...")


# ======================================================================
# 4. _parse_date
# ======================================================================


class TestParseDate:
    """Tests for _parse_date helper."""

    def test_iso_utc(self) -> None:
        assert _parse_date("2025-06-15T10:30:00Z") == date(2025, 6, 15)

    def test_iso_with_tz(self) -> None:
        assert _parse_date("2025-06-15T10:30:00+00:00") == date(2025, 6, 15)

    def test_rfc2822(self) -> None:
        assert _parse_date("Mon, 15 Jun 2025 10:30:00 GMT") == date(2025, 6, 15)

    def test_date_only(self) -> None:
        assert _parse_date("2025-06-15") == date(2025, 6, 15)

    def test_invalid_string(self) -> None:
        assert _parse_date("not a date") is None

    def test_empty_string(self) -> None:
        assert _parse_date("") is None

    def test_whitespace_stripped(self) -> None:
        assert _parse_date("  2025-06-15  ") == date(2025, 6, 15)


# ======================================================================
# 5. _fetch_newsapi
# ======================================================================


class TestFetchNewsapi:
    """Tests for NewsIngester._fetch_newsapi."""

    def test_empty_key_returns_empty(self, ingester_no_key: NewsIngester) -> None:
        result = ingester_no_key._fetch_newsapi()
        assert result == []

    @patch("src.data.news_ingestion.requests.get")
    def test_ok_response_returns_articles(
        self, mock_get: MagicMock, ingester: NewsIngester
    ) -> None:
        articles_payload = [
            {
                "title": "NVIDIA beats earnings",
                "description": "<p>Strong Q4</p>",
                "content": "Full content here",
                "publishedAt": "2025-06-15T10:00:00Z",
                "url": "https://example.com/nvda",
                "source": {"name": "TestSource"},
            }
        ]
        mock_get.return_value = _make_newsapi_response(articles_payload)

        result = ingester._fetch_newsapi()

        assert len(result) > 0
        first = result[0]
        assert first["title"] == "NVIDIA beats earnings"
        assert first["url"] == "https://example.com/nvda"
        assert first["source"].startswith("NewsAPI/")
        assert first["date"] == "2025-06-15"
        assert first["ticker"] == "NVDA"

    @patch("src.data.news_ingestion.time.sleep")
    @patch("src.data.news_ingestion.requests.get")
    def test_retry_on_request_exception(
        self,
        mock_get: MagicMock,
        mock_sleep: MagicMock,
        ingester: NewsIngester,
    ) -> None:
        import requests as req

        mock_get.side_effect = req.RequestException("timeout")

        result = ingester._fetch_newsapi()

        assert result == []
        # sleep called between retries (not on last attempt)
        assert mock_sleep.call_count > 0


# ======================================================================
# 6. _fetch_rss
# ======================================================================


class TestFetchRss:
    """Tests for NewsIngester._fetch_rss."""

    @patch("src.data.news_ingestion.time.sleep")
    @patch("src.data.news_ingestion.feedparser.parse")
    def test_valid_feed_returns_articles(
        self,
        mock_parse: MagicMock,
        mock_sleep: MagicMock,
        ingester: NewsIngester,
    ) -> None:
        entry = {
            "title": "Apple launches new iPhone",
            "summary": "<b>Revolutionary</b> device",
            "published": "2025-06-15T10:00:00Z",
            "link": "https://example.com/apple",
        }
        mock_parse.return_value = _make_rss_feed([entry])

        result = ingester._fetch_rss()

        assert len(result) > 0
        apple_articles = [a for a in result if a["ticker"] == "AAPL"]
        assert len(apple_articles) > 0
        art = apple_articles[0]
        assert art["url"] == "https://example.com/apple"
        assert art["date"] == "2025-06-15"
        assert "<b>" not in art["summary"]

    # ------------------------------------------------------------------
    # 7. _fetch_rss retry
    # ------------------------------------------------------------------

    @patch("src.data.news_ingestion.time.sleep")
    @patch("src.data.news_ingestion.feedparser.parse")
    def test_rss_retry_success_on_third_attempt(
        self,
        mock_parse: MagicMock,
        mock_sleep: MagicMock,
        ingester: NewsIngester,
    ) -> None:
        bozo_feed = _make_rss_feed([], bozo=True)
        good_feed = _make_rss_feed([{
            "title": "Market update",
            "summary": "S&P 500 rises",
            "published": "2025-06-15T10:00:00Z",
            "link": "https://example.com/spy",
        }])

        # For each of the 3 RSS feeds, fail twice then succeed
        mock_parse.side_effect = [
            bozo_feed, bozo_feed, good_feed,  # feed 1
            bozo_feed, bozo_feed, good_feed,  # feed 2
            bozo_feed, bozo_feed, good_feed,  # feed 3
        ]

        result = ingester._fetch_rss()

        # Each feed should produce 1 article on the 3rd attempt
        assert len(result) == 3
        assert mock_sleep.call_count > 0


# ======================================================================
# 8. _save_to_postgres
# ======================================================================


class TestSaveToPostgres:
    """Tests for NewsIngester._save_to_postgres."""

    def test_empty_df_returns_zero(self, ingester: NewsIngester) -> None:
        df = pl.DataFrame(schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "title": pl.Utf8,
            "source": pl.Utf8,
            "url": pl.Utf8,
            "summary": pl.Utf8,
        })
        assert ingester._save_to_postgres(df) == 0

    def test_inserts_rows(
        self, ingester: NewsIngester, mock_engine: MagicMock
    ) -> None:
        df = pl.DataFrame({
            "date": [date(2025, 6, 15)],
            "ticker": ["NVDA"],
            "title": ["NVIDIA earnings"],
            "source": ["TestSource"],
            "url": ["https://example.com/1"],
            "summary": ["Summary text"],
        })

        conn = mock_engine.begin.return_value.__enter__.return_value
        conn.execute.return_value = MagicMock(rowcount=1)

        result = ingester._save_to_postgres(df)

        assert result == 1
        conn.execute.assert_called_once()
        # Verify the SQL contains ON CONFLICT
        sql_arg = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in str(sql_arg)


# ======================================================================
# 9. _ensure_table
# ======================================================================


class TestEnsureTable:
    """Tests for NewsIngester._ensure_table."""

    def test_executes_create_table(
        self, ingester: NewsIngester, mock_engine: MagicMock
    ) -> None:
        ingester._ensure_table()

        conn = mock_engine.begin.return_value.__enter__.return_value
        conn.execute.assert_called_once()
        sql_arg = str(conn.execute.call_args[0][0])
        assert "CREATE TABLE IF NOT EXISTS" in sql_arg


# ======================================================================
# 10-11. run()
# ======================================================================


class TestRun:
    """Tests for NewsIngester.run orchestration."""

    def test_no_articles_returns_empty_df_with_schema(
        self, ingester: NewsIngester
    ) -> None:
        with patch.object(ingester, "_ensure_table"), \
             patch.object(ingester, "_fetch_newsapi", return_value=[]), \
             patch.object(ingester, "_fetch_rss", return_value=[]):

            result = ingester.run()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0
        expected_cols = {"date", "ticker", "title", "source", "url", "summary"}
        assert set(result.columns) == expected_cols

    def test_run_filters_invalid_dates_and_dedup(
        self, ingester: NewsIngester
    ) -> None:
        articles = [
            {
                "date": "2025-06-15",
                "ticker": "NVDA",
                "title": "NVIDIA news",
                "source": "TestSource",
                "url": "https://example.com/1",
                "summary": "Summary 1",
            },
            {
                "date": None,  # invalid date -- should be filtered
                "ticker": "AAPL",
                "title": "Apple news",
                "source": "TestSource",
                "url": "https://example.com/2",
                "summary": "Summary 2",
            },
            {
                "date": "2025-06-15",
                "ticker": "NVDA",
                "title": "NVIDIA news duplicate",
                "source": "TestSource",
                "url": "https://example.com/1",  # duplicate URL
                "summary": "Summary 3",
            },
        ]

        with patch.object(ingester, "_ensure_table"), \
             patch.object(ingester, "_fetch_newsapi", return_value=articles), \
             patch.object(ingester, "_fetch_rss", return_value=[]), \
             patch.object(ingester, "_save_to_postgres", return_value=1) as mock_save:

            result = ingester.run()

        # Null date filtered, duplicate URL removed -> 1 row
        assert result.height == 1
        assert result["url"][0] == "https://example.com/1"
        mock_save.assert_called_once()

    def test_run_filters_empty_titles(self, ingester: NewsIngester) -> None:
        articles = [
            {
                "date": "2025-06-15",
                "ticker": "NVDA",
                "title": "",
                "source": "TestSource",
                "url": "https://example.com/empty-title",
                "summary": "Has summary but no title",
            },
            {
                "date": "2025-06-15",
                "ticker": "AAPL",
                "title": "Valid title",
                "source": "TestSource",
                "url": "https://example.com/valid",
                "summary": "Summary",
            },
        ]

        with patch.object(ingester, "_ensure_table"), \
             patch.object(ingester, "_fetch_newsapi", return_value=articles), \
             patch.object(ingester, "_fetch_rss", return_value=[]), \
             patch.object(ingester, "_save_to_postgres", return_value=1):

            result = ingester.run()

        assert result.height == 1
        assert result["title"][0] == "Valid title"

    def test_run_calls_ensure_table_and_save(
        self, ingester: NewsIngester
    ) -> None:
        articles = [
            {
                "date": "2025-06-15",
                "ticker": "SPY",
                "title": "Market update",
                "source": "RSS",
                "url": "https://example.com/spy",
                "summary": "S&P 500 summary",
            },
        ]

        with patch.object(ingester, "_ensure_table") as mock_table, \
             patch.object(ingester, "_fetch_newsapi", return_value=[]), \
             patch.object(ingester, "_fetch_rss", return_value=articles), \
             patch.object(ingester, "_save_to_postgres", return_value=1) as mock_save:

            result = ingester.run()

        mock_table.assert_called_once()
        mock_save.assert_called_once()
        assert result.height == 1
