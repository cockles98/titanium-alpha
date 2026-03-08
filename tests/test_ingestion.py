"""Tests for ``src.data.ingestion.MarketDataIngester``.

All external I/O (yfinance, PostgreSQL, sleep) is mocked so these
tests run offline and fast.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, call, patch

import pandas as pd
import polars as pl
import pytest

from src.data.ingestion import OHLCV_TABLE, MarketDataIngester


# ---------------------------------------------------------------------------
# 1. Happy path: _download_ticker returns correct Polars DataFrame
# ---------------------------------------------------------------------------

class TestDownloadTickerHappyPath:
    """Verify the happy path for a single-ticker download."""

    @patch("src.data.ingestion.yf.download")
    def test_returns_polars_dataframe(
        self, mock_yf_download: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert isinstance(result, pl.DataFrame)

    @patch("src.data.ingestion.yf.download")
    def test_has_ticker_column_filled(
        self, mock_yf_download: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert result["ticker"].to_list() == ["SPY"] * result.height

    @patch("src.data.ingestion.yf.download")
    def test_row_count_matches_source(
        self, mock_yf_download: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert result.height == len(sample_yf_dataframe)


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    """Ensure the output DataFrame has the exact columns and types."""

    @patch("src.data.ingestion.yf.download")
    def test_columns_match_schema(
        self,
        mock_yf_download: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
        expected_polars_schema: dict[str, pl.DataType],
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine)

        result = ingester._download_ticker("SPY")

        assert list(result.columns) == list(expected_polars_schema.keys())

    @patch("src.data.ingestion.yf.download")
    def test_dtypes_match_schema(
        self,
        mock_yf_download: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
        expected_polars_schema: dict[str, pl.DataType],
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine)

        result = ingester._download_ticker("SPY")

        for col_name, expected_dtype in expected_polars_schema.items():
            assert result[col_name].dtype == expected_dtype, (
                f"Column '{col_name}': expected {expected_dtype}, got {result[col_name].dtype}"
            )


# ---------------------------------------------------------------------------
# 3. yfinance returns empty DataFrame -> ValueError
# ---------------------------------------------------------------------------

class TestEmptyDownload:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_empty_dataframe_raises_runtime_error(
        self, mock_yf_download: MagicMock, mock_sleep: MagicMock, mock_engine: MagicMock,
    ) -> None:
        """When yfinance returns an empty DF every attempt, a RuntimeError
        is raised (wrapping the ValueError from each attempt)."""
        mock_yf_download.return_value = pd.DataFrame()
        ingester = MarketDataIngester(engine=mock_engine, tickers=["BAD"])

        with pytest.raises(RuntimeError, match="Failed to download BAD"):
            ingester._download_ticker("BAD")


# ---------------------------------------------------------------------------
# 4. Retry: fails 2x then succeeds on 3rd
# ---------------------------------------------------------------------------

class TestRetrySuccess:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_succeeds_on_third_attempt(
        self,
        mock_yf_download: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = [
            ConnectionError("network down"),
            ConnectionError("network down again"),
            sample_yf_dataframe,
        ]
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert isinstance(result, pl.DataFrame)
        assert result.height == len(sample_yf_dataframe)
        assert mock_yf_download.call_count == 3
        # Sleep is called with exponential backoff: 2^1, 2^2
        assert mock_sleep.call_args_list == [call(2), call(4)]


# ---------------------------------------------------------------------------
# 5. Retry: fails 3x -> RuntimeError
# ---------------------------------------------------------------------------

class TestRetryExhausted:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_raises_after_all_retries_exhausted(
        self, mock_yf_download: MagicMock, mock_sleep: MagicMock, mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = ConnectionError("always fails")
        ingester = MarketDataIngester(engine=mock_engine, tickers=["FAIL"], max_retries=3)

        with pytest.raises(RuntimeError, match="Failed to download FAIL after 3 attempts"):
            ingester._download_ticker("FAIL")

        assert mock_yf_download.call_count == 3
        # Sleep called twice (after attempt 1 and 2, not after the last failure)
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# 6. _save_to_postgres: engine.begin() is called with upsert SQL
# ---------------------------------------------------------------------------

class TestSaveToPostgres:

    def test_upsert_calls_engine_begin_and_execute(self, mock_engine: MagicMock) -> None:
        df = pl.DataFrame(
            {
                "date": ["2024-01-02"],
                "ticker": ["SPY"],
                "open": [450.0],
                "high": [455.0],
                "low": [448.0],
                "close": [452.0],
                "volume": [80_000_000],
                "adj_close": [451.0],
            }
        )
        ingester = MarketDataIngester(engine=mock_engine)

        rows = ingester._save_to_postgres(df)

        assert rows == 1
        mock_engine.begin.assert_called_once()
        conn = mock_engine.begin.return_value.__enter__.return_value
        conn.execute.assert_called_once()

        # Verify the SQL text contains ON CONFLICT (our upsert marker)
        executed_sql = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in str(executed_sql)
        assert OHLCV_TABLE in str(executed_sql)

    def test_empty_dataframe_returns_zero(self, mock_engine: MagicMock) -> None:
        df = pl.DataFrame(
            {
                "date": [],
                "ticker": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "adj_close": [],
            }
        )
        ingester = MarketDataIngester(engine=mock_engine)

        rows = ingester._save_to_postgres(df)

        assert rows == 0
        mock_engine.begin.assert_not_called()


# ---------------------------------------------------------------------------
# 7. _ensure_table: verifies CREATE TABLE is executed
# ---------------------------------------------------------------------------

class TestEnsureTable:

    def test_executes_create_table(self, mock_engine: MagicMock) -> None:
        ingester = MarketDataIngester(engine=mock_engine)

        ingester._ensure_table()

        mock_engine.begin.assert_called_once()
        conn = mock_engine.begin.return_value.__enter__.return_value
        conn.execute.assert_called_once()

        executed_sql = str(conn.execute.call_args[0][0])
        assert "CREATE TABLE IF NOT EXISTS" in executed_sql
        assert OHLCV_TABLE in executed_sql


# ---------------------------------------------------------------------------
# 8. run(): orchestrates download + save for all tickers
# ---------------------------------------------------------------------------

class TestRun:

    @patch("src.data.ingestion.yf.download")
    def test_run_sequential_orchestrates_all_tickers(
        self,
        mock_yf_download: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = lambda *a, **kw: sample_yf_dataframe.copy()
        tickers = ["SPY", "NVDA"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=False)

        # _ensure_table (1) + _save per ticker (2) = 3
        assert mock_engine.begin.call_count == 3
        assert mock_yf_download.call_count == len(tickers)
        assert isinstance(result, pl.DataFrame)
        assert result.height == len(sample_yf_dataframe) * len(tickers)

    @patch("src.data.ingestion.yf.download")
    def test_run_returns_correct_ticker_values(
        self,
        mock_yf_download: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = lambda *a, **kw: sample_yf_dataframe.copy()
        tickers = ["SPY", "AAPL"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=False)

        result_tickers = result["ticker"].unique().sort().to_list()
        assert result_tickers == sorted(tickers)


# ---------------------------------------------------------------------------
# 9. Edge cases: single data point
# ---------------------------------------------------------------------------

class TestSingleRow:

    @patch("src.data.ingestion.yf.download")
    def test_single_row_dataframe(
        self, mock_yf_download: MagicMock, mock_engine: MagicMock,
    ) -> None:
        dates = pd.DatetimeIndex(["2024-01-02"], name="Date")
        ticker = "SPY"
        data = {
            ("Open", ticker): [450.0],
            ("High", ticker): [455.0],
            ("Low", ticker): [448.0],
            ("Close", ticker): [452.0],
            ("Volume", ticker): [80_000_000],
            ("Adj Close", ticker): [451.0],
        }
        columns = pd.MultiIndex.from_arrays(
            [["Open", "High", "Low", "Close", "Volume", "Adj Close"],
             [ticker] * 6]
        )
        pdf = pd.DataFrame(data, index=dates, columns=columns)
        mock_yf_download.return_value = pdf

        ingester = MarketDataIngester(engine=mock_engine)
        result = ingester._download_ticker("SPY")

        assert result.height == 1
        assert result["ticker"][0] == "SPY"


# ---------------------------------------------------------------------------
# 10. Edge case: NaN values in OHLCV data
# ---------------------------------------------------------------------------

class TestNaNHandling:

    @patch("src.data.ingestion.yf.download")
    def test_nan_values_are_preserved_as_null(
        self, mock_yf_download: MagicMock, mock_engine: MagicMock,
    ) -> None:
        """yfinance can return NaN for some fields (e.g. Adj Close on
        certain instruments).  The pipeline should not crash -- nulls are
        acceptable and will be handled downstream."""
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
        ticker = "SPY"
        data = {
            ("Open", ticker): [450.0, float("nan")],
            ("High", ticker): [455.0, 460.0],
            ("Low", ticker): [448.0, 449.0],
            ("Close", ticker): [452.0, 458.0],
            ("Volume", ticker): [80_000_000, 90_000_000],
            ("Adj Close", ticker): [float("nan"), 457.0],
        }
        columns = pd.MultiIndex.from_arrays(
            [["Open", "High", "Low", "Close", "Volume", "Adj Close"],
             [ticker] * 6]
        )
        pdf = pd.DataFrame(data, index=dates, columns=columns)
        mock_yf_download.return_value = pdf

        ingester = MarketDataIngester(engine=mock_engine)
        result = ingester._download_ticker("SPY")

        assert result.height == 2
        # NaN in Polars becomes null
        assert result["open"][1] is None
        assert result["adj_close"][0] is None


# ---------------------------------------------------------------------------
# 11. Parallel download
# ---------------------------------------------------------------------------


class TestParallelDownload:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_parallel_downloads_all_tickers(
        self,
        mock_yf_download: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = lambda *a, **kw: sample_yf_dataframe.copy()
        tickers = ["SPY", "NVDA", "AAPL"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=True)

        assert mock_yf_download.call_count == 3
        assert isinstance(result, pl.DataFrame)
        assert result["ticker"].unique().sort().to_list() == sorted(tickers)

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_parallel_partial_failure_skips_bad_ticker(
        self,
        mock_yf_download: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        """When one ticker fails all retries, others still succeed."""
        def _side_effect(*args: object, **kwargs: object) -> pd.DataFrame:
            # yf.download is called with ticker as first positional arg
            called_ticker = args[0] if args else kwargs.get("tickers", "")
            if called_ticker == "BAD":
                raise ConnectionError("always fails")
            return sample_yf_dataframe.copy()

        mock_yf_download.side_effect = _side_effect
        tickers = ["SPY", "BAD", "AAPL"]
        ingester = MarketDataIngester(
            engine=mock_engine, tickers=tickers, max_retries=1
        )

        result = ingester.run(parallel=True)

        # BAD failed, but SPY and AAPL succeeded
        result_tickers = result["ticker"].unique().sort().to_list()
        assert "BAD" not in result_tickers
        assert "SPY" in result_tickers
        assert "AAPL" in result_tickers

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_parallel_all_fail_raises(
        self,
        mock_yf_download: MagicMock,
        mock_sleep: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.side_effect = ConnectionError("down")
        ingester = MarketDataIngester(
            engine=mock_engine, tickers=["A", "B"], max_retries=1
        )

        with pytest.raises(RuntimeError, match="No tickers could be downloaded"):
            ingester.run(parallel=True)

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.download")
    def test_single_ticker_uses_sequential(
        self,
        mock_yf_download: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        """With a single ticker, parallel=True still works (sequential path)."""
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester.run(parallel=True)

        assert result.height == len(sample_yf_dataframe)


# ---------------------------------------------------------------------------
# 12. start_date / end_date support
# ---------------------------------------------------------------------------


class TestDateRange:

    def test_start_date_overrides_years(self, mock_engine: MagicMock) -> None:
        ingester = MarketDataIngester(
            engine=mock_engine,
            tickers=["SPY"],
            start_date=date(2016, 1, 1),
            end_date=date(2026, 1, 1),
            years=5,  # should be ignored
        )
        assert ingester.start_date == date(2016, 1, 1)
        assert ingester.end_date == date(2026, 1, 1)

    def test_years_fallback(self, mock_engine: MagicMock) -> None:
        ingester = MarketDataIngester(
            engine=mock_engine, tickers=["SPY"], years=10
        )
        expected_start = ingester.end_date - timedelta(days=10 * 365)
        assert ingester.start_date == expected_start

    def test_end_date_defaults_to_today(self, mock_engine: MagicMock) -> None:
        ingester = MarketDataIngester(
            engine=mock_engine, tickers=["SPY"]
        )
        assert ingester.end_date == date.today()

    @patch("src.data.ingestion.yf.download")
    def test_download_uses_configured_dates(
        self,
        mock_yf_download: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_yf_download.return_value = sample_yf_dataframe
        ingester = MarketDataIngester(
            engine=mock_engine,
            tickers=["SPY"],
            start_date=date(2020, 6, 1),
            end_date=date(2025, 6, 1),
        )

        ingester._download_ticker("SPY")

        _, kwargs = mock_yf_download.call_args
        assert kwargs["start"] == "2020-06-01"
        assert kwargs["end"] == "2025-06-01"
