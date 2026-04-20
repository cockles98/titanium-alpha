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

from src.data.ingestion import OHLCV_TABLE, MarketDataIngester, _resolve_tickers


def _make_ticker_mock(history_rv: pd.DataFrame) -> MagicMock:
    """Create a mock ``yf.Ticker`` whose ``.history()`` returns *history_rv*."""
    mock = MagicMock()
    mock.history.return_value = history_rv
    return mock


# ---------------------------------------------------------------------------
# 1. Happy path: _download_ticker returns correct Polars DataFrame
# ---------------------------------------------------------------------------

class TestDownloadTickerHappyPath:
    """Verify the happy path for a single-ticker download."""

    @patch("src.data.ingestion.yf.Ticker")
    def test_returns_polars_dataframe(
        self, mock_ticker_cls: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert isinstance(result, pl.DataFrame)

    @patch("src.data.ingestion.yf.Ticker")
    def test_has_ticker_column_filled(
        self, mock_ticker_cls: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert result["ticker"].to_list() == ["SPY"] * result.height

    @patch("src.data.ingestion.yf.Ticker")
    def test_row_count_matches_source(
        self, mock_ticker_cls: MagicMock, sample_yf_dataframe: pd.DataFrame, mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert result.height == len(sample_yf_dataframe)


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    """Ensure the output DataFrame has the exact columns and types."""

    @patch("src.data.ingestion.yf.Ticker")
    def test_columns_match_schema(
        self,
        mock_ticker_cls: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
        expected_polars_schema: dict[str, pl.DataType],
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
        ingester = MarketDataIngester(engine=mock_engine)

        result = ingester._download_ticker("SPY")

        assert list(result.columns) == list(expected_polars_schema.keys())

    @patch("src.data.ingestion.yf.Ticker")
    def test_dtypes_match_schema(
        self,
        mock_ticker_cls: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
        expected_polars_schema: dict[str, pl.DataType],
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
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
    @patch("src.data.ingestion.yf.Ticker")
    def test_empty_dataframe_raises_runtime_error(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock, mock_engine: MagicMock,
    ) -> None:
        """When yfinance returns an empty DF every attempt, a RuntimeError
        is raised (wrapping the ValueError from each attempt)."""
        mock_ticker_cls.return_value = _make_ticker_mock(pd.DataFrame())
        ingester = MarketDataIngester(engine=mock_engine, tickers=["BAD"])

        with pytest.raises(RuntimeError, match="Failed to download BAD"):
            ingester._download_ticker("BAD")


# ---------------------------------------------------------------------------
# 4. Retry: fails 2x then succeeds on 3rd
# ---------------------------------------------------------------------------

class TestRetrySuccess:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.Ticker")
    def test_succeeds_on_third_attempt(
        self,
        mock_ticker_cls: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = [
            ConnectionError("network down"),
            ConnectionError("network down again"),
            sample_yf_dataframe,
        ]
        mock_ticker_cls.return_value = mock_ticker
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])

        result = ingester._download_ticker("SPY")

        assert isinstance(result, pl.DataFrame)
        assert result.height == len(sample_yf_dataframe)
        assert mock_ticker.history.call_count == 3
        # Sleep is called with exponential backoff: 2^1, 2^2
        assert mock_sleep.call_args_list == [call(2), call(4)]


# ---------------------------------------------------------------------------
# 5. Retry: fails 3x -> RuntimeError
# ---------------------------------------------------------------------------

class TestRetryExhausted:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.Ticker")
    def test_raises_after_all_retries_exhausted(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock, mock_engine: MagicMock,
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ConnectionError("always fails")
        mock_ticker_cls.return_value = mock_ticker
        ingester = MarketDataIngester(engine=mock_engine, tickers=["FAIL"], max_retries=3)

        with pytest.raises(RuntimeError, match="Failed to download FAIL after 3 attempts"):
            ingester._download_ticker("FAIL")

        assert mock_ticker.history.call_count == 3
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

    @patch("src.data.ingestion.yf.Ticker")
    def test_run_sequential_orchestrates_all_tickers(
        self,
        mock_ticker_cls: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe.copy())
        tickers = ["SPY", "NVDA"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=False)

        # _ensure_table (1) + _save per ticker (2) = 3
        assert mock_engine.begin.call_count == 3
        assert mock_ticker_cls.call_count == len(tickers)
        assert isinstance(result, pl.DataFrame)
        assert result.height == len(sample_yf_dataframe) * len(tickers)

    @patch("src.data.ingestion.yf.Ticker")
    def test_run_returns_correct_ticker_values(
        self,
        mock_ticker_cls: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe.copy())
        tickers = ["SPY", "AAPL"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=False)

        result_tickers = result["ticker"].unique().sort().to_list()
        assert result_tickers == sorted(tickers)


# ---------------------------------------------------------------------------
# 9. Edge cases: single data point
# ---------------------------------------------------------------------------

class TestSingleRow:

    @patch("src.data.ingestion.yf.Ticker")
    def test_single_row_dataframe(
        self, mock_ticker_cls: MagicMock, mock_engine: MagicMock,
    ) -> None:
        dates = pd.DatetimeIndex(["2024-01-02"], name="Date")
        pdf = pd.DataFrame(
            {
                "Open": [450.0],
                "High": [455.0],
                "Low": [448.0],
                "Close": [452.0],
                "Volume": [80_000_000],
                "Dividends": [0.0],
                "Stock Splits": [0.0],
            },
            index=dates,
        )
        mock_ticker_cls.return_value = _make_ticker_mock(pdf)

        ingester = MarketDataIngester(engine=mock_engine)
        result = ingester._download_ticker("SPY")

        assert result.height == 1
        assert result["ticker"][0] == "SPY"


# ---------------------------------------------------------------------------
# 10. Edge case: NaN values in OHLCV data
# ---------------------------------------------------------------------------

class TestNaNHandling:

    @patch("src.data.ingestion.yf.Ticker")
    def test_nan_values_are_preserved_as_null(
        self, mock_ticker_cls: MagicMock, mock_engine: MagicMock,
    ) -> None:
        """yfinance can return NaN for some fields. The pipeline should
        not crash -- nulls are acceptable and handled downstream."""
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
        pdf = pd.DataFrame(
            {
                "Open": [450.0, float("nan")],
                "High": [455.0, 460.0],
                "Low": [448.0, 449.0],
                "Close": [452.0, 458.0],
                "Volume": [80_000_000, 90_000_000],
                "Dividends": [0.0, 0.0],
                "Stock Splits": [0.0, 0.0],
            },
            index=dates,
        )
        mock_ticker_cls.return_value = _make_ticker_mock(pdf)

        ingester = MarketDataIngester(engine=mock_engine)
        result = ingester._download_ticker("SPY")

        assert result.height == 2
        # NaN in Polars becomes null
        assert result["open"][1] is None
        # adj_close = close (no NaN expected since Close is valid)
        assert result["adj_close"][0] is not None


# ---------------------------------------------------------------------------
# 11. Parallel download
# ---------------------------------------------------------------------------


class TestParallelDownload:

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.Ticker")
    def test_parallel_downloads_all_tickers(
        self,
        mock_ticker_cls: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe.copy())
        tickers = ["SPY", "NVDA", "AAPL"]
        ingester = MarketDataIngester(engine=mock_engine, tickers=tickers)

        result = ingester.run(parallel=True)

        assert mock_ticker_cls.call_count == 3
        assert isinstance(result, pl.DataFrame)
        assert result["ticker"].unique().sort().to_list() == sorted(tickers)

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.Ticker")
    def test_parallel_partial_failure_skips_bad_ticker(
        self,
        mock_ticker_cls: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        """When one ticker fails all retries, others still succeed."""
        def _make_mock(ticker: str) -> MagicMock:
            mock = MagicMock()
            if ticker == "BAD":
                mock.history.side_effect = ConnectionError("always fails")
            else:
                mock.history.return_value = sample_yf_dataframe.copy()
            return mock

        mock_ticker_cls.side_effect = _make_mock
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
    @patch("src.data.ingestion.yf.Ticker")
    def test_parallel_all_fail_raises(
        self,
        mock_ticker_cls: MagicMock,
        mock_sleep: MagicMock,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ConnectionError("down")
        mock_ticker_cls.return_value = mock_ticker
        ingester = MarketDataIngester(
            engine=mock_engine, tickers=["A", "B"], max_retries=1
        )

        with pytest.raises(RuntimeError, match="No tickers could be downloaded"):
            ingester.run(parallel=True)

    @patch("src.data.ingestion.time.sleep")
    @patch("src.data.ingestion.yf.Ticker")
    def test_single_ticker_uses_sequential(
        self,
        mock_ticker_cls: MagicMock,
        mock_sleep: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        """With a single ticker, parallel=True still works (sequential path)."""
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
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

    @patch("src.data.ingestion.yf.Ticker")
    def test_download_uses_configured_dates(
        self,
        mock_ticker_cls: MagicMock,
        sample_yf_dataframe: pd.DataFrame,
        mock_engine: MagicMock,
    ) -> None:
        mock_ticker_cls.return_value = _make_ticker_mock(sample_yf_dataframe)
        ingester = MarketDataIngester(
            engine=mock_engine,
            tickers=["SPY"],
            start_date=date(2020, 6, 1),
            end_date=date(2025, 6, 1),
        )

        ingester._download_ticker("SPY")

        # Verify Ticker was constructed and history called with correct dates
        mock_ticker = mock_ticker_cls.return_value
        _, kwargs = mock_ticker.history.call_args
        assert kwargs["start"] == "2020-06-01"
        assert kwargs["end"] == "2025-06-01"


# ---------------------------------------------------------------------------
# 13. _resolve_tickers includes benchmark
# ---------------------------------------------------------------------------


class TestResolveTickers:

    @patch("src.data.ingestion.load_ticker_config")
    def test_benchmark_included_from_config(
        self, mock_config: MagicMock,
    ) -> None:
        """Benchmark ticker (SPY) must be included even though it's
        not in the tickers list of the config file."""
        mock_config.return_value = {
            "tickers": ["AAPL", "MSFT"],
            "benchmark": "SPY",
            "market": "US",
        }
        result = _resolve_tickers()
        assert "SPY" in result
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("src.data.ingestion.load_ticker_config")
    def test_benchmark_not_duplicated_if_already_in_tickers(
        self, mock_config: MagicMock,
    ) -> None:
        mock_config.return_value = {
            "tickers": ["AAPL", "SPY", "MSFT"],
            "benchmark": "SPY",
            "market": "US",
        }
        result = _resolve_tickers()
        assert result.count("SPY") == 1

    def test_explicit_tickers_deduped(self) -> None:
        result = _resolve_tickers(["AAPL", "AAPL", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_explicit_tickers_returned_as_is(self) -> None:
        result = _resolve_tickers(["GOOG", "NVDA"])
        assert result == ["GOOG", "NVDA"]


# ---------------------------------------------------------------------------
# 14. _validate_ohlcv: data integrity checks
# ---------------------------------------------------------------------------


class TestValidateOhlcv:
    """Tests for OHLCV data integrity validation."""

    def test_valid_data_passes_through(self, mock_engine: MagicMock) -> None:
        df = pl.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["SPY", "SPY"],
            "open": [450.0, 452.0],
            "high": [455.0, 458.0],
            "low": [448.0, 449.0],
            "close": [452.0, 456.0],
            "volume": [80_000_000, 90_000_000],
            "adj_close": [452.0, 456.0],
        })
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 2

    @staticmethod
    def _make_valid_ohlcv(n: int = 25) -> pl.DataFrame:
        """Helper: generate n valid OHLCV rows for SPY."""
        return pl.DataFrame({
            "date": [date(2024, 1, i + 1) for i in range(n)],
            "ticker": ["SPY"] * n,
            "open": [450.0 + i for i in range(n)],
            "high": [455.0 + i for i in range(n)],
            "low": [448.0 + i for i in range(n)],
            "close": [452.0 + i for i in range(n)],
            "volume": [80_000_000] * n,
            "adj_close": [452.0 + i for i in range(n)],
        })

    def test_zero_close_dropped(self, mock_engine: MagicMock) -> None:
        df = self._make_valid_ohlcv(25)
        # Inject 1 bad row (close=0) — 1/26 ≈ 3.8% < 5% threshold
        bad = pl.DataFrame({
            "date": [date(2024, 2, 1)],
            "ticker": ["SPY"],
            "open": [0.0], "high": [0.0], "low": [0.0], "close": [0.0],
            "volume": [90_000_000], "adj_close": [0.0],
        })
        df = pl.concat([df, bad])
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 25
        assert (result["close"] > 0).all()

    def test_negative_close_dropped(self, mock_engine: MagicMock) -> None:
        df = self._make_valid_ohlcv(25)
        bad = pl.DataFrame({
            "date": [date(2024, 2, 1)],
            "ticker": ["SPY"],
            "open": [-1.0], "high": [-1.0], "low": [-1.0], "close": [-5.0],
            "volume": [90_000_000], "adj_close": [-5.0],
        })
        df = pl.concat([df, bad])
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 25

    def test_high_less_than_low_dropped(self, mock_engine: MagicMock) -> None:
        df = self._make_valid_ohlcv(25)
        bad = pl.DataFrame({
            "date": [date(2024, 2, 1)],
            "ticker": ["SPY"],
            "open": [452.0], "high": [449.0], "low": [458.0],  # high < low
            "close": [456.0], "volume": [90_000_000], "adj_close": [456.0],
        })
        df = pl.concat([df, bad])
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 25

    def test_negative_volume_dropped(self, mock_engine: MagicMock) -> None:
        """Rows with volume < 0 must be filtered out (not just counted)."""
        df = self._make_valid_ohlcv(25)
        bad = pl.DataFrame({
            "date": [date(2024, 2, 1)],
            "ticker": ["SPY"],
            "open": [452.0], "high": [458.0], "low": [449.0],
            "close": [456.0], "volume": [-100], "adj_close": [456.0],
        })
        df = pl.concat([df, bad])
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 25
        assert (result["volume"].cast(pl.Int64) >= 0).all()

    def test_multi_violation_row_counted_once(self, mock_engine: MagicMock) -> None:
        """A row violating both close<=0 AND high<low counts as 1 dropped row."""
        df = self._make_valid_ohlcv(25)
        bad = pl.DataFrame({
            "date": [date(2024, 2, 1)],
            "ticker": ["SPY"],
            "open": [0.0], "high": [1.0], "low": [5.0],  # close=0 AND high<low
            "close": [0.0], "volume": [90_000_000], "adj_close": [0.0],
        })
        df = pl.concat([df, bad])
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        # Only 1 row dropped, not 2 (old code would double-count)
        assert result.height == 25

    def test_null_volume_preserved(self, mock_engine: MagicMock) -> None:
        """Null volume is acceptable (propagates as null features downstream)."""
        df = pl.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["SPY", "SPY"],
            "open": [450.0, 452.0],
            "high": [455.0, 458.0],
            "low": [448.0, 449.0],
            "close": [452.0, 456.0],
            "volume": [80_000_000, None],
            "adj_close": [452.0, 456.0],
        })
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        result = ingester._validate_ohlcv(df, "SPY")
        assert result.height == 2

    def test_massive_corruption_raises(self, mock_engine: MagicMock) -> None:
        """If >5% of rows are invalid, raise ValueError."""
        # 10 rows, 6 with close=0 → 60% corruption
        df = pl.DataFrame({
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "ticker": ["SPY"] * 10,
            "open": [450.0] * 4 + [0.0] * 6,
            "high": [455.0] * 4 + [0.0] * 6,
            "low": [448.0] * 4 + [0.0] * 6,
            "close": [452.0] * 4 + [0.0] * 6,
            "volume": [80_000_000] * 10,
            "adj_close": [452.0] * 4 + [0.0] * 6,
        })
        ingester = MarketDataIngester(engine=mock_engine, tickers=["SPY"])
        with pytest.raises(ValueError, match="systematic corruption"):
            ingester._validate_ohlcv(df, "SPY")
