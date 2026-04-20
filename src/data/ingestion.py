"""Market data ingestion pipeline.

Downloads OHLCV data from Yahoo Finance, transforms with Polars,
and persists to PostgreSQL.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import polars as pl
import yfinance as yf
from loguru import logger
from sqlalchemy import Engine, text

from src.config import load_ticker_config, load_tickers
from src.utils.db import get_postgres_engine

DEFAULT_TICKERS: list[str] = ["SPY", "NVDA", "AAPL", "QQQ"]


def _resolve_tickers(tickers: list[str] | None = None) -> list[str]:
    """Resolve tickers from argument, config file, or hardcoded fallback.

    When loading from config, the benchmark ticker (e.g. SPY) is included
    automatically so that downstream benchmark calculations have data.

    Args:
        tickers: Explicit list of tickers.  If ``None``, tries to load
            from ``config/tickers.json``, falling back to
            ``DEFAULT_TICKERS``.

    Returns:
        List of ticker symbols (deduplicated).
    """
    if tickers is not None:
        return list(dict.fromkeys(tickers))
    try:
        config = load_ticker_config()
        all_tickers = list(config["tickers"])
        benchmark = config.get("benchmark")
        if benchmark and benchmark not in all_tickers:
            all_tickers.append(benchmark)
        return all_tickers
    except Exception:
        return DEFAULT_TICKERS
OHLCV_TABLE: str = "market_ohlcv"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {OHLCV_TABLE} (
    date        DATE        NOT NULL,
    ticker      VARCHAR(10) NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION,
    PRIMARY KEY (date, ticker)
);
"""


class MarketDataIngester:
    """Downloads OHLCV market data and stores it in PostgreSQL.

    Args:
        engine: SQLAlchemy engine. If None, creates one from env vars.
        tickers: List of ticker symbols to download.
        years: Number of years of history to fetch (ignored when
            ``start_date`` is provided).
        max_retries: Max download attempts per ticker on network failure.
        start_date: Explicit start date.  When set, ``years`` is ignored.
        end_date: Explicit end date.  Defaults to today.
        max_workers: Number of parallel download threads.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        tickers: list[str] | None = None,
        years: int = 5,
        max_retries: int = 3,
        start_date: date | None = None,
        end_date: date | None = None,
        max_workers: int = 5,
    ) -> None:
        self.engine = engine or get_postgres_engine()
        self.tickers = _resolve_tickers(tickers)
        self.years = years
        self.max_retries = max_retries
        self.max_workers = max_workers

        self.end_date = end_date or date.today()
        if start_date is not None:
            self.start_date = start_date
        else:
            self.start_date = self.end_date - timedelta(days=self.years * 365)

    def _ensure_table(self) -> None:
        """Create the OHLCV table if it doesn't exist."""
        with self.engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
        logger.info("Table '{}' ready", OHLCV_TABLE)

    @staticmethod
    def _validate_ohlcv(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
        """Validate OHLCV data integrity and drop corrupt rows.

        Guards against upstream data issues (yfinance API glitches,
        corporate action misapplication) that would cascade as -inf/NaN
        through downstream ``log()`` calls in features and walk-forward.

        Checks:
            - ``close > 0`` (critical: ``log(0)`` = -inf)
            - ``high >= low`` (impossible OHLCV invariant)
            - ``volume >= 0``

        Args:
            df: Downloaded OHLCV DataFrame for a single ticker.
            ticker: Ticker symbol (for log messages).

        Returns:
            Cleaned DataFrame with invalid rows removed.

        Raises:
            ValueError: If more than 5% of rows are invalid,
                suggesting systematic data corruption.
        """
        n_before = df.height

        # Count individual violation types for diagnostics
        invalid_close = df.filter(
            pl.col("close").is_null() | (pl.col("close") <= 0)
        ).height
        invalid_hl = df.filter(pl.col("high") < pl.col("low")).height
        invalid_vol = df.filter(pl.col("volume") < 0).height

        # Fast path: no violations detected
        if invalid_close == 0 and invalid_hl == 0 and invalid_vol == 0:
            return df

        # Drop rows that would break downstream math
        # (volume null is OK — propagates as null features; volume < 0 is not)
        df_clean = df.filter(
            pl.col("close").is_not_null()
            & (pl.col("close") > 0)
            & (pl.col("high") >= pl.col("low"))
            & (pl.col("volume").is_null() | (pl.col("volume") >= 0))
        )

        # Use actual row-count delta (avoids double-counting rows
        # that violate multiple conditions simultaneously)
        n_dropped = n_before - df_clean.height

        logger.warning(
            "{}: dropped {} invalid rows (close<=0: {}, high<low: {}, "
            "vol<0: {}) | {} → {} rows",
            ticker,
            n_dropped,
            invalid_close,
            invalid_hl,
            invalid_vol,
            n_before,
            df_clean.height,
        )

        corruption_pct = n_dropped / n_before if n_before > 0 else 0
        if corruption_pct > 0.05:
            raise ValueError(
                f"{ticker}: {corruption_pct:.1%} of rows invalid "
                f"({n_dropped}/{n_before}) — likely systematic corruption"
            )

        return df_clean

    def _download_ticker(self, ticker: str) -> pl.DataFrame:
        """Download OHLCV for a single ticker with retry logic.

        Args:
            ticker: The ticker symbol (e.g. "SPY").

        Returns:
            Polars DataFrame with columns:
            [date, ticker, open, high, low, close, volume, adj_close].

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Downloading {} ({} → {}) | attempt {}/{}",
                    ticker,
                    self.start_date,
                    self.end_date,
                    attempt,
                    self.max_retries,
                )
                # Use yf.Ticker().history() instead of yf.download()
                # to avoid thread-safety issues with yfinance's shared
                # session/cache that cause adjacent tickers to get
                # identical data in parallel downloads.
                t = yf.Ticker(ticker)
                pdf = t.history(
                    start=str(self.start_date),
                    end=str(self.end_date),
                    auto_adjust=True,
                )

                if pdf.empty:
                    raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

                # yf.Ticker.history() returns simple column names
                # (Open, High, Low, Close, Volume, Dividends, Stock Splits)
                pdf = pdf.reset_index()

                # Convert Pandas → Polars immediately
                df = pl.from_pandas(pdf)

                # Normalize column names to our schema.
                # With auto_adjust=True, OHLC are already split+dividend
                # adjusted — no separate "Adj Close" column exists.
                df = df.rename(
                    {
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                # adj_close = close when auto_adjust=True
                df = df.with_columns([
                    pl.lit(ticker).alias("ticker"),
                    pl.col("close").alias("adj_close"),
                ])

                # Select and cast to final schema
                df = df.select(
                    pl.col("date").cast(pl.Date),
                    pl.col("ticker").cast(pl.Utf8),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                    pl.col("adj_close").cast(pl.Float64),
                )

                df = self._validate_ohlcv(df, ticker)

                logger.info(
                    "{} downloaded | {} rows | {} → {}",
                    ticker,
                    df.height,
                    df["date"].min(),
                    df["date"].max(),
                )
                return df

            except Exception as exc:
                logger.warning(
                    "Attempt {}/{} failed for {}: {}",
                    attempt,
                    self.max_retries,
                    ticker,
                    exc,
                )
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to download {ticker} after {self.max_retries} attempts"
                    ) from exc
                time.sleep(2**attempt)

        # Unreachable, but satisfies type checker
        raise RuntimeError(f"Failed to download {ticker}")  # pragma: no cover

    def _save_to_postgres(self, df: pl.DataFrame) -> int:
        """Upsert a Polars DataFrame into PostgreSQL.

        Uses INSERT ... ON CONFLICT to avoid duplicates on re-runs.

        Args:
            df: DataFrame matching the OHLCV schema.

        Returns:
            Number of rows upserted.
        """
        rows = df.to_dicts()
        if not rows:
            return 0

        upsert_sql = text(f"""
            INSERT INTO {OHLCV_TABLE}
                (date, ticker, open, high, low, close, volume, adj_close)
            VALUES
                (:date, :ticker, :open, :high, :low, :close, :volume, :adj_close)
            ON CONFLICT (date, ticker) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adj_close = EXCLUDED.adj_close
        """)

        with self.engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        logger.info("Upserted {} rows into '{}'", len(rows), OHLCV_TABLE)
        return len(rows)

    def _download_batch(self) -> list[pl.DataFrame]:
        """Download all tickers in parallel using ThreadPoolExecutor.

        Submissions are staggered by 0.5 s to respect yfinance rate
        limits.  Each ticker retries independently via
        ``_download_ticker``.

        Returns:
            List of DataFrames (one per successfully downloaded ticker).
            Tickers that fail after all retries are logged and skipped.
        """
        frames: list[pl.DataFrame] = []
        failed: list[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, ticker in enumerate(self.tickers):
                if i > 0:
                    time.sleep(0.5)
                futures[pool.submit(self._download_ticker, ticker)] = ticker

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    frames.append(future.result())
                except RuntimeError:
                    logger.error("Skipping {} — all retries exhausted", ticker)
                    failed.append(ticker)

        if failed:
            logger.warning(
                "Failed tickers ({}/{}): {}",
                len(failed),
                len(self.tickers),
                failed,
            )
        return frames

    def run(self, *, parallel: bool = True) -> pl.DataFrame:
        """Execute the full ingestion pipeline.

        Downloads data for all tickers, saves to PostgreSQL,
        and returns the combined DataFrame.

        Args:
            parallel: If ``True`` (default), downloads tickers in
                parallel using ``ThreadPoolExecutor``.  Set to
                ``False`` for sequential downloads (backward compat).

        Returns:
            Combined Polars DataFrame with all tickers.

        Raises:
            RuntimeError: If no tickers could be downloaded.
        """
        self._ensure_table()

        if parallel and len(self.tickers) > 1:
            all_frames = self._download_batch()
        else:
            all_frames = []
            for ticker in self.tickers:
                try:
                    all_frames.append(self._download_ticker(ticker))
                except RuntimeError:
                    logger.error("Skipping {} — all retries exhausted", ticker)

        if not all_frames:
            raise RuntimeError("No tickers could be downloaded")

        for df in all_frames:
            self._save_to_postgres(df)

        combined = pl.concat(all_frames)
        logger.info(
            "Ingestion complete | {} tickers | {} total rows",
            len(all_frames),
            combined.height,
        )
        return combined


if __name__ == "__main__":
    ingester = MarketDataIngester(years=12)
    result = ingester.run()
    logger.info("Result shape: {} rows x {} cols", result.height, result.width)
    logger.info("Tickers: {}", result["ticker"].unique().to_list())
