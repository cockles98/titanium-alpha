"""Market data ingestion pipeline.

Downloads OHLCV data from Yahoo Finance, transforms with Polars,
and persists to PostgreSQL.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import polars as pl
import yfinance as yf
from loguru import logger
from sqlalchemy import Engine, text

from src.utils.db import get_postgres_engine

DEFAULT_TICKERS: list[str] = ["SPY", "NVDA", "AAPL", "QQQ"]
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
        years: Number of years of history to fetch.
        max_retries: Max download attempts per ticker on network failure.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        tickers: list[str] | None = None,
        years: int = 5,
        max_retries: int = 3,
    ) -> None:
        self.engine = engine or get_postgres_engine()
        self.tickers = tickers or DEFAULT_TICKERS
        self.years = years
        self.max_retries = max_retries

    def _ensure_table(self) -> None:
        """Create the OHLCV table if it doesn't exist."""
        with self.engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
        logger.info("Table '{}' ready", OHLCV_TABLE)

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
        end_date = date.today()
        start_date = end_date - timedelta(days=self.years * 365)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Downloading {} ({} → {}) | attempt {}/{}",
                    ticker,
                    start_date,
                    end_date,
                    attempt,
                    self.max_retries,
                )
                pdf = yf.download(
                    ticker,
                    start=str(start_date),
                    end=str(end_date),
                    auto_adjust=False,
                    progress=False,
                )

                if pdf.empty:
                    raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

                # yfinance returns MultiIndex columns: (Price, Ticker)
                # Flatten by taking the first level
                pdf.columns = [col[0] for col in pdf.columns]
                pdf = pdf.reset_index()

                # Convert Pandas → Polars immediately
                df = pl.from_pandas(pdf)

                # Normalize column names to our schema
                df = df.rename(
                    {
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                        "Adj Close": "adj_close",
                    }
                )
                df = df.with_columns(pl.lit(ticker).alias("ticker"))

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

    def run(self) -> pl.DataFrame:
        """Execute the full ingestion pipeline.

        Downloads data for all tickers, saves to PostgreSQL,
        and returns the combined DataFrame.

        Returns:
            Combined Polars DataFrame with all tickers.
        """
        self._ensure_table()

        all_frames: list[pl.DataFrame] = []
        for ticker in self.tickers:
            df = self._download_ticker(ticker)
            self._save_to_postgres(df)
            all_frames.append(df)

        combined = pl.concat(all_frames)
        logger.info(
            "Ingestion complete | {} tickers | {} total rows",
            len(self.tickers),
            combined.height,
        )
        return combined


if __name__ == "__main__":
    ingester = MarketDataIngester()
    result = ingester.run()
    logger.info("Result shape: {} rows x {} cols", result.height, result.width)
    logger.info("Tickers: {}", result["ticker"].unique().to_list())
