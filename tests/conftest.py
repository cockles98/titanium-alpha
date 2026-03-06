"""Shared pytest fixtures for Titanium Alpha."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest


@pytest.fixture()
def mock_engine() -> MagicMock:
    """A fake SQLAlchemy Engine that supports context-manager ``begin()``."""
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine


@pytest.fixture()
def sample_yf_dataframe() -> pd.DataFrame:
    """Return a Pandas DataFrame that mimics yfinance MultiIndex output.

    Columns are a MultiIndex with levels (Price, Ticker), and the index
    is a DatetimeIndex named ``Date`` -- exactly what ``yf.download``
    returns when ``auto_adjust=False``.
    """
    n = 5
    base_date = date(2024, 1, 2)
    dates = pd.DatetimeIndex(
        [base_date + timedelta(days=i) for i in range(n)],
        name="Date",
    )
    ticker = "SPY"

    arrays = [
        ["Open", "High", "Low", "Close", "Volume", "Adj Close"],
        [ticker] * 6,
    ]
    columns = pd.MultiIndex.from_arrays(arrays)

    data = {
        ("Open", ticker): [450.0 + i for i in range(n)],
        ("High", ticker): [455.0 + i for i in range(n)],
        ("Low", ticker): [448.0 + i for i in range(n)],
        ("Close", ticker): [452.0 + i for i in range(n)],
        ("Volume", ticker): [80_000_000 + i * 1_000_000 for i in range(n)],
        ("Adj Close", ticker): [451.0 + i for i in range(n)],
    }

    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture()
def sample_ohlcv_df() -> pl.DataFrame:
    """Return a realistic OHLCV Polars DataFrame for feature engineering tests.

    Generates 100 trading days of synthetic data with:
    - Upward-trending close prices with some noise
    - Reasonable OHLC relationships (high >= close >= low, etc.)
    - Volume with variation
    """
    import random

    random.seed(42)
    n = 100
    base_date = date(2023, 1, 2)
    dates = [base_date + timedelta(days=i) for i in range(n)]

    # Generate a random-walk close series
    closes: list[float] = [450.0]
    for _ in range(n - 1):
        change = random.gauss(0.2, 2.0)
        closes.append(max(closes[-1] + change, 10.0))

    opens = [c + random.gauss(0, 0.5) for c in closes]
    highs = [max(o, c) + abs(random.gauss(0, 1.5)) for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(random.gauss(0, 1.5)) for o, c in zip(opens, closes)]
    volumes = [int(80_000_000 + random.gauss(0, 5_000_000)) for _ in range(n)]
    adj_closes = [c * 0.998 for c in closes]

    return pl.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * n,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "adj_close": adj_closes,
        }
    )


@pytest.fixture()
def sample_features_df(sample_ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """Return OHLCV DataFrame with all 9 technical features computed.

    Uses ``compute_all_features`` from the features module to generate
    a realistic 17-column DataFrame suitable for PatchTST tests.
    Nulls from feature warmup are dropped.
    """
    from src.models.features import compute_all_features

    return compute_all_features(sample_ohlcv_df).drop_nulls()


@pytest.fixture()
def expected_polars_schema() -> dict[str, pl.DataType]:
    """The canonical column-name -> Polars dtype mapping after ingestion."""
    return {
        "date": pl.Date,
        "ticker": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "adj_close": pl.Float64,
    }
