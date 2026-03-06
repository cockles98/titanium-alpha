"""Feature engineering for financial time series.

Computes technical indicators from OHLCV data to feed PatchTST.
All functions use backward-only windows to prevent look-ahead bias.
"""

from __future__ import annotations

import math

import polars as pl
from loguru import logger

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_OHLCV_REQUIRED_COLS: list[str] = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def _validate_ohlcv(df: pl.DataFrame, required_cols: list[str] | None = None) -> None:
    """Raise ``ValueError`` if *df* is missing required columns.

    Args:
        df: DataFrame to validate.
        required_cols: Column names to check. Defaults to ``_OHLCV_REQUIRED_COLS``.
    """
    cols = required_cols or _OHLCV_REQUIRED_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _log_null_summary(result: pl.DataFrame | pl.Series, name: str) -> None:
    """Log a debug-level summary of null counts in *result*.

    Args:
        result: A Polars Series or DataFrame.
        name: Human-readable label for the log line.
    """
    if isinstance(result, pl.Series):
        nulls = result.null_count()
        logger.debug("{}: {} nulls out of {} rows", name, nulls, len(result))
    else:
        for col in result.columns:
            nulls = result[col].null_count()
            logger.debug("{}.{}: {} nulls out of {} rows", name, col, nulls, len(result))


# ---------------------------------------------------------------------------
# Public API — standalone indicators
# ---------------------------------------------------------------------------


def rsi(series: pl.Series, period: int = 14) -> pl.Series:
    """Compute RSI (SMA-based variant).

    Args:
        series: Price series (typically close prices).
        period: Look-back window size.

    Returns:
        Series named ``rsi_{period}`` with values in [0, 100].
        The first *period* values are null.
    """
    delta = series.diff()

    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    avg_gain = gain.rolling_mean(window_size=period, min_samples=period)
    avg_loss = loss.rolling_mean(window_size=period, min_samples=period)

    rs = avg_gain / avg_loss

    rsi_values = pl.Series(
        name=f"rsi_{period}",
        values=[
            None if r is None else (
                None if (g == 0 and l == 0) else
                100.0 if l == 0 else
                100.0 - 100.0 / (1.0 + r)
            )
            for r, g, l in zip(
                rs.to_list(), avg_gain.to_list(), avg_loss.to_list()
            )
        ],
    )

    _log_null_summary(rsi_values, f"rsi_{period}")
    return rsi_values


def bollinger_bands(
    series: pl.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pl.DataFrame:
    """Compute Bollinger Bands.

    Args:
        series: Price series (typically close prices).
        period: Window for the simple moving average.
        num_std: Number of standard deviations for the bands.

    Returns:
        DataFrame with columns ``bb_upper``, ``bb_middle``, ``bb_lower``.
    """
    middle = series.rolling_mean(window_size=period, min_samples=period)
    std = series.rolling_std(window_size=period, min_samples=period)

    upper = middle + num_std * std
    lower = middle - num_std * std

    result = pl.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
        }
    )

    _log_null_summary(result, "bollinger_bands")
    return result


def realized_volatility(series: pl.Series, window: int = 21) -> pl.Series:
    """Compute annualized realized volatility from log returns.

    Args:
        series: Price series (typically close prices).
        window: Rolling window for standard deviation of log returns.

    Returns:
        Series named ``realized_vol_{window}``. The first ``window + 1``
        values are null (1 for the log-return diff, then *window* for
        the rolling std).
    """
    log_returns = series.log().diff()

    vol = log_returns.rolling_std(window_size=window, min_samples=window)

    annualized = vol * math.sqrt(252)

    result = annualized.alias(f"realized_vol_{window}")
    _log_null_summary(result, f"realized_vol_{window}")
    return result


def volume_profile(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """Compute volume-based features from an OHLCV DataFrame.

    Args:
        ohlcv_df: DataFrame with at least ``open``, ``high``, ``low``,
            ``close``, and ``volume`` columns.

    Returns:
        DataFrame with columns: ``volume_sma``, ``relative_volume``,
        ``vwap`` (cumulative), ``obv``.
    """
    _validate_ohlcv(ohlcv_df)

    volume = ohlcv_df["volume"].cast(pl.Float64)

    # Volume SMA (20-day)
    volume_sma = volume.rolling_mean(window_size=20, min_samples=20).alias("volume_sma")

    # Relative volume (current / SMA)
    relative_volume = (volume / volume_sma).alias("relative_volume")

    # Cumulative VWAP: cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (
        ohlcv_df["high"].cast(pl.Float64)
        + ohlcv_df["low"].cast(pl.Float64)
        + ohlcv_df["close"].cast(pl.Float64)
    ) / 3.0
    tp_vol = typical_price * volume
    vwap = (tp_vol.cum_sum() / volume.cum_sum()).alias("vwap")

    # On-Balance Volume
    close = ohlcv_df["close"].cast(pl.Float64)
    sign = close.diff().sign()
    # First value of sign is null from diff; set to 0 for OBV
    sign = sign.fill_null(0)
    obv = (sign * volume).cum_sum().alias("obv")

    result = pl.DataFrame(
        {
            "volume_sma": volume_sma,
            "relative_volume": relative_volume,
            "vwap": vwap,
            "obv": obv,
        }
    )

    _log_null_summary(result, "volume_profile")
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_features(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """Compute all technical features and concatenate with the original data.

    Args:
        ohlcv_df: OHLCV DataFrame (must contain date, ticker, open, high,
            low, close, volume columns).

    Returns:
        DataFrame with the original columns plus 9 feature columns
        (17 total when adj_close is present, 16 without).
    """
    _validate_ohlcv(ohlcv_df)

    close = ohlcv_df["close"].cast(pl.Float64)

    rsi_s = rsi(close)
    bb_df = bollinger_bands(close)
    rvol_s = realized_volatility(close)
    vp_df = volume_profile(ohlcv_df)

    result = pl.concat(
        [
            ohlcv_df,
            rsi_s.to_frame(),
            bb_df,
            rvol_s.to_frame(),
            vp_df,
        ],
        how="horizontal",
    )

    logger.info(
        "compute_all_features: {} rows × {} cols",
        result.height,
        result.width,
    )
    return result
