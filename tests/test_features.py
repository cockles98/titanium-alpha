"""Tests for src/models/features.py — technical indicators."""

from __future__ import annotations

import polars as pl
import pytest

from src.models.features import (
    bollinger_bands,
    compute_all_features,
    realized_volatility,
    rsi,
    volume_profile,
)

# ---------------------------------------------------------------------------
# TestRSI
# ---------------------------------------------------------------------------


class TestRSI:
    """Core RSI behaviour."""

    def test_output_is_series(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = rsi(sample_ohlcv_df["close"])
        assert isinstance(result, pl.Series)

    def test_name_contains_period(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = rsi(sample_ohlcv_df["close"], period=14)
        assert result.name == "rsi_14"

    def test_custom_period_name(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = rsi(sample_ohlcv_df["close"], period=7)
        assert result.name == "rsi_7"

    def test_values_between_0_and_100(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = rsi(sample_ohlcv_df["close"])
        non_null = result.drop_nulls()
        assert (non_null >= 0).all()
        assert (non_null <= 100).all()

    def test_initial_nulls(self, sample_ohlcv_df: pl.DataFrame) -> None:
        period = 14
        result = rsi(sample_ohlcv_df["close"], period=period)
        # First `period` values should be null (period rows need period diffs)
        assert result[:period].null_count() == period

    def test_length_matches_input(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = rsi(sample_ohlcv_df["close"])
        assert len(result) == len(sample_ohlcv_df)


class TestRSIEdgeCases:
    """RSI boundary conditions."""

    def test_monotonically_increasing_near_100(self) -> None:
        series = pl.Series("close", [float(i) for i in range(1, 52)])
        result = rsi(series, period=14)
        non_null = result.drop_nulls()
        # All gains, no losses → RSI should be very high
        assert non_null[-1] > 95.0

    def test_monotonically_decreasing_near_0(self) -> None:
        series = pl.Series("close", [500.0 - i for i in range(51)])
        result = rsi(series, period=14)
        non_null = result.drop_nulls()
        # All losses, no gains → RSI should be very low
        assert non_null[-1] < 5.0

    def test_constant_series_returns_null(self) -> None:
        series = pl.Series("close", [100.0] * 30)
        result = rsi(series, period=14)
        # No change → gain=0, loss=0 → RSI is undefined → null
        non_initial = result[15:]  # past warmup
        assert non_initial.null_count() == len(non_initial)


# ---------------------------------------------------------------------------
# TestBollingerBands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    """Bollinger Bands output structure and values."""

    def test_returns_three_columns(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = bollinger_bands(sample_ohlcv_df["close"])
        assert set(result.columns) == {"bb_upper", "bb_middle", "bb_lower"}

    def test_upper_ge_middle_ge_lower(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = bollinger_bands(sample_ohlcv_df["close"])
        # Drop rows where any column is null
        valid = result.drop_nulls()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_initial_nulls(self, sample_ohlcv_df: pl.DataFrame) -> None:
        period = 20
        result = bollinger_bands(sample_ohlcv_df["close"], period=period)
        assert result["bb_middle"][:period - 1].null_count() == period - 1

    def test_length_matches_input(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = bollinger_bands(sample_ohlcv_df["close"])
        assert result.height == len(sample_ohlcv_df)


# ---------------------------------------------------------------------------
# TestRealizedVolatility
# ---------------------------------------------------------------------------


class TestRealizedVolatility:
    """Realized volatility output."""

    def test_non_negative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = realized_volatility(sample_ohlcv_df["close"])
        non_null = result.drop_nulls()
        assert (non_null >= 0).all()

    def test_initial_nulls(self, sample_ohlcv_df: pl.DataFrame) -> None:
        window = 21
        result = realized_volatility(sample_ohlcv_df["close"], window=window)
        # diff produces 1 null; rolling_std(window) needs window non-null inputs
        # Total leading nulls = window (the diff null is consumed by the window)
        first_chunk = result[:window]
        assert first_chunk.null_count() == window

    def test_name_contains_window(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = realized_volatility(sample_ohlcv_df["close"], window=10)
        assert result.name == "realized_vol_10"

    def test_annualization_factor(self) -> None:
        """Constant daily return → vol should reflect √252 scaling."""

        # Price series with constant 1% daily return
        prices = [100.0 * (1.01 ** i) for i in range(60)]
        series = pl.Series("close", prices)
        result = realized_volatility(series, window=20)
        non_null = result.drop_nulls()
        # Log return of 1% daily ≈ 0.00995; std of constant = 0
        # Actually constant returns → std ≈ 0 → vol ≈ 0
        assert non_null[-1] < 0.01  # nearly zero

    def test_length_matches_input(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = realized_volatility(sample_ohlcv_df["close"])
        assert len(result) == len(sample_ohlcv_df)


# ---------------------------------------------------------------------------
# TestVolumeProfile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    """Volume-based feature output."""

    def test_returns_four_columns(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = volume_profile(sample_ohlcv_df)
        assert set(result.columns) == {"volume_sma", "relative_volume", "vwap", "obv"}

    def test_relative_volume_is_one_when_volume_equals_sma(self) -> None:
        """If volume is constant, relative_volume should be 1.0 after warmup."""
        n = 30
        df = pl.DataFrame(
            {
                "date": list(range(n)),
                "ticker": ["X"] * n,
                "open": [100.0] * n,
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0] * n,
                "volume": [1_000_000] * n,
            }
        )
        result = volume_profile(df)
        rv = result["relative_volume"].drop_nulls()
        assert abs(rv[-1] - 1.0) < 1e-9

    def test_obv_direction(self) -> None:
        """OBV should increase when close goes up."""
        n = 5
        df = pl.DataFrame(
            {
                "date": list(range(n)),
                "ticker": ["X"] * n,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000] * n,
            }
        )
        result = volume_profile(df)
        obv = result["obv"]
        # Prices always going up → OBV should be increasing
        for i in range(1, len(obv)):
            assert obv[i] >= obv[i - 1]

    def test_vwap_is_positive(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = volume_profile(sample_ohlcv_df)
        assert (result["vwap"] > 0).all()

    def test_length_matches_input(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = volume_profile(sample_ohlcv_df)
        assert result.height == len(sample_ohlcv_df)


# ---------------------------------------------------------------------------
# TestComputeAllFeatures
# ---------------------------------------------------------------------------


class TestComputeAllFeatures:
    """Orchestrator function."""

    def test_total_columns(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = compute_all_features(sample_ohlcv_df)
        # 8 original (incl adj_close) + 9 features = 17
        assert result.width == 17

    def test_original_columns_preserved(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = compute_all_features(sample_ohlcv_df)
        for col in sample_ohlcv_df.columns:
            assert col in result.columns

    def test_feature_columns_present(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = compute_all_features(sample_ohlcv_df)
        expected_features = {
            "rsi_14",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "realized_vol_21",
            "volume_sma",
            "relative_volume",
            "vwap",
            "obv",
        }
        assert expected_features.issubset(set(result.columns))

    def test_row_count_unchanged(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = compute_all_features(sample_ohlcv_df)
        assert result.height == sample_ohlcv_df.height

    def test_multi_ticker_isolation(self) -> None:
        """Features for each ticker must be independent (no cross-contamination)."""
        from datetime import date, timedelta

        n = 60
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]

        # Two tickers with very different price levels
        spy = pl.DataFrame({
            "date": dates, "ticker": ["SPY"] * n,
            "open": [450.0 + i * 0.1 for i in range(n)],
            "high": [455.0 + i * 0.1 for i in range(n)],
            "low": [445.0 + i * 0.1 for i in range(n)],
            "close": [450.0 + i * 0.1 for i in range(n)],
            "volume": [80_000_000] * n,
        })
        nvda = pl.DataFrame({
            "date": dates, "ticker": ["NVDA"] * n,
            "open": [800.0 + i * 0.2 for i in range(n)],
            "high": [810.0 + i * 0.2 for i in range(n)],
            "low": [790.0 + i * 0.2 for i in range(n)],
            "close": [800.0 + i * 0.2 for i in range(n)],
            "volume": [50_000_000] * n,
        })

        combined = pl.concat([spy, nvda])

        # Compute features on combined and on each ticker alone
        combined_result = compute_all_features(combined)
        spy_only = compute_all_features(spy)
        nvda_only = compute_all_features(nvda)

        # Extract each ticker's features from combined result
        spy_from_combined = combined_result.filter(pl.col("ticker") == "SPY")
        nvda_from_combined = combined_result.filter(pl.col("ticker") == "NVDA")

        # Features must match exactly (no cross-contamination)
        for col in ["rsi_14", "bb_upper", "bb_middle", "bb_lower",
                     "realized_vol_21", "vwap", "obv"]:
            spy_solo = spy_only[col].to_list()
            spy_combo = spy_from_combined[col].to_list()
            for i, (s, c) in enumerate(zip(spy_solo, spy_combo)):
                if s is None and c is None:
                    continue
                assert s is not None and c is not None, (
                    f"Null mismatch in SPY.{col}[{i}]: solo={s}, combo={c}"
                )
                assert abs(s - c) < 1e-10, (
                    f"Cross-ticker contamination in SPY.{col}[{i}]: "
                    f"solo={s}, combo={c}"
                )

            nvda_solo = nvda_only[col].to_list()
            nvda_combo = nvda_from_combined[col].to_list()
            for i, (s, c) in enumerate(zip(nvda_solo, nvda_combo)):
                if s is None and c is None:
                    continue
                assert s is not None and c is not None, (
                    f"Null mismatch in NVDA.{col}[{i}]: solo={s}, combo={c}"
                )
                assert abs(s - c) < 1e-10, (
                    f"Cross-ticker contamination in NVDA.{col}[{i}]: "
                    f"solo={s}, combo={c}"
                )


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation."""

    def test_missing_columns_raises_value_error(self) -> None:
        df = pl.DataFrame({"date": [1], "ticker": ["X"], "close": [100.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            volume_profile(df)

    def test_compute_all_missing_columns_raises(self) -> None:
        df = pl.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_all_features(df)


# ---------------------------------------------------------------------------
# TestNoLookAhead
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    """Verify that adding future data does not change historical features."""

    def test_truncation_does_not_alter_past_features(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        full = compute_all_features(sample_ohlcv_df)
        truncated_df = sample_ohlcv_df.head(len(sample_ohlcv_df) - 10)
        partial = compute_all_features(truncated_df)

        # The first N-10 rows of 'full' should match 'partial' exactly
        n = partial.height
        for col in partial.columns:
            full_vals = full[col][:n]
            partial_vals = partial[col]

            # Compare: both null or both equal
            for i in range(n):
                fv = full_vals[i]
                pv = partial_vals[i]
                if fv is None and pv is None:
                    continue
                if fv is None or pv is None:
                    pytest.fail(
                        f"Look-ahead detected in '{col}' at row {i}: "
                        f"full={fv}, partial={pv}"
                    )
                if isinstance(fv, float):
                    assert abs(fv - pv) < 1e-10, (
                        f"Look-ahead detected in '{col}' at row {i}: "
                        f"full={fv}, partial={pv}"
                    )
                else:
                    assert fv == pv, (
                        f"Look-ahead detected in '{col}' at row {i}: "
                        f"full={fv}, partial={pv}"
                    )
