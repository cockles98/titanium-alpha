"""Tests for src/models/predict.py — PredictionPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data.ingestion import _resolve_tickers
from src.models.predict import PredictionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, tickers: list[str] | None = None) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    from datetime import date, timedelta
    import random

    random.seed(42)
    tickers = tickers or ["SPY"]
    frames: list[pl.DataFrame] = []

    for ticker in tickers:
        base = date(2020, 1, 2)
        dates = [base + timedelta(days=i) for i in range(n)]
        closes: list[float] = [450.0]
        for _ in range(n - 1):
            closes.append(max(closes[-1] + random.gauss(0.1, 2.0), 10.0))

        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "ticker": [ticker] * n,
                    "open": [c + random.gauss(0, 0.5) for c in closes],
                    "high": [c + abs(random.gauss(0, 1.5)) for c in closes],
                    "low": [c - abs(random.gauss(0, 1.5)) for c in closes],
                    "close": closes,
                    "volume": [int(80e6 + random.gauss(0, 5e6)) for _ in range(n)],
                    "adj_close": [c * 0.998 for c in closes],
                }
            )
        )

    return pl.concat(frames)


# ---------------------------------------------------------------------------
# TestPredictionPipelineInit
# ---------------------------------------------------------------------------


class TestPredictionPipelineInit:
    """Constructor and defaults."""

    @patch("src.models.predict.get_postgres_engine")
    def test_default_params(self, mock_engine_fn: MagicMock) -> None:
        mock_engine_fn.return_value = MagicMock()
        p = PredictionPipeline()
        assert p.val_size == 63
        assert p.max_steps == 5000
        assert p.tickers == _resolve_tickers(None)

    def test_custom_params(self) -> None:
        engine = MagicMock()
        p = PredictionPipeline(
            engine=engine,
            output_dir="/tmp/test",
            tickers=["SPY"],
            val_size=30,
            max_steps=100,
        )
        assert p.engine is engine
        assert p.tickers == ["SPY"]
        assert p.val_size == 30


# ---------------------------------------------------------------------------
# TestLoadOHLCV
# ---------------------------------------------------------------------------


class TestLoadOHLCV:
    """Data loading from PostgreSQL."""

    @patch("src.models.predict.get_postgres_engine")
    def test_load_returns_dataframe(self, mock_engine_fn: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine
        ohlcv = _make_ohlcv(50, ["SPY"])

        with patch("polars.read_database", return_value=ohlcv):
            p = PredictionPipeline(engine=mock_engine, tickers=["SPY"])
            result = p.load_ohlcv()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 50

    @patch("src.models.predict.get_postgres_engine")
    def test_empty_data_raises(self, mock_engine_fn: MagicMock) -> None:
        mock_engine = MagicMock()
        mock_engine_fn.return_value = mock_engine
        empty = pl.DataFrame(
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

        with patch("polars.read_database", return_value=empty):
            p = PredictionPipeline(engine=mock_engine)
            with pytest.raises(ValueError, match="No OHLCV data found"):
                p.load_ohlcv()


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """MAE and RMSE computation."""

    def test_perfect_forecast(self) -> None:
        """Zero error when forecast dates match actuals exactly."""
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5, 10)),
                "PatchTST-q0.5": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        actual = pl.DataFrame(
            {
                "date": list(range(10)),
                "ticker": ["SPY"] * 10,
                "close": [90.0] * 5 + [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        result = PredictionPipeline.compute_metrics(forecast, actual, h=5)
        assert result["mae"][0] == pytest.approx(0.0)
        assert result["rmse"][0] == pytest.approx(0.0)

    def test_known_error(self) -> None:
        """MAE and RMSE with known offsets via date-aligned join."""
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 3,
                "ds": [2, 3, 4],
                "PatchTST-q0.5": [102.0, 103.0, 104.0],
            }
        )
        actual = pl.DataFrame(
            {
                "date": list(range(5)),
                "ticker": ["SPY"] * 5,
                "close": [90.0, 91.0, 100.0, 101.0, 102.0],
            }
        )
        result = PredictionPipeline.compute_metrics(forecast, actual, h=3)
        # Joined on dates 2,3,4: forecast [102,103,104] vs actual [100,101,102]
        # Errors: [2, 2, 2] → MAE=2, RMSE=2
        assert result["mae"][0] == pytest.approx(2.0)
        assert result["rmse"][0] == pytest.approx(2.0)

    def test_multiple_tickers(self) -> None:
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY", "NVDA"],
                "ds": [0, 0],
                "PatchTST-q0.5": [105.0, 205.0],
            }
        )
        actual = pl.DataFrame(
            {
                "date": [0, 0],
                "ticker": ["SPY", "NVDA"],
                "close": [100.0, 200.0],
            }
        )
        result = PredictionPipeline.compute_metrics(forecast, actual, h=1)
        assert result.height == 2

    def test_no_overlap_returns_empty(self) -> None:
        """Forecast dates outside actual range → empty metrics (production case)."""
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(100, 105)),
                "PatchTST-q0.5": [110.0] * 5,
            }
        )
        actual = pl.DataFrame(
            {
                "date": list(range(10)),
                "ticker": ["SPY"] * 10,
                "close": [100.0] * 10,
            }
        )
        result = PredictionPipeline.compute_metrics(forecast, actual, h=5)
        assert result.height == 0

    def test_missing_median_col(self) -> None:
        """No median column → empty metrics."""
        forecast = pl.DataFrame(
            {"unique_id": ["SPY"], "ds": [0], "PatchTST-q0.1": [100.0]}
        )
        actual = pl.DataFrame({"date": [0], "ticker": ["SPY"], "close": [100.0]})
        result = PredictionPipeline.compute_metrics(forecast, actual)
        assert result.height == 0


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------


class TestRun:
    """Full pipeline execution with mocks."""

    @patch("src.models.predict.TitaniumForecaster")
    @patch("src.models.predict.compute_all_features")
    def test_run_end_to_end(
        self,
        mock_features: MagicMock,
        mock_forecaster_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        ohlcv = _make_ohlcv(200, ["SPY"])
        features = _make_ohlcv(200, ["SPY"])  # Simplified, just needs same structure
        mock_features.return_value = features

        # Mock forecaster
        mock_fc = MagicMock()
        mock_forecaster_class.return_value = mock_fc

        last_close = features.filter(pl.col("ticker") == "SPY")["close"][-1]
        mock_fc.h = 5
        mock_fc.predict.return_value = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.5": [last_close + 5] * 5,
            }
        )
        mock_fc.predict_proba.return_value = pl.DataFrame(
            {"ticker": ["SPY"], "prob_up": [0.8], "expected_return": [0.01]}
        )

        engine = MagicMock()
        p = PredictionPipeline(
            engine=engine,
            output_dir=str(tmp_path),
            tickers=["SPY"],
        )
        # Mock load_ohlcv to return our data
        p.load_ohlcv = MagicMock(return_value=ohlcv)

        result = p.run()

        assert isinstance(result, pl.DataFrame)
        assert "prob_up" in result.columns
        mock_fc.fit.assert_called_once()
        mock_fc.predict.assert_called_once()
        mock_fc.predict_proba.assert_called_once()

    @patch("src.models.predict.TitaniumForecaster")
    @patch("src.models.predict.compute_all_features")
    def test_run_saves_parquet_files(
        self,
        mock_features: MagicMock,
        mock_forecaster_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        ohlcv = _make_ohlcv(200, ["SPY"])
        mock_features.return_value = ohlcv

        mock_fc = MagicMock()
        mock_forecaster_class.return_value = mock_fc
        mock_fc.h = 5
        mock_fc.predict.return_value = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.5": [100.0] * 5,
            }
        )
        mock_fc.predict_proba.return_value = pl.DataFrame(
            {"ticker": ["SPY"], "prob_up": [0.6], "expected_return": [0.005]}
        )

        engine = MagicMock()
        p = PredictionPipeline(
            engine=engine,
            output_dir=str(tmp_path),
            tickers=["SPY"],
        )
        p.load_ohlcv = MagicMock(return_value=ohlcv)

        p.run()

        assert (tmp_path / "predictions.parquet").exists()
        assert (tmp_path / "forecast.parquet").exists()
        assert (tmp_path / "metrics.parquet").exists()

    @patch("src.models.predict.TitaniumForecaster")
    @patch("src.models.predict.compute_all_features")
    def test_run_passes_max_steps(
        self,
        mock_features: MagicMock,
        mock_forecaster_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        ohlcv = _make_ohlcv(200, ["SPY"])
        mock_features.return_value = ohlcv

        mock_fc = MagicMock()
        mock_forecaster_class.return_value = mock_fc
        mock_fc.h = 5
        mock_fc.predict.return_value = pl.DataFrame(
            {"unique_id": ["SPY"], "ds": [0], "PatchTST-q0.5": [100.0]}
        )
        mock_fc.predict_proba.return_value = pl.DataFrame(
            {"ticker": ["SPY"], "prob_up": [0.5], "expected_return": [0.0]}
        )

        engine = MagicMock()
        p = PredictionPipeline(
            engine=engine,
            output_dir=str(tmp_path),
            tickers=["SPY"],
            max_steps=100,
        )
        p.load_ohlcv = MagicMock(return_value=ohlcv)
        p.run()

        mock_forecaster_class.assert_called_once_with(max_steps=100)


# ---------------------------------------------------------------------------
# TestOutputReload
# ---------------------------------------------------------------------------


class TestOutputReload:
    """Verify saved Parquet files can be read back correctly."""

    @patch("src.models.predict.TitaniumForecaster")
    @patch("src.models.predict.compute_all_features")
    def test_predictions_parquet_roundtrip(
        self,
        mock_features: MagicMock,
        mock_forecaster_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        ohlcv = _make_ohlcv(200, ["SPY", "NVDA"])
        mock_features.return_value = ohlcv

        mock_fc = MagicMock()
        mock_forecaster_class.return_value = mock_fc
        mock_fc.h = 5
        mock_fc.predict.return_value = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5 + ["NVDA"] * 5,
                "ds": list(range(5)) * 2,
                "PatchTST-q0.5": [100.0] * 10,
            }
        )
        mock_fc.predict_proba.return_value = pl.DataFrame(
            {
                "ticker": ["SPY", "NVDA"],
                "prob_up": [0.8, 0.4],
                "expected_return": [0.02, -0.01],
            }
        )

        engine = MagicMock()
        p = PredictionPipeline(
            engine=engine,
            output_dir=str(tmp_path),
            tickers=["SPY", "NVDA"],
        )
        p.load_ohlcv = MagicMock(return_value=ohlcv)
        p.run()

        # Read back and verify
        loaded = pl.read_parquet(tmp_path / "predictions.parquet")
        assert loaded.height == 2
        assert set(loaded.columns) == {"ticker", "prob_up", "expected_return"}
