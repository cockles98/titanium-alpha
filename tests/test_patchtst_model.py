"""Tests for src/models/patchtst_model.py — TitaniumForecaster."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.models.patchtst_model import TitaniumForecaster, _REQUIRED_COLS


# ---------------------------------------------------------------------------
# TestTitaniumForecasterInit
# ---------------------------------------------------------------------------


class TestTitaniumForecasterInit:
    """Constructor defaults and custom parameters."""

    def test_default_params(self) -> None:
        f = TitaniumForecaster()
        assert f.h == 5
        assert f.input_size == 60
        assert f.batch_size == 32
        assert f.max_steps == 5000
        assert f.learning_rate == 1e-4
        assert f.quantiles == [0.1, 0.25, 0.5, 0.75, 0.9]
        assert f.random_seed == 42
        assert not f._is_fitted

    def test_custom_params(self) -> None:
        f = TitaniumForecaster(
            h=10,
            input_size=120,
            batch_size=64,
            quantiles=[0.25, 0.5, 0.75],
        )
        assert f.h == 10
        assert f.input_size == 120
        assert f.batch_size == 64
        assert f.quantiles == [0.25, 0.5, 0.75]

    def test_not_fitted_on_init(self) -> None:
        f = TitaniumForecaster()
        assert f._nf is None
        assert not f._is_fitted


# ---------------------------------------------------------------------------
# TestPrepareDF
# ---------------------------------------------------------------------------


class TestPrepareDF:
    """DataFrame validation and preparation."""

    def test_selects_required_columns(self, sample_features_df: pl.DataFrame) -> None:
        result = TitaniumForecaster._prepare_df(sample_features_df)
        assert set(result.columns) == set(_REQUIRED_COLS)

    def test_missing_columns_raises(self) -> None:
        df = pl.DataFrame({"date": [1], "ticker": ["X"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            TitaniumForecaster._prepare_df(df)

    def test_drops_extra_columns(self, sample_features_df: pl.DataFrame) -> None:
        result = TitaniumForecaster._prepare_df(sample_features_df)
        assert "adj_close" not in result.columns
        assert "rsi_14" not in result.columns
        assert "bb_upper" not in result.columns

    def test_no_nulls_in_output(self, sample_features_df: pl.DataFrame) -> None:
        result = TitaniumForecaster._prepare_df(sample_features_df)
        for col in result.columns:
            assert result[col].null_count() == 0

    def test_preserves_row_count_when_no_nulls(
        self, sample_features_df: pl.DataFrame
    ) -> None:
        """sample_features_df already has nulls dropped."""
        result = TitaniumForecaster._prepare_df(sample_features_df)
        assert result.height == sample_features_df.height


# ---------------------------------------------------------------------------
# TestBuildModel
# ---------------------------------------------------------------------------


class TestBuildModel:
    """PatchTST model instantiation."""

    def test_returns_patchtst(self) -> None:
        from neuralforecast.models import PatchTST

        f = TitaniumForecaster()
        model = f._build_model()
        assert isinstance(model, PatchTST)

    def test_respects_h_param(self) -> None:
        f = TitaniumForecaster(h=10)
        model = f._build_model()
        assert model.h == 10

    def test_respects_input_size(self) -> None:
        f = TitaniumForecaster(input_size=120)
        model = f._build_model()
        assert model.input_size == 120


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Input validation and error handling."""

    def test_missing_columns_raises(self) -> None:
        df = pl.DataFrame({"date": [1], "ticker": ["X"]})
        f = TitaniumForecaster()
        with pytest.raises(ValueError, match="Missing required columns"):
            f._validate_df(df)

    def test_too_few_rows_raises(self) -> None:
        """fit() should reject DataFrames smaller than input_size + val_size."""
        from datetime import date, timedelta

        n = 10  # way less than input_size(60) + val_size(63)
        df = pl.DataFrame(
            {
                "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(n)],
                "ticker": ["SPY"] * n,
                "close": [100.0 + i for i in range(n)],
            }
        )
        f = TitaniumForecaster()
        with pytest.raises(ValueError, match="Need at least"):
            f.fit(df, val_size=63)

    def test_predict_before_fit_raises(self) -> None:
        f = TitaniumForecaster()
        with pytest.raises(RuntimeError, match="not fitted"):
            f.predict()

    def test_save_before_fit_raises(self) -> None:
        f = TitaniumForecaster()
        with pytest.raises(RuntimeError, match="not fitted"):
            f.save()


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------


class TestFit:
    """Model training with mocked NeuralForecast."""

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_fit_calls_nf(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)

        mock_nf_instance.fit.assert_called_once()
        call_kwargs = mock_nf_instance.fit.call_args
        assert call_kwargs.kwargs["val_size"] == 10
        assert call_kwargs.kwargs["id_col"] == "ticker"
        assert call_kwargs.kwargs["time_col"] == "date"
        assert call_kwargs.kwargs["target_col"] == "close"
        assert f._is_fitted

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_fit_sets_fitted_flag(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        mock_nf_class.return_value = MagicMock()
        f = TitaniumForecaster()
        assert not f._is_fitted
        f.fit(sample_features_df, val_size=10)
        assert f._is_fitted

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_fit_df_has_only_required_cols(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)

        # Verify the df passed to NF.fit has only required columns
        fit_call = mock_nf_instance.fit.call_args
        passed_df = fit_call.kwargs["df"]
        assert set(passed_df.columns) == set(_REQUIRED_COLS)


# ---------------------------------------------------------------------------
# TestPredict
# ---------------------------------------------------------------------------


class TestPredict:
    """Prediction with mocked NeuralForecast."""

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_predict_calls_nf(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance

        mock_forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.5": [100.0] * 5,
            }
        )
        mock_nf_instance.predict.return_value = mock_forecast

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)
        result = f.predict()

        mock_nf_instance.predict.assert_called_once()
        assert isinstance(result, pl.DataFrame)

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_predict_with_new_data(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance
        mock_nf_instance.predict.return_value = pl.DataFrame(
            {"unique_id": ["SPY"], "ds": [1], "PatchTST-q0.5": [100.0]}
        )

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)
        result = f.predict(df=sample_features_df)

        call_kwargs = mock_nf_instance.predict.call_args.kwargs
        assert "df" in call_kwargs


# ---------------------------------------------------------------------------
# TestPredictProba
# ---------------------------------------------------------------------------


class TestPredictProba:
    """Probability computation from quantile forecasts."""

    def _make_fitted_forecaster(
        self,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
        forecast_df: pl.DataFrame,
    ) -> TitaniumForecaster:
        """Helper to create a fitted forecaster with mocked predict."""
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance
        mock_nf_instance.predict.return_value = forecast_df

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)
        return f

    @patch("src.models.patchtst_model.PatchTST")
    @patch("src.models.patchtst_model.NeuralForecast")
    def test_prob_up_between_0_and_1(
        self,
        mock_nf_class: MagicMock,
        mock_patchtst_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        last_close = sample_features_df["close"][-1]
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.1": [last_close + 10] * 5,
                "PatchTST-q0.25": [last_close + 20] * 5,
                "PatchTST-q0.5": [last_close + 30] * 5,
                "PatchTST-q0.75": [last_close + 40] * 5,
                "PatchTST-q0.9": [last_close + 50] * 5,
            }
        )
        f = self._make_fitted_forecaster(mock_nf_class, sample_features_df, forecast)
        result = f.predict_proba(sample_features_df)

        assert "prob_up" in result.columns
        assert "expected_return" in result.columns
        assert "ticker" in result.columns
        assert (result["prob_up"] >= 0).all()
        assert (result["prob_up"] <= 1).all()

    @patch("src.models.patchtst_model.PatchTST")
    @patch("src.models.patchtst_model.NeuralForecast")
    def test_all_quantiles_above_gives_prob_1(
        self,
        mock_nf_class: MagicMock,
        mock_patchtst_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        last_close = sample_features_df["close"][-1]
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.1": [last_close + 100] * 5,
                "PatchTST-q0.25": [last_close + 100] * 5,
                "PatchTST-q0.5": [last_close + 100] * 5,
                "PatchTST-q0.75": [last_close + 100] * 5,
                "PatchTST-q0.9": [last_close + 100] * 5,
            }
        )
        f = self._make_fitted_forecaster(mock_nf_class, sample_features_df, forecast)
        result = f.predict_proba(sample_features_df)
        assert result["prob_up"][0] == 1.0

    @patch("src.models.patchtst_model.PatchTST")
    @patch("src.models.patchtst_model.NeuralForecast")
    def test_all_quantiles_below_gives_prob_0(
        self,
        mock_nf_class: MagicMock,
        mock_patchtst_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        last_close = sample_features_df["close"][-1]
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.1": [last_close - 100] * 5,
                "PatchTST-q0.25": [last_close - 100] * 5,
                "PatchTST-q0.5": [last_close - 100] * 5,
                "PatchTST-q0.75": [last_close - 100] * 5,
                "PatchTST-q0.9": [last_close - 100] * 5,
            }
        )
        f = self._make_fitted_forecaster(mock_nf_class, sample_features_df, forecast)
        result = f.predict_proba(sample_features_df)
        assert result["prob_up"][0] == 0.0

    @patch("src.models.patchtst_model.PatchTST")
    @patch("src.models.patchtst_model.NeuralForecast")
    def test_expected_return_positive_when_median_above(
        self,
        mock_nf_class: MagicMock,
        mock_patchtst_class: MagicMock,
        sample_features_df: pl.DataFrame,
    ) -> None:
        last_close = sample_features_df["close"][-1]
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.1": [last_close + 5] * 5,
                "PatchTST-q0.25": [last_close + 10] * 5,
                "PatchTST-q0.5": [last_close + 15] * 5,
                "PatchTST-q0.75": [last_close + 20] * 5,
                "PatchTST-q0.9": [last_close + 25] * 5,
            }
        )
        f = self._make_fitted_forecaster(mock_nf_class, sample_features_df, forecast)
        result = f.predict_proba(sample_features_df)
        assert result["expected_return"][0] > 0


# ---------------------------------------------------------------------------
# TestComputeProbUp
# ---------------------------------------------------------------------------


class TestComputeProbUp:
    """Static method _compute_prob_up."""

    def test_mixed_quantiles(self) -> None:
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.1": [90.0] * 5,
                "PatchTST-q0.25": [95.0] * 5,
                "PatchTST-q0.5": [105.0] * 5,
                "PatchTST-q0.75": [110.0] * 5,
                "PatchTST-q0.9": [115.0] * 5,
            }
        )
        result = TitaniumForecaster._compute_prob_up(
            forecast,
            last_closes={"SPY": 100.0},
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        )
        # q0.5, q0.75, q0.9 above 100 → 3/5 = 0.6
        assert result["prob_up"][0] == pytest.approx(0.6)

    def test_multiple_tickers(self) -> None:
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5 + ["NVDA"] * 5,
                "ds": list(range(5)) * 2,
                "PatchTST-q0.5": [110.0] * 5 + [200.0] * 5,
            }
        )
        result = TitaniumForecaster._compute_prob_up(
            forecast,
            last_closes={"SPY": 100.0, "NVDA": 250.0},
            quantiles=[0.5],
        )
        assert result.height == 2
        spy_row = result.filter(pl.col("ticker") == "SPY")
        nvda_row = result.filter(pl.col("ticker") == "NVDA")
        assert spy_row["prob_up"][0] == 1.0  # 110 > 100
        assert nvda_row["prob_up"][0] == 0.0  # 200 < 250

    def test_skips_ticker_without_close(self) -> None:
        forecast = pl.DataFrame(
            {
                "unique_id": ["SPY"] * 5,
                "ds": list(range(5)),
                "PatchTST-q0.5": [110.0] * 5,
            }
        )
        result = TitaniumForecaster._compute_prob_up(
            forecast,
            last_closes={},  # no close for SPY
            quantiles=[0.5],
        )
        assert result.height == 0


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Model persistence."""

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_save_creates_metadata(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance

        f = TitaniumForecaster(h=5, input_size=60)
        f.fit(sample_features_df, val_size=10)

        save_path = str(tmp_path / "test_ckpt")
        f.save(path=save_path)

        metadata_file = Path(save_path) / "metadata.json"
        assert metadata_file.exists()

        metadata = json.loads(metadata_file.read_text())
        assert metadata["model"] == "PatchTST"
        assert metadata["h"] == 5
        assert metadata["input_size"] == 60
        assert metadata["quantiles"] == [0.1, 0.25, 0.5, 0.75, 0.9]

    @patch("src.models.patchtst_model.NeuralForecast")
    @patch("src.models.patchtst_model.PatchTST")
    def test_save_calls_nf_save(
        self,
        mock_patchtst_class: MagicMock,
        mock_nf_class: MagicMock,
        sample_features_df: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        mock_nf_instance = MagicMock()
        mock_nf_class.return_value = mock_nf_instance

        f = TitaniumForecaster()
        f.fit(sample_features_df, val_size=10)
        f.save(path=str(tmp_path / "ckpt"))

        mock_nf_instance.save.assert_called_once()

    @patch("src.models.patchtst_model.NeuralForecast")
    def test_load_restores_params(
        self,
        mock_nf_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        metadata = {
            "model": "PatchTST",
            "h": 10,
            "input_size": 120,
            "batch_size": 64,
            "max_steps": 3000,
            "learning_rate": 0.001,
            "quantiles": [0.25, 0.5, 0.75],
            "random_seed": 99,
        }
        (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

        mock_nf_class.load.return_value = MagicMock()

        f = TitaniumForecaster.load(path=str(ckpt_dir))
        assert f.h == 10
        assert f.input_size == 120
        assert f.quantiles == [0.25, 0.5, 0.75]
        assert f._is_fitted

    def test_load_missing_metadata_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            TitaniumForecaster.load(path=str(tmp_path / "nonexistent"))
