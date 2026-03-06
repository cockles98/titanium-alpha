"""PatchTST forecaster for multi-asset return prediction.

Wraps NeuralForecast's PatchTST model to predict forward close prices
and compute up/down probabilities per ticker using quantile forecasts.

Note: PatchTST operates on the close price series only (channel-independent
design). Technical features from ``features.py`` are consumed by the
LangGraph agent layer, not by PatchTST directly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import PatchTST

_REQUIRED_COLS: list[str] = ["date", "ticker", "close"]


class TitaniumForecaster:
    """PatchTST forecaster for multi-asset return prediction.

    Uses NeuralForecast's PatchTST with quantile loss to produce
    probabilistic forecasts of close prices h days ahead.

    PatchTST is channel-independent: it learns patterns from the close
    price series only. Technical indicators (RSI, BB, etc.) are used by
    the LangGraph agents for qualitative analysis, not by this model.

    Args:
        h: Forecast horizon in trading days.
        input_size: Number of historical bars the model sees.
        batch_size: Mini-batch size for training.
        max_steps: Maximum training iterations.
        learning_rate: Adam learning rate.
        early_stop_patience_steps: Steps without val improvement before stop.
        val_check_steps: Frequency of validation checks.
        quantiles: Quantile levels for MQLoss.
        random_seed: Reproducibility seed.
    """

    DEFAULT_QUANTILES: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]

    def __init__(
        self,
        h: int = 5,
        input_size: int = 60,
        batch_size: int = 32,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        early_stop_patience_steps: int = 10,
        val_check_steps: int = 50,
        quantiles: list[float] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.random_seed = random_seed

        self._nf: NeuralForecast | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_df(df: pl.DataFrame) -> None:
        """Raise ``ValueError`` if *df* is missing required columns."""
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _prepare_df(df: pl.DataFrame) -> pl.DataFrame:
        """Validate, select columns, and drop nulls.

        Args:
            df: DataFrame with at least date, ticker, close columns.

        Returns:
            Clean DataFrame with only the columns NeuralForecast needs.
        """
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        prepared = df.select(_REQUIRED_COLS).drop_nulls()

        logger.debug(
            "_prepare_df: {} → {} rows (dropped {} nulls)",
            df.height,
            prepared.height,
            df.height - prepared.height,
        )
        return prepared

    def _build_model(self) -> PatchTST:
        """Instantiate a PatchTST model with the configured hyperparameters.

        Returns:
            Configured PatchTST instance.
        """
        model = PatchTST(
            h=self.h,
            input_size=self.input_size,
            loss=MQLoss(quantiles=self.quantiles),
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            early_stop_patience_steps=self.early_stop_patience_steps,
            val_check_steps=self.val_check_steps,
            random_seed=self.random_seed,
        )
        logger.debug(
            "PatchTST built: h={}, input_size={}, quantiles={}",
            self.h,
            self.input_size,
            self.quantiles,
        )
        return model

    @staticmethod
    def _compute_prob_up(
        forecast_df: pl.DataFrame,
        last_closes: dict[str, float],
        quantiles: list[float],
    ) -> pl.DataFrame:
        """Compute probability of price increase from quantile forecasts.

        For each ticker, counts the fraction of quantile predictions at
        horizon h (the last forecast step) that exceed the current close.

        Args:
            forecast_df: NeuralForecast predict output with quantile columns.
            last_closes: Mapping of ticker → last known close price.
            quantiles: Quantile levels used in MQLoss.

        Returns:
            DataFrame with columns: ticker, prob_up, expected_return.
        """
        q_cols = [f"PatchTST-q{q}" for q in quantiles]

        tickers: list[str] = []
        prob_ups: list[float] = []
        expected_returns: list[float] = []

        unique_ids = forecast_df["unique_id"].unique().to_list()
        for ticker in unique_ids:
            ticker_rows = forecast_df.filter(pl.col("unique_id") == ticker)
            last_row = ticker_rows.tail(1)

            current_close = last_closes.get(ticker)
            if current_close is None or current_close == 0:
                logger.warning("No last close for ticker {}, skipping", ticker)
                continue

            q_values = [last_row[col][0] for col in q_cols if col in last_row.columns]
            if not q_values:
                continue

            n_above = sum(1 for v in q_values if v > current_close)
            prob_up = n_above / len(q_values)

            median_col = "PatchTST-q0.5"
            if median_col in last_row.columns:
                median_pred = last_row[median_col][0]
                exp_ret = (median_pred - current_close) / current_close
            else:
                exp_ret = 0.0

            tickers.append(ticker)
            prob_ups.append(prob_up)
            expected_returns.append(exp_ret)

        return pl.DataFrame(
            {
                "ticker": tickers,
                "prob_up": prob_ups,
                "expected_return": expected_returns,
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame, val_size: int = 63) -> None:
        """Train the PatchTST model.

        Args:
            df: DataFrame with at least date, ticker, close columns.
            val_size: Number of recent observations per ticker held out
                for validation (~3 months of daily data).
        """
        prepared = self._prepare_df(df)

        min_rows = self.input_size + val_size
        if prepared.height < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows, got {prepared.height}. "
                f"(input_size={self.input_size} + val_size={val_size})"
            )

        model = self._build_model()
        self._nf = NeuralForecast(models=[model], freq="1bd")

        logger.info(
            "Fitting PatchTST: {} rows, val_size={}",
            prepared.height,
            val_size,
        )

        self._nf.fit(
            df=prepared,
            val_size=val_size,
            id_col="ticker",
            time_col="date",
            target_col="close",
        )

        self._is_fitted = True
        logger.info("PatchTST training complete")

    def predict(self, df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Generate quantile forecasts for the next h days.

        Args:
            df: Optional new data. If None, uses the training data tail.

        Returns:
            NeuralForecast forecast DataFrame with quantile columns.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        kwargs: dict[str, Any] = {
            "id_col": "ticker",
            "time_col": "date",
            "target_col": "close",
        }
        if df is not None:
            kwargs["df"] = self._prepare_df(df)

        forecast = self._nf.predict(**kwargs)

        logger.info("Predict: {} rows generated", forecast.height)
        return forecast

    def predict_proba(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute probability of price increase per ticker.

        Runs ``predict`` and then compares the quantile forecasts at
        horizon h against the last known close price to estimate
        the probability of an upward move.

        Args:
            df: DataFrame with at least date, ticker, close columns.

        Returns:
            DataFrame with columns: ``ticker``, ``prob_up``,
            ``expected_return``.
        """
        prepared = self._prepare_df(df)

        last_closes: dict[str, float] = {}
        for ticker in prepared["ticker"].unique().to_list():
            ticker_data = prepared.filter(pl.col("ticker") == ticker).sort("date")
            last_closes[ticker] = ticker_data["close"][-1]

        forecast = self.predict(df=df)

        return self._compute_prob_up(forecast, last_closes, self.quantiles)

    def save(self, path: str = "models/checkpoints") -> None:
        """Save the fitted model and metadata to disk.

        Args:
            path: Directory for saving model files.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted or self._nf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._nf.save(path=str(save_dir))

        metadata: dict[str, Any] = {
            "model": "PatchTST",
            "h": self.h,
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "quantiles": self.quantiles,
            "random_seed": self.random_seed,
            "saved_date": str(date.today()),
        }

        metadata_path = save_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Model saved to {}", save_dir)

    @classmethod
    def load(cls, path: str = "models/checkpoints") -> TitaniumForecaster:
        """Load a previously saved model from disk.

        Args:
            path: Directory containing the saved model files.

        Returns:
            A fitted TitaniumForecaster instance.

        Raises:
            FileNotFoundError: If the path or metadata file does not exist.
        """
        load_dir = Path(path)

        metadata_path = load_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        forecaster = cls(
            h=metadata.get("h", 5),
            input_size=metadata.get("input_size", 60),
            batch_size=metadata.get("batch_size", 32),
            max_steps=metadata.get("max_steps", 5000),
            learning_rate=metadata.get("learning_rate", 1e-4),
            quantiles=metadata.get("quantiles"),
            random_seed=metadata.get("random_seed", 42),
        )

        forecaster._nf = NeuralForecast.load(path=str(load_dir))
        forecaster._is_fitted = True

        logger.info("Model loaded from {}", load_dir)
        return forecaster
