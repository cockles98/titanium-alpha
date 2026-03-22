"""PatchTST forecaster for multi-asset return prediction.

Wraps NeuralForecast's PatchTST model to predict forward close prices
and compute up/down probabilities per ticker using quantile forecasts.

Note: PatchTST operates on the close price series only (channel-independent
design). Technical features from ``features.py`` are consumed by the
LangGraph agent layer, not by PatchTST directly.
"""

from __future__ import annotations

import json
import math
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
            start_padding_enabled=True,
        )
        logger.debug(
            "PatchTST built: h={}, input_size={}, quantiles={}",
            self.h,
            self.input_size,
            self.quantiles,
        )
        return model

    @staticmethod
    def _interpolate_prob_up(
        q_levels: list[float],
        q_values: list[float],
        close: float,
    ) -> float:
        """Interpolate P(price > close) from quantile forecasts.

        Uses linear interpolation on the empirical CDF defined by the
        quantile levels and their predicted values.  Returns ``1 - CDF(close)``.

        With 5 quantiles [0.1, 0.25, 0.5, 0.75, 0.9] this gives continuous
        probabilities instead of the old discrete {0.0, 0.2, ..., 1.0}.

        Args:
            q_levels: Quantile levels (e.g. [0.1, 0.25, 0.5, 0.75, 0.9]).
            q_values: Predicted values at each quantile level.
            close: Last known close price.

        Returns:
            Probability in [0, 1] that the price exceeds ``close``.
        """
        # Sort values and levels independently to guarantee CDF monotonicity.
        # MQLoss does not enforce monotone quantile predictions, so crossing
        # can occur (e.g. Q(0.25) > Q(0.5)).  The rearrangement approach maps
        # the i-th smallest predicted value to the i-th smallest quantile
        # level, producing the optimal monotone CDF estimate.
        vals = sorted(q_values)
        probs = sorted(q_levels)

        # close strictly below all quantile predictions
        if close < vals[0]:
            return 1.0 - probs[0] / 2.0  # midpoint of [0, q_min]

        # close strictly above all quantile predictions
        if close > vals[-1]:
            return (1.0 - probs[-1]) / 2.0  # midpoint of [q_max, 1]

        # Exact match: close equals one or more quantile predictions.
        # Use the max matching quantile level as CDF — conservative when
        # multiple quantiles collapse (model uncertainty → less conviction).
        matching = [p for v, p in zip(vals, probs) if v == close]
        if matching:
            return 1.0 - max(matching)

        # Strict linear interpolation (division by zero impossible here
        # because close ≠ any val, so adjacent equal vals can't bracket it)
        for i in range(len(vals) - 1):
            if vals[i] < close < vals[i + 1]:
                frac = (close - vals[i]) / (vals[i + 1] - vals[i])
                cdf_at_close = probs[i] + frac * (probs[i + 1] - probs[i])
                return 1.0 - cdf_at_close

        return 0.5  # fallback (should not reach here)

    @staticmethod
    def _compute_prob_up(
        forecast_df: pl.DataFrame,
        last_closes: dict[str, float],
        quantiles: list[float],
    ) -> pl.DataFrame:
        """Compute probability of price increase from quantile forecasts.

        For each ticker, interpolates the empirical CDF at the last known
        close price using the quantile predictions at horizon h (last
        forecast step).  Returns ``1 - CDF(close)`` as a continuous
        probability.

        Args:
            forecast_df: NeuralForecast predict output with quantile columns.
            last_closes: Mapping of ticker → last known close price.
            quantiles: Quantile levels used in MQLoss.

        Returns:
            DataFrame with columns: ticker, prob_up, expected_return,
            last_close.
        """
        # Detect quantile column names (supports both old and new NeuralForecast naming)
        q_cols = [f"PatchTST-q{q}" for q in quantiles]
        q_cols_present = [c for c in q_cols if c in forecast_df.columns]

        # Map column name → quantile level for old naming
        col_to_level: dict[str, float] = {}
        if q_cols_present:
            for q, c in zip(quantiles, q_cols):
                if c in forecast_df.columns:
                    col_to_level[c] = q
        else:
            # New NeuralForecast naming: PatchTST-lo-80.0, PatchTST-lo-50.0,
            # PatchTST-median, PatchTST-hi-50.0, PatchTST-hi-80.0
            q_cols_present = [
                c for c in forecast_df.columns
                if c.startswith("PatchTST-") and c not in ("ticker", "date")
            ]
            # Derive quantile levels from column names
            for c in q_cols_present:
                if "median" in c:
                    col_to_level[c] = 0.5
                elif "-lo-" in c:
                    # e.g. "PatchTST-lo-80.0" → quantile = (100 - 80) / 200 = 0.1
                    try:
                        pct = float(c.split("-lo-")[1])
                    except ValueError:
                        logger.warning("Skipping unparseable column: {}", c)
                        continue
                    col_to_level[c] = (100.0 - pct) / 200.0
                elif "-hi-" in c:
                    # e.g. "PatchTST-hi-50.0" → quantile = (100 + 50) / 200 = 0.75
                    try:
                        pct = float(c.split("-hi-")[1])
                    except ValueError:
                        logger.warning("Skipping unparseable column: {}", c)
                        continue
                    col_to_level[c] = (100.0 + pct) / 200.0

        # Detect median column
        median_col = next(
            (c for c in ("PatchTST-q0.5", "PatchTST-median") if c in forecast_df.columns),
            None,
        )

        tickers: list[str] = []
        prob_ups: list[float] = []
        expected_returns: list[float] = []
        closes: list[float] = []

        id_col = "ticker" if "ticker" in forecast_df.columns else "unique_id"
        unique_ids = forecast_df[id_col].unique().to_list()
        for ticker in unique_ids:
            ticker_rows = forecast_df.filter(pl.col(id_col) == ticker)
            last_row = ticker_rows.tail(1)

            current_close = last_closes.get(ticker)
            if current_close is None or current_close == 0:
                logger.warning("No last close for ticker {}, skipping", ticker)
                continue

            recognized_cols = [
                col for col in q_cols_present
                if col in last_row.columns and col in col_to_level
            ]
            if not recognized_cols:
                logger.warning(
                    "Ticker {} skipped: no recognized quantile columns", ticker,
                )
                continue

            q_values_raw = [last_row[col][0] for col in recognized_cols]
            q_levels_raw = [col_to_level[col] for col in recognized_cols]

            # Filter non-finite predictions (NeuralForecast can emit NaN/inf
            # under numerical instability; sorted() is undefined with NaN).
            valid = [
                (v, l) for v, l in zip(q_values_raw, q_levels_raw)
                if v is not None and math.isfinite(v)
            ]
            if not valid:
                logger.warning(
                    "Ticker {} skipped: all quantile predictions non-finite",
                    ticker,
                )
                continue
            q_values = [v for v, _ in valid]
            q_levels = [l for _, l in valid]

            prob_up = TitaniumForecaster._interpolate_prob_up(
                q_levels, q_values, current_close
            )

            if median_col and median_col in last_row.columns:
                median_pred = last_row[median_col][0]
                if median_pred is not None and math.isfinite(median_pred):
                    exp_ret = (median_pred - current_close) / current_close
                else:
                    exp_ret = 0.0
            else:
                exp_ret = 0.0

            tickers.append(ticker)
            prob_ups.append(prob_up)
            expected_returns.append(exp_ret)
            closes.append(current_close)

        return pl.DataFrame(
            {
                "ticker": tickers,
                "prob_up": prob_ups,
                "expected_return": expected_returns,
                "last_close": closes,
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
        ticker_counts = prepared.group_by("ticker").agg(pl.len().alias("n"))
        short_tickers = ticker_counts.filter(pl.col("n") < min_rows)
        if short_tickers.height > 0:
            examples = short_tickers.head(5).to_dicts()
            raise ValueError(
                f"Tickers with fewer than {min_rows} rows "
                f"(input_size={self.input_size} + val_size={val_size}): "
                f"{examples}"
            )

        model = self._build_model()
        self._nf = NeuralForecast(models=[model], freq="1d")

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

        if df is not None:
            forecast = self._nf.predict(df=self._prepare_df(df))
        else:
            forecast = self._nf.predict()

        logger.info("Predict: {} rows generated", forecast.height)
        return forecast

    def predict_proba(
        self,
        df: pl.DataFrame,
        forecast: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Compute probability of price increase per ticker.

        Compares quantile forecasts at horizon h against the last known
        close price to estimate the probability of an upward move.

        Args:
            df: DataFrame with at least date, ticker, close columns.
            forecast: Pre-computed forecast DataFrame. If None, calls
                ``predict(df=df)`` internally.

        Returns:
            DataFrame with columns: ``ticker``, ``prob_up``,
            ``expected_return``.
        """
        prepared = self._prepare_df(df)

        last_closes: dict[str, float] = {}
        for ticker in prepared["ticker"].unique().to_list():
            ticker_data = prepared.filter(pl.col("ticker") == ticker).sort("date")
            last_closes[ticker] = ticker_data["close"][-1]

        if forecast is None:
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
