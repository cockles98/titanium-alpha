"""End-to-end prediction pipeline.

Loads OHLCV data from PostgreSQL, computes features, trains PatchTST,
generates probabilistic forecasts, logs metrics, and saves results
to Parquet files.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import polars as pl
from loguru import logger
from sqlalchemy import Engine

from src.data.ingestion import OHLCV_TABLE, _resolve_tickers
from src.models.features import compute_all_features
from src.models.patchtst_model import TitaniumForecaster
from src.utils.db import get_postgres_engine


class PredictionPipeline:
    """End-to-end pipeline: PostgreSQL → Features → PatchTST → Parquet.

    Args:
        engine: SQLAlchemy engine. If None, creates one from env vars.
        output_dir: Directory for Parquet output files.
        tickers: Ticker filter. If None, loads all tickers from DB.
        val_size: Validation set size for PatchTST training.
        max_steps: Maximum training steps for PatchTST.
    """

    def __init__(
        self,
        *,
        engine: Engine | None = None,
        output_dir: str = "data/outputs",
        tickers: list[str] | None = None,
        val_size: int = 63,
        max_steps: int = 5000,
    ) -> None:
        self.engine = engine or get_postgres_engine()
        self.output_dir = Path(output_dir)
        self.tickers = _resolve_tickers(tickers)
        self.val_size = val_size
        self.max_steps = max_steps

    def load_ohlcv(self) -> pl.DataFrame:
        """Load OHLCV data from PostgreSQL.

        Returns:
            Polars DataFrame with OHLCV columns, sorted by ticker and date.

        Raises:
            ValueError: If no data is found for the configured tickers.
        """
        _TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")
        for t in self.tickers:
            if not _TICKER_RE.match(t):
                raise ValueError(f"Invalid ticker symbol: {t!r}")

        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        query = (
            f"SELECT date, ticker, open, high, low, close, volume, adj_close "
            f"FROM {OHLCV_TABLE} "
            f"WHERE ticker IN ({ticker_list}) "
            f"ORDER BY ticker, date"
        )

        df = pl.read_database(query, connection=self.engine)

        if df.height == 0:
            raise ValueError(
                f"No OHLCV data found for tickers {self.tickers} "
                f"in table '{OHLCV_TABLE}'"
            )

        logger.info(
            "Loaded {} rows from PostgreSQL | tickers: {}",
            df.height,
            df["ticker"].unique().to_list(),
        )
        return df

    @staticmethod
    def compute_metrics(
        forecast_df: pl.DataFrame,
        actual_df: pl.DataFrame,
        h: int = 5,
    ) -> pl.DataFrame:
        """Compute MAE and RMSE per ticker from quantile forecasts.

        Joins forecast and actual DataFrames on (ticker, date) to ensure
        temporal alignment.  Returns empty metrics when no overlapping
        dates exist (e.g., during a production run where future actuals
        are not yet available).

        Args:
            forecast_df: NeuralForecast output with quantile columns.
            actual_df: DataFrame with date, ticker, close columns.
            h: Forecast horizon (unused — kept for API compat).

        Returns:
            DataFrame with columns: ticker, mae, rmse.
        """
        median_col = next(
            (c for c in ("PatchTST-q0.5", "PatchTST-median") if c in forecast_df.columns),
            None,
        )
        if median_col is None:
            logger.warning("No median column in forecast; skipping metrics")
            return pl.DataFrame({"ticker": [], "mae": [], "rmse": []})

        # Normalise column names for join
        id_col = "ticker" if "ticker" in forecast_df.columns else "unique_id"
        time_col = "ds" if "ds" in forecast_df.columns else "date"

        fc = forecast_df.select([
            pl.col(id_col).alias("ticker"),
            pl.col(time_col).alias("date"),
            pl.col(median_col).alias("predicted"),
        ])

        act = actual_df.select(["ticker", "date", "close"])

        # Guard against type mismatch (e.g. integer ds vs datetime date)
        if fc["date"].dtype != act["date"].dtype:
            logger.info(
                "Date type mismatch (forecast: {}, actual: {}); "
                "metrics unavailable",
                fc["date"].dtype, act["date"].dtype,
            )
            return pl.DataFrame({"ticker": [], "mae": [], "rmse": []})

        # Inner join: only dates present in both forecast and actuals
        joined = fc.join(act, on=["ticker", "date"], how="inner")

        if joined.height == 0:
            logger.info(
                "No overlapping dates between forecast and actuals; "
                "metrics will be available after actuals are observed"
            )
            return pl.DataFrame({"ticker": [], "mae": [], "rmse": []})

        tickers: list[str] = []
        maes: list[float] = []
        rmses: list[float] = []

        for ticker in joined["ticker"].unique().to_list():
            rows = joined.filter(pl.col("ticker") == ticker)
            errors = (rows["predicted"] - rows["close"]).to_list()
            n = len(errors)

            mae = sum(abs(e) for e in errors) / n
            rmse = math.sqrt(sum(e**2 for e in errors) / n)

            tickers.append(ticker)
            maes.append(mae)
            rmses.append(rmse)

            logger.info(
                "Metrics {}: MAE={:.4f}, RMSE={:.4f} (n={})",
                ticker, mae, rmse, n,
            )

        return pl.DataFrame({"ticker": tickers, "mae": maes, "rmse": rmses})

    def run(self) -> pl.DataFrame:
        """Execute the full prediction pipeline.

        Steps:
            1. Load OHLCV from PostgreSQL
            2. Compute technical features
            3. Train PatchTST
            4. Generate forecasts and probabilities
            5. Compute and log metrics
            6. Save results to Parquet

        Returns:
            DataFrame with per-ticker probabilities (ticker, prob_up,
            expected_return).
        """
        # 1. Load data
        ohlcv_df = self.load_ohlcv()

        # 2. Compute features
        features_df = compute_all_features(ohlcv_df)
        logger.info(
            "Features computed: {} rows × {} cols",
            features_df.height,
            features_df.width,
        )

        # 3. Train PatchTST (or load from checkpoint)
        checkpoint_dir = str(self.output_dir / "model_checkpoint")
        expected = TitaniumForecaster(max_steps=self.max_steps).get_params()
        try:
            forecaster = TitaniumForecaster.load(
                checkpoint_dir, expect_params=expected,
            )
            logger.info("Loaded PatchTST from checkpoint — skipping training")
        except Exception as exc:
            logger.info("No valid checkpoint ({}); training from scratch", exc)
            forecaster = TitaniumForecaster(
                max_steps=self.max_steps,
            )
            forecaster.fit(features_df, val_size=self.val_size)
            forecaster.save(checkpoint_dir)
            logger.info("Checkpoint saved to {}", checkpoint_dir)

        # 4. Generate forecasts (single predict call, reused for prob_up)
        forecast_df = forecaster.predict()
        proba_df = forecaster.predict_proba(features_df, forecast=forecast_df)

        logger.info("Predictions:\n{}", proba_df)

        # 5. Compute metrics (date-aligned join; empty until actuals observed)
        metrics_df = self.compute_metrics(forecast_df, features_df, h=forecaster.h)

        # 6. Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.output_dir / "predictions.parquet"
        forecast_path = self.output_dir / "forecast.parquet"
        metrics_path = self.output_dir / "metrics.parquet"
        features_path = self.output_dir / "features.parquet"

        proba_df.write_parquet(predictions_path)
        forecast_df.write_parquet(forecast_path)
        metrics_df.write_parquet(metrics_path)

        # Save latest feature snapshot per ticker for the agent pipeline
        _FEATURE_COLS = [
            "rsi_14", "bb_upper", "bb_middle", "bb_lower",
            "realized_vol_21", "volume_sma", "relative_volume", "vwap", "obv",
        ]
        feature_cols_present = [c for c in _FEATURE_COLS if c in features_df.columns]
        if feature_cols_present:
            latest_features = (
                features_df.sort("date")
                .group_by("ticker")
                .last()
                .select(["ticker"] + feature_cols_present)
            )
            latest_features.write_parquet(features_path)
            logger.info(
                "Features snapshot saved: {} tickers × {} indicators",
                latest_features.height,
                len(feature_cols_present),
            )

        logger.info(
            "Outputs saved to {}: predictions, forecast, metrics, features",
            self.output_dir,
        )

        return proba_df


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    result = pipeline.run()
    logger.info("Pipeline complete | {} tickers processed", result.height)
