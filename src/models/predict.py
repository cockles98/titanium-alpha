"""End-to-end prediction pipeline.

Loads OHLCV data from PostgreSQL, computes features, trains PatchTST,
generates probabilistic forecasts, logs metrics, and saves results
to Parquet files.
"""

from __future__ import annotations

import math
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

        Compares the median forecast (PatchTST-q0.5) at horizon h
        against the last h known close prices per ticker.

        Args:
            forecast_df: NeuralForecast output with quantile columns.
            actual_df: DataFrame with date, ticker, close columns.
            h: Forecast horizon (number of steps to compare).

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

        tickers: list[str] = []
        maes: list[float] = []
        rmses: list[float] = []

        id_col = "ticker" if "ticker" in forecast_df.columns else "unique_id"
        unique_ids = forecast_df[id_col].unique().to_list()
        for ticker in unique_ids:
            # Get forecast values (h rows per ticker)
            fc_rows = forecast_df.filter(pl.col(id_col) == ticker)
            predicted = fc_rows[median_col].to_list()

            # Get last h actual closes
            actual_rows = actual_df.filter(pl.col("ticker") == ticker).sort("date")
            actual_closes = actual_rows["close"].tail(h).to_list()

            n = min(len(predicted), len(actual_closes))
            if n == 0:
                continue

            errors = [p - a for p, a in zip(predicted[:n], actual_closes[:n])]
            mae = sum(abs(e) for e in errors) / n
            rmse = math.sqrt(sum(e**2 for e in errors) / n)

            tickers.append(ticker)
            maes.append(mae)
            rmses.append(rmse)

            logger.info(
                "Metrics {}: MAE={:.4f}, RMSE={:.4f}",
                ticker,
                mae,
                rmse,
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

        # 3. Train PatchTST
        forecaster = TitaniumForecaster(
            max_steps=self.max_steps,
        )
        forecaster.fit(features_df, val_size=self.val_size)

        # 4. Generate forecasts
        forecast_df = forecaster.predict()
        proba_df = forecaster.predict_proba(features_df)

        logger.info("Predictions:\n{}", proba_df)

        # 5. Compute metrics (in-sample: compare forecast vs last h closes)
        metrics_df = self.compute_metrics(forecast_df, features_df, h=forecaster.h)

        # 6. Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.output_dir / "predictions.parquet"
        forecast_path = self.output_dir / "forecast.parquet"
        metrics_path = self.output_dir / "metrics.parquet"

        proba_df.write_parquet(predictions_path)
        forecast_df.write_parquet(forecast_path)
        metrics_df.write_parquet(metrics_path)

        logger.info(
            "Outputs saved to {}: predictions.parquet, forecast.parquet, metrics.parquet",
            self.output_dir,
        )

        return proba_df


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    result = pipeline.run()
    logger.info("Pipeline complete | {} tickers processed", result.height)
