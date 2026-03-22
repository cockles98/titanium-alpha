"""Walk-forward benchmark orchestrator.

Runs the full benchmark pipeline: load config → load OHLCV → walk-forward
→ compute metrics → generate PDF report → save outputs.

Usage::

    # From Python
    from src.backtest.run_benchmark import run_us_benchmark
    result = run_us_benchmark(use_patchtst=False)  # fast mode

    # From CLI
    python -m src.backtest.run_benchmark          # PatchTST (slow)
    python -m src.backtest.run_benchmark --naive   # NaiveModelFactory (fast)
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.backtest.benchmark_metrics import compute_benchmark_metrics
from src.backtest.benchmark_report import BenchmarkReport
from src.backtest.cpcv import TransactionCosts
from src.backtest.walk_forward import (
    NaiveModelFactory,
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
)
from src.portfolio.hrp import HRPConfig  # noqa: F401 — used by _PatchTSTModelFactory
from src.config import load_benchmark, load_ticker_config, load_tickers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("data/outputs")
_OHLCV_TABLE = "market_ohlcv"


# ---------------------------------------------------------------------------
# OHLCV loader
# ---------------------------------------------------------------------------


def _load_ohlcv_from_postgres(
    tickers: list[str],
    benchmark: str,
) -> pl.DataFrame:
    """Load OHLCV data from PostgreSQL.

    Args:
        tickers: List of tradeable ticker symbols.
        benchmark: Benchmark ticker symbol.

    Returns:
        Polars DataFrame with OHLCV data for all tickers + benchmark.

    Raises:
        RuntimeError: If no data is found.
    """
    from src.utils.db import get_postgres_engine

    engine = get_postgres_engine()
    all_tickers = list(set(tickers + [benchmark]))
    ticker_list = ", ".join(f"'{t}'" for t in all_tickers)
    query = (
        f"SELECT date, ticker, open, high, low, close, volume "
        f"FROM {_OHLCV_TABLE} "
        f"WHERE ticker IN ({ticker_list}) "
        f"ORDER BY ticker, date"
    )

    df = pl.read_database(query, connection=engine)
    if df.height == 0:
        raise RuntimeError(
            f"No OHLCV data in table '{_OHLCV_TABLE}' "
            f"for tickers {all_tickers}"
        )

    logger.info(
        "Loaded {} rows from PostgreSQL | {} tickers",
        df.height,
        df["ticker"].n_unique(),
    )
    return df


def _filter_oos_period(
    df: pl.DataFrame,
    n_years: int,
) -> pl.DataFrame:
    """Filter DataFrame to the out-of-sample period.

    Keeps only the last ``n_years`` of data based on the maximum date
    in the DataFrame. Correctly accounts for leap years.

    Args:
        df: OHLCV DataFrame with a ``date`` column.
        n_years: Number of years to keep.

    Returns:
        Filtered DataFrame.
    """
    max_date = df["date"].max()
    
    # Calcula o cutoff com segurança para anos bissextos (ex: 29 de fevereiro)
    try:
        cutoff = max_date.replace(year=max_date.year - n_years)
    except ValueError:
        # Cai aqui se max_date for 29 de fev e o ano alvo não for bissexto
        cutoff = max_date.replace(year=max_date.year - n_years, day=28)

    filtered = df.filter(pl.col("date") >= cutoff)
    
    logger.info(
        "OOS filter: {} years | cutoff={} | {} → {} rows",
        n_years,
        cutoff,
        df.height,
        filtered.height,
    )
    return filtered


# ---------------------------------------------------------------------------
# Model factory resolver
# ---------------------------------------------------------------------------


def _resolve_model_factory(use_patchtst: bool) -> Any:
    """Instantiate the appropriate model factory.

    Args:
        use_patchtst: If ``True``, uses PatchTST-based factory.
            If ``False``, uses ``NaiveModelFactory`` for fast validation.

    Returns:
        An object implementing the ``ModelFactory`` protocol.
    """
    if not use_patchtst:
        logger.info("Using NaiveModelFactory (fast mode)")
        return NaiveModelFactory(lookback=5)

    # Lazy import to avoid heavy deps when running in fast mode
    try:
        from src.models.patchtst_model import TitaniumForecaster

        logger.info("Using PatchTST model factory")
        return _PatchTSTModelFactory()
    except ImportError as exc:
        logger.warning(
            "PatchTST not available ({}); falling back to NaiveModelFactory",
            exc,
        )
        return NaiveModelFactory()


class _PatchTSTModelFactory:
    """ModelFactory adapter wrapping TitaniumForecaster with fallback.

    Implements the ``ModelFactory`` protocol by delegating to
    ``TitaniumForecaster.fit()`` and ``TitaniumForecaster.predict_proba()``.
    Falls back to NaiveModelFactory if series is too short.
    """

    def __init__(self) -> None:
        self._forecaster: Any = None
        self._fallback: Any = None

    def train(self, train_df: pl.DataFrame) -> None:
        """Train PatchTST on historical OHLCV data, with fallback for short series."""
        from src.models.patchtst_model import TitaniumForecaster

        try:
            self._forecaster = TitaniumForecaster()
            self._forecaster.fit(train_df)
            self._fallback = None
        except Exception as e:
            if "too short" in str(e).lower():
                logger.warning("Series too short for PatchTST; using NaiveModelFactory fallback")
                self._fallback = NaiveModelFactory(lookback=5)
                self._forecaster = None
            else:
                raise

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Generate per-ticker confidence scores via PatchTST or fallback.

        Returns:
            ``{ticker: P(up)}`` where P(up) is the probability of
            price increase over the forecast horizon.
        """
        if self._fallback is not None:
            return self._fallback.predict(df)

        if self._forecaster is None:
            # Not yet trained — return neutral
            tickers = df["ticker"].unique().sort().to_list()
            return {t: 0.5 for t in tickers}

        proba = self._forecaster.predict_proba(df)
        # predict_proba returns a DataFrame; convert to {ticker: prob_up} dict
        return dict(zip(
            proba["ticker"].to_list(),
            proba["prob_up"].to_list(),
        ))


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------


def _save_outputs(
    result: WalkForwardResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Save benchmark results to disk.

    Saves:
        - ``benchmark_equity.parquet``: daily equity curve
        - ``benchmark_metrics.json``: computed metrics
        - ``benchmark_weights.parquet``: rebalance weight history

    Args:
        result: Walk-forward result.
        output_dir: Directory for output files.

    Returns:
        Dict mapping output name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Equity curve
    equity_path = output_dir / "benchmark_equity.parquet"
    result.equity_curve.write_parquet(equity_path)
    paths["equity"] = equity_path

    # Metrics
    metrics_path = output_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)
    paths["metrics"] = metrics_path

    # Weight history
    if result.rebalance_history:
        weight_rows: list[dict[str, Any]] = []
        for rec in result.rebalance_history:
            for ticker, weight in rec.weights.items():
                weight_rows.append({
                    "date": rec.date,
                    "ticker": ticker,
                    "weight": weight,
                    "turnover": rec.turnover,
                    "costs": rec.costs,
                    "retrained": rec.retrained,
                })
        weights_df = pl.DataFrame(weight_rows)
        weights_path = output_dir / "benchmark_weights.parquet"
        weights_df.write_parquet(weights_path)
        paths["weights"] = weights_path

    logger.info("Outputs saved to {}: {}", output_dir, list(paths.keys()))
    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_us_benchmark(
    *,
    config_path: str = "config/tickers.json",
    output_dir: str = "data/outputs",
    use_patchtst: bool = True,
    n_years: int = 30,
    ohlcv: pl.DataFrame | None = None,
) -> WalkForwardResult:
    """Run the full US benchmark pipeline.

    Steps:
        1. Load config (tickers, benchmark, market params)
        2. Load OHLCV from PostgreSQL (or use provided DataFrame)
        3. Filter to OOS period (last ``n_years``)
        4. Instantiate model factory (PatchTST or Naive)
        5. Run walk-forward backtest (weekly rebalance, semi-annual retrain)
        6. Generate PDF report
        7. Save outputs (Parquet + JSON)

    Args:
        config_path: Path to tickers.json config file.
        output_dir: Directory for output files.
        use_patchtst: Use PatchTST model (slow) or NaiveModelFactory (fast).
        n_years: Number of years for out-of-sample period.
        ohlcv: Pre-loaded OHLCV data.  If ``None``, loads from PostgreSQL.

    Returns:
        Walk-forward result with metrics populated.
    """
    out_path = Path(output_dir)

    # Step 1: Load config
    logger.info("Step 1/7: Loading config from {}", config_path)
    config = load_ticker_config(config_path)
    tickers = config["tickers"]
    benchmark = config["benchmark"]
    rf = config.get("risk_free_rate", 0.05)
    trading_days = config.get("trading_days_per_year", 252)

    logger.info(
        "Config: {} tickers, benchmark={}, rf={}, trading_days={}",
        len(tickers),
        benchmark,
        rf,
        trading_days,
    )

    # Step 2: Load OHLCV
    logger.info("Step 2/7: Loading OHLCV data")
    if ohlcv is None:
        ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)

    # Step 3: Filter OOS period
    logger.info("Step 3/7: Filtering OOS period ({} years)", n_years)
    ohlcv = _filter_oos_period(ohlcv, n_years)

    # Step 4: Model factory
    logger.info("Step 4/7: Initializing model factory")
    model_factory = _resolve_model_factory(use_patchtst)

    # Step 5: Walk-forward backtest
    logger.info("Step 5/7: Running walk-forward backtest")
    n = len(tickers)
    wf_config = WalkForwardConfig(
        rebalance_every=13,            # ~2.5 weeks (CPCV-OOS validated)
        retrain_every=126,             # Semestral
        lookback_days=756,             # ~3 anos de covariância (CPCV-OOS validated)
        initial_capital=1_000_000.0,
        costs=TransactionCosts(
            slippage_bps=5.0,
            commission_bps=10.0,
        ),
        min_rebalance_delta=0.02,      # 2% threshold
        trading_days_per_year=trading_days,
        rf=rf,
        hrp_config=HRPConfig(
            linkage_method="ward",     # CPCV-OOS validated (+0.03 vs single)
            shrinkage=True,            # Ledoit-Wolf (CPCV-OOS validated +0.03)
            max_weight=min(0.25, 2 / n),
        ),
    )

    backtester = WalkForwardBacktester(config=wf_config)
    result = backtester.run(ohlcv, tickers, benchmark, model_factory)

    # Step 6: Generate PDF report
    logger.info("Step 6/7: Generating PDF report")
    try:
        report = BenchmarkReport(
            result,
            benchmark_name=f"{benchmark}",
            output_dir=out_path,
        )
        report.generate()
    except Exception as exc:
        logger.warning("PDF report generation failed: {}", exc)

    # Step 7: Save outputs
    logger.info("Step 7/7: Saving outputs")
    _save_outputs(result, out_path)

    logger.info(
        "Benchmark complete | Sharpe={:.3f} | CAGR={:.2%} | MaxDD={:.2%}",
        result.metrics.get("sharpe_ratio", 0.0),
        result.metrics.get("cagr", 0.0),
        result.metrics.get("max_drawdown", 0.0),
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    use_naive = "--naive" in sys.argv
    run_us_benchmark(use_patchtst=not use_naive)
