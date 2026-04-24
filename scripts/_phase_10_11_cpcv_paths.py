"""Generate data/outputs/cpcv_paths.parquet for dashboard Phases 10 & 11.

Runs a single CPCV-OOS validation on the walk-forward **champion** config
(identical to ``src.backtest.run_benchmark.run_us_benchmark``) with
``collect_equity=True`` and persists the 15 per-path equity curves in
the long format consumed by
``src.dashboard.app.load_cpcv_paths``.

The model factory is ``NaiveModelFactory(lookback=5)`` — the documented
CPCV-OOS proxy that makes the 15-path sweep feasible (~10-20 min total
vs. multiple hours with PatchTST).

Outputs:
    - ``data/outputs/cpcv_paths.parquet`` (columns: ``config``, ``path_id``,
      ``date``, ``equity``, ``sharpe``). Consumed by the spaghetti chart
      (Phase 10) and the Sharpe violin chart (Phase 11).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import CPCVParameterValidator, save_cpcv_paths
from src.backtest.run_benchmark import _filter_oos_period, _load_ohlcv_from_postgres
from src.backtest.walk_forward import NaiveModelFactory, WalkForwardConfig
from src.config import load_ticker_config
from src.portfolio.hrp import HRPConfig

_OUTPUT_PATH = Path("data/outputs/cpcv_paths.parquet")
_CONFIG_NAME = "champion"


def _build_champion_config(n_tickers: int, rf: float, trading_days: int) -> WalkForwardConfig:
    """Replicate the champion walk-forward config from ``run_benchmark``."""
    return WalkForwardConfig(
        rebalance_every=15,
        retrain_every=126,
        lookback_days=756,
        initial_capital=1_000_000.0,
        costs=TransactionCosts(slippage_bps=5.0, commission_bps=10.0),
        min_rebalance_delta=0.02,
        target_vol=0.10,
        vol_lookback=63,
        min_leverage=0.5,
        max_leverage=1.0,
        trading_days_per_year=trading_days,
        rf=rf,
        hrp_config=HRPConfig(
            linkage_method="ward",
            shrinkage=True,
            max_weight=min(0.06, 2 / n_tickers),
        ),
        top_n=None,
    )


def main(
    *,
    config_path: str = "config/tickers.json",
    n_years: int = 13,
    output_path: Path = _OUTPUT_PATH,
) -> Path:
    logger.info("Phase 10/11 — generating {}", output_path)

    config = load_ticker_config(config_path)
    tickers = config["tickers"]
    benchmark = config["benchmark"]
    rf = config.get("risk_free_rate", 0.05)
    trading_days = config.get("trading_days_per_year", 252)

    ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)
    ohlcv = _filter_oos_period(ohlcv, n_years)

    wf_config = _build_champion_config(len(tickers), rf, trading_days)

    validator = CPCVParameterValidator(
        ohlcv=ohlcv,
        tickers=tickers,
        benchmark_ticker=benchmark,
        n_splits=6,
        n_test_groups=2,
        embargo_pct=0.01,
        purge_days=5,
    )

    logger.info(
        "Running CPCV-OOS on champion (NaiveModelFactory lookback=5, "
        "{} paths) — this may take ~10-20 min",
        len(validator._paths),
    )

    result = validator.validate(
        config=wf_config,
        model_factory=NaiveModelFactory(lookback=5),
        collect_equity=True,
        n_trials=1,  # path-level equity for a single selected config
    )

    logger.info(
        "CPCV complete: mean_sharpe={:.3f}, std={:.3f}, pct_positive={:.1%}",
        result.mean_sharpe,
        result.std_sharpe,
        result.pct_positive,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = save_cpcv_paths(result, output_path, config_name=_CONFIG_NAME)

    logger.info("Per-path Sharpes: {}", [round(s, 3) for s in result.per_path_sharpe])
    logger.info("Saved paths parquet → {}", written)
    return written


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
