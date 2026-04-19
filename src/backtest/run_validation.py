"""Comprehensive fine-tuning grid search for walk-forward strategy parameters.

Three-tier systematic approach designed for ~18h per tier:

  Tier 1 (~165 configs): Single-axis sweeps + key 2-way interactions
  Tier 2 (~165 configs): Multi-axis factorial combinations (3/4-way)
  Tier 3 (~170 configs): Adaptive champion refinement from Tier 1+2 winners

Features:

  - Resume: skips already-computed configs on restart
  - Incremental saves: persists every 5 configs (crash-safe)
  - ETA: rolling-average time estimation
  - Cross-tier dedup: same parameter fingerprint never runs twice
  - Holdout validation: champion tested on unseen temporal holdout (n_trials=1)

Time budget (C(6,2)=15 CPCV-OOS paths):

  - NaiveModelFactory: ~375s per config (25s x 15 paths), ~172 configs per 18h
  - PatchTST (cached): ~inference-only per path, slower but uses real signal

Usage::

    python -m src.backtest.run_validation --tier 1        # Day 1
    python -m src.backtest.run_validation --tier 2        # Day 2
    python -m src.backtest.run_validation --tier 3        # Day 3
    python -m src.backtest.run_validation --tier all       # Run all sequentially
    python -m src.backtest.run_validation --holdout        # Holdout validation (requires prior tier results)
    python -m src.backtest.run_validation --holdout --holdout-years=3
    python -m src.backtest.run_validation --estimate       # Time 1 config
    python -m src.backtest.run_validation --dry-run        # Print grid, no execution
    python -m src.backtest.run_validation --tier 1 --patchtst  # Use cached PatchTST

PatchTST cache interaction
--------------------------
When using ``_PatchTSTModelFactory`` instead of ``NaiveModelFactory``, each
``train()`` call caches the fitted model to ``models/wf_cache/<hash>/``.
The hash is computed from the **training window + PatchTST hyperparameters**,
so the cache is hit only when ALL of the following match a previous run:

  Parameters that INVALIDATE the cache (trigger full retrain):

    Data window (determined by WalkForwardConfig):
      - ``retrain_every``  — shifts *when* each retrain happens
      - ``lookback_days``  — changes the window size (start date + row count)
      - ``n_years``        — changes the OOS period, shifting retrain dates
      - ticker list        — adding/removing tickers changes the window

    PatchTST hyperparameters (TitaniumForecaster defaults):
      - ``h``              — forecast horizon (default 5)
      - ``input_size``     — lookback window for the model (default 60)
      - ``batch_size``     — training batch size (default 32)
      - ``max_steps``      — max training iterations (default 5000)
      - ``learning_rate``  — Adam lr (default 1e-4)
      - ``quantiles``      — quantile levels (default [0.1, 0.25, 0.5, 0.75, 0.9])
      - ``random_seed``    — reproducibility seed (default 42)

  Parameters that DO NOT invalidate the cache (safe to sweep freely):
    - ``rebalance_every``       — rebalance frequency (post-prediction)
    - ``top_n``                 — ticker selection (post-prediction)
    - ``hrp_config`` (all)      — linkage, shrinkage, max_weight, tilt, etc.
    - ``target_vol``            — vol targeting (post-allocation)
    - ``killswitch``            — drawdown killswitch (post-allocation)
    - ``min_rebalance_delta``   — turnover filter (post-allocation)
    - ``costs``                 — transaction costs (post-allocation)
    - ``momentum_lookback``     — only used by NaiveModelFactory

To clear the cache, delete ``models/wf_cache/``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import (
    CPCVParameterValidator,
    ValidationResult,
    deflated_sharpe_ratio,
)
from src.backtest.walk_forward import (
    KillswitchConfig,
    ModelFactory,
    NaiveModelFactory,
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
)
from src.portfolio.hrp import HRPConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path("data/outputs")
_BASE_COSTS = TransactionCosts(slippage_bps=5.0, commission_bps=10.0)
_SAVE_EVERY = 5
_DEFAULT_HOLDOUT_YEARS = 2

# Model factory toggle — set by CLI (--patchtst) or programmatically.
# When True, uses _PatchTSTModelFactory (cached) instead of NaiveModelFactory.
_USE_PATCHTST: bool = False


# ---------------------------------------------------------------------------
# Holdout result
# ---------------------------------------------------------------------------


@dataclass
class HoldoutResult:
    """Output of a holdout temporal validation.

    Attributes:
        champion_name: Name of the champion config from grid search.
        champion_params: Full parameter dict of the champion.
        grid_search_sharpe: Mean Sharpe from CPCV-OOS grid search.
        grid_search_dsr: DSR p-value from grid search (penalised by n_trials).
        holdout_sharpe: Sharpe computed on the holdout period only.
        holdout_dsr: DSR p-value with n_trials=1 on the holdout.
        holdout_accepted: Whether holdout_dsr > 0.95.
        holdout_n_obs: Number of daily return observations in holdout.
        holdout_metrics: Full walk-forward metrics on the holdout period.
        wf_result: Full WalkForwardResult from the holdout run.
        metadata: Extra info (holdout_years, holdout_start, etc.).
    """

    champion_name: str
    champion_params: dict[str, Any]
    grid_search_sharpe: float
    grid_search_dsr: float
    holdout_sharpe: float
    holdout_dsr: float
    holdout_accepted: bool
    holdout_n_obs: int
    holdout_metrics: dict[str, float] = field(default_factory=dict)
    wf_result: WalkForwardResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Holdout data helpers
# ---------------------------------------------------------------------------


def _holdout_cutoff(df: pl.DataFrame, holdout_years: int) -> date:
    """Compute the holdout start date from the data.

    Args:
        df: OHLCV DataFrame with a ``date`` column.
        holdout_years: Number of years to reserve for holdout.

    Returns:
        The cutoff date (holdout starts at this date, inclusive).

    Raises:
        ValueError: If holdout_years exceeds the data span.
    """
    if holdout_years < 1:
        raise ValueError(f"holdout_years must be >= 1, got {holdout_years}")
    max_date = df["date"].max()
    min_date = df["date"].min()
    try:
        cutoff = max_date.replace(year=max_date.year - holdout_years)
    except ValueError:
        # Leap year edge case (Feb 29 → Feb 28)
        cutoff = max_date.replace(year=max_date.year - holdout_years, day=28)
    if cutoff <= min_date:
        raise ValueError(
            f"holdout_years={holdout_years} exceeds data span "
            f"({min_date} to {max_date}); cutoff {cutoff} is at or before data start"
        )
    return cutoff


def filter_before_holdout(
    df: pl.DataFrame,
    holdout_years: int,
) -> pl.DataFrame:
    """Filter OHLCV to exclude the holdout period.

    Keeps only data **before** the holdout cutoff so the grid search
    never sees holdout returns.

    Args:
        df: OHLCV DataFrame with a ``date`` column.
        holdout_years: Number of years to reserve for holdout.

    Returns:
        Filtered DataFrame (grid search period only).
    """
    cutoff = _holdout_cutoff(df, holdout_years)
    filtered = df.filter(pl.col("date") < cutoff)
    logger.info(
        "Holdout filter: removed last {} years | cutoff={} | {} → {} rows",
        holdout_years, cutoff, df.height, filtered.height,
    )
    return filtered


def prepare_holdout_ohlcv(
    df: pl.DataFrame,
    holdout_years: int,
    lookback_days: int,
) -> tuple[pl.DataFrame, date]:
    """Prepare OHLCV for holdout walk-forward run.

    The holdout backtester needs a lookback buffer before the holdout
    start date for covariance estimation and model warm-up.  This
    function returns OHLCV starting from ``holdout_start - buffer``
    and the holdout start date.

    Args:
        df: Full OHLCV DataFrame.
        holdout_years: Number of years in the holdout period.
        lookback_days: Trading days needed for lookback (e.g. 756).

    Returns:
        Tuple of (filtered OHLCV with buffer, holdout_start date).
    """
    holdout_start = _holdout_cutoff(df, holdout_years)
    # Convert trading days to calendar days with margin
    buffer_calendar = int(lookback_days * 365 / 252) + 30
    buffer_start = holdout_start - timedelta(days=buffer_calendar)
    filtered = df.filter(pl.col("date") >= buffer_start)
    logger.info(
        "Holdout OHLCV: holdout_start={} | buffer_start={} | {} rows",
        holdout_start, buffer_start, filtered.height,
    )
    return filtered, holdout_start


# ---------------------------------------------------------------------------
# Champion identification
# ---------------------------------------------------------------------------


def identify_champion(
    output_dir: Path = _OUTPUT_DIR,
) -> tuple[str, dict[str, Any], float, float]:
    """Find the best config from all tier results.

    Ranks by ``mean_sharpe`` (not DSR, since DSR is uniformly penalised
    by n_trials — the ranking is identical).

    Args:
        output_dir: Directory containing tier result JSON files.

    Returns:
        Tuple of (name, params, mean_sharpe, dsr_pvalue).

    Raises:
        RuntimeError: If no tier results are found.
    """
    results = _load_all_results(output_dir)
    if not results:
        raise RuntimeError(
            f"No tier results found in {output_dir}. "
            "Run --tier 1/2/3 before --holdout."
        )
    # Sort by mean_sharpe descending (DSR ranking is identical)
    results.sort(key=lambda x: x.get("mean_sharpe", 0), reverse=True)
    best = results[0]
    return (
        best["name"],
        best.get("params", {}),
        best.get("mean_sharpe", 0.0),
        best.get("deflated_sharpe", 0.0),
    )


# ---------------------------------------------------------------------------
# Holdout validation
# ---------------------------------------------------------------------------


def run_holdout_validation(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    benchmark_ticker: str = "SPY",
    holdout_years: int = _DEFAULT_HOLDOUT_YEARS,
    output_dir: str = "data/outputs",
) -> HoldoutResult:
    """Validate the grid search champion on an unseen holdout period.

    The champion (best mean Sharpe from tiers 1-3) is run through a
    full walk-forward backtest on the holdout period.  DSR is computed
    with ``n_trials=1`` (no multiple-testing penalty), testing whether
    the out-of-sample Sharpe is genuinely positive.

    Args:
        ohlcv: Full OHLCV data (including both grid search and holdout periods).
        tickers: List of ticker symbols.
        benchmark_ticker: Benchmark ticker (e.g. ``"SPY"``).
        holdout_years: Number of years reserved for holdout.
        output_dir: Directory for output files.

    Returns:
        HoldoutResult with acceptance verdict.
    """
    out_path = Path(output_dir)
    n = len(tickers)

    # 1. Identify champion
    champ_name, champ_params, gs_sharpe, gs_dsr = identify_champion(out_path)
    logger.info(
        "Champion: {} | grid Sharpe={:.3f} | grid DSR={:.3f}",
        champ_name, gs_sharpe, gs_dsr,
    )

    # 2. Reconstruct trial
    trial = _trial_from_params(champ_name, champ_params, n)
    lookback = trial.wf_config.lookback_days

    # 3. Prepare holdout data (with lookback buffer)
    holdout_ohlcv, holdout_start = prepare_holdout_ohlcv(
        ohlcv, holdout_years, lookback,
    )

    # 4. Run walk-forward on holdout period
    logger.info("Running walk-forward on holdout ({} → end)...", holdout_start)
    backtester = WalkForwardBacktester(config=trial.wf_config)
    wf_result = backtester.run(
        holdout_ohlcv, tickers, benchmark_ticker, trial.factory,
    )

    # 5. Filter daily returns to holdout period only
    holdout_returns = wf_result.daily_returns.filter(
        pl.col("date") >= holdout_start
    )
    n_obs = holdout_returns.height
    logger.info("Holdout observations: {} days", n_obs)

    if n_obs < 2:
        logger.warning("Insufficient holdout data ({} obs); cannot compute Sharpe", n_obs)
        return HoldoutResult(
            champion_name=champ_name,
            champion_params=champ_params,
            grid_search_sharpe=gs_sharpe,
            grid_search_dsr=gs_dsr,
            holdout_sharpe=0.0,
            holdout_dsr=0.0,
            holdout_accepted=False,
            holdout_n_obs=n_obs,
            metadata={"holdout_years": holdout_years, "error": "insufficient_data"},
        )

    # 6. Compute Sharpe on holdout returns
    port_returns = holdout_returns["portfolio_return"].to_list()
    rf = trial.wf_config.rf
    trading_days = trial.wf_config.trading_days_per_year
    rf_daily = (1.0 + rf) ** (1.0 / trading_days) - 1.0
    excess = [r - rf_daily for r in port_returns]

    mean_ex = sum(excess) / n_obs
    var = sum((e - mean_ex) ** 2 for e in excess) / (n_obs - 1)
    holdout_sharpe = (mean_ex / math.sqrt(var)) * math.sqrt(trading_days) if var > 0 else 0.0

    # 7. Compute DSR with n_trials=1
    # Skew/kurt computed on excess returns (same series as Sharpe) per Lo (2002)
    skew = _holdout_skewness(excess)
    kurt = _holdout_kurtosis(excess)
    holdout_dsr = deflated_sharpe_ratio(
        observed_sharpe=holdout_sharpe,
        n_trials=1,
        n_observations=n_obs,
        skewness=skew,
        kurtosis=kurt,
        sharpe_benchmark=0.0,
        periods_per_year=trading_days,
    )

    accepted = holdout_dsr > 0.95

    logger.info(
        "HOLDOUT RESULT: Sharpe={:.3f} | DSR={:.3f} | n_obs={} | accepted={}",
        holdout_sharpe, holdout_dsr, n_obs, accepted,
    )

    result = HoldoutResult(
        champion_name=champ_name,
        champion_params=champ_params,
        grid_search_sharpe=gs_sharpe,
        grid_search_dsr=gs_dsr,
        holdout_sharpe=holdout_sharpe,
        holdout_dsr=holdout_dsr,
        holdout_accepted=accepted,
        holdout_n_obs=n_obs,
        holdout_metrics=wf_result.metrics,
        wf_result=wf_result,
        metadata={
            "holdout_years": holdout_years,
            "holdout_start": str(holdout_start),
            "skewness": skew,
            "kurtosis": kurt,
            "grid_search_n_configs": len(_load_all_results(out_path)),
        },
    )

    # 8. Save outputs
    _save_holdout_results(result, out_path)
    return result


def _holdout_skewness(values: list[float]) -> float:
    """Sample skewness for holdout returns."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (m2 ** 1.5)


def _holdout_kurtosis(values: list[float]) -> float:
    """Sample kurtosis (full, not excess) for holdout returns."""
    n = len(values)
    if n < 4:
        return 3.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 3.0
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m4 / (m2 ** 2)


def _save_holdout_results(result: HoldoutResult, output_dir: Path) -> None:
    """Save holdout validation results to JSON and Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_dir / "validation_holdout_results.json"
    data = {
        "champion_name": result.champion_name,
        "champion_params": result.champion_params,
        "grid_search_sharpe": result.grid_search_sharpe,
        "grid_search_dsr": result.grid_search_dsr,
        "holdout_sharpe": result.holdout_sharpe,
        "holdout_dsr": result.holdout_dsr,
        "holdout_accepted": result.holdout_accepted,
        "holdout_n_obs": result.holdout_n_obs,
        "holdout_metrics": result.holdout_metrics,
        "metadata": result.metadata,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Holdout results saved to {}", json_path)

    # Equity curve parquet
    if result.wf_result is not None:
        eq_path = output_dir / "validation_holdout_equity.parquet"
        result.wf_result.equity_curve.write_parquet(eq_path)
        logger.info("Holdout equity curve saved to {}", eq_path)

    # Markdown summary
    md_path = output_dir / "validation_holdout_summary.md"
    verdict = "ACCEPTED" if result.holdout_accepted else "REJECTED"
    lines = [
        "# Holdout Validation Results", "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
        f"## Champion: {result.champion_name}", "",
        "| Metric | Grid Search | Holdout |",
        "|--------|-------------|---------|",
        f"| Sharpe | {result.grid_search_sharpe:.3f} | {result.holdout_sharpe:.3f} |",
        f"| DSR p-value | {result.grid_search_dsr:.3f} | {result.holdout_dsr:.3f} |",
        f"| n_trials | {result.metadata.get('grid_search_n_configs', '?')} | 1 |",
        f"| Observations | — | {result.holdout_n_obs} |", "",
        f"## Verdict: **{verdict}**", "",
        f"Holdout period: {result.metadata.get('holdout_years', '?')} years "
        f"(from {result.metadata.get('holdout_start', '?')})", "",
    ]
    if result.holdout_metrics:
        lines.extend(["## Holdout Walk-Forward Metrics", ""])
        for k, v in sorted(result.holdout_metrics.items()):
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Holdout summary saved to {}", md_path)


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """Single config + factory pair for CPCV-OOS validation."""

    name: str
    wf_config: WalkForwardConfig
    factory: ModelFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dyn_maxw(n: int) -> float:
    """Dynamic max_weight = min(0.25, 2/n)."""
    return min(0.25, 2.0 / max(n, 1))


def _make_factory() -> ModelFactory:
    """Create the model factory based on ``_USE_PATCHTST`` toggle.

    When ``_USE_PATCHTST`` is True, imports and returns a
    ``_PatchTSTModelFactory`` (with cache enabled) from
    ``run_benchmark``.  Otherwise returns a ``NaiveModelFactory``.
    """
    if _USE_PATCHTST:
        from src.backtest.run_benchmark import _PatchTSTModelFactory

        return _PatchTSTModelFactory(cache_enabled=True)
    return NaiveModelFactory(lookback=5)


def _hrp(n: int, **kw: Any) -> HRPConfig:
    """HRPConfig with baseline defaults + overrides.

    Baseline matches CPCV-OOS validated config: ward + Ledoit-Wolf shrinkage.
    """
    d: dict[str, Any] = {
        "linkage_method": "ward",
        "correlation_method": "pearson",
        "shrinkage": True,
        "confidence_tilt_cap": 0.20,
        "min_weight": 0.0,
        "max_weight": _dyn_maxw(n),
        "turnover_threshold": 0.02,
    }
    d.update(kw)
    return HRPConfig(**d)


def _wf(**kw: Any) -> WalkForwardConfig:
    """WalkForwardConfig with baseline defaults + overrides.

    Baseline matches CPCV-OOS validated config: rb=13, rt=126, cov=756.
    retrain_every and lookback_days are fixed to avoid PatchTST cache
    invalidation — do NOT sweep these parameters.
    """
    d: dict[str, Any] = {
        "rebalance_every": 13,
        "retrain_every": 126,
        "lookback_days": 756,
        "initial_capital": 1_000_000.0,
        "costs": _BASE_COSTS,
        "min_rebalance_delta": 0.02,
        "trading_days_per_year": 252,
        "rf": 0.05,
    }
    d.update(kw)
    return WalkForwardConfig(**d)


def _t(n: int, name: str, **wf_kw: Any) -> Trial:
    """Shorthand: create Trial with validated baseline defaults + overrides.

    Always sets ``hrp_config`` to the validated baseline (ward + Ledoit-Wolf
    shrinkage) unless explicitly overridden, so walk-forward execution
    matches the parameter fingerprint.  Momentum is fixed at 5 (PatchTST-safe).
    """
    wf_kw.setdefault("hrp_config", _hrp(n))
    return Trial(name=name, wf_config=_wf(**wf_kw), factory=_make_factory())


def _trial_params(trial: Trial, n: int) -> dict[str, Any]:
    """Extract all tunable parameters into a flat dict for fingerprinting."""
    cfg = trial.wf_config
    hrp = cfg.hrp_config or HRPConfig(max_weight=_dyn_maxw(n))
    ks = cfg.killswitch
    costs = cfg.costs or TransactionCosts()
    return {
        "momentum_lookback": getattr(trial.factory, "lookback", None),
        "rebalance_every": cfg.rebalance_every,
        "retrain_every": cfg.retrain_every,
        "lookback_days": cfg.lookback_days,
        "linkage_method": hrp.linkage_method,
        "shrinkage": hrp.shrinkage,
        "correlation_method": hrp.correlation_method,
        "confidence_tilt_cap": hrp.confidence_tilt_cap,
        "max_weight": round(hrp.max_weight, 6),
        "min_weight": hrp.min_weight,
        "turnover_threshold": hrp.turnover_threshold,
        "target_vol": cfg.target_vol,
        "vol_lookback": cfg.vol_lookback,
        "min_leverage": cfg.min_leverage,
        "max_leverage": cfg.max_leverage,
        "killswitch_max_dd": ks.max_drawdown_pct if ks else None,
        "killswitch_recovery": ks.recovery_threshold_pct if ks else None,
        "killswitch_ramp": ks.ramp_up_days if ks else None,
        "min_rebalance_delta": cfg.min_rebalance_delta,
        "slippage_bps": costs.slippage_bps,
        "commission_bps": costs.commission_bps,
        "market_impact_bps": costs.market_impact_bps,
        "top_n": cfg.top_n,
    }


def _param_fp(params: dict[str, Any]) -> str:
    """Canonical JSON string for dedup (parameter fingerprint)."""
    norm: dict[str, Any] = {}
    for k in sorted(params):
        v = params[k]
        norm[k] = round(v, 6) if isinstance(v, float) else v
    return json.dumps(norm, sort_keys=True)


def _dedup(
    trials: list[Trial], n: int, existing_fps: set[str] | None = None,
) -> list[Trial]:
    """Remove duplicate trials by parameter fingerprint."""
    seen = set(existing_fps) if existing_fps else set()
    unique: list[Trial] = []
    for t in trials:
        fp = _param_fp(_trial_params(t, n))
        if fp not in seen:
            seen.add(fp)
            unique.append(t)
    return unique


def _trial_from_params(name: str, params: dict[str, Any], n: int) -> Trial:
    """Reconstruct a Trial from a saved parameter dict."""
    hrp = _hrp(
        n,
        linkage_method=params.get("linkage_method", "single"),
        correlation_method=params.get("correlation_method", "pearson"),
        shrinkage=params.get("shrinkage", False),
        confidence_tilt_cap=params.get("confidence_tilt_cap", 0.20),
        max_weight=params.get("max_weight", _dyn_maxw(n)),
        min_weight=params.get("min_weight", 0.0),
        turnover_threshold=params.get("turnover_threshold", 0.02),
    )
    ks = None
    if params.get("killswitch_max_dd") is not None:
        ks = KillswitchConfig(
            max_drawdown_pct=params["killswitch_max_dd"],
            recovery_threshold_pct=params.get("killswitch_recovery", -0.05),
            ramp_up_days=params.get("killswitch_ramp", 21),
        )
    costs = TransactionCosts(
        slippage_bps=params.get("slippage_bps", 5.0),
        commission_bps=params.get("commission_bps", 10.0),
        market_impact_bps=params.get("market_impact_bps", 0.0),
    )
    wf = WalkForwardConfig(
        rebalance_every=params.get("rebalance_every", 5),
        retrain_every=params.get("retrain_every", 126),
        lookback_days=params.get("lookback_days", 504),
        initial_capital=1_000_000.0,
        costs=costs,
        min_rebalance_delta=params.get("min_rebalance_delta", 0.02),
        trading_days_per_year=252,
        rf=0.05,
        target_vol=params.get("target_vol"),
        vol_lookback=params.get("vol_lookback", 63),
        max_leverage=params.get("max_leverage", 1.0),
        min_leverage=params.get("min_leverage", 0.5),
        killswitch=ks,
        hrp_config=hrp,
        top_n=params.get("top_n"),
    )
    factory = _make_factory()
    return Trial(name=name, wf_config=wf, factory=factory)


# ---------------------------------------------------------------------------
# Tier 1: Single-axis sweeps + 2-way interactions (~165 configs)
# ---------------------------------------------------------------------------


def build_tier1(n: int) -> list[Trial]:
    """Tier 1: isolate marginal effect of each PatchTST-safe dimension, then test key pairs.

    Fixed (PatchTST cache-safe): retrain_every=126, lookback_days=756, momentum=5.
    Swept: rebalance_every, top_n, HRP params, target_vol, killswitch, costs, delta.

    Note on top_n x max_weight: when top_n is set, walk_forward.py dynamically
    overrides max_weight to min(0.25, 2/n_sel), so these interactions are excluded.
    """
    trials: list[Trial] = []

    # === BASELINE (rb=13, rt=126, cov=756, ward+shrink) ===
    trials.append(_t(n,"baseline"))

    # === A. SINGLE-AXIS SWEEPS ===

    # A1. Rebalance frequency (8 new, baseline=13)
    for rb in [1, 2, 3, 5, 7, 10, 15, 21]:
        trials.append(_t(n,f"s_rebal{rb:03d}", rebalance_every=rb))

    # A2. Top-N ticker selection (8 new, baseline=None=all)
    for tn in [5, 10, 15, 20, 25, 30, 35, 40]:
        trials.append(_t(n,f"s_topn{tn:03d}", top_n=tn))

    # A3. HRP linkage x shrinkage (3 new, baseline=ward+shrink)
    trials.append(_t(n,"s_single", hrp_config=_hrp(n, linkage_method="single")))
    trials.append(_t(n,"s_noshrink", hrp_config=_hrp(n, shrinkage=False)))
    trials.append(_t(n,"s_single_noshrink", hrp_config=_hrp(n, linkage_method="single", shrinkage=False)))

    # A4. Confidence tilt cap (8 new, baseline=0.20)
    for tc in [0.0, 0.05, 0.10, 0.15, 0.30, 0.50, 0.75, 1.0]:
        trials.append(_t(n,f"s_tilt{int(tc * 100):03d}", hrp_config=_hrp(n, confidence_tilt_cap=tc)))

    # A5. Target volatility (6 new, baseline=None)
    for tv in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        trials.append(_t(n,f"s_vol{int(tv * 100):03d}", target_vol=tv))

    # A6. Max weight (6 new, baseline=dyn_maxw)
    for mw in [0.04, 0.06, 0.08, 0.10, 0.15, 0.25]:
        trials.append(_t(n,f"s_maxw{int(mw * 100):03d}", hrp_config=_hrp(n, max_weight=mw)))

    # A7. Min rebalance delta (6 new, baseline=0.02)
    for d in [0.005, 0.01, 0.015, 0.03, 0.04, 0.05]:
        trials.append(_t(n,f"s_delta{int(d * 1000):03d}", min_rebalance_delta=d))

    # A8. Killswitch threshold (6 new, baseline=None)
    for dd in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        trials.append(_t(n,f"s_ks{int(dd * 100):03d}", killswitch=KillswitchConfig(max_drawdown_pct=-dd)))

    # A9. Correlation method (1 new, baseline=pearson)
    trials.append(_t(n,"s_spearman", hrp_config=_hrp(n, correlation_method="spearman")))

    # A10. HRP turnover threshold (5 new, baseline=0.02)
    for to in [0.0, 0.005, 0.01, 0.03, 0.05]:
        trials.append(_t(n,f"s_to{int(to * 1000):03d}", hrp_config=_hrp(n, turnover_threshold=to)))

    # A11. Vol lookback (4 new, with target_vol=0.12 to activate)
    for vlb in [21, 42, 84, 126]:
        trials.append(_t(n,f"s_vlb{vlb:03d}", target_vol=0.12, vol_lookback=vlb))

    # A12. Min leverage (4 new, with target_vol=0.12)
    for ml in [0.0, 0.2, 0.3, 0.7]:
        trials.append(_t(n,f"s_mlev{int(ml * 10):02d}", target_vol=0.12, min_leverage=ml))

    # A13. Killswitch recovery threshold (3 new, with max_dd=-0.15)
    for rec in [0.02, 0.08, 0.10]:
        trials.append(_t(n,
            f"s_ksrec{int(rec * 100):03d}",
            killswitch=KillswitchConfig(max_drawdown_pct=-0.15, recovery_threshold_pct=-rec),
        ))

    # A14. Killswitch ramp days (4 new, with max_dd=-0.15)
    for rd in [5, 10, 15, 42]:
        trials.append(_t(n,
            f"s_ksramp{rd:03d}",
            killswitch=KillswitchConfig(max_drawdown_pct=-0.15, ramp_up_days=rd),
        ))

    # === B. 2-WAY INTERACTIONS ===

    # B1. Rebalance x Tilt (16)
    for rb in [1, 5, 10, 21]:
        for tc in [0.0, 0.10, 0.50, 1.0]:
            trials.append(_t(n,
                f"x_rebal{rb:03d}_tilt{int(tc * 100):03d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, confidence_tilt_cap=tc),
            ))

    # B2. single+noShrink x Tilt (4, baseline=ward+shrink so test opposite)
    for tc in [0.0, 0.10, 0.50, 1.0]:
        trials.append(_t(n,
            f"x_singlenoshrink_tilt{int(tc * 100):03d}",
            hrp_config=_hrp(n, linkage_method="single", shrinkage=False, confidence_tilt_cap=tc),
        ))

    # B3. Vol target x Killswitch (4)
    for tv in [0.10, 0.15]:
        for dd in [0.15, 0.20]:
            trials.append(_t(n,
                f"x_vol{int(tv * 100):03d}_ks{int(dd * 100):03d}",
                target_vol=tv,
                killswitch=KillswitchConfig(max_drawdown_pct=-dd),
            ))

    # B4. Rebalance x Delta (6)
    for rb in [1, 5]:
        for d in [0.01, 0.03, 0.05]:
            trials.append(_t(n,
                f"x_rebal{rb:03d}_delta{int(d * 1000):03d}",
                rebalance_every=rb,
                min_rebalance_delta=d,
            ))

    # B5. top_n x Rebalance (9)
    for tn in [10, 20, 30]:
        for rb in [5, 10, 21]:
            trials.append(_t(n,
                f"x_topn{tn:03d}_rebal{rb:03d}",
                top_n=tn, rebalance_every=rb,
            ))

    # B6. top_n x Tilt (9)
    for tn in [10, 20, 30]:
        for tc in [0.0, 0.50, 1.0]:
            trials.append(_t(n,
                f"x_topn{tn:03d}_tilt{int(tc * 100):03d}",
                top_n=tn,
                hrp_config=_hrp(n, confidence_tilt_cap=tc),
            ))

    # B7. top_n x Killswitch (6)
    for tn in [10, 20, 30]:
        for dd in [0.12, 0.20]:
            trials.append(_t(n,
                f"x_topn{tn:03d}_ks{int(dd * 100):03d}",
                top_n=tn,
                killswitch=KillswitchConfig(max_drawdown_pct=-dd),
            ))

    # B8. Rebalance x Max weight (9, top_n=None so max_weight is used)
    for rb in [1, 10, 21]:
        for mw in [0.04, 0.10, 0.25]:
            trials.append(_t(n,
                f"x_rebal{rb:03d}_maxw{int(mw * 100):03d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, max_weight=mw),
            ))

    # B9. top_n x HRP variant (6)
    for tn in [10, 20, 30]:
        for lnk, shr, hname in [
            ("single", False, "single_noshrink"),
            ("ward", False, "ward_noshrink"),
        ]:
            trials.append(_t(n,
                f"x_topn{tn:03d}_{hname}",
                top_n=tn,
                hrp_config=_hrp(n, linkage_method=lnk, shrinkage=shr),
            ))

    # B10. Rebalance x Killswitch (6)
    for rb in [5, 10, 21]:
        for dd in [0.12, 0.20]:
            trials.append(_t(n,
                f"x_rebal{rb:03d}_ks{int(dd * 100):03d}",
                rebalance_every=rb,
                killswitch=KillswitchConfig(max_drawdown_pct=-dd),
            ))

    # === C. STRATEGIC 3-WAY COMBOS ===

    # C1. top_n x Rebalance x Tilt (6)
    for tn, rb, tc in [
        (20, 13, 0.50), (20, 13, 1.0), (30, 10, 0.50),
        (10, 5, 1.0), (15, 13, 0.20), (20, 10, 1.0),
    ]:
        trials.append(_t(n,
            f"x3_topn{tn:03d}_rebal{rb:03d}_tilt{int(tc * 100):03d}",
            top_n=tn, rebalance_every=rb,
            hrp_config=_hrp(n, confidence_tilt_cap=tc),
        ))

    # C2. top_n x Rebalance x KS (4)
    for tn, rb, dd in [(20, 13, 0.15), (30, 13, 0.15), (20, 10, 0.20), (10, 10, 0.15)]:
        trials.append(_t(n,
            f"x3_topn{tn:03d}_rebal{rb:03d}_ks{int(dd * 100):03d}",
            top_n=tn, rebalance_every=rb,
            killswitch=KillswitchConfig(max_drawdown_pct=-dd),
        ))

    # C3. top_n + target vol (2)
    trials.append(_t(n,"x3_topn020_vol012", top_n=20, target_vol=0.12))
    trials.append(_t(n,
        "x3_topn020_vol012_ks015",
        top_n=20, target_vol=0.12,
        killswitch=KillswitchConfig(max_drawdown_pct=-0.15),
    ))

    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Tier 2: Multi-axis factorial combinations (~165 configs)
# ---------------------------------------------------------------------------


def build_tier2(n: int) -> list[Trial]:
    """Tier 2: data-driven from Tier 1 results — focus on winner region.

    Tier 1 key findings:
    - top_n HURTS (all configs rank bottom-half) → REMOVED entirely
    - rb=10-21 is the sweet spot (rb=10 is #1 overall)
    - single linkage competitive with ward; single+shrink at 0.777
    - max_weight relaxation (0.10-0.25) very promising at rb=10 (0.774)
    - turnover_threshold has zero impact → REMOVED
    - killswitch hurts more than helps → de-emphasized
    - tilt has minimal marginal effect → de-emphasized
    - delta=0.005 slightly best → fine-grain

    Fixed: retrain_every=126, lookback_days=756, momentum=5, top_n=None.
    Focus: rb fine-grain x HRP structure x max_weight interaction surface.
    """
    trials: list[Trial] = []

    # === A. Rebalance fine-grain in sweet spot (14) ===
    # Tier 1 showed rb=10 (#1), rb=15 (#12), rb=21 (#17), rb=13 (baseline)
    for rb in [7, 8, 9, 10, 11, 12, 14, 15, 17, 19, 21, 25, 30, 42]:
        trials.append(_t(n, f"t2a_rb{rb:03d}", rebalance_every=rb))

    # === B. HRP structure factorial at winner rb values (24) ===
    # Tier 1: single+shrink=0.777, single+noshrink=0.745, ward+noshrink=0.743
    for link in ["single", "ward"]:
        for shrk in [False, True]:
            for rb in [10, 13, 15]:
                for corr in ["pearson", "spearman"]:
                    lk = "W" if link == "ward" else "S"
                    sk = "Y" if shrk else "N"
                    trials.append(_t(n,
                        f"t2b_{lk}{sk}_{corr[0]}_r{rb:02d}",
                        rebalance_every=rb,
                        hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk,
                                        correlation_method=corr),
                    ))

    # === C. Max weight exploration (20) ===
    # Tier 1: x_rebal010_maxw010/025=0.774 — relaxing max_weight is very promising
    for mw in [0.06, 0.08, 0.10, 0.15, 0.25]:
        for rb in [8, 10, 13, 15]:
            trials.append(_t(n,
                f"t2c_mw{int(mw * 100):02d}_r{rb:02d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, max_weight=mw),
            ))

    # === D. 3-WAY: HRP structure x Max weight x Rebalance (24) ===
    # Combine the two strongest signals from Tier 1
    for link, shrk, hname in [
        ("single", True, "SY"), ("single", False, "SN"),
        ("ward", True, "WY"), ("ward", False, "WN"),
    ]:
        for mw in [0.10, 0.25]:
            for rb in [10, 13, 15]:
                trials.append(_t(n,
                    f"t2d_{hname}_mw{int(mw * 100):02d}_r{rb:02d}",
                    rebalance_every=rb,
                    hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk, max_weight=mw),
                ))

    # === E. HRP structure at extended rb range (16) ===
    # Test winning HRP combos at rb values outside sweet spot
    for link, shrk, hname in [("single", True, "SY"), ("single", False, "SN")]:
        for rb in [5, 7, 21, 25, 30, 42, 63, 126]:
            trials.append(_t(n,
                f"t2e_{hname}_r{rb:03d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk),
            ))

    # === F. Delta fine-grain (8) ===
    # Tier 1: delta=0.005 best (Sharpe=0.687 vs baseline 0.662)
    for d in [0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030]:
        trials.append(_t(n,
            f"t2f_delta{int(d * 1000):03d}",
            rebalance_every=13, min_rebalance_delta=d,
        ))

    # === G. Delta x Rebalance at best combos (12) ===
    for d in [0.005, 0.010]:
        for rb in [8, 10, 12, 13, 15, 21]:
            trials.append(_t(n,
                f"t2g_d{int(d * 1000):03d}_r{rb:02d}",
                rebalance_every=rb, min_rebalance_delta=d,
            ))

    # === H. 4-WAY: HRP x Max weight x Rebalance x Delta (16) ===
    for link, shrk, hname in [("single", True, "SY"), ("ward", True, "WY")]:
        for mw in [0.10, 0.25]:
            for rb in [10, 15]:
                for d in [0.005, 0.010]:
                    trials.append(_t(n,
                        f"t2h_{hname}_mw{int(mw * 100):02d}_r{rb:02d}_d{int(d * 1000):03d}",
                        rebalance_every=rb, min_rebalance_delta=d,
                        hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk, max_weight=mw),
                    ))

    # === I. Spearman x Max weight at winner rb (8) ===
    # Spearman was ~neutral in Tier 1 but worth testing with relaxed max_weight
    for mw in [0.10, 0.25]:
        for rb in [10, 13, 15, 21]:
            trials.append(_t(n,
                f"t2i_spear_mw{int(mw * 100):02d}_r{rb:02d}",
                rebalance_every=rb,
                hrp_config=_hrp(n, correlation_method="spearman", max_weight=mw),
            ))

    # === J. Risk overlay confirmation on best configs (12) ===
    # KS generally hurts, but test mild levels on top configs
    for rb in [10, 13, 15]:
        for dd in [0.15, 0.20]:
            trials.append(_t(n,
                f"t2j_r{rb:02d}_ks{int(dd * 100):03d}",
                rebalance_every=rb,
                killswitch=KillswitchConfig(max_drawdown_pct=-dd),
            ))
    # Vol targeting on best configs
    for rb in [10, 13, 15]:
        for tv in [0.10, 0.15]:
            trials.append(_t(n,
                f"t2j_r{rb:02d}_vol{int(tv * 100):03d}",
                rebalance_every=rb, target_vol=tv,
            ))

    # === K. Max weight x Tilt x Rebalance (12) ===
    # Tilt has minimal impact but test with relaxed max_weight
    for mw in [0.10, 0.25]:
        for tc in [0.50, 1.0]:
            for rb in [10, 13, 15]:
                trials.append(_t(n,
                    f"t2k_mw{int(mw * 100):02d}_t{int(tc * 100):03d}_r{rb:02d}",
                    rebalance_every=rb,
                    hrp_config=_hrp(n, max_weight=mw, confidence_tilt_cap=tc),
                ))

    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Tier 3: Adaptive champion refinement (~170 configs)
# ---------------------------------------------------------------------------


def _load_all_results(output_dir: Path = _OUTPUT_DIR) -> list[dict[str, Any]]:
    """Load all previous tier results as a flat list sorted by DSR."""
    results: list[dict[str, Any]] = []
    for tier_num in [1, 2, 3]:
        path = output_dir / f"validation_tier{tier_num}_results.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Corrupt file {} — skipping", path)
            continue
        for name, r in data.get("configs", {}).items():
            results.append({"name": name, **r})
    results.sort(key=lambda x: x.get("deflated_sharpe", 0), reverse=True)
    return results


def _load_existing_fps(output_dir: Path = _OUTPUT_DIR) -> set[str]:
    """Load parameter fingerprints from all completed validations."""
    fps: set[str] = set()
    for r in _load_all_results(output_dir):
        params = r.get("params")
        if params:
            fps.add(_param_fp(params))
    return fps


def build_tier3(n: int) -> list[Trial]:
    """Tier 3: targeted factorial on the dimensions that matter.

    Key findings from Tier 1+2 driving this design:
    - #1 overall: rb=15 + vol_target=0.10 (Sharpe=0.806)
    - rb=15-25 sweet spot (shifted higher than Tier 1)
    - spearman competitive at rb=15+ (T2: 0.772 #12)
    - max_weight ≥0.06 saturates (default 0.038 too tight)
    - HRP: single+shrink competitive (T2: 0.778 #4)
    - top_n, killswitch, delta, turnover_threshold: zero/negative impact → REMOVED

    Sections:
    A. rb × vol_target fine-grain (~57) — the #1 interaction
    B. rb × vol × spearman (~12) — strong T2 signal
    C. HRP structure at winner rb × vol (~24) — single+shrink, ward+noshrink
    D. Max weight at winner combos (~16) — relax from 0.038
    E. 3-way HRP × mw × rb at vol=0.10 (~18) — combine top findings
    F. Cost stress on T1+T2 champions (~25) — robustness
    G. HRP + spearman combos on champions (~25) — T2 signal
    """
    prev = _load_all_results()
    if not prev:
        logger.warning("No previous results found; using fallback Tier 3 grid")
        return _build_tier3_fallback(n)

    trials: list[Trial] = []

    # === A. RB × VOL_TARGET FINE-GRAIN (~57) ===
    # The #1 interaction from T1+T2: rb=15 + vol=0.10 = Sharpe 0.806
    for rb in [12, 13, 14, 15, 16, 17, 19, 21, 25]:
        for tv in [0.08, 0.09, 0.10, 0.11, 0.12]:
            trials.append(_t(n,
                f"t3a_rb{rb:03d}_vol{int(tv * 100):03d}",
                rebalance_every=rb, target_vol=tv,
            ))
    # Extend to wider rb range at vol=0.10 (the sweet spot)
    for rb in [7, 8, 9, 10, 11, 30, 42]:
        trials.append(_t(n,
            f"t3a_rb{rb:03d}_vol010",
            rebalance_every=rb, target_vol=0.10,
        ))

    # === B. RB × VOL × SPEARMAN (~12) ===
    # T2 showed spearman competitive at rb=15+ — test with vol_target
    for rb in [15, 17, 21, 25]:
        for tv in [0.09, 0.10, 0.11]:
            trials.append(_t(n,
                f"t3b_rb{rb:03d}_vol{int(tv * 100):03d}_spear",
                rebalance_every=rb, target_vol=tv,
                hrp_config=_hrp(n, correlation_method="spearman"),
            ))

    # === C. HRP STRUCTURE AT WINNER RB × VOL COMBOS (~24) ===
    # Test non-baseline HRP at the top rb × vol pairs
    for rb in [12, 15, 17, 21]:
        for tv in [0.10, None]:
            tv_str = f"vol{int(tv * 100)}" if tv else "novol"
            for link, shrk, hname in [
                ("single", True, "SY"),
                ("single", False, "SN"),
                ("ward", False, "WN"),
            ]:
                trials.append(_t(n,
                    f"t3c_{hname}_rb{rb:03d}_{tv_str}",
                    rebalance_every=rb, target_vol=tv,
                    hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk),
                ))

    # === D. MAX WEIGHT AT WINNER COMBOS (~16) ===
    # Default dyn_maxw ≈ 0.038 is too tight; relax to 0.06/0.10
    for rb in [12, 15, 17, 21]:
        for tv in [0.10, None]:
            tv_str = f"vol{int(tv * 100)}" if tv else "novol"
            for mw in [0.06, 0.10]:
                trials.append(_t(n,
                    f"t3d_mw{int(mw * 100):02d}_rb{rb:03d}_{tv_str}",
                    rebalance_every=rb, target_vol=tv,
                    hrp_config=_hrp(n, max_weight=mw),
                ))

    # === E. 3-WAY: HRP × MAX_WEIGHT × RB AT VOL=0.10 (~18) ===
    # Combine the strongest signals: relaxed mw + HRP structure + vol targeting
    for link, shrk, hname in [
        ("single", True, "SY"),
        ("ward", False, "WN"),
        ("single", False, "SN"),
    ]:
        for mw in [0.06, 0.10]:
            for rb in [15, 21, 25]:
                trials.append(_t(n,
                    f"t3e_{hname}_mw{int(mw * 100):02d}_rb{rb:03d}_vol010",
                    rebalance_every=rb, target_vol=0.10,
                    hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk, max_weight=mw),
                ))

    # === F. COST STRESS ON T1+T2 CHAMPIONS (~25) ===
    # Sanitize champion params to PatchTST-safe values before cost stress
    _SAFE_OVERRIDES: dict[str, Any] = {
        "retrain_every": 126, "lookback_days": 756,
        "momentum_lookback": 5, "top_n": None,
    }
    seen_fps: set[str] = set()
    safe_champs: list[dict[str, Any]] = []
    for r in prev:
        p = {**r["params"], **_SAFE_OVERRIDES}
        fp = _param_fp(p)
        if fp not in seen_fps:
            seen_fps.add(fp)
            safe_champs.append(p)

    cost_regimes = [
        ("low", TransactionCosts(slippage_bps=2.0, commission_bps=5.0)),
        ("2x", TransactionCosts(slippage_bps=10.0, commission_bps=20.0)),
        ("3x", TransactionCosts(slippage_bps=15.0, commission_bps=30.0)),
        ("impact5", TransactionCosts(slippage_bps=5.0, commission_bps=10.0, market_impact_bps=5.0)),
        ("zero", TransactionCosts(slippage_bps=0.0, commission_bps=0.0)),
    ]
    for rank, p in enumerate(safe_champs[:5]):
        tag = f"t3f{rank}"
        for cname, costs in cost_regimes:
            new_p = {
                **p,
                "slippage_bps": costs.slippage_bps,
                "commission_bps": costs.commission_bps,
                "market_impact_bps": costs.market_impact_bps,
            }
            trials.append(_trial_from_params(f"{tag}_{cname}", new_p, n))

    # === G. HRP + SPEARMAN COMBOS ON CHAMPIONS (~25) ===
    hrp_combos = [
        ("ward_spear", {"linkage_method": "ward", "correlation_method": "spearman"}),
        ("single_shrink_spear", {"linkage_method": "single", "shrinkage": True,
                                 "correlation_method": "spearman"}),
        ("ward_shrink_spear", {"linkage_method": "ward", "shrinkage": True,
                               "correlation_method": "spearman"}),
        ("single_shrink", {"linkage_method": "single", "shrinkage": True}),
        ("single_noshrink_spear", {"linkage_method": "single", "shrinkage": False,
                                   "correlation_method": "spearman"}),
    ]
    for rank, p in enumerate(safe_champs[:5]):
        tag = f"t3g{rank}"
        for hname, overrides in hrp_combos:
            new_p = {**p, **overrides}
            trials.append(_trial_from_params(f"{tag}_{hname}", new_p, n))

    return _dedup(trials, n)


def _build_tier3_fallback(n: int) -> list[Trial]:
    """Fallback Tier 3 when no previous results exist.

    Explores rb x vol_target interaction + HRP structure + cost stress.
    """
    trials: list[Trial] = []
    # rb x vol_target grid (strongest interaction from T1+T2)
    for rb in [10, 13, 15, 17, 21, 25]:
        for tv in [None, 0.08, 0.10, 0.12]:
            tv_str = f"vol{int(tv * 100)}" if tv else "novol"
            trials.append(_t(n, f"t3f_rb{rb}_{tv_str}",
                             rebalance_every=rb, target_vol=tv))
    # HRP structure at best rb values
    for rb in [15, 21, 25]:
        for link, shrk in [("single", True), ("single", False), ("ward", False)]:
            lk = "s" if link == "single" else "w"
            sk = "y" if shrk else "n"
            trials.append(_t(n, f"t3f_rb{rb}_hrp_{lk}{sk}",
                             rebalance_every=rb,
                             hrp_config=_hrp(n, linkage_method=link, shrinkage=shrk)))
    # Spearman at best rb values
    for rb in [15, 21, 25]:
        trials.append(_t(n, f"t3f_rb{rb}_spear",
                         rebalance_every=rb,
                         hrp_config=_hrp(n, correlation_method="spearman")))
    # Max weight relaxation at best rb
    for rb in [15, 21]:
        for mw in [0.06, 0.10, 0.15]:
            trials.append(_t(n, f"t3f_rb{rb}_mw{int(mw * 100):02d}",
                             rebalance_every=rb,
                             hrp_config=_hrp(n, max_weight=mw)))
    # Cost stress on winner profile
    for slip in [2, 10, 15, 25]:
        trials.append(_t(n,
            f"t3f_winner_slip{slip:03d}",
            rebalance_every=15, target_vol=0.10,
            costs=TransactionCosts(slippage_bps=float(slip), commission_bps=10.0),
        ))
    return _dedup(trials, n)


# ---------------------------------------------------------------------------
# Runner with resume, incremental saves, and ETA
# ---------------------------------------------------------------------------


def run_tier(
    tier_num: int,
    trials: list[Trial],
    validator: CPCVParameterValidator,
    n_tickers: int,
    output_dir: Path = _OUTPUT_DIR,
) -> dict[str, dict[str, Any]]:
    """Run a tier of trials with resume support, periodic saves, and ETA."""
    results_path = output_dir / f"validation_tier{tier_num}_results.json"

    # Load existing results for resume
    existing: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        try:
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("configs", {})
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt results file {} — starting fresh: {}", results_path, e)
            existing = {}

    # Load cross-tier fingerprints for dedup
    all_fps = _load_existing_fps(output_dir)

    # Filter pending trials (resume + cross-tier dedup)
    pending: list[Trial] = []
    for t in trials:
        if t.name in existing:
            continue
        fp = _param_fp(_trial_params(t, n_tickers))
        if fp in all_fps:
            logger.debug("Cross-tier dedup: skip {}", t.name)
            continue
        pending.append(t)

    # Count total trials for DSR correction (cumulative across all tiers)
    # prev_results = completed configs from OTHER tiers (exclude current tier's results)
    all_prev = _load_all_results(output_dir)
    prev_other_tier = sum(
        1 for r in all_prev if r["name"] not in existing
    )
    # Total = configs from other tiers + all configs scheduled for this tier
    # (both completed + pending, since they're all from the same testing family)
    n_total_trials = prev_other_tier + len(existing) + len(pending)
    # Tier 3 inflation: adaptive configs are statistically dependent on Tier 1+2
    # champions, so the effective number of independent trials is higher.
    # We inflate by 2x for Tier 3 to partially compensate for data snooping.
    if tier_num == 3 and prev_other_tier > 0:
        n_total_trials = n_total_trials + prev_other_tier  # count prior trials twice

    total = len(existing) + len(pending)
    logger.info(
        "=== TIER {} === {} total | {} pending | {} completed | n_trials_dsr={}",
        tier_num, total, len(pending), len(existing), n_total_trials,
    )

    if not pending:
        logger.info("All configs already computed. Nothing to do.")
        _save_all_outputs(tier_num, existing, total, output_dir)
        return existing

    # Run pending trials with ETA
    times: deque[float] = deque(maxlen=20)
    tier_start = time.monotonic()

    for i, trial in enumerate(pending):
        t0 = time.monotonic()

        try:
            result = validator.validate(
                config=trial.wf_config,
                model_factory=trial.factory,
                n_trials=n_total_trials,
            )
        except Exception as e:
            logger.error("[{}/{}] {} FAILED: {}", len(existing) + 1, total, trial.name, e)
            continue

        elapsed = time.monotonic() - t0
        times.append(elapsed)

        # Store result
        existing[trial.name] = {
            "params": _trial_params(trial, n_tickers),
            "mean_sharpe": result.mean_sharpe,
            "std_sharpe": result.std_sharpe,
            "pct_positive": result.pct_positive,
            "per_path_sharpe": result.per_path_sharpe,
            "deflated_sharpe": result.deflated_sharpe,
            "p_value": result.p_value,
            "accepted": result.accepted,
            "elapsed_seconds": round(elapsed, 1),
        }

        # Periodic save
        done = len(existing)
        if done % _SAVE_EVERY == 0 or i == len(pending) - 1:
            _save_tier_json(tier_num, existing, total, output_dir)

        # ETA
        avg_t = sum(times) / len(times)
        remaining = len(pending) - (i + 1)
        eta = str(timedelta(seconds=int(remaining * avg_t)))
        wall = str(timedelta(seconds=int(time.monotonic() - tier_start)))

        acc = "OK" if result.accepted else "--"
        logger.info(
            "[{}/{}] {} | Sharpe={:.3f} | DSR={:.3f} | {} | {:.0f}s | ETA {} | wall {}",
            done, total, trial.name,
            result.mean_sharpe, result.p_value,
            acc, elapsed, eta, wall,
        )

    # Final save
    _save_all_outputs(tier_num, existing, total, output_dir)
    return existing


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _save_tier_json(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    total: int,
    output_dir: Path,
) -> None:
    """Incremental JSON save (atomic write via temp file + rename)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_results.json"
    tmp_path = path.with_suffix(".tmp")
    data = {
        "tier": tier_num,
        "generated_at": datetime.now().isoformat(),
        "total_trials": total,
        "completed": len(configs),
        "configs": configs,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _save_all_outputs(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    total: int,
    output_dir: Path,
) -> None:
    """Save JSON, Markdown summary, and per-path Parquet."""
    _save_tier_json(tier_num, configs, total, output_dir)
    _save_summary_md(tier_num, configs, output_dir)
    _save_paths_parquet(tier_num, configs, output_dir)


def _save_summary_md(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate ranked Markdown summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_summary.md"

    ranked = sorted(configs.items(), key=lambda x: x[1].get("deflated_sharpe", 0), reverse=True)

    n_accepted = sum(1 for _, r in ranked if r.get("accepted"))
    best_name = ranked[0][0] if ranked else "N/A"
    best_sharpe = ranked[0][1].get("mean_sharpe", 0) if ranked else 0

    lines = [
        f"# Tier {tier_num} Validation Results", "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"| {len(configs)} configs | {n_accepted} accepted", "",
        f"> Best: **{best_name}** (Sharpe={best_sharpe:.3f})", "",
        "## Rankings (sorted by DSR p-value)", "",
        "| # | Config | Sharpe | Std | %+ | DSR | OK |",
        "|---|--------|--------|-----|-----|-----|-----|",
    ]
    for i, (name, r) in enumerate(ranked, 1):
        acc = "YES" if r.get("accepted") else "no"
        lines.append(
            f"| {i} | {name} | {r.get('mean_sharpe', 0):.3f} "
            f"| {r.get('std_sharpe', 0):.3f} "
            f"| {r.get('pct_positive', 0):.0%} "
            f"| {r.get('deflated_sharpe', 0):.3f} | {acc} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Summary saved to {}", path)


def _save_paths_parquet(
    tier_num: int,
    configs: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save per-path Sharpe values for distribution analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"validation_tier{tier_num}_paths.parquet"
    rows: list[dict[str, Any]] = []
    for name, r in configs.items():
        for pid, sharpe in enumerate(r.get("per_path_sharpe", [])):
            rows.append({
                "config": name,
                "path_id": pid,
                "sharpe": sharpe,
                "accepted": r.get("accepted", False),
            })
    if rows:
        pl.DataFrame(rows).write_parquet(path)
        logger.info("Per-path data: {} rows -> {}", len(rows), path)


# ---------------------------------------------------------------------------
# Estimate mode
# ---------------------------------------------------------------------------


def _run_estimate(
    validator: CPCVParameterValidator,
    n: int,
) -> None:
    """Time 1 config and print ETA for all tiers."""
    trial = _t(n=n, name="estimate_probe")
    logger.info("Running 1 config for time estimation...")

    t0 = time.monotonic()
    validator.validate(config=trial.wf_config, model_factory=trial.factory, n_trials=1)
    elapsed = time.monotonic() - t0

    logger.info("Single config: {:.1f}s", elapsed)

    t1 = len(build_tier1(n))
    t2 = len(build_tier2(n))
    # Tier 3 count depends on results, estimate 170
    t3_est = 170

    for label, count in [("Tier 1", t1), ("Tier 2", t2), ("Tier 3 (est.)", t3_est)]:
        hours = count * elapsed / 3600
        logger.info("  {} ({} configs): {:.1f}h", label, count, hours)

    total = t1 + t2 + t3_est
    logger.info("  Total ({} configs): {:.1f}h ({:.1f} days at 18h/day)",
                total, total * elapsed / 3600, total * elapsed / 3600 / 18)


# ---------------------------------------------------------------------------
# Public API (backwards-compatible)
# ---------------------------------------------------------------------------


def run_improvement_validation(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    benchmark_ticker: str = "SPY",
    output_dir: str = "data/outputs",
    subset: str = "all",
) -> dict[str, ValidationResult]:
    """Legacy API: runs Tier 1 only (backwards compatible).

    For the full 3-tier grid, use the CLI with ``--tier``.
    """
    out = Path(output_dir)
    n = len(tickers)
    validator = CPCVParameterValidator(
        ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark_ticker,
    )
    trials = build_tier1(n)
    existing_fps = _load_existing_fps(out)
    trials = _dedup(trials, n, existing_fps)
    run_tier(1, trials, validator, n, out)
    # Return empty dict for API compat — real results are in JSON
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """Entry point for ``python -m src.backtest.run_validation``."""
    global _USE_PATCHTST  # noqa: PLW0603
    if "--patchtst" in sys.argv:
        _USE_PATCHTST = True
        logger.info("Using PatchTST model factory (cached)")

    tier_str = "1"
    if "--tier" in sys.argv:
        idx = sys.argv.index("--tier")
        if idx + 1 < len(sys.argv):
            tier_str = sys.argv[idx + 1]

    # Legacy --subset support
    if "--subset" in sys.argv:
        idx = sys.argv.index("--subset")
        if idx + 1 < len(sys.argv):
            tier_str = "1"  # legacy mode runs tier 1

    # Parse holdout args
    holdout_mode = "--holdout" in sys.argv
    holdout_years = _DEFAULT_HOLDOUT_YEARS
    for arg in sys.argv:
        if arg.startswith("--holdout-years="):
            holdout_years = int(arg.split("=", 1)[1])

    # Load data
    from src.backtest.run_benchmark import _filter_oos_period, _load_ohlcv_from_postgres
    from src.config import load_benchmark, load_tickers

    tickers = load_tickers()
    benchmark = load_benchmark()
    n = len(tickers)

    # Dry run: print grid size
    if "--dry-run" in sys.argv:
        for t_num, builder in [(1, build_tier1), (2, build_tier2)]:
            trials = builder(n)
            logger.info("Tier {}: {} configs", t_num, len(trials))
            for t in trials[:5]:
                logger.info("  {}", t.name)
            if len(trials) > 5:
                logger.info("  ... and {} more", len(trials) - 5)
        logger.info("Tier 3: ~170 (adaptive, depends on Tier 1+2 results)")
        return

    logger.info("Loading OHLCV data...")
    ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)
    ohlcv = _filter_oos_period(ohlcv, n_years=10)

    # Holdout-only mode: skip grid search, validate champion directly
    if holdout_mode and "--tier" not in sys.argv:
        logger.info("=== HOLDOUT VALIDATION (champion from prior tiers) ===")
        result = run_holdout_validation(
            ohlcv=ohlcv,
            tickers=tickers,
            benchmark_ticker=benchmark,
            holdout_years=holdout_years,
        )
        verdict = "ACCEPTED" if result.holdout_accepted else "REJECTED"
        logger.info(
            "Holdout verdict: {} | Sharpe={:.3f} | DSR={:.3f}",
            verdict, result.holdout_sharpe, result.holdout_dsr,
        )
        return

    # If holdout mode with tiers: filter grid search data to exclude holdout
    grid_ohlcv = ohlcv
    if holdout_mode:
        logger.info("Holdout mode: reserving last {} years for validation", holdout_years)
        grid_ohlcv = filter_before_holdout(ohlcv, holdout_years)

    validator = CPCVParameterValidator(
        ohlcv=grid_ohlcv, tickers=tickers, benchmark_ticker=benchmark,
    )

    # Estimate mode
    if "--estimate" in sys.argv:
        _run_estimate(validator, n)
        return

    # Determine tiers to run
    if tier_str == "all":
        tiers_to_run = [1, 2, 3]
    else:
        tiers_to_run = [int(tier_str)]

    for t_num in tiers_to_run:
        if t_num == 1:
            trials = build_tier1(n)
        elif t_num == 2:
            trials = build_tier2(n)
        elif t_num == 3:
            trials = build_tier3(n)
        else:
            logger.error("Unknown tier: {}", t_num)
            continue

        # Cross-tier dedup
        existing_fps = _load_existing_fps(_OUTPUT_DIR)
        trials = _dedup(trials, n, existing_fps)

        logger.info("Tier {}: {} configs after dedup", t_num, len(trials))
        run_tier(t_num, trials, validator, n)

    # Final cross-tier summary
    all_results = _load_all_results()
    if all_results:
        logger.info("=== FINAL RANKINGS (all tiers) ===")
        n_accepted = sum(1 for r in all_results if r.get("accepted"))
        logger.info("{}/{} configs accepted across all tiers", n_accepted, len(all_results))
        for i, r in enumerate(all_results[:10], 1):
            acc = "OK" if r.get("accepted") else "--"
            logger.info(
                "  #{}: {} | Sharpe={:.3f} | DSR={:.3f} | {}",
                i, r["name"], r.get("mean_sharpe", 0), r.get("deflated_sharpe", 0), acc,
            )

    # Run holdout validation after grid search
    if holdout_mode:
        logger.info("=== HOLDOUT VALIDATION ===")
        result = run_holdout_validation(
            ohlcv=ohlcv,
            tickers=tickers,
            benchmark_ticker=benchmark,
            holdout_years=holdout_years,
        )
        verdict = "ACCEPTED" if result.holdout_accepted else "REJECTED"
        logger.info(
            "Holdout verdict: {} | Sharpe={:.3f} | DSR={:.3f}",
            verdict, result.holdout_sharpe, result.holdout_dsr,
        )


if __name__ == "__main__":
    _cli_main()
