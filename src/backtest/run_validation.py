"""CPCV-OOS validation of walk-forward strategy improvements.

Builds a grid of configurations (baseline + improvements) and validates
each via CPCV-OOS, producing ranked results with Deflated Sharpe Ratio
correction for multiple testing.

Outputs::

    data/outputs/
        validation_results.json      — full metrics per config
        validation_summary.md        — readable ranking table
        validation_per_path.parquet  — per-path Sharpe for analysis

Usage::

    # From Python
    from src.backtest.run_validation import run_improvement_validation
    results = run_improvement_validation(ohlcv=df, tickers=tickers)

    # From CLI
    python -m src.backtest.run_validation               # all configs
    python -m src.backtest.run_validation --subset tier3 # tier 3 only
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import CPCVParameterValidator, ValidationResult
from src.backtest.walk_forward import (
    KillswitchConfig,
    ModelFactory,
    NaiveModelFactory,
    WalkForwardConfig,
)
from src.portfolio.hrp import HRPConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("data/outputs")

# Shared base costs (consistent with run_benchmark.py)
_BASE_COSTS = TransactionCosts(slippage_bps=5.0, commission_bps=10.0)


# ---------------------------------------------------------------------------
# Config grid builder
# ---------------------------------------------------------------------------


def _base_config(**overrides: Any) -> WalkForwardConfig:
    """Create a WalkForwardConfig with baseline defaults + overrides."""
    defaults: dict[str, Any] = {
        "rebalance_every": 5,
        "retrain_every": 126,
        "lookback_days": 504,
        "initial_capital": 1_000_000.0,
        "costs": _BASE_COSTS,
        "min_rebalance_delta": 0.02,
        "trading_days_per_year": 252,
        "rf": 0.05,
    }
    defaults.update(overrides)
    return WalkForwardConfig(**defaults)


def build_tier1_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    """Build Tier 1 configurations (structural + risk preference)."""
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}

    configs["baseline"] = _base_config()
    configs["vol_target_08"] = _base_config(target_vol=0.08)
    configs["vol_target_10"] = _base_config(target_vol=0.10)
    configs["vol_target_12"] = _base_config(target_vol=0.12)

    configs["ward_linkage"] = _base_config(
        hrp_config=HRPConfig(linkage_method="ward", max_weight=dynamic_max_weight)
    )
    configs["shrinkage"] = _base_config(
        hrp_config=HRPConfig(shrinkage=True, max_weight=dynamic_max_weight)
    )
    configs["ward_shrinkage"] = _base_config(
        hrp_config=HRPConfig(linkage_method="ward", shrinkage=True, max_weight=dynamic_max_weight)
    )

    return configs


def build_tier2_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    """Build Tier 2 configurations (parameter tuning)."""
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}

    configs["rebalance_10d"] = _base_config(rebalance_every=10)
    configs["rebalance_21d"] = _base_config(rebalance_every=21)

    configs["no_tilt"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.0, max_weight=dynamic_max_weight)
    )
    configs["tilt_010"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.10, max_weight=dynamic_max_weight)
    )
    configs["tilt_030"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.30, max_weight=dynamic_max_weight)
    )

    configs["lookback_252"] = _base_config(lookback_days=252)
    configs["lookback_756"] = _base_config(lookback_days=756)

    configs["killswitch_15"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.15))
    configs["killswitch_20"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.20))
    configs["killswitch_25"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.25))

    return configs


def build_tier3_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    """Build Tier 3 configurations (ultrafast, aggressive tilts, higher vol)."""
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}

    # Frequências ultracurtas e filtros de microestrutura
    configs["rebalance_1d"] = _base_config(rebalance_every=1)
    configs["rebalance_2d"] = _base_config(rebalance_every=2)
    configs["rebalance_3d"] = _base_config(rebalance_every=3)
    configs["rebalance_5d_delta_03"] = _base_config(min_rebalance_delta=0.03)
    configs["rebalance_5d_delta_05"] = _base_config(min_rebalance_delta=0.05)

    # Janelas de Covariância Curtas
    configs["lookback_63"] = _base_config(lookback_days=63)
    configs["lookback_126"] = _base_config(lookback_days=126)

    # Tilts Agressivos
    configs["tilt_040"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.40, max_weight=dynamic_max_weight)
    )
    configs["tilt_050"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.50, max_weight=dynamic_max_weight)
    )
    configs["tilt_075"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=0.75, max_weight=dynamic_max_weight)
    )
    configs["tilt_100"] = _base_config(
        hrp_config=HRPConfig(confidence_tilt_cap=1.0, max_weight=dynamic_max_weight)
    )

    # Escala de Volatilidade
    configs["vol_target_15"] = _base_config(target_vol=0.15)
    configs["vol_target_20"] = _base_config(target_vol=0.20)

    return configs


def build_momentum_factories() -> dict[str, NaiveModelFactory]:
    """Build NaiveModelFactory variants with different lookback periods."""
    return {
        "momentum_1d": NaiveModelFactory(lookback=1),
        "momentum_2d": NaiveModelFactory(lookback=2),
        "momentum_3d": NaiveModelFactory(lookback=3),
        "momentum_4d": NaiveModelFactory(lookback=4),
        "momentum_5d": NaiveModelFactory(lookback=5),
        "momentum_21d": NaiveModelFactory(lookback=21),
        "momentum_63d": NaiveModelFactory(lookback=63),
        "momentum_126d": NaiveModelFactory(lookback=126),
    }


def build_all_configs(
    subset: str = "all",
    n_tickers: int = 52,
) -> dict[str, WalkForwardConfig]:
    """Build the full grid of configurations to validate."""
    configs: dict[str, WalkForwardConfig] = {}

    if subset in ("all", "tier1"):
        configs.update(build_tier1_configs(n_tickers))

    if subset in ("all", "tier2"):
        configs.update(build_tier2_configs(n_tickers))
        
    if subset in ("all", "tier3"):
        configs.update(build_tier3_configs(n_tickers))

    # Always ensure baseline is present if we are testing a specific tier
    # so we have a reference point, unless it's explicitly omitted by the user logic.
    if "baseline" not in configs:
        configs["baseline"] = _base_config()

    return configs


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _save_results_json(results: list[tuple[str, ValidationResult]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_results.json"
    data: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "n_configs": len(results),
        "configs": {name: result.to_dict() for name, result in results},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Validation results saved to {}", path)
    return path


def _save_summary_md(
    results: list[tuple[str, ValidationResult]],
    momentum_results: list[tuple[str, ValidationResult]] | None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_summary.md"
    lines = [
        "# CPCV-OOS Validation Summary", "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
        "## Configuration Grid", "",
        "| # | Config | Mean Sharpe | Std | Pct+ | DSR p-value | Accepted |",
        "|---|--------|-------------|-----|------|-------------|----------|",
    ]
    for i, (name, r) in enumerate(results, 1):
        accepted_str = "YES" if r.accepted else "no"
        lines.append(f"| {i} | {name} | {r.mean_sharpe:.3f} | {r.std_sharpe:.3f} | {r.pct_positive:.1%} | {r.p_value:.3f} | {accepted_str} |")

    if momentum_results:
        lines.extend([
            "", "## Momentum Lookback Variants", "",
            "| # | Factory | Mean Sharpe | Std | Pct+ | DSR p-value | Accepted |",
            "|---|---------|-------------|-----|------|-------------|----------|",
        ])
        for i, (name, r) in enumerate(momentum_results, 1):
            accepted_str = "YES" if r.accepted else "no"
            lines.append(f"| {i} | {name} | {r.mean_sharpe:.3f} | {r.std_sharpe:.3f} | {r.pct_positive:.1%} | {r.p_value:.3f} | {accepted_str} |")

    lines.extend([
        "", "## Legend", "",
        "- **Pct+**: Fraction of CPCV paths with Sharpe > 0",
        "- **DSR p-value**: Deflated Sharpe Ratio (>0.95 = significant)",
        "- **Accepted**: Pct+ >= 66.7% AND DSR p-value > 0.95", ""
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Validation summary saved to {}", path)
    return path


def _save_per_path_parquet(
    results: list[tuple[str, ValidationResult]],
    momentum_results: list[tuple[str, ValidationResult]] | None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_per_path.parquet"
    rows = []
    all_results = list(results)
    if momentum_results:
        all_results.extend(momentum_results)

    for name, r in all_results:
        for path_idx, sharpe in enumerate(r.per_path_sharpe):
            rows.append({"config": name, "path_id": path_idx, "sharpe": sharpe, "accepted": r.accepted})

    if rows:
        df = pl.DataFrame(rows)
        df.write_parquet(path)
        logger.info("Per-path data saved to {} ({} rows)", path, df.height)
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_improvement_validation(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    benchmark_ticker: str = "SPY",
    output_dir: str = "data/outputs",
    subset: str = "all",
) -> dict[str, ValidationResult]:
    out_path = Path(output_dir)

    logger.info("Step 1/6: Building configuration grid (subset={})", subset)
    configs = build_all_configs(subset, n_tickers=len(tickers))
    logger.info("  {} configurations to validate", len(configs))

    logger.info("Step 2/6: Building momentum factory variants")
    factories = build_momentum_factories()
    logger.info("  {} momentum variants", len(factories))

    logger.info("Step 3/6: Creating CPCV-OOS validator")
    validator = CPCVParameterValidator(ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark_ticker)
    n_total_trials = len(configs) + len(factories)

    logger.info("Step 4/6: Running config grid search ({} configs)", len(configs))
    config_results = validator.grid_search(configs=configs, model_factory=NaiveModelFactory(), n_trials=n_total_trials)

    logger.info("Step 5/6: Running momentum factory grid search ({} variants)", len(factories))
    baseline_config = _base_config()
    momentum_results = []
    for i, (name, factory) in enumerate(factories.items()):
        logger.info("  Momentum [{}/{}]: {}", i + 1, len(factories), name)
        result = validator.validate(config=baseline_config, model_factory=factory, n_trials=n_total_trials)
        momentum_results.append((name, result))

    momentum_results.sort(key=lambda x: x[1].deflated_sharpe, reverse=True)

    logger.info("Step 6/6: Saving outputs")
    _save_results_json(config_results + momentum_results, out_path)
    _save_summary_md(config_results, momentum_results, out_path)
    _save_per_path_parquet(config_results, momentum_results, out_path)

    all_results = {name: result for name, result in config_results + momentum_results}
    n_accepted = sum(1 for r in all_results.values() if r.accepted)
    logger.info("Validation complete: {}/{} configs accepted", n_accepted, len(all_results))

    ranked = sorted(all_results.items(), key=lambda x: x[1].deflated_sharpe, reverse=True)
    for i, (name, r) in enumerate(ranked[:5], 1):
        logger.info(
            "  #{}: {} | Sharpe={:.3f} | Pct+={:.0%} | DSR={:.3f} | {}",
            i, name, r.mean_sharpe, r.pct_positive, r.p_value,
            "ACCEPTED" if r.accepted else "rejected",
        )

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    subset = "all"
    if "--subset" in sys.argv:
        idx = sys.argv.index("--subset")
        if idx + 1 < len(sys.argv):
            subset = sys.argv[idx + 1]

    from src.backtest.run_benchmark import _filter_oos_period, _load_ohlcv_from_postgres
    from src.config import load_benchmark, load_tickers

    tickers = load_tickers()
    benchmark = load_benchmark()
    ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)
    ohlcv = _filter_oos_period(ohlcv, n_years=10)

    run_improvement_validation(
        ohlcv=ohlcv,
        tickers=tickers,
        benchmark_ticker=benchmark,
        subset=subset,
    )


if __name__ == "__main__":
    _cli_main()