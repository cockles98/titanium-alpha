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
    python -m src.backtest.run_validation --subset tier4 # tier 4 only
"""

from __future__ import annotations

import itertools
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
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}
    configs["baseline"] = _base_config()
    configs["vol_target_08"] = _base_config(target_vol=0.08)
    configs["vol_target_10"] = _base_config(target_vol=0.10)
    configs["vol_target_12"] = _base_config(target_vol=0.12)
    configs["ward_linkage"] = _base_config(hrp_config=HRPConfig(linkage_method="ward", max_weight=dynamic_max_weight))
    configs["shrinkage"] = _base_config(hrp_config=HRPConfig(shrinkage=True, max_weight=dynamic_max_weight))
    configs["ward_shrinkage"] = _base_config(hrp_config=HRPConfig(linkage_method="ward", shrinkage=True, max_weight=dynamic_max_weight))
    return configs


def build_tier2_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}
    configs["rebalance_10d"] = _base_config(rebalance_every=10)
    configs["rebalance_21d"] = _base_config(rebalance_every=21)
    configs["no_tilt"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.0, max_weight=dynamic_max_weight))
    configs["tilt_010"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.10, max_weight=dynamic_max_weight))
    configs["tilt_030"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.30, max_weight=dynamic_max_weight))
    configs["lookback_252"] = _base_config(lookback_days=252)
    configs["lookback_756"] = _base_config(lookback_days=756)
    configs["killswitch_15"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.15))
    configs["killswitch_20"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.20))
    configs["killswitch_25"] = _base_config(killswitch=KillswitchConfig(max_drawdown_pct=-0.25))
    return configs


def build_tier3_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}
    configs["rebalance_1d"] = _base_config(rebalance_every=1)
    configs["rebalance_2d"] = _base_config(rebalance_every=2)
    configs["rebalance_3d"] = _base_config(rebalance_every=3)
    configs["rebalance_5d_delta_03"] = _base_config(min_rebalance_delta=0.03)
    configs["rebalance_5d_delta_05"] = _base_config(min_rebalance_delta=0.05)
    configs["lookback_63"] = _base_config(lookback_days=63)
    configs["lookback_126"] = _base_config(lookback_days=126)
    configs["tilt_040"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.40, max_weight=dynamic_max_weight))
    configs["tilt_050"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.50, max_weight=dynamic_max_weight))
    configs["tilt_075"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=0.75, max_weight=dynamic_max_weight))
    configs["tilt_100"] = _base_config(hrp_config=HRPConfig(confidence_tilt_cap=1.0, max_weight=dynamic_max_weight))
    configs["vol_target_15"] = _base_config(target_vol=0.15)
    configs["vol_target_20"] = _base_config(target_vol=0.20)
    return configs


def build_tier4_configs(n_tickers: int = 52) -> dict[str, WalkForwardConfig]:
    """Build Tier 4 configurations (Cost & Risk Optimization)."""
    configs: dict[str, WalkForwardConfig] = {}
    
    # Base God Mode params para vermos o impacto do delta no cenário de alto giro
    base_kwargs = {
        "rebalance_every": 1,
        "lookback_days": 63,
    }
    
    # 1. Filtros de Giro Finos (min_rebalance_delta)
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    for delta in [0.005, 0.010, 0.015, 0.025]:
        configs[f"t4_delta_{int(delta*1000):03d}"] = _base_config(
            **base_kwargs,
            min_rebalance_delta=delta,
            hrp_config=HRPConfig(confidence_tilt_cap=1.0, max_weight=dynamic_max_weight)
        )

    # 2. Concentração Máxima (max_weight por ativo)
    for max_wt in [0.05, 0.10, 0.15, 0.20]:
        configs[f"t4_maxwt_{int(max_wt*100):02d}"] = _base_config(
            **base_kwargs,
            min_rebalance_delta=0.01, # assume um delta razoável de 1%
            hrp_config=HRPConfig(confidence_tilt_cap=1.0, max_weight=max_wt)
        )

    # 3. Target Volatility Rápido
    configs["t4_vol_lookback_21"] = _base_config(
        **base_kwargs,
        target_vol=0.15,
        vol_lookback=21,
        hrp_config=HRPConfig(confidence_tilt_cap=1.0, max_weight=dynamic_max_weight)
    )

    return configs


def build_momentum_factories() -> dict[str, NaiveModelFactory]:
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


def build_god_combinations(n_tickers: int = 52) -> dict[str, tuple[WalkForwardConfig, NaiveModelFactory]]:
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    params = {
        "rebal1d": {"rebalance_every": 1},
        "cov63": {"lookback_days": 63},
        "tilt100": {"hrp_config": HRPConfig(confidence_tilt_cap=1.0, max_weight=dynamic_max_weight)},
        "mom1d": {"lookback": 1}
    }
    combinations: dict[str, tuple[WalkForwardConfig, NaiveModelFactory]] = {}
    keys = list(params.keys())
    for r in [2, 3, 4]:
        for combo in itertools.combinations(keys, r):
            name = "god_" + "_".join(combo)
            c_kwargs: dict[str, Any] = {}
            f_kwargs: dict[str, int] = {"lookback": 5}
            for key in combo:
                if key == "mom1d":
                    f_kwargs.update(params[key])
                else:
                    c_kwargs.update(params[key])
            config = _base_config(**c_kwargs)
            factory = NaiveModelFactory(**f_kwargs)
            combinations[name] = (config, factory)
    return combinations


def build_all_configs(
    subset: str = "all",
    n_tickers: int = 52,
) -> dict[str, WalkForwardConfig]:
    configs: dict[str, WalkForwardConfig] = {}

    if subset in ("all", "tier1"):
        configs.update(build_tier1_configs(n_tickers))
    if subset in ("all", "tier2"):
        configs.update(build_tier2_configs(n_tickers))
    if subset in ("all", "tier3"):
        configs.update(build_tier3_configs(n_tickers))
    if subset in ("all", "tier4"):
        configs.update(build_tier4_configs(n_tickers))

    if "baseline" not in configs and subset != "god":
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
    for name, r in list(results) + (momentum_results or []):
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

    logger.info("Step 1/8: Building configuration grid (subset={})", subset)
    configs = build_all_configs(subset, n_tickers=len(tickers))
    logger.info("  {} base configurations to validate", len(configs))

    logger.info("Step 2/8: Building momentum factory variants")
    factories = {}
    if subset not in ("god", "tier4"):
        factories = build_momentum_factories()
    logger.info("  {} momentum variants", len(factories))

    logger.info("Step 3/8: Building God Mode combinations")
    god_combinations = {}
    if subset in ("all", "god"):
        god_combinations = build_god_combinations(n_tickers=len(tickers))
    logger.info("  {} god combinations", len(god_combinations))

    logger.info("Step 4/8: Creating CPCV-OOS validator")
    validator = CPCVParameterValidator(ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark_ticker)
    n_total_trials = len(configs) + len(factories) + len(god_combinations)

    logger.info("Step 5/8: Running base config grid search ({} configs)", len(configs))
    # Se estamos no Tier 4, usamos o momentum de 1 dia como fábrica padrão para validar os filtros
    default_factory = NaiveModelFactory(lookback=1) if subset == "tier4" else NaiveModelFactory()
    config_results = validator.grid_search(configs=configs, model_factory=default_factory, n_trials=n_total_trials)

    logger.info("Step 6/8: Running momentum factory grid search ({} variants)", len(factories))
    baseline_config = _base_config()
    momentum_results = []
    for i, (name, factory) in enumerate(factories.items()):
        logger.info("  Momentum [{}/{}]: {}", i + 1, len(factories), name)
        result = validator.validate(config=baseline_config, model_factory=factory, n_trials=n_total_trials)
        momentum_results.append((name, result))
    momentum_results.sort(key=lambda x: x[1].deflated_sharpe, reverse=True)

    logger.info("Step 7/8: Running God Mode grid search ({} variants)", len(god_combinations))
    god_results = []
    for i, (name, (config, factory)) in enumerate(god_combinations.items()):
        logger.info("  God Mode [{}/{}]: {}", i + 1, len(god_combinations), name)
        result = validator.validate(config=config, model_factory=factory, n_trials=n_total_trials)
        god_results.append((name, result))
    god_results.sort(key=lambda x: x[1].deflated_sharpe, reverse=True)

    config_results.extend(god_results)

    logger.info("Step 8/8: Saving outputs")
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