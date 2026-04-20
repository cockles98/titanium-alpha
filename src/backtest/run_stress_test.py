"""Liquidity stress test for the God Mode configuration.

Tests the strategy's resilience to increasing transaction costs (slippage)
to find the break-even point where the Sharpe ratio drops below 1.0.

Usage::
    python -m src.backtest.run_stress_test
"""

import json
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.backtest.cpcv_oos import CPCVParameterValidator
from src.backtest.walk_forward import NaiveModelFactory, WalkForwardConfig
from src.portfolio.hrp import HRPConfig


def build_stress_configs(n_tickers: int) -> dict[str, WalkForwardConfig]:
    """Build God Mode configs with escalating slippage."""
    dynamic_max_weight = min(0.25, 2.0 / max(n_tickers, 1))
    configs: dict[str, WalkForwardConfig] = {}
    
    # Derrapagem em basis points (bps) para testar
    slippages = [5.0, 15.0, 30.0, 50.0, 75.0, 100.0]
    
    for slip in slippages:
        name = f"god_slip_{int(slip):03d}bps"
        configs[name] = WalkForwardConfig(
            rebalance_every=1,
            retrain_every=126,
            lookback_days=63,
            initial_capital=1_000_000.0,
            costs=TransactionCosts(slippage_bps=slip, commission_bps=10.0),
            min_rebalance_delta=0.02,
            trading_days_per_year=252,
            rf=0.05,
            hrp_config=HRPConfig(
                confidence_tilt_cap=1.0, 
                max_weight=dynamic_max_weight
            )
        )
    return configs


def run_stress_test() -> None:
    # 1. Load Data
    from src.backtest.run_benchmark import _filter_oos_period, _load_ohlcv_from_postgres
    from src.config import load_benchmark, load_tickers

    logger.info("Carregando dados para o Stress Test de Liquidez...")
    tickers = load_tickers()
    benchmark = load_benchmark()
    ohlcv = _load_ohlcv_from_postgres(tickers, benchmark)
    ohlcv = _filter_oos_period(ohlcv, n_years=10)

    # 2. Build Configs & Validator
    configs = build_stress_configs(len(tickers))
    factory = NaiveModelFactory(lookback=1)  # Sinal fixado em 1 dia
    validator = CPCVParameterValidator(ohlcv=ohlcv, tickers=tickers, benchmark_ticker=benchmark)
    
    logger.info(f"Rodando teste de estresse em {len(configs)} níveis de custo...")
    
    # 3. Execute Grid Search
    results = validator.grid_search(
        configs=configs, 
        model_factory=factory, 
        n_trials=len(configs)
    )
    
    # 4. Exibir e Salvar Resultados
    logger.info("\n--- RESULTADOS DO STRESS TEST ---")
    print(f"{'Configuração':<20} | {'Sharpe':<8} | {'DSR p-value':<12} | {'Aprovado?'}")
    print("-" * 60)
    
    # Ordenar por slippage crescente
    results.sort(key=lambda x: x[0]) 
    
    output_data = {}
    for name, r in results:
        aprovado = "SIM" if r.accepted else "NÃO"
        print(f"{name:<20} | {r.mean_sharpe:>8.3f} | {r.p_value:>12.3f} | {aprovado}")
        output_data[name] = {"sharpe": r.mean_sharpe, "p_value": r.p_value, "accepted": r.accepted}

    out_path = Path("data/outputs/stress_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Resultados detalhados salvos em {out_path}")

if __name__ == "__main__":
    run_stress_test()