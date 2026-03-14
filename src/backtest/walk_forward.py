"""Walk-forward backtester for portfolio-level strategy validation.

Simulates the portfolio operating over time by periodically retraining
the prediction model (slow cycle) and rebalancing weights via HRP
(fast cycle). Produces an equity curve comparable to a buy-and-hold
benchmark.

Key design choices:

- **Two-cycle architecture**: model retraining (every ``retrain_every``
  days) is separated from rebalancing (every ``rebalance_every`` days)
  to keep compute costs manageable.
- **Strict Temporal Isolation**: Mark-to-market is applied before any
  trading decision is made for the next day, completely eliminating
  look-ahead bias.
- **Cost of Capital**: Cash yields the risk-free rate, and leverage
  (negative cash) incurs a margin spread cost.
- **Transaction costs**: applied on absolute weight changes at each
  rebalance. Benchmark is cost-free buy-and-hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

import polars as pl
from loguru import logger

from src.backtest.cpcv import TransactionCosts
from src.portfolio.hrp import HRPConfig, HRPOptimizer


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelFactory(Protocol):
    """Interface for train/predict used by the walk-forward loop."""

    def train(self, train_df: pl.DataFrame) -> None:
        """Train (or retrain) the model on historical OHLCV data."""
        ...

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Generate per-ticker scores from the most recent data."""
        ...


# ---------------------------------------------------------------------------
# Naive model factory (for pipeline validation)
# ---------------------------------------------------------------------------


class NaiveModelFactory:
    """Simple momentum-based model for fast pipeline validation."""

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def train(self, train_df: pl.DataFrame) -> None:
        """No-op — NaiveModelFactory does not train."""

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Return momentum-based confidence per ticker."""
        scores: dict[str, float] = {}
        for ticker in df["ticker"].unique().sort().to_list():
            closes = (
                df.filter(pl.col("ticker") == ticker)
                .sort("date")["close"]
                .tail(self.lookback + 1)
            )
            if closes.len() < 2:
                scores[ticker] = 0.5
                continue
            ret = (closes[-1] - closes[0]) / closes[0]
            scaling = 50.0 / max(self.lookback, 1)
            conf = max(0.05, min(0.95, 0.5 + ret * scaling))
            scores[ticker] = conf
        return scores


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KillswitchConfig:
    """Configuration for the drawdown killswitch."""

    max_drawdown_pct: float = -0.15
    recovery_threshold_pct: float = -0.05
    ramp_up_days: int = 21


@dataclass(frozen=True)
class WalkForwardConfig:
    """Immutable configuration for the walk-forward backtester."""

    retrain_every: int = 126
    rebalance_every: int = 5
    lookback_days: int = 504
    initial_capital: float = 1_000_000.0
    costs: TransactionCosts | None = None
    min_rebalance_delta: float = 0.0
    trading_days_per_year: int = 252
    rf: float = 0.05
    margin_spread: float = 0.02  # Spread over RF for borrowing cash
    target_vol: float | None = None
    vol_lookback: int = 63
    max_leverage: float = 1.0
    min_leverage: float = 0.5
    killswitch: KillswitchConfig | None = None
    hrp_config: HRPConfig | None = None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event."""

    date: date
    weights: dict[str, float]
    turnover: float
    costs: float
    retrained: bool


@dataclass
class WalkForwardResult:
    """Full output of a walk-forward backtest run."""

    equity_curve: pl.DataFrame
    daily_returns: pl.DataFrame
    rebalance_history: list[RebalanceRecord]
    metrics: dict[str, float] = field(default_factory=dict)
    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WalkForwardBacktester
# ---------------------------------------------------------------------------


class WalkForwardBacktester:
    """Walk-forward portfolio backtester."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()
        cfg = self.config
        if cfg.target_vol is not None:
            if cfg.vol_lookback < 2:
                raise ValueError(f"vol_lookback must be >= 2, got {cfg.vol_lookback}")
            if cfg.max_leverage < cfg.min_leverage:
                raise ValueError(
                    f"max_leverage ({cfg.max_leverage}) must be >= "
                    f"min_leverage ({cfg.min_leverage})"
                )
        if cfg.killswitch is not None:
            if cfg.killswitch.ramp_up_days < 1:
                raise ValueError(
                    f"ramp_up_days must be >= 1, got {cfg.killswitch.ramp_up_days}"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_daily_returns(ohlcv: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
        filtered = ohlcv.filter(pl.col("ticker").is_in(tickers))
        returns_long = (
            filtered.sort(["ticker", "date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1)
                .alias("daily_return")
            )
            .drop_nulls(subset=["daily_return"])
        )

        wide = (
            returns_long.pivot(on="ticker", index="date", values="daily_return")
            .sort("date")
            .fill_null(0.0)
        )
        return wide

    @staticmethod
    def _compute_log_returns_for_hrp(
        ohlcv: pl.DataFrame, tickers: list[str], max_date: date, lookback: int
    ) -> pl.DataFrame:
        filtered = ohlcv.filter(
            (pl.col("ticker").is_in(tickers)) & (pl.col("date") <= max_date)
        )
        log_ret = (
            filtered.sort(["ticker", "date"])
            .with_columns(
                pl.col("close").log().diff().over("ticker").alias("log_return")
            )
            .drop_nulls(subset=["log_return"])
        )

        wide = log_ret.pivot(on="ticker", index="date", values="log_return").sort("date")

        ticker_cols = [c for c in wide.columns if c != "date"]
        all_present = pl.all_horizontal(pl.col(c).is_not_null() for c in ticker_cols)
        first_complete_idx = wide.with_row_index("_idx").filter(all_present)["_idx"]
        
        if first_complete_idx.len() > 0:
            wide = wide.slice(int(first_complete_idx[0]))
        wide = wide.fill_null(0.0)

        if wide.height > lookback:
            wide = wide.tail(lookback)

        return wide.select(ticker_cols)

    @staticmethod
    def _apply_costs(
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        portfolio_value: float,
        costs: TransactionCosts | None,
    ) -> tuple[float, float]:
        all_tickers = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in all_tickers
        )

        if costs is None or turnover == 0:
            return turnover, 0.0

        dollar_cost = turnover * portfolio_value * costs.fixed_cost_frac
        return turnover, dollar_cost

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv: pl.DataFrame,
        tickers: list[str],
        benchmark_ticker: str = "SPY",
        model_factory: ModelFactory | None = None,
    ) -> WalkForwardResult:
        cfg = self.config
        if model_factory is None:
            model_factory = NaiveModelFactory()

        all_tickers = list(set(tickers) | {benchmark_ticker})
        returns_wide = self._compute_daily_returns(ohlcv, all_tickers)

        available_tickers = [t for t in tickers if t in returns_wide.columns]
        if not available_tickers:
            raise ValueError("No tradeable tickers found in OHLCV data")
        if benchmark_ticker not in returns_wide.columns:
            raise ValueError(f"Benchmark ticker '{benchmark_ticker}' not in OHLCV data")

        trading_dates = returns_wide["date"].to_list()

        if len(trading_dates) <= cfg.lookback_days:
            raise ValueError(
                f"Need > {cfg.lookback_days} trading days for warmup, "
                f"got {len(trading_dates)}"
            )

        warmup_end = cfg.lookback_days
        active_dates = trading_dates[warmup_end:]

        logger.info(
            "Walk-forward: {} active days, {} tickers, "
            "rebalance every {} days, retrain every {} days",
            len(active_dates),
            len(available_tickers),
            cfg.rebalance_every,
            cfg.retrain_every,
        )

        # ---- Daily rates for Cost of Capital (Compound Interest)
        rf_daily = (1.0 + cfg.rf) ** (1.0 / cfg.trading_days_per_year) - 1.0
        margin_rate_daily = (
            (1.0 + cfg.rf + cfg.margin_spread) ** (1.0 / cfg.trading_days_per_year) - 1.0
        )

        # ---- State variables
        equal_w = 1.0 / len(available_tickers)
        holdings: dict[str, float] = {
            t: cfg.initial_capital * equal_w for t in available_tickers
        }
        cash = 0.0
        portfolio_value = cfg.initial_capital
        benchmark_value = cfg.initial_capital

        rebalance_history: list[RebalanceRecord] = []
        equity_dates: list[date] = []
        equity_portfolio: list[float] = []
        equity_benchmark: list[float] = []
        returns_port: list[float] = []
        returns_bench: list[float] = []

        days_since_retrain = cfg.retrain_every
        days_since_rebalance = cfg.rebalance_every
        total_retrains = 0

        n = len(available_tickers)
        hrp_config = cfg.hrp_config or HRPConfig(max_weight=min(0.25, 2.0 / n))

        in_cash = False
        days_recovering = 0
        peak_value = cfg.initial_capital
        bench_peak = cfg.initial_capital

        # ---- Main loop
        for i, current_date in enumerate(active_dates):
            row_idx = warmup_end + i
            row = returns_wide.row(row_idx, named=True)

            # =================================================================
            # FASE 1: MARCAÇÃO A MERCADO (O mercado move de t-1 para t)
            # =================================================================
            portfolio_value_start_of_day = portfolio_value
            bench_ret = row.get(benchmark_ticker, 0.0)
            benchmark_value *= (1.0 + bench_ret)

            if in_cash:
                cash *= (1.0 + rf_daily)
                portfolio_value = cash
                port_ret = (
                    (portfolio_value / portfolio_value_start_of_day) - 1.0
                    if portfolio_value_start_of_day > 0 else 0.0
                )
            else:
                # 1. Yield/Cost on cash from previous day
                if cash >= 0:
                    cash *= (1.0 + rf_daily)
                else:
                    cash *= (1.0 + margin_rate_daily)

                # 2. Asset returns applied to previous day's holdings
                for t in available_tickers:
                    asset_ret = row.get(t, 0.0)
                    holdings[t] *= (1.0 + asset_ret)

                # 3. New Portfolio Value
                portfolio_value = sum(holdings.values()) + cash
                
                port_ret = (
                    (portfolio_value / portfolio_value_start_of_day - 1.0)
                    if portfolio_value_start_of_day > 0 else 0.0
                )

            # =================================================================
            # FASE 2: MÉTRICAS E AVALIAÇÃO DO KILLSWITCH
            # =================================================================
            if cfg.killswitch is not None:
                peak_value = max(peak_value, portfolio_value)
                bench_peak = max(bench_peak, benchmark_value)
                dd = (portfolio_value / peak_value) - 1.0 if peak_value > 0 else 0.0

                if dd <= cfg.killswitch.max_drawdown_pct and not in_cash:
                    # Liquidate
                    effective_weights = {
                        t: holdings[t] / portfolio_value for t in available_tickers
                    } if portfolio_value > 0 else {t: 0.0 for t in available_tickers}
                    
                    zero_weights = {t: 0.0 for t in available_tickers}
                    _, exit_cost = self._apply_costs(
                        effective_weights, zero_weights, portfolio_value, cfg.costs
                    )
                    
                    portfolio_value -= exit_cost
                    holdings = {t: 0.0 for t in available_tickers}
                    cash = portfolio_value
                    in_cash = True
                    days_recovering = 0
                    logger.warning("KILLSWITCH ON {}: DD={:.2%}, cost=${:.2f}", current_date, dd, exit_cost)

                elif in_cash:
                    bench_dd = ((benchmark_value / bench_peak) - 1.0 if bench_peak > 0 else 0.0)
                    if bench_dd >= cfg.killswitch.recovery_threshold_pct:
                        days_recovering += 1
                        ramp = min(1.0, days_recovering / cfg.killswitch.ramp_up_days)
                        if ramp >= 1.0:
                            in_cash = False
                            peak_value = portfolio_value
                            days_since_rebalance = cfg.rebalance_every
                            logger.info("KILLSWITCH OFF {}: resuming trading", current_date)
                    else:
                        days_recovering = 0

            equity_dates.append(current_date)
            equity_portfolio.append(portfolio_value)
            equity_benchmark.append(benchmark_value)
            returns_port.append(port_ret)
            returns_bench.append(bench_ret)

            # =================================================================
            # FASE 3: DECISÃO E REBALANCEAMENTO (Prepara exposição para t+1)
            # =================================================================
            
            # -- Retrain
            retrained = False
            if not in_cash and days_since_retrain >= cfg.retrain_every:
                train_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers)) & (pl.col("date") <= current_date)
                ).sort(["ticker", "date"])

                train_data = train_data.with_columns(
                    pl.col("date").rank("ordinal").over("ticker").alias("_rank")
                )
                max_ranks = train_data.group_by("ticker").agg(
                    pl.col("_rank").max().alias("_max_rank")
                )
                train_data = (
                    train_data.join(max_ranks, on="ticker")
                    .filter(pl.col("_rank") > (pl.col("_max_rank") - cfg.lookback_days))
                    .drop(["_rank", "_max_rank"])
                )

                model_factory.train(train_data)
                days_since_retrain = 0
                retrained = True
                total_retrains += 1
                logger.debug("Retrained model on {}", current_date)

            days_since_retrain += 1

            # -- Rebalance
            rebalanced = False
            if not in_cash and days_since_rebalance >= cfg.rebalance_every:
                predict_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers)) & (pl.col("date") <= current_date)
                ).sort(["ticker", "date"])

                confidences = model_factory.predict(predict_data)
                log_returns = self._compute_log_returns_for_hrp(
                    ohlcv, available_tickers, current_date, cfg.lookback_days
                )

                if log_returns.height >= 2:
                    optimizer = HRPOptimizer(config=hrp_config)
                    hrp_result = optimizer.optimize(log_returns, confidences=confidences)
                    new_weights = hrp_result.weights
                else:
                    new_weights = {t: 1.0 / len(available_tickers) for t in available_tickers}

                effective_weights = {
                    t: holdings[t] / portfolio_value for t in available_tickers
                } if portfolio_value > 0 else {
                    t: 1.0 / len(available_tickers) for t in available_tickers
                }

                total_delta = sum(
                    abs(new_weights.get(t, 0.0) - effective_weights.get(t, 0.0))
                    for t in set(new_weights) | set(effective_weights)
                )

                if total_delta >= cfg.min_rebalance_delta:
                    turnover, dollar_cost = self._apply_costs(
                        effective_weights, new_weights, portfolio_value, cfg.costs
                    )
                    
                    portfolio_value -= dollar_cost
                    
                    # Distribute new holdings based on post-cost portfolio value
                    holdings = {
                        t: portfolio_value * new_weights.get(t, 0.0) for t in available_tickers
                    }
                    cash = portfolio_value - sum(holdings.values())

                    rebalance_history.append(
                        RebalanceRecord(
                            date=current_date,
                            weights=dict(new_weights),
                            turnover=turnover,
                            costs=dollar_cost,
                            retrained=retrained,
                        )
                    )
                    rebalanced = True
                    days_since_rebalance = 0

                    logger.debug(
                        "Rebalanced on {}: turnover={:.4f} cost=${:.2f}",
                        current_date, turnover, dollar_cost,
                    )

            if not rebalanced:
                days_since_rebalance += 1

            # -- Volatility targeting (daily exposure adjustment)
            if not in_cash and cfg.target_vol is not None and len(returns_port) >= cfg.vol_lookback:
                recent_rets = returns_port[-cfg.vol_lookback:]
                n_rets = len(recent_rets)
                mean_ret = sum(recent_rets) / n_rets
                var = sum((r - mean_ret) ** 2 for r in recent_rets) / (n_rets - 1)
                realized_vol = math.sqrt(var) * math.sqrt(cfg.trading_days_per_year)

                if realized_vol > 0:
                    raw_leverage = cfg.target_vol / realized_vol
                    leverage = max(cfg.min_leverage, min(cfg.max_leverage, raw_leverage))

                    invested = sum(holdings.values())
                    if invested > 0:
                        scale = leverage * portfolio_value / invested
                        for t in available_tickers:
                            holdings[t] *= scale
                        
                        # Cash absorbs the difference (will go negative if leverage > 1.0)
                        cash = portfolio_value - sum(holdings.values())

        # ---- Build results
        equity_curve = pl.DataFrame({
            "date": equity_dates,
            "portfolio_value": equity_portfolio,
            "benchmark_value": equity_benchmark,
        })

        daily_returns = pl.DataFrame({
            "date": equity_dates,
            "portfolio_return": returns_port,
            "benchmark_return": returns_bench,
        })

        metadata: dict[str, Any] = {
            "n_tickers": len(available_tickers),
            "benchmark_ticker": benchmark_ticker,
            "start_date": str(equity_dates[0]) if equity_dates else None,
            "end_date": str(equity_dates[-1]) if equity_dates else None,
            "n_trading_days": len(equity_dates),
            "n_rebalances": len(rebalance_history),
            "n_retrains": total_retrains,
        }

        logger.info(
            "Walk-forward complete: {} days, {} rebalances, "
            "final portfolio=${:,.0f}, final benchmark=${:,.0f}",
            len(equity_dates),
            len(rebalance_history),
            portfolio_value,
            benchmark_value,
        )

        from src.backtest.benchmark_metrics import compute_benchmark_metrics
        metrics = compute_benchmark_metrics(
            portfolio_returns=daily_returns["portfolio_return"],
            benchmark_returns=daily_returns["benchmark_return"],
            rebalance_history=rebalance_history,
            dates=daily_returns["date"],
            rf=cfg.rf,
            trading_days=cfg.trading_days_per_year,
        )

        return WalkForwardResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            rebalance_history=rebalance_history,
            metrics=metrics,
            config=cfg,
            metadata=metadata,
        )