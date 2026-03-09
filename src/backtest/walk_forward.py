"""Walk-forward backtester for portfolio-level strategy validation.

Simulates the portfolio operating over time by periodically retraining
the prediction model (slow cycle) and rebalancing weights via HRP
(fast cycle).  Produces an equity curve comparable to a buy-and-hold
benchmark.

Key design choices:

- **Two-cycle architecture**: model retraining (every ``retrain_every``
  days) is separated from rebalancing (every ``rebalance_every`` days)
  to keep compute costs manageable.
- **Zero look-ahead**: at every step, only data up to the current date
  is visible.  The ``model_factory`` receives a strict temporal cutoff.
- **Transaction costs**: applied on absolute weight changes at each
  rebalance.  Benchmark is cost-free buy-and-hold.

Usage::

    from src.backtest.walk_forward import (
        WalkForwardBacktester,
        WalkForwardConfig,
        NaiveModelFactory,
    )

    backtester = WalkForwardBacktester(
        config=WalkForwardConfig(rebalance_every=5, retrain_every=126),
    )
    result = backtester.run(
        ohlcv=ohlcv_df,
        tickers=["AAPL", "MSFT", "GOOG"],
        benchmark_ticker="SPY",
        model_factory=NaiveModelFactory(),
    )
    print(result.metrics)
"""

from __future__ import annotations

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
    """Interface for train/predict used by the walk-forward loop.

    Implementations must be **stateful**: ``train()`` stores model
    state internally and ``predict()`` uses the last trained state.
    """

    def train(self, train_df: pl.DataFrame) -> None:
        """Train (or retrain) the model on historical OHLCV data.

        Args:
            train_df: OHLCV data in long format (columns: date, ticker,
                open, high, low, close, volume).  Sorted by ticker/date.
        """
        ...

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Generate per-ticker scores from the most recent data.

        Args:
            df: OHLCV data up to the current date (long format).

        Returns:
            ``{ticker: score}`` where higher score = more bullish.
            Used as ``confidences`` input to HRP.
        """
        ...


# ---------------------------------------------------------------------------
# Naive model factory (for pipeline validation)
# ---------------------------------------------------------------------------


class NaiveModelFactory:
    """Simple momentum-based model for fast pipeline validation.

    ``predict()`` returns the cumulative return over the last
    ``lookback`` days per ticker, mapped to a 0-1 confidence via
    a sigmoid-like clamp.

    Args:
        lookback: Number of trailing days for momentum calculation.
    """

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def train(self, train_df: pl.DataFrame) -> None:
        """No-op — NaiveModelFactory does not train."""

    def predict(self, df: pl.DataFrame) -> dict[str, float]:
        """Return momentum-based confidence per ticker.

        Returns:
            ``{ticker: confidence}`` where confidence is in ``[0, 1]``.
            Positive momentum → confidence > 0.5.
        """
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
            # Scale inversely with lookback so that longer windows
            # don't saturate the clamp.  lookback=5 → scaling=10 (backward compat).
            scaling = 50.0 / max(self.lookback, 1)
            conf = max(0.05, min(0.95, 0.5 + ret * scaling))
            scores[ticker] = conf
        return scores


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Immutable configuration for the walk-forward backtester.

    Attributes:
        retrain_every: Retrain model every N trading days.
        rebalance_every: Rebalance portfolio every N trading days.
        lookback_days: Size of the trailing window for model training
            and HRP covariance estimation.
        initial_capital: Starting portfolio value.
        costs: Transaction cost model.  ``None`` means zero costs.
        min_rebalance_delta: Skip rebalance if total absolute weight
            change is below this threshold.
        trading_days_per_year: For annualisation of metrics.
        rf: Annual risk-free rate.
    """

    retrain_every: int = 126
    rebalance_every: int = 5
    lookback_days: int = 504
    initial_capital: float = 1_000_000.0
    costs: TransactionCosts | None = None
    min_rebalance_delta: float = 0.0
    trading_days_per_year: int = 252
    rf: float = 0.05


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RebalanceRecord:
    """Record of a single rebalance event.

    Attributes:
        date: Date of the rebalance.
        weights: Portfolio weights after rebalance.
        turnover: Sum of absolute weight changes.
        costs: Dollar cost of this rebalance.
        retrained: Whether the model was retrained on this date.
    """

    date: date
    weights: dict[str, float]
    turnover: float
    costs: float
    retrained: bool


@dataclass
class WalkForwardResult:
    """Full output of a walk-forward backtest run.

    Attributes:
        equity_curve: DataFrame with columns ``date``,
            ``portfolio_value``, ``benchmark_value``.
        daily_returns: DataFrame with columns ``date``,
            ``portfolio_return``, ``benchmark_return``.
        rebalance_history: Ordered list of rebalance events.
        metrics: Summary metrics (populated by benchmark_metrics).
        config: The configuration used.
        metadata: Extra info (n_tickers, period, etc.).
    """

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
    """Walk-forward portfolio backtester.

    Simulates portfolio evolution by iterating over trading dates,
    periodically retraining the model and rebalancing via HRP.

    Args:
        config: Backtester configuration.  Uses defaults when ``None``.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_daily_returns(
        ohlcv: pl.DataFrame, tickers: list[str]
    ) -> pl.DataFrame:
        """Compute daily simple returns from close prices.

        Args:
            ohlcv: OHLCV in long format (date, ticker, close, ...).
            tickers: Tickers to include.

        Returns:
            Wide DataFrame: columns = tickers, rows = dates,
            values = simple daily returns.  First row dropped (NaN).
        """
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
            returns_long.pivot(
                on="ticker", index="date", values="daily_return"
            )
            .sort("date")
            .fill_null(0.0)
        )
        return wide

    @staticmethod
    def _compute_log_returns_for_hrp(
        ohlcv: pl.DataFrame,
        tickers: list[str],
        max_date: date,
        lookback: int,
    ) -> pl.DataFrame:
        """Compute log returns for HRP covariance estimation.

        Args:
            ohlcv: OHLCV in long format.
            tickers: Tickers to include.
            max_date: Cutoff date (inclusive).
            lookback: Number of trailing rows to keep.

        Returns:
            Wide DataFrame of log returns (cols=tickers).
        """
        filtered = ohlcv.filter(
            (pl.col("ticker").is_in(tickers))
            & (pl.col("date") <= max_date)
        )
        log_ret = (
            filtered.sort(["ticker", "date"])
            .with_columns(
                pl.col("close")
                .log()
                .diff()
                .over("ticker")
                .alias("log_return")
            )
            .drop_nulls(subset=["log_return"])
        )

        wide = (
            log_ret.pivot(on="ticker", index="date", values="log_return")
            .sort("date")
        )

        # Drop leading rows where any ticker has null (incomplete history),
        # then fill remaining interior nulls with 0.0.  This avoids both
        # the old drop_nulls() (too aggressive — drops rows if ANY null)
        # and pure fill_null(0.0) (deflates variance for short-history tickers).
        ticker_cols = [c for c in wide.columns if c != "date"]
        all_present = pl.all_horizontal(
            pl.col(c).is_not_null() for c in ticker_cols
        )
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
        """Compute turnover and dollar cost of a rebalance.

        Uses ``TransactionCosts.fixed_cost_frac`` (slippage + commission).
        ``market_impact_bps`` is not applied here because per-asset
        volume data is not available at the portfolio level; use the
        CPCV backtester for per-ticker market impact analysis.

        Args:
            old_weights: Weights before rebalance (drift-adjusted).
            new_weights: Target weights after rebalance.
            portfolio_value: Current portfolio value.
            costs: Cost model (None = zero).

        Returns:
            Tuple of (turnover, dollar_cost).
        """
        all_tickers = set(old_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
            for t in all_tickers
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
        """Execute the walk-forward backtest.

        Args:
            ohlcv: Full OHLCV data in long format (date, ticker, open,
                high, low, close, volume).  Must include both tradeable
                tickers and the benchmark ticker.
            tickers: Tradeable ticker symbols.
            benchmark_ticker: Buy-and-hold benchmark ticker.
            model_factory: Model for generating predictions.  Defaults
                to ``NaiveModelFactory()`` if ``None``.

        Returns:
            WalkForwardResult with equity curve, metrics, and history.

        Raises:
            ValueError: If OHLCV data is insufficient for warmup.
        """
        cfg = self.config
        if model_factory is None:
            model_factory = NaiveModelFactory()

        # ---- Prepare daily returns (wide) for all tickers + benchmark
        all_tickers = list(set(tickers) | {benchmark_ticker})
        returns_wide = self._compute_daily_returns(ohlcv, all_tickers)

        # Ensure all requested tickers are present
        available_tickers = [
            t for t in tickers if t in returns_wide.columns
        ]
        if not available_tickers:
            raise ValueError("No tradeable tickers found in OHLCV data")

        if benchmark_ticker not in returns_wide.columns:
            raise ValueError(
                f"Benchmark ticker '{benchmark_ticker}' not in OHLCV data"
            )

        trading_dates = returns_wide["date"].to_list()

        # Warmup: skip first lookback_days
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

        # ---- State variables
        # Track per-asset dollar values (weight drift between rebalances)
        equal_w = 1.0 / len(available_tickers)
        holdings: dict[str, float] = {
            t: cfg.initial_capital * equal_w for t in available_tickers
        }
        portfolio_value = cfg.initial_capital
        benchmark_value = cfg.initial_capital

        rebalance_history: list[RebalanceRecord] = []
        equity_dates: list[date] = []
        equity_portfolio: list[float] = []
        equity_benchmark: list[float] = []
        returns_port: list[float] = []
        returns_bench: list[float] = []

        days_since_retrain = cfg.retrain_every  # force initial train
        days_since_rebalance = cfg.rebalance_every  # force initial rebalance
        total_retrains = 0

        # HRP config with dynamic max_weight
        n = len(available_tickers)
        hrp_config = HRPConfig(max_weight=min(0.25, 2.0 / n))

        # ---- Main loop
        for i, current_date in enumerate(active_dates):
            # Row index in returns_wide
            row_idx = warmup_end + i

            # -- Retrain check
            retrained = False
            if days_since_retrain >= cfg.retrain_every:
                train_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers))
                    & (pl.col("date") <= current_date)
                ).sort(["ticker", "date"])

                # Trim to lookback
                train_data = (
                    train_data.with_columns(
                        pl.col("date")
                        .rank("ordinal")
                        .over("ticker")
                        .alias("_rank")
                    )
                )
                max_ranks = train_data.group_by("ticker").agg(
                    pl.col("_rank").max().alias("_max_rank")
                )
                train_data = (
                    train_data.join(max_ranks, on="ticker")
                    .filter(
                        pl.col("_rank")
                        > (pl.col("_max_rank") - cfg.lookback_days)
                    )
                    .drop(["_rank", "_max_rank"])
                )

                model_factory.train(train_data)
                days_since_retrain = 0
                retrained = True
                total_retrains += 1
                logger.debug("Retrained model on {}", current_date)

            days_since_retrain += 1

            # -- Rebalance check
            rebalanced = False
            if days_since_rebalance >= cfg.rebalance_every:
                # Predict
                predict_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers))
                    & (pl.col("date") <= current_date)
                ).sort(["ticker", "date"])

                confidences = model_factory.predict(predict_data)

                # HRP
                log_returns = self._compute_log_returns_for_hrp(
                    ohlcv, available_tickers, current_date, cfg.lookback_days
                )

                if log_returns.height >= 2:
                    optimizer = HRPOptimizer(config=hrp_config)
                    hrp_result = optimizer.optimize(
                        log_returns, confidences=confidences
                    )
                    new_weights = hrp_result.weights
                else:
                    # Fallback: equal weight
                    new_weights = {
                        t: 1.0 / len(available_tickers)
                        for t in available_tickers
                    }
                    logger.warning(
                        "Insufficient data for HRP on {}; using equal weights",
                        current_date,
                    )

                # Effective weights from current holdings (drift-adjusted)
                effective_weights = {
                    t: holdings[t] / portfolio_value
                    for t in available_tickers
                } if portfolio_value > 0 else {
                    t: 1.0 / len(available_tickers)
                    for t in available_tickers
                }

                # Check min_rebalance_delta threshold
                total_delta = sum(
                    abs(new_weights.get(t, 0.0) - effective_weights.get(t, 0.0))
                    for t in set(new_weights) | set(effective_weights)
                )

                if total_delta >= cfg.min_rebalance_delta:
                    turnover, dollar_cost = self._apply_costs(
                        effective_weights,
                        new_weights,
                        portfolio_value,
                        cfg.costs,
                    )
                    portfolio_value -= dollar_cost

                    # Redistribute holdings to new target weights
                    holdings = {
                        t: portfolio_value * new_weights.get(t, 0.0)
                        for t in available_tickers
                    }

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
                        current_date,
                        turnover,
                        dollar_cost,
                    )

            if not rebalanced:
                days_since_rebalance += 1

            # -- Compute daily returns (drift-adjusted holdings)
            row = returns_wide.row(row_idx, named=True)

            # Update per-asset holdings with today's return
            old_portfolio_value = portfolio_value
            for t in available_tickers:
                asset_ret = row.get(t, 0.0)
                holdings[t] *= (1.0 + asset_ret)

            portfolio_value = sum(holdings.values())
            port_ret = (
                (portfolio_value / old_portfolio_value - 1.0)
                if old_portfolio_value > 0
                else 0.0
            )

            bench_ret = row.get(benchmark_ticker, 0.0)
            benchmark_value *= (1.0 + bench_ret)

            equity_dates.append(current_date)
            equity_portfolio.append(portfolio_value)
            equity_benchmark.append(benchmark_value)
            returns_port.append(port_ret)
            returns_bench.append(bench_ret)

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

        # ---- Compute metrics
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
