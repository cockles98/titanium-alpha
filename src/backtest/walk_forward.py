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
        """Return momentum-based confidence per ticker using a true sigmoid.

        Returns:
            {ticker: confidence} where confidence asymptotically approaches
            0 and 1. Positive momentum → confidence > 0.5.
        """
        import math

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

            # Fator k (steepness) da sigmóide, escalado inversamente pelo lookback
            k = 50.0 / max(self.lookback, 1)

            # Aplicação da função logística (Sigmóide)
            # Retorna um valor no intervalo (0, 1) suavemente
            conf = 1.0 / (1.0 + math.exp(-k * ret))

            # Opcional: manter os limites estritos [0.05, 0.95] se a otimização
            # do HRP exigir que nenhum peso chegue a zero absoluto tão rápido.
            scores[ticker] = max(0.05, min(0.95, conf))

        return scores


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KillswitchConfig:
    """Configuration for the drawdown killswitch.

    When portfolio drawdown breaches ``max_drawdown_pct``, all holdings
    are liquidated (moved to cash).  Re-entry occurs after the
    **benchmark** drawdown recovers above ``recovery_threshold_pct``
    for ``ramp_up_days`` consecutive trading days.

    Using the benchmark (not the portfolio) for recovery avoids the
    logical bug where the portfolio never exits cash because its own
    drawdown is frozen while in cash.

    Upon re-entry, ``peak_value`` is reset to the current portfolio
    value to avoid immediate re-triggering from stale peaks.

    Attributes:
        max_drawdown_pct: Drawdown threshold to trigger cash exit
            (negative, e.g. ``-0.15`` for -15%).
        recovery_threshold_pct: Benchmark drawdown level at which
            recovery countdown begins (negative, e.g. ``-0.05``).
        ramp_up_days: Number of consecutive days the benchmark must
            stay above ``recovery_threshold_pct`` before re-entry.
            Must be >= 1.
    """

    max_drawdown_pct: float = -0.15
    recovery_threshold_pct: float = -0.05
    ramp_up_days: int = 21


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
        target_vol: Target annualised portfolio volatility.  When
            ``None`` (default), no volatility targeting is applied.
            Example: ``0.10`` for 10% annual vol.
        vol_lookback: Rolling window (trading days) for realised
            volatility estimation.  Default 63 (~one quarter).
        max_leverage: Upper bound on the vol-targeting leverage
            factor.  ``1.0`` means no leverage (long-only).
        min_leverage: Lower bound on exposure.  ``0.5`` means the
            portfolio never goes below 50% invested.  Set to ``0.0``
            for a full volatility killswitch.
        margin_spread: Annual spread above ``rf`` charged on borrowed
            margin (leverage > 1).  For example ``0.015`` for 150 bps.
            The total annual borrow cost is ``rf + margin_spread``.
            Cash (leverage < 1) earns ``rf`` pro-rata.  When the
            killswitch is active, the full portfolio earns ``rf``.
        killswitch: Drawdown killswitch configuration.  When ``None``
            (default), no killswitch is active.
        hrp_config: HRP optimizer configuration.  When ``None``
            (default), a dynamic config is created with
            ``max_weight=min(0.25, 2/n)``.
        top_n: Number of top-scoring tickers to include at each
            rebalance.  When ``None`` (default), all available
            tickers are used.  Must be >= 1 when set.
    """

    retrain_every: int = 126
    rebalance_every: int = 5
    lookback_days: int = 504
    initial_capital: float = 1_000_000.0
    costs: TransactionCosts | None = None
    min_rebalance_delta: float = 0.02
    trading_days_per_year: int = 252
    rf: float = 0.05
    target_vol: float | None = None
    vol_lookback: int = 63
    max_leverage: float = 1.0
    min_leverage: float = 0.5
    margin_spread: float = 0.015
    killswitch: KillswitchConfig | None = None
    hrp_config: HRPConfig | None = None
    top_n: int | None = None


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
        ticker_returns: Optional wide DataFrame (date × ticker) of simple
            daily returns over the active backtest window, useful for
            downstream contribution-to-return analysis. ``None`` when the
            backtester was not asked to retain it.
    """

    equity_curve: pl.DataFrame
    daily_returns: pl.DataFrame
    rebalance_history: list[RebalanceRecord]
    metrics: dict[str, float] = field(default_factory=dict)
    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    ticker_returns: pl.DataFrame | None = None


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
        cfg = self.config
        if cfg.target_vol is not None:
            if cfg.vol_lookback < 2:
                raise ValueError(
                    f"vol_lookback must be >= 2, got {cfg.vol_lookback}"
                )
            if cfg.max_leverage < cfg.min_leverage:
                raise ValueError(
                    f"max_leverage ({cfg.max_leverage}) must be >= "
                    f"min_leverage ({cfg.min_leverage})"
                )
        if cfg.top_n is not None and cfg.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {cfg.top_n}")
        if cfg.killswitch is not None:
            if cfg.killswitch.ramp_up_days < 1:
                raise ValueError(
                    f"ramp_up_days must be >= 1, "
                    f"got {cfg.killswitch.ramp_up_days}"
                )

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

        # Remove duplicatas preservando a ordem
        tickers = list(dict.fromkeys(tickers))

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
        # Inicialização institucional: Fundo nasce 100% em caixa.
        # O primeiro rebalanceamento pagará o custo total de montagem da carteira.
        holdings: dict[str, float] = {
            t: 0.0 for t in available_tickers
        }
        cash = cfg.initial_capital
        portfolio_value = cfg.initial_capital
        benchmark_value = cfg.initial_capital

        # Vol-targeting state (Phase 18): the leverage actually committed
        # at the most recent rebalance.  Updated only when a rebalance
        # passes the ``min_rebalance_delta`` gate, so the daily series is
        # a step function.  Killswitch overrides this to 0.0 at end-of-day.
        current_leverage = 1.0

        rebalance_history: list[RebalanceRecord] = []
        equity_dates: list[date] = []
        equity_portfolio: list[float] = []
        equity_benchmark: list[float] = []
        equity_leverage: list[float] = []
        equity_realized_vol: list[float | None] = []
        returns_port: list[float] = []
        returns_bench: list[float] = []

        days_since_retrain = cfg.retrain_every  # force initial train
        days_since_rebalance = cfg.rebalance_every  # force initial rebalance
        total_retrains = 0

        # HRP config: use provided or dynamic max_weight
        n = len(available_tickers)
        if cfg.hrp_config is not None:
            hrp_config = cfg.hrp_config
        else:
            hrp_config = HRPConfig(max_weight=min(0.25, 2.0 / n))

        # Killswitch state
        in_cash = False
        days_recovering = 0
        peak_value = cfg.initial_capital
        bench_peak = cfg.initial_capital

        # ---- Main loop
        for i, current_date in enumerate(active_dates):
            # Row index in returns_wide
            row_idx = warmup_end + i

            # The decision cutoff is the PREVIOUS trading day.
            # On day t we decide using data up to t-1, then earn
            # day-t's return (close_t / close_{t-1} - 1).  Using
            # current_date would leak close_t into the signal.
            decision_date = trading_dates[row_idx - 1]

            # Capture start-of-day value before costs/rebalance so that
            # port_ret correctly includes transaction cost drag.
            portfolio_value_start_of_day = portfolio_value

            # -- Retrain check (skip when in cash — nothing to trade)
            retrained = False
            if not in_cash and days_since_retrain >= cfg.retrain_every:
                train_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers))
                    & (pl.col("date") <= decision_date)
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

            # -- Rebalance check (skip when in cash)
            if not in_cash and days_since_rebalance >= cfg.rebalance_every:
                # Reset counter unconditionally so we check again in
                # rebalance_every days, even when min_rebalance_delta
                # prevents actual trading.  Without this reset, predict+HRP
                # would run every subsequent day (days_since_rebalance keeps
                # growing past the threshold).
                days_since_rebalance = 0

                # Predict — use decision_date (t-1) to avoid leaking
                # day-t's close into the signal.
                predict_data = ohlcv.filter(
                    (pl.col("ticker").is_in(available_tickers))
                    & (pl.col("date") <= decision_date)
                ).sort(["ticker", "date"])

                confidences = model_factory.predict(predict_data)

                # -- Top-N ticker selection
                if cfg.top_n is not None:
                    effective_top_n = min(cfg.top_n, len(confidences))
                    sorted_tickers = sorted(
                        confidences.keys(),
                        key=lambda t: (
                            confidences[t]
                            if math.isfinite(confidences[t])
                            else float("-inf")
                        ),
                        reverse=True,
                    )
                    selected_tickers = sorted_tickers[:effective_top_n]
                    confidences = {
                        t: confidences[t] for t in selected_tickers
                    }
                    logger.debug(
                        "Top-{} selected on {}: {}",
                        effective_top_n,
                        current_date,
                        selected_tickers,
                    )
                else:
                    selected_tickers = list(available_tickers)

                # Adapt HRP max_weight to the selected universe
                n_sel = len(selected_tickers)
                if cfg.top_n is not None:
                    rebalance_hrp_config = HRPConfig(
                        linkage_method=hrp_config.linkage_method,
                        correlation_method=hrp_config.correlation_method,
                        shrinkage=hrp_config.shrinkage,
                        confidence_tilt_cap=hrp_config.confidence_tilt_cap,
                        min_weight=hrp_config.min_weight,
                        max_weight=min(0.25, 2.0 / n_sel),
                        turnover_threshold=hrp_config.turnover_threshold,
                    )
                else:
                    rebalance_hrp_config = hrp_config

                # HRP — covariance estimated up to decision_date (t-1)
                log_returns = self._compute_log_returns_for_hrp(
                    ohlcv, selected_tickers, decision_date, cfg.lookback_days
                )

                if log_returns.height >= 2:
                    optimizer = HRPOptimizer(config=rebalance_hrp_config)
                    hrp_result = optimizer.optimize(
                        log_returns, confidences=confidences
                    )
                    raw_weights = hrp_result.weights
                else:
                    # Fallback: equal weight
                    raw_weights = {
                        t: 1.0 / n_sel for t in selected_tickers
                    }
                    logger.warning(
                        "Insufficient data for HRP on {}; using equal weights",
                        current_date,
                    )

                # -- Volatility targeting (Ex-Ante Risk of target allocation)
                leverage = 1.0
                if cfg.target_vol is not None and log_returns.height >= cfg.vol_lookback:
                    # Utiliza os retornos simples já calculados para a simulação exata
                    recent_simple_rets = returns_wide.filter(
                        pl.col("date") <= decision_date
                    ).tail(cfg.vol_lookback)

                    simulated_port_rets = []
                    for row_dict in recent_simple_rets.iter_rows(named=True):
                        # Pondera os retornos discretos/simples
                        day_ret = sum(row_dict.get(t, 0.0) * w for t, w in raw_weights.items())
                        simulated_port_rets.append(day_ret)

                    n_rets = len(simulated_port_rets)
                    mean_ret = sum(simulated_port_rets) / n_rets
                    var = sum((r - mean_ret) ** 2 for r in simulated_port_rets) / (n_rets - 1)
                    ex_ante_vol = math.sqrt(var) * math.sqrt(cfg.trading_days_per_year)

                    if ex_ante_vol > 0:
                        raw_leverage = cfg.target_vol / ex_ante_vol
                        leverage = max(
                            cfg.min_leverage,
                            min(cfg.max_leverage, raw_leverage),
                        )

                # Apply leverage to raw HRP weights (sum will now equal leverage, not 1.0)
                new_weights = {t: w * leverage for t, w in raw_weights.items()}

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

                    # Record rebalance
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
                    # Commit the leverage that was actually applied; the
                    # daily series will hold this until the next rebalance.
                    current_leverage = leverage

                    logger.debug(
                        "Rebalanced on {}: turnover={:.4f} cost=${:.2f}",
                        current_date,
                        turnover,
                        dollar_cost,
                    )

            days_since_rebalance += 1

            # -- Compute daily returns (drift-adjusted holdings)
            row = returns_wide.row(row_idx, named=True)
            bench_ret = row.get(benchmark_ticker, 0.0)
            benchmark_value *= (1.0 + bench_ret)

            # Daily risk-free rate for cash carry / margin cost (Geometric compounding)
            daily_rf = (1.0 + cfg.rf) ** (1.0 / cfg.trading_days_per_year) - 1.0

            if in_cash:
                # Killswitch active: full portfolio parked in cash, earns rf
                portfolio_value *= (1.0 + daily_rf)
                port_ret = daily_rf
            else:
                # Update per-asset holdings with today's return
                for t in available_tickers:
                    asset_ret = row.get(t, 0.0)
                    holdings[t] *= (1.0 + asset_ret)

                # Cash carry: positive cash earns rf, negative cash
                # (margin) costs rf + spread.
                if cash >= 0:
                    cash *= (1.0 + daily_rf)
                else:
                    annual_borrow_rate = cfg.rf + cfg.margin_spread
                    daily_borrow = (1.0 + annual_borrow_rate) ** (1.0 / cfg.trading_days_per_year) - 1.0
                    cash *= (1.0 + daily_borrow)

                portfolio_value = sum(holdings.values()) + cash

            # Safeguard de Ruína Matemática
            if portfolio_value <= 0:
                logger.error("BANKRUPTCY TRIGERRED no dia {}. Capital atingiu <= 0.", current_date)
                portfolio_value = 0.0
                port_ret = -1.0
                equity_dates.append(current_date)
                equity_portfolio.append(0.0)
                equity_benchmark.append(benchmark_value)
                equity_leverage.append(0.0)
                equity_realized_vol.append(None)
                returns_port.append(port_ret)
                returns_bench.append(bench_ret)
                break # Encerra o backtest imediatamente

            # -- Drawdown killswitch (after returns are applied)
            if cfg.killswitch is not None:
                peak_value = max(peak_value, portfolio_value)
                bench_peak = max(bench_peak, benchmark_value)
                dd = (portfolio_value / peak_value) - 1.0 if peak_value > 0 else 0.0

                if dd <= cfg.killswitch.max_drawdown_pct and not in_cash:
                    # Exit: liquidate all holdings
                    effective_weights = {
                        t: holdings[t] / portfolio_value
                        for t in available_tickers
                    } if portfolio_value > 0 else {
                        t: 0.0 for t in available_tickers
                    }
                    zero_weights = {t: 0.0 for t in available_tickers}
                    _, exit_cost = self._apply_costs(
                        effective_weights, zero_weights,
                        portfolio_value, cfg.costs,
                    )
                    portfolio_value -= exit_cost
                    holdings = {t: 0.0 for t in available_tickers}
                    cash = portfolio_value
                    in_cash = True
                    days_recovering = 0
                    logger.warning(
                        "KILLSWITCH ON {}: DD={:.2%}, cost=${:.2f}",
                        current_date, dd, exit_cost,
                    )

                elif in_cash:
                    # Recovery: use benchmark drawdown as proxy
                    bench_dd = (
                        (benchmark_value / bench_peak) - 1.0
                        if bench_peak > 0
                        else 0.0
                    )
                    if bench_dd >= cfg.killswitch.recovery_threshold_pct:
                        days_recovering += 1
                        ramp = min(
                            1.0,
                            days_recovering / cfg.killswitch.ramp_up_days,
                        )
                        if ramp >= 1.0:
                            in_cash = False
                            # Reset peak to avoid immediate re-trigger
                            # from stale pre-crash high-water mark
                            peak_value = portfolio_value
                            # Force rebalance on next iteration
                            days_since_rebalance = cfg.rebalance_every
                            logger.info(
                                "KILLSWITCH OFF {}: ramp complete, "
                                "resuming trading",
                                current_date,
                            )
                    else:
                        # Benchmark still in drawdown — reset ramp
                        days_recovering = 0

            # Use start-of-day value as base so that rebalance
            # costs are correctly reflected in the return series.
            port_ret = (
                (portfolio_value / portfolio_value_start_of_day - 1.0)
                if portfolio_value_start_of_day > 0
                else 0.0
            )

            equity_dates.append(current_date)
            equity_portfolio.append(portfolio_value)
            equity_benchmark.append(benchmark_value)
            returns_port.append(port_ret)
            returns_bench.append(bench_ret)

            # Track end-of-day risk diagnostics (Phase 18).  Killswitch
            # overrides the committed leverage to 0.0 since holdings are
            # liquidated.  Realized vol is NaN until ``vol_lookback``
            # daily returns have accumulated.
            equity_leverage.append(0.0 if in_cash else current_leverage)
            if len(returns_port) >= cfg.vol_lookback:
                recent = returns_port[-cfg.vol_lookback:]
                mean_r = sum(recent) / cfg.vol_lookback
                var = sum((r - mean_r) ** 2 for r in recent) / (
                    cfg.vol_lookback - 1
                )
                equity_realized_vol.append(
                    math.sqrt(var) * math.sqrt(cfg.trading_days_per_year)
                )
            else:
                equity_realized_vol.append(None)

        # ---- Build results
        equity_curve = pl.DataFrame({
            "date": equity_dates,
            "portfolio_value": equity_portfolio,
            "benchmark_value": equity_benchmark,
            "leverage": equity_leverage,
            "realized_vol_63d": equity_realized_vol,
        })

        daily_returns = pl.DataFrame({
            "date": equity_dates,
            "portfolio_return": returns_port,
            "benchmark_return": returns_bench,
        })

        metadata: dict[str, Any] = {
            "n_tickers": len(available_tickers),
            "top_n": cfg.top_n,
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

        # Slice the wide returns matrix to the active window so downstream
        # contribution analysis (Phase 9 dashboard waterfall) only sees
        # dates that actually appear in equity_curve / rebalance_history.
        if equity_dates:
            ticker_returns_active = returns_wide.filter(
                pl.col("date").is_in(equity_dates)
            ).sort("date")
        else:
            ticker_returns_active = None

        return WalkForwardResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            rebalance_history=rebalance_history,
            metrics=metrics,
            config=cfg,
            metadata=metadata,
            ticker_returns=ticker_returns_active,
        )
