"""Portfolio vs benchmark performance metrics.

Computes a comprehensive set of risk-adjusted, absolute, and relative
metrics for a portfolio equity curve compared to a buy-and-hold benchmark.

Metrics computed:

- **Return**: CAGR, total_return
- **Risk**: annualized_volatility, max_drawdown, max_drawdown_duration_days,
  calmar_ratio
- **Risk-adjusted**: sharpe_ratio, sortino_ratio, information_ratio
- **vs Benchmark**: alpha, beta, tracking_error, hit_rate_monthly
- **Turnover**: avg_annual_turnover, avg_positions

Usage::

    from src.backtest.benchmark_metrics import compute_benchmark_metrics

    metrics = compute_benchmark_metrics(
        portfolio_returns=result.daily_returns["portfolio_return"],
        benchmark_returns=result.daily_returns["benchmark_return"],
        rebalance_history=result.rebalance_history,
    )
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl
from loguru import logger

from src.backtest.walk_forward import RebalanceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cumulative_from_returns(returns: list[float]) -> list[float]:
    """Build a cumulative return series (starting at 1.0) from daily returns.

    Args:
        returns: List of simple daily returns.

    Returns:
        Cumulative series where ``cum[i] = prod(1 + r_j for j <= i)``.
    """
    cum: list[float] = []
    value = 1.0
    for r in returns:
        value *= (1.0 + r)
        cum.append(value)
    return cum


def _compute_drawdown_series(cumulative: list[float]) -> list[float]:
    """Compute drawdown at each point in a cumulative return series.

    Args:
        cumulative: Cumulative return values starting from 1.0.

    Returns:
        Drawdown series (non-positive values).  E.g. -0.10 = 10% DD.
    """
    if not cumulative:
        return []

    peak = cumulative[0]
    dd: list[float] = []
    for val in cumulative:
        if val > peak:
            peak = val
        dd.append((val - peak) / peak if peak != 0 else 0.0)
    return dd


def _max_drawdown_duration(drawdown_series: list[float]) -> int:
    """Find the longest drawdown duration in trading days.

    A drawdown period starts when DD < 0 and ends when DD returns to 0.

    Args:
        drawdown_series: Output of ``_compute_drawdown_series``.

    Returns:
        Duration in trading days of the longest drawdown.  0 if no DD.
    """
    max_dur = 0
    current_dur = 0
    for dd in drawdown_series:
        if dd < 0:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def _compute_monthly_returns(
    daily_returns: list[float],
    dates: list[date],
) -> list[float]:
    """Aggregate daily returns into monthly returns.

    Groups by (year, month) and compounds within each group.

    Args:
        daily_returns: Simple daily returns.
        dates: Corresponding dates (same length as daily_returns).

    Returns:
        List of monthly compounded returns.
    """
    if not daily_returns or not dates:
        return []

    # Group by (year, month)
    monthly: dict[tuple[int, int], float] = {}
    for d, r in zip(dates, daily_returns):
        key = (d.year, d.month)
        monthly[key] = monthly.get(key, 1.0) * (1.0 + r)

    return [v - 1.0 for v in monthly.values()]


def _capm_regression(
    port_excess: list[float],
    bench_excess: list[float],
) -> tuple[float, float]:
    """Simple OLS regression for Jensen's alpha and beta.

    Regresses ``port_excess = alpha + beta * bench_excess + eps``.

    Args:
        port_excess: Portfolio excess returns (over rf).
        bench_excess: Benchmark excess returns (over rf).

    Returns:
        ``(alpha, beta)`` tuple.  Returns ``(0.0, 0.0)`` if fewer than
        2 observations or zero benchmark variance.
    """
    n = len(port_excess)
    if n < 2 or len(bench_excess) != n:
        return 0.0, 0.0

    mean_x = sum(bench_excess) / n
    mean_y = sum(port_excess) / n

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(bench_excess, port_excess)) / n
    var_x = sum((x - mean_x) ** 2 for x in bench_excess) / n

    if var_x == 0.0:
        return 0.0, 0.0

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    return alpha, beta


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def compute_benchmark_metrics(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    *,
    rebalance_history: list[RebalanceRecord] | None = None,
    dates: pl.Series | None = None,
    rf: float = 0.05,
    trading_days: int = 252,
) -> dict[str, float]:
    """Compute a comprehensive set of portfolio vs benchmark metrics.

    Args:
        portfolio_returns: Daily simple returns of the portfolio.
        benchmark_returns: Daily simple returns of the benchmark.
        rebalance_history: List of rebalance events (for turnover metrics).
        dates: Date series corresponding to the returns (for monthly
            aggregation).  If ``None``, monthly metrics are skipped.
        rf: Annualised risk-free rate.
        trading_days: Number of trading days per year (for annualisation).

    Returns:
        Dictionary with all computed metrics.
    """
    port_ret = portfolio_returns.to_list()
    bench_ret = benchmark_returns.to_list()
    n = len(port_ret)

    if n < 2:
        logger.warning("Fewer than 2 returns — metrics will be zeros")
        return _empty_metrics()

    # ---- Cumulative series
    port_cum = _cumulative_from_returns(port_ret)
    bench_cum = _cumulative_from_returns(bench_ret)

    # ---- Return metrics
    total_return = port_cum[-1] - 1.0
    years = n / trading_days
    cagr = port_cum[-1] ** (1.0 / years) - 1.0 if port_cum[-1] > 0 and years > 0 else -1.0

    bench_total_return = bench_cum[-1] - 1.0

    # ---- Risk metrics
    ann_vol = _std(port_ret) * math.sqrt(trading_days)

    dd_series = _compute_drawdown_series(port_cum)
    max_dd = min(dd_series) if dd_series else 0.0
    max_dd_duration = _max_drawdown_duration(dd_series)

    calmar = cagr / abs(max_dd) if max_dd != 0.0 else 0.0

    # ---- Risk-adjusted metrics
    rf_daily = rf / trading_days
    excess_port = [r - rf_daily for r in port_ret]
    excess_bench = [r - rf_daily for r in bench_ret]

    mean_excess = sum(excess_port) / n
    sharpe = (mean_excess / _std(excess_port)) * math.sqrt(trading_days) if _std(excess_port) > 0 else 0.0

    # Sortino: downside deviation per Sortino (1994) — E[min(r, 0)^2] over ALL obs
    downside_sq_sum = sum(min(r, 0.0) ** 2 for r in excess_port)
    downside_vol = math.sqrt(downside_sq_sum / n) * math.sqrt(trading_days)
    sortino = (sum(excess_port) / n * trading_days) / downside_vol if downside_vol > 0 else 0.0

    # ---- Relative metrics (vs benchmark)
    active_ret = [p - b for p, b in zip(port_ret, bench_ret)]
    tracking_error = _std(active_ret) * math.sqrt(trading_days)
    info_ratio = (
        (sum(active_ret) / n * math.sqrt(trading_days)) / _std(active_ret)
        if _std(active_ret) > 0
        else 0.0
    )

    alpha_daily, beta = _capm_regression(excess_port, excess_bench)
    alpha = alpha_daily * trading_days  # annualise

    # ---- Monthly hit rate
    hit_rate = 0.0
    if dates is not None:
        date_list = dates.to_list()
        port_monthly = _compute_monthly_returns(port_ret, date_list)
        bench_monthly = _compute_monthly_returns(bench_ret, date_list)
        if port_monthly and bench_monthly and len(port_monthly) == len(bench_monthly):
            hits = sum(1 for p, b in zip(port_monthly, bench_monthly) if p > b)
            hit_rate = hits / len(port_monthly)

    # ---- Turnover metrics
    avg_annual_turnover = 0.0
    avg_positions = 0.0
    if rebalance_history:
        total_turnover = sum(r.turnover for r in rebalance_history)
        n_rebalances = len(rebalance_history)
        avg_positions = sum(
            sum(1 for w in r.weights.values() if w > 1e-6)
            for r in rebalance_history
        ) / n_rebalances

        # Annualise turnover
        if years > 0:
            avg_annual_turnover = total_turnover / years

    metrics = {
        # Return
        "cagr": cagr,
        "total_return": total_return,
        "benchmark_total_return": bench_total_return,
        # Risk
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": float(max_dd_duration),
        "calmar_ratio": calmar,
        # Risk-adjusted
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        # vs Benchmark
        "information_ratio": info_ratio,
        "alpha": alpha,
        "beta": beta,
        "tracking_error": tracking_error,
        "hit_rate_monthly": hit_rate,
        # Turnover
        "avg_annual_turnover": avg_annual_turnover,
        "avg_positions": avg_positions,
    }

    logger.info(
        "Metrics computed: Sharpe={:.3f}, CAGR={:.2%}, MaxDD={:.2%}, "
        "Alpha={:.4f}, Beta={:.3f}",
        sharpe,
        cagr,
        max_dd,
        alpha,
        beta,
    )

    return metrics


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _empty_metrics() -> dict[str, float]:
    """Return metrics dict with all zeros."""
    return {
        "cagr": 0.0,
        "total_return": 0.0,
        "benchmark_total_return": 0.0,
        "annualized_volatility": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0.0,
        "calmar_ratio": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "information_ratio": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "tracking_error": 0.0,
        "hit_rate_monthly": 0.0,
        "avg_annual_turnover": 0.0,
        "avg_positions": 0.0,
    }
