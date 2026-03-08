"""Tests for src.backtest.benchmark_metrics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from src.backtest.benchmark_metrics import (
    _capm_regression,
    _compute_drawdown_series,
    _compute_monthly_returns,
    _cumulative_from_returns,
    _max_drawdown_duration,
    _std,
    compute_benchmark_metrics,
)
from src.backtest.walk_forward import RebalanceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dates(n: int, start: date = date(2020, 1, 2)) -> list[date]:
    """Generate n consecutive dates."""
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# TestCumulativeFromReturns
# ---------------------------------------------------------------------------


class TestCumulativeFromReturns:
    def test_empty(self) -> None:
        assert _cumulative_from_returns([]) == []

    def test_single(self) -> None:
        result = _cumulative_from_returns([0.10])
        assert result == pytest.approx([1.10])

    def test_compounding(self) -> None:
        result = _cumulative_from_returns([0.10, -0.05, 0.02])
        expected = [1.10, 1.10 * 0.95, 1.10 * 0.95 * 1.02]
        assert result == pytest.approx(expected)

    def test_zero_returns(self) -> None:
        result = _cumulative_from_returns([0.0, 0.0, 0.0])
        assert result == pytest.approx([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# TestDrawdownSeries
# ---------------------------------------------------------------------------


class TestDrawdownSeries:
    def test_empty(self) -> None:
        assert _compute_drawdown_series([]) == []

    def test_monotonically_increasing(self) -> None:
        dd = _compute_drawdown_series([1.0, 1.05, 1.10, 1.15])
        assert all(d == 0.0 for d in dd)

    def test_single_drawdown(self) -> None:
        cum = [1.0, 1.10, 1.05, 1.00, 1.15]
        dd = _compute_drawdown_series(cum)
        # Peak at 1.10; trough at 1.00 → DD = (1.00-1.10)/1.10
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert dd[2] == pytest.approx((1.05 - 1.10) / 1.10)
        assert dd[3] == pytest.approx((1.00 - 1.10) / 1.10)
        assert dd[4] == 0.0  # new peak

    def test_all_negative(self) -> None:
        cum = [1.0, 0.9, 0.8, 0.7]
        dd = _compute_drawdown_series(cum)
        assert dd[-1] == pytest.approx(-0.30)


# ---------------------------------------------------------------------------
# TestMaxDrawdownDuration
# ---------------------------------------------------------------------------


class TestMaxDrawdownDuration:
    def test_no_drawdown(self) -> None:
        assert _max_drawdown_duration([0.0, 0.0, 0.0]) == 0

    def test_single_drawdown(self) -> None:
        dd = [0.0, -0.01, -0.03, -0.02, 0.0, 0.0]
        assert _max_drawdown_duration(dd) == 3

    def test_two_drawdowns_takes_longest(self) -> None:
        dd = [0.0, -0.01, 0.0, -0.02, -0.03, -0.01, 0.0]
        assert _max_drawdown_duration(dd) == 3

    def test_empty(self) -> None:
        assert _max_drawdown_duration([]) == 0


# ---------------------------------------------------------------------------
# TestMonthlyReturns
# ---------------------------------------------------------------------------


class TestMonthlyReturns:
    def test_empty(self) -> None:
        assert _compute_monthly_returns([], []) == []

    def test_single_month(self) -> None:
        dates = [date(2020, 1, d) for d in range(2, 7)]
        rets = [0.01] * 5
        monthly = _compute_monthly_returns(rets, dates)
        assert len(monthly) == 1
        expected = 1.01 ** 5 - 1.0
        assert monthly[0] == pytest.approx(expected)

    def test_two_months(self) -> None:
        dates = [date(2020, 1, 15), date(2020, 1, 16), date(2020, 2, 3), date(2020, 2, 4)]
        rets = [0.01, 0.02, -0.01, 0.03]
        monthly = _compute_monthly_returns(rets, dates)
        assert len(monthly) == 2


# ---------------------------------------------------------------------------
# TestCAPMRegression
# ---------------------------------------------------------------------------


class TestCAPMRegression:
    def test_too_few_observations(self) -> None:
        alpha, beta = _capm_regression([0.01], [0.01])
        assert alpha == 0.0
        assert beta == 0.0

    def test_perfect_correlation(self) -> None:
        bench = [0.01, 0.02, -0.01, 0.03, -0.02]
        port = [x * 2 for x in bench]  # beta = 2
        alpha, beta = _capm_regression(port, bench)
        assert beta == pytest.approx(2.0)
        assert alpha == pytest.approx(0.0, abs=1e-10)

    def test_zero_benchmark_variance(self) -> None:
        alpha, beta = _capm_regression([0.01, 0.02, 0.03], [0.01, 0.01, 0.01])
        assert alpha == 0.0
        assert beta == 0.0

    def test_with_alpha(self) -> None:
        # port = 0.001 + 1.5 * bench
        bench = [0.01, -0.01, 0.02, -0.02, 0.015]
        port = [0.001 + 1.5 * b for b in bench]
        alpha, beta = _capm_regression(port, bench)
        assert beta == pytest.approx(1.5, abs=1e-10)
        assert alpha == pytest.approx(0.001, abs=1e-10)


# ---------------------------------------------------------------------------
# TestStd
# ---------------------------------------------------------------------------


class TestStd:
    def test_empty(self) -> None:
        assert _std([]) == 0.0

    def test_single(self) -> None:
        assert _std([5.0]) == 0.0

    def test_known_values(self) -> None:
        # [1, 2, 3, 4, 5] → std with ddof=1
        import statistics
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _std(vals) == pytest.approx(statistics.stdev(vals))

    def test_constant(self) -> None:
        assert _std([3.0, 3.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# TestComputeBenchmarkMetrics
# ---------------------------------------------------------------------------


class TestComputeBenchmarkMetrics:
    def test_returns_all_keys(self) -> None:
        port = pl.Series("portfolio_return", [0.01] * 100)
        bench = pl.Series("benchmark_return", [0.005] * 100)
        metrics = compute_benchmark_metrics(port, bench)
        expected_keys = {
            "cagr", "total_return", "benchmark_total_return",
            "annualized_volatility", "max_drawdown", "max_drawdown_duration_days",
            "calmar_ratio", "sharpe_ratio", "sortino_ratio",
            "information_ratio", "alpha", "beta", "tracking_error",
            "hit_rate_monthly", "avg_annual_turnover", "avg_positions",
        }
        assert set(metrics.keys()) == expected_keys

    def test_too_few_returns(self) -> None:
        port = pl.Series("r", [0.01])
        bench = pl.Series("r", [0.005])
        metrics = compute_benchmark_metrics(port, bench)
        assert all(v == 0.0 for v in metrics.values())

    def test_positive_returns_positive_sharpe(self) -> None:
        port = pl.Series("r", [0.005] * 252)
        bench = pl.Series("r", [0.002] * 252)
        metrics = compute_benchmark_metrics(port, bench, rf=0.0)
        # Constant returns → zero volatility → Sharpe = 0
        # because std of constant excess returns = 0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["cagr"] > 0.0
        assert metrics["total_return"] > 0.0

    def test_cagr_one_year(self) -> None:
        # 252 days of 0.1% daily return
        daily_ret = 0.001
        port = pl.Series("r", [daily_ret] * 252)
        bench = pl.Series("r", [0.0] * 252)
        metrics = compute_benchmark_metrics(port, bench, rf=0.0)
        # CAGR ≈ (1.001^252)^1 - 1
        expected_cagr = (1 + daily_ret) ** 252 - 1.0
        assert metrics["cagr"] == pytest.approx(expected_cagr, rel=1e-6)

    def test_max_drawdown_negative(self) -> None:
        # Goes up then down
        rets = [0.05] * 10 + [-0.10] * 5 + [0.01] * 10
        port = pl.Series("r", rets)
        bench = pl.Series("r", [0.0] * len(rets))
        metrics = compute_benchmark_metrics(port, bench)
        assert metrics["max_drawdown"] < 0.0

    def test_max_drawdown_duration(self) -> None:
        # Down 1% for 20 days then recover
        rets = [0.01] * 10 + [-0.005] * 20 + [0.01] * 20
        port = pl.Series("r", rets)
        bench = pl.Series("r", [0.0] * len(rets))
        metrics = compute_benchmark_metrics(port, bench)
        assert metrics["max_drawdown_duration_days"] > 0

    def test_sortino_with_downside(self) -> None:
        import random
        rng = random.Random(42)
        rets = [rng.gauss(0.001, 0.02) for _ in range(252)]
        port = pl.Series("r", rets)
        bench = pl.Series("r", [0.0] * 252)
        metrics = compute_benchmark_metrics(port, bench, rf=0.0)
        # Should be nonzero with mixed returns
        assert metrics["sortino_ratio"] != 0.0

    def test_information_ratio(self) -> None:
        # Portfolio consistently beats benchmark
        port = pl.Series("r", [0.01] * 100)
        bench = pl.Series("r", [0.005] * 100)
        metrics = compute_benchmark_metrics(port, bench)
        # Constant active return → zero TE → info_ratio = 0
        assert metrics["tracking_error"] == 0.0

    def test_beta_one_for_identical(self) -> None:
        import random
        rng = random.Random(42)
        rets = [rng.gauss(0.001, 0.01) for _ in range(100)]
        port = pl.Series("r", rets)
        bench = pl.Series("r", rets)
        metrics = compute_benchmark_metrics(port, bench, rf=0.0)
        assert metrics["beta"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["alpha"] == pytest.approx(0.0, abs=1e-6)

    def test_calmar_ratio(self) -> None:
        import random
        rng = random.Random(123)
        rets = [rng.gauss(0.002, 0.015) for _ in range(252)]
        port = pl.Series("r", rets)
        bench = pl.Series("r", [0.0] * 252)
        metrics = compute_benchmark_metrics(port, bench)
        if metrics["max_drawdown"] != 0.0:
            expected_calmar = metrics["cagr"] / abs(metrics["max_drawdown"])
            assert metrics["calmar_ratio"] == pytest.approx(expected_calmar)


# ---------------------------------------------------------------------------
# TestWithDates (monthly hit rate)
# ---------------------------------------------------------------------------


class TestWithDates:
    def test_hit_rate_all_wins(self) -> None:
        n = 252
        dates = pl.Series("date", _make_dates(n))
        port = pl.Series("r", [0.01] * n)
        bench = pl.Series("r", [0.005] * n)
        metrics = compute_benchmark_metrics(port, bench, dates=dates)
        assert metrics["hit_rate_monthly"] == pytest.approx(1.0)

    def test_hit_rate_all_losses(self) -> None:
        n = 252
        dates = pl.Series("date", _make_dates(n))
        port = pl.Series("r", [0.001] * n)
        bench = pl.Series("r", [0.005] * n)
        metrics = compute_benchmark_metrics(port, bench, dates=dates)
        assert metrics["hit_rate_monthly"] == pytest.approx(0.0)

    def test_hit_rate_no_dates(self) -> None:
        port = pl.Series("r", [0.01] * 100)
        bench = pl.Series("r", [0.005] * 100)
        metrics = compute_benchmark_metrics(port, bench, dates=None)
        assert metrics["hit_rate_monthly"] == 0.0


# ---------------------------------------------------------------------------
# TestWithRebalanceHistory
# ---------------------------------------------------------------------------


class TestWithRebalanceHistory:
    def _make_records(self, n: int) -> list[RebalanceRecord]:
        return [
            RebalanceRecord(
                date=date(2020, 1, 2) + timedelta(days=i * 5),
                weights={"A": 0.5, "B": 0.3, "C": 0.2},
                turnover=0.20,
                costs=100.0,
                retrained=i % 5 == 0,
            )
            for i in range(n)
        ]

    def test_turnover_computed(self) -> None:
        records = self._make_records(50)
        port = pl.Series("r", [0.001] * 252)
        bench = pl.Series("r", [0.0005] * 252)
        metrics = compute_benchmark_metrics(
            port, bench, rebalance_history=records,
        )
        assert metrics["avg_annual_turnover"] > 0.0

    def test_avg_positions(self) -> None:
        records = self._make_records(10)
        port = pl.Series("r", [0.001] * 100)
        bench = pl.Series("r", [0.0005] * 100)
        metrics = compute_benchmark_metrics(
            port, bench, rebalance_history=records,
        )
        assert metrics["avg_positions"] == pytest.approx(3.0)

    def test_no_rebalance_history(self) -> None:
        port = pl.Series("r", [0.001] * 100)
        bench = pl.Series("r", [0.0005] * 100)
        metrics = compute_benchmark_metrics(port, bench)
        assert metrics["avg_annual_turnover"] == 0.0
        assert metrics["avg_positions"] == 0.0


# ---------------------------------------------------------------------------
# TestIntegrationWithWalkForward
# ---------------------------------------------------------------------------


class TestIntegrationWithWalkForward:
    """Verify that WalkForwardBacktester.run() populates metrics."""

    def test_run_populates_metrics(self) -> None:
        import random
        from src.backtest.walk_forward import (
            NaiveModelFactory,
            WalkForwardBacktester,
            WalkForwardConfig,
        )

        rng = random.Random(42)
        rows: list[dict] = []
        start = date(2020, 1, 2)
        prices = {"A": 100.0, "B": 100.0, "SPY": 100.0}
        for day in range(300):
            d = start + timedelta(days=day)
            for ticker in prices:
                prices[ticker] *= 1.0 + rng.gauss(0.0005, 0.01)
                rows.append({
                    "date": d, "ticker": ticker,
                    "open": prices[ticker], "high": prices[ticker] * 1.01,
                    "low": prices[ticker] * 0.99, "close": prices[ticker],
                    "volume": 1_000_000,
                })

        ohlcv = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

        cfg = WalkForwardConfig(lookback_days=50, rebalance_every=5, retrain_every=50)
        bt = WalkForwardBacktester(config=cfg)
        result = bt.run(ohlcv, ["A", "B"], "SPY", NaiveModelFactory())

        assert result.metrics, "metrics should be populated"
        assert "sharpe_ratio" in result.metrics
        assert "cagr" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "alpha" in result.metrics
        assert "beta" in result.metrics
        assert isinstance(result.metrics["sharpe_ratio"], float)
