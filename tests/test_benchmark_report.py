"""Tests for src.backtest.benchmark_report."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.backtest.benchmark_report import (
    BenchmarkReport,
    _compute_drawdown,
    _format_metrics_rows,
    _rolling_sharpe,
)
from src.backtest.walk_forward import RebalanceRecord, WalkForwardConfig, WalkForwardResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    n_days: int = 300,
    n_rebalances: int = 10,
    n_tickers: int = 3,
) -> WalkForwardResult:
    """Create a realistic WalkForwardResult for testing."""
    import random

    rng = random.Random(42)
    start = date(2020, 1, 2)
    dates: list[date] = []
    portfolio_values: list[float] = []
    benchmark_values: list[float] = []
    port_returns: list[float] = []
    bench_returns: list[float] = []

    port_val = 1_000_000.0
    bench_val = 1_000_000.0

    for day in range(n_days):
        d = start + timedelta(days=day)
        dates.append(d)

        p_ret = rng.gauss(0.0005, 0.015)
        b_ret = rng.gauss(0.0003, 0.012)

        port_val *= (1 + p_ret)
        bench_val *= (1 + b_ret)

        portfolio_values.append(port_val)
        benchmark_values.append(bench_val)
        port_returns.append(p_ret)
        bench_returns.append(b_ret)

    equity_curve = pl.DataFrame({
        "date": dates,
        "portfolio_value": portfolio_values,
        "benchmark_value": benchmark_values,
    })

    daily_returns = pl.DataFrame({
        "date": dates,
        "portfolio_return": port_returns,
        "benchmark_return": bench_returns,
    })

    tickers = [f"T{i}" for i in range(n_tickers)]
    rebalance_history: list[RebalanceRecord] = []
    step = max(1, n_days // n_rebalances)
    for i in range(n_rebalances):
        w: dict[str, float] = {}
        remaining = 1.0
        for j, t in enumerate(tickers):
            if j == len(tickers) - 1:
                w[t] = remaining
            else:
                share = rng.uniform(0.05, remaining / (len(tickers) - j))
                w[t] = share
                remaining -= share
        rebalance_history.append(RebalanceRecord(
            date=start + timedelta(days=i * step),
            weights=w,
            turnover=rng.uniform(0.05, 0.5),
            costs=rng.uniform(100, 5000),
            retrained=i % 3 == 0,
        ))

    metrics = {
        "cagr": 0.12,
        "total_return": 0.35,
        "benchmark_total_return": 0.20,
        "annualized_volatility": 0.18,
        "max_drawdown": -0.15,
        "max_drawdown_duration_days": 45.0,
        "calmar_ratio": 0.80,
        "sharpe_ratio": 0.85,
        "sortino_ratio": 1.20,
        "information_ratio": 0.45,
        "alpha": 0.03,
        "beta": 0.92,
        "tracking_error": 0.08,
        "hit_rate_monthly": 0.60,
        "avg_annual_turnover": 5.2,
        "avg_positions": 3.0,
    }

    return WalkForwardResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        rebalance_history=rebalance_history,
        metrics=metrics,
        config=WalkForwardConfig(),
        metadata={"n_tickers": n_tickers},
    )


# ---------------------------------------------------------------------------
# TestComputeDrawdown
# ---------------------------------------------------------------------------


class TestComputeDrawdown:
    def test_empty(self) -> None:
        assert _compute_drawdown([]) == []

    def test_monotonic_increase(self) -> None:
        dd = _compute_drawdown([100, 110, 120, 130])
        assert all(d == 0.0 for d in dd)

    def test_drawdown_values(self) -> None:
        dd = _compute_drawdown([100, 110, 100, 90, 120])
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert dd[2] == pytest.approx((100 - 110) / 110)
        assert dd[3] == pytest.approx((90 - 110) / 110)
        assert dd[4] == 0.0  # new peak


# ---------------------------------------------------------------------------
# TestRollingSharpe
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def test_output_length(self) -> None:
        rets = [0.001] * 100
        result = _rolling_sharpe(rets, window=20, rf=0.05, trading_days=252)
        assert len(result) == 100 - 20 + 1

    def test_constant_returns_zero_sharpe(self) -> None:
        # Constant returns → zero std → zero Sharpe
        rets = [0.001] * 50
        result = _rolling_sharpe(rets, window=20, rf=0.0, trading_days=252)
        assert all(s == 0.0 for s in result)

    def test_positive_returns_positive_sharpe(self) -> None:
        import random
        rng = random.Random(42)
        rets = [rng.gauss(0.002, 0.01) for _ in range(100)]
        result = _rolling_sharpe(rets, window=50, rf=0.0, trading_days=252)
        # Most windows should have positive Sharpe with positive mean
        positive_count = sum(1 for s in result if s > 0)
        assert positive_count > len(result) * 0.5


# ---------------------------------------------------------------------------
# TestFormatMetricsRows
# ---------------------------------------------------------------------------


class TestFormatMetricsRows:
    def test_returns_all_16_rows(self) -> None:
        metrics = {k: 0.0 for k in [
            "cagr", "total_return", "benchmark_total_return",
            "annualized_volatility", "max_drawdown", "max_drawdown_duration_days",
            "calmar_ratio", "sharpe_ratio", "sortino_ratio",
            "information_ratio", "alpha", "beta", "tracking_error",
            "hit_rate_monthly", "avg_annual_turnover", "avg_positions",
        ]}
        rows = _format_metrics_rows(metrics)
        assert len(rows) == 16

    def test_formatting(self) -> None:
        metrics = {"cagr": 0.1234, "sharpe_ratio": 1.567}
        rows = _format_metrics_rows(metrics)
        cagr_row = [r for r in rows if r[0] == "CAGR"][0]
        assert cagr_row[1] == "12.34%"
        sharpe_row = [r for r in rows if r[0] == "Sharpe Ratio"][0]
        assert sharpe_row[1] == "1.567"

    def test_categories_assigned(self) -> None:
        metrics = {"cagr": 0.1, "max_drawdown": -0.15, "beta": 1.0}
        rows = _format_metrics_rows(metrics)
        categories = {r[2] for r in rows}
        assert "Return" in categories
        assert "Risk" in categories
        assert "vs Benchmark" in categories


# ---------------------------------------------------------------------------
# TestBenchmarkReportInit
# ---------------------------------------------------------------------------


class TestBenchmarkReportInit:
    def test_valid_init(self) -> None:
        result = _make_result()
        report = BenchmarkReport(result)
        assert report.benchmark_name == "S&P 500 (SPY)"
        assert report.output_dir == Path("data/outputs")

    def test_custom_name_and_dir(self) -> None:
        result = _make_result()
        report = BenchmarkReport(result, benchmark_name="IBOV", output_dir="/tmp/reports")
        assert report.benchmark_name == "IBOV"
        assert report.output_dir == Path("/tmp/reports")

    def test_insufficient_data_raises(self) -> None:
        result = WalkForwardResult(
            equity_curve=pl.DataFrame({"date": [date(2020, 1, 1)],
                                       "portfolio_value": [100.0],
                                       "benchmark_value": [100.0]}),
            daily_returns=pl.DataFrame({"date": [date(2020, 1, 1)],
                                        "portfolio_return": [0.0],
                                        "benchmark_return": [0.0]}),
            rebalance_history=[],
        )
        with pytest.raises(ValueError, match="insufficient data"):
            BenchmarkReport(result)


# ---------------------------------------------------------------------------
# TestBenchmarkReportGenerate
# ---------------------------------------------------------------------------


class TestBenchmarkReportGenerate:
    def test_generates_pdf(self, tmp_path: Path) -> None:
        result = _make_result()
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        assert pdf_path.stat().st_size > 0

    def test_pdf_filename(self, tmp_path: Path) -> None:
        result = _make_result()
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.name == "benchmark_report.pdf"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "subdir" / "reports"
        result = _make_result()
        report = BenchmarkReport(result, output_dir=out)
        pdf_path = report.generate()
        assert out.exists()
        assert pdf_path.exists()

    def test_no_metrics_still_generates(self, tmp_path: Path) -> None:
        result = _make_result()
        result.metrics = {}
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()

    def test_few_rebalances_still_generates(self, tmp_path: Path) -> None:
        result = _make_result(n_rebalances=1)
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()

    def test_short_data_no_rolling_sharpe(self, tmp_path: Path) -> None:
        """Data shorter than rolling window still generates PDF."""
        result = _make_result(n_days=50)
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()

    def test_many_tickers_heatmap(self, tmp_path: Path) -> None:
        """Report with 20+ tickers shows top 15 in heatmap."""
        result = _make_result(n_tickers=20, n_rebalances=15)
        report = BenchmarkReport(result, output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
