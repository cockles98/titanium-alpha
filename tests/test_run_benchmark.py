"""Tests for src.backtest.run_benchmark."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.backtest.run_benchmark import (
    _filter_oos_period,
    _resolve_model_factory,
    _save_outputs,
    run_us_benchmark,
)
from src.backtest.walk_forward import (
    NaiveModelFactory,
    RebalanceRecord,
    WalkForwardConfig,
    WalkForwardResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    tickers: list[str],
    n_days: int = 600,
    start: date = date(2018, 1, 2),
) -> pl.DataFrame:
    """Generate synthetic OHLCV data with noise."""
    rng = random.Random(42)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        price = 100.0
        for day in range(n_days):
            d = start + timedelta(days=day)
            price *= 1.0 + rng.gauss(0.0003, 0.015)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * (1 + rng.gauss(0, 0.002)),
                "high": price * (1 + abs(rng.gauss(0, 0.005))),
                "low": price * (1 - abs(rng.gauss(0, 0.005))),
                "close": price,
                "volume": int(1_000_000 * (1 + rng.gauss(0, 0.3))),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _make_result(n_days: int = 100) -> WalkForwardResult:
    """Create a minimal WalkForwardResult for save tests."""
    rng = random.Random(42)
    dates = [date(2020, 1, 2) + timedelta(days=i) for i in range(n_days)]
    port_val = 1_000_000.0
    bench_val = 1_000_000.0
    equity_dates: list[date] = []
    port_vals: list[float] = []
    bench_vals: list[float] = []
    port_rets: list[float] = []
    bench_rets: list[float] = []

    for d in dates:
        p_ret = rng.gauss(0.0005, 0.01)
        b_ret = rng.gauss(0.0003, 0.008)
        port_val *= (1 + p_ret)
        bench_val *= (1 + b_ret)
        equity_dates.append(d)
        port_vals.append(port_val)
        bench_vals.append(bench_val)
        port_rets.append(p_ret)
        bench_rets.append(b_ret)

    rebalance_history = [
        RebalanceRecord(
            date=dates[0],
            weights={"A": 0.5, "B": 0.3, "C": 0.2},
            turnover=1.0,
            costs=150.0,
            retrained=True,
        ),
        RebalanceRecord(
            date=dates[5],
            weights={"A": 0.4, "B": 0.35, "C": 0.25},
            turnover=0.2,
            costs=30.0,
            retrained=False,
        ),
    ]

    return WalkForwardResult(
        equity_curve=pl.DataFrame({
            "date": equity_dates,
            "portfolio_value": port_vals,
            "benchmark_value": bench_vals,
        }),
        daily_returns=pl.DataFrame({
            "date": equity_dates,
            "portfolio_return": port_rets,
            "benchmark_return": bench_rets,
        }),
        rebalance_history=rebalance_history,
        metrics={"sharpe_ratio": 0.85, "cagr": 0.12, "max_drawdown": -0.10},
        config=WalkForwardConfig(),
        metadata={"n_tickers": 3},
    )


# ---------------------------------------------------------------------------
# TestFilterOOSPeriod
# ---------------------------------------------------------------------------


class TestFilterOOSPeriod:
    def test_filters_to_n_years(self) -> None:
        df = _make_ohlcv(["A"], n_days=800, start=date(2015, 1, 1))
        filtered = _filter_oos_period(df, n_years=1)
        max_date = df["date"].max()
        min_filtered = filtered["date"].min()
        assert (max_date - min_filtered).days <= 366  # ~1 year

    def test_keeps_all_if_short(self) -> None:
        df = _make_ohlcv(["A"], n_days=100)
        filtered = _filter_oos_period(df, n_years=10)
        assert filtered.height == df.height

    def test_returns_polars_dataframe(self) -> None:
        df = _make_ohlcv(["A"], n_days=200)
        filtered = _filter_oos_period(df, n_years=1)
        assert isinstance(filtered, pl.DataFrame)


# ---------------------------------------------------------------------------
# TestResolveModelFactory
# ---------------------------------------------------------------------------


class TestResolveModelFactory:
    def test_naive_mode(self) -> None:
        factory = _resolve_model_factory(use_patchtst=False)
        assert isinstance(factory, NaiveModelFactory)

    def test_patchtst_fallback_when_not_installed(self) -> None:
        """When PatchTST is not available, falls back to Naive."""
        with patch.dict("sys.modules", {"src.models.patchtst_model": None}):
            factory = _resolve_model_factory(use_patchtst=True)
            assert isinstance(factory, NaiveModelFactory)


# ---------------------------------------------------------------------------
# TestSaveOutputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def test_saves_equity_parquet(self, tmp_path: Path) -> None:
        result = _make_result()
        paths = _save_outputs(result, tmp_path)
        assert "equity" in paths
        assert paths["equity"].exists()
        df = pl.read_parquet(paths["equity"])
        assert "portfolio_value" in df.columns

    def test_saves_metrics_json(self, tmp_path: Path) -> None:
        result = _make_result()
        paths = _save_outputs(result, tmp_path)
        assert "metrics" in paths
        with open(paths["metrics"]) as f:
            metrics = json.load(f)
        assert "sharpe_ratio" in metrics

    def test_saves_weights_parquet(self, tmp_path: Path) -> None:
        result = _make_result()
        paths = _save_outputs(result, tmp_path)
        assert "weights" in paths
        df = pl.read_parquet(paths["weights"])
        assert "ticker" in df.columns
        assert "weight" in df.columns
        assert "turnover" in df.columns

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir"
        result = _make_result()
        paths = _save_outputs(result, out)
        assert out.exists()
        assert all(p.exists() for p in paths.values())

    def test_no_rebalance_skips_weights(self, tmp_path: Path) -> None:
        result = _make_result()
        result.rebalance_history = []
        paths = _save_outputs(result, tmp_path)
        assert "weights" not in paths


# ---------------------------------------------------------------------------
# TestRunUSBenchmark (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunUSBenchmark:
    def test_end_to_end_with_naive(self, tmp_path: Path) -> None:
        """Full pipeline with pre-loaded OHLCV and NaiveModelFactory."""
        ohlcv = _make_ohlcv(["A", "B", "SPY"], n_days=800)

        config = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["A", "B"],
            "trading_days_per_year": 252,
            "risk_free_rate": 0.05,
        }

        with patch("src.backtest.run_benchmark.load_ticker_config", return_value=config):
            result = run_us_benchmark(
                output_dir=str(tmp_path),
                use_patchtst=False,
                n_years=2,
                ohlcv=ohlcv,
            )

        assert isinstance(result, WalkForwardResult)
        assert result.metrics, "Metrics should be populated"
        assert "sharpe_ratio" in result.metrics

        # Check output files
        assert (tmp_path / "benchmark_equity.parquet").exists()
        assert (tmp_path / "benchmark_metrics.json").exists()
        assert (tmp_path / "benchmark_report.pdf").exists()

    def test_metrics_json_valid(self, tmp_path: Path) -> None:
        """Metrics JSON is valid and contains expected keys."""
        ohlcv = _make_ohlcv(["X", "Y", "SPY"], n_days=800)

        config = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["X", "Y"],
            "trading_days_per_year": 252,
            "risk_free_rate": 0.05,
        }

        with patch("src.backtest.run_benchmark.load_ticker_config", return_value=config):
            run_us_benchmark(
                output_dir=str(tmp_path),
                use_patchtst=False,
                n_years=2,
                ohlcv=ohlcv,
            )

        with open(tmp_path / "benchmark_metrics.json") as f:
            metrics = json.load(f)

        assert "sharpe_ratio" in metrics
        assert "cagr" in metrics
        assert "max_drawdown" in metrics
        assert "alpha" in metrics

    def test_config_from_file(self, tmp_path: Path) -> None:
        """Config is loaded from the specified path."""
        ohlcv = _make_ohlcv(["A", "SPY"], n_days=800)

        config = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["A"],
            "trading_days_per_year": 252,
            "risk_free_rate": 0.03,
        }

        with patch("src.backtest.run_benchmark.load_ticker_config", return_value=config) as mock_load:
            run_us_benchmark(
                config_path="custom/path.json",
                output_dir=str(tmp_path),
                use_patchtst=False,
                n_years=2,
                ohlcv=ohlcv,
            )
            mock_load.assert_called_once_with("custom/path.json")

    def test_weights_parquet_has_rebalances(self, tmp_path: Path) -> None:
        """Weights parquet contains rebalance history."""
        ohlcv = _make_ohlcv(["A", "B", "SPY"], n_days=800)

        config = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["A", "B"],
            "trading_days_per_year": 252,
            "risk_free_rate": 0.05,
        }

        with patch("src.backtest.run_benchmark.load_ticker_config", return_value=config):
            run_us_benchmark(
                output_dir=str(tmp_path),
                use_patchtst=False,
                n_years=2,
                ohlcv=ohlcv,
            )

        weights_path = tmp_path / "benchmark_weights.parquet"
        assert weights_path.exists()
        df = pl.read_parquet(weights_path)
        assert df.height > 0
        assert set(df.columns) >= {"date", "ticker", "weight", "turnover", "costs", "retrained"}

    def test_pdf_report_failure_non_fatal(self, tmp_path: Path) -> None:
        """If PDF generation fails, pipeline still completes."""
        ohlcv = _make_ohlcv(["A", "SPY"], n_days=800)

        config = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["A"],
            "trading_days_per_year": 252,
            "risk_free_rate": 0.05,
        }

        with (
            patch("src.backtest.run_benchmark.load_ticker_config", return_value=config),
            patch("src.backtest.run_benchmark.BenchmarkReport") as mock_report,
        ):
            mock_report.side_effect = RuntimeError("PDF failed")
            result = run_us_benchmark(
                output_dir=str(tmp_path),
                use_patchtst=False,
                n_years=2,
                ohlcv=ohlcv,
            )

        assert isinstance(result, WalkForwardResult)
        assert (tmp_path / "benchmark_equity.parquet").exists()
        assert (tmp_path / "benchmark_metrics.json").exists()
