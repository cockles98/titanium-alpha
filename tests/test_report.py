"""Tests for src/backtest/report.py — BacktestReport PDF generation.

Covers initialization validation, PDF generation, individual plot methods,
and edge cases with empty or minimal data.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # Non-interactive backend before any pyplot import

from src.backtest.cpcv import BacktestResult, FoldResult
from src.backtest.report import BacktestReport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fold_result(
    path_id: int = 0,
    sharpe: float = 1.0,
    max_drawdown: float = -0.05,
    cagr: float = 0.10,
    n_trades: int = 5,
    total_costs: float = 0.005,
    equity_curve: list[float] | None = None,
) -> FoldResult:
    """Create a FoldResult with realistic defaults and an equity curve."""
    if equity_curve is None:
        equity_curve = [1.0, 1.01, 1.03, 1.02, 1.05, 1.04, 1.07]
    return FoldResult(
        path_id=path_id,
        test_groups=(path_id, path_id + 1),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        cagr=cagr,
        n_train=200,
        n_test=80,
        n_returns=len(equity_curve) - 1,
        n_trades=n_trades,
        total_costs=total_costs,
        equity_curve=equity_curve,
    )


def _make_backtest_result(n_folds: int = 5) -> BacktestResult:
    """Create a BacktestResult with multiple folds for report tests."""
    folds = [
        _make_fold_result(
            path_id=i,
            sharpe=0.5 + i * 0.3 - (0.8 if i == 2 else 0.0),
            max_drawdown=-0.03 - i * 0.02,
            cagr=0.08 + i * 0.01,
            n_trades=3 + i,
            total_costs=0.002 * (i + 1),
        )
        for i in range(n_folds)
    ]
    sharpes = [f.sharpe for f in folds]
    n = len(sharpes)
    mean_s = sum(sharpes) / n
    return BacktestResult(
        fold_results=folds,
        n_paths=n,
        mean_sharpe=mean_s,
        std_sharpe=0.5,
        pct_positive_sharpe=sum(1 for s in sharpes if s > 0) / n,
        mean_max_drawdown=sum(f.max_drawdown for f in folds) / n,
        mean_cagr=sum(f.cagr for f in folds) / n,
        mean_n_trades=sum(f.n_trades for f in folds) / n,
        mean_total_costs=sum(f.total_costs for f in folds) / n,
    )


# ---------------------------------------------------------------------------
# TestBacktestReportInit
# ---------------------------------------------------------------------------


class TestBacktestReportInit:
    """Tests for BacktestReport.__init__ validation."""

    def test_valid_init(self) -> None:
        """Valid result + ticker creates a BacktestReport without error."""
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="SPY")
        assert report.ticker == "SPY"
        assert report.result is result

    def test_empty_result_raises(self) -> None:
        """ValueError when fold_results is empty."""
        empty_result = BacktestResult(fold_results=[], n_paths=0)
        with pytest.raises(ValueError, match="no fold_results"):
            BacktestReport(empty_result, ticker="SPY")

    def test_default_output_dir(self) -> None:
        """Default output_dir is data/outputs."""
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="SPY")
        assert report.output_dir == Path("data/outputs")

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """Custom output_dir is stored as a Path."""
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="NVDA", output_dir=tmp_path)
        assert report.output_dir == tmp_path


# ---------------------------------------------------------------------------
# TestBacktestReportGenerate
# ---------------------------------------------------------------------------


class TestBacktestReportGenerate:
    """Tests for BacktestReport.generate() PDF creation."""

    def test_generates_pdf(self, tmp_path: Path) -> None:
        """generate() returns a Path to the created PDF."""
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path == tmp_path / "backtest_report_SPY.pdf"

    def test_pdf_file_exists(self, tmp_path: Path) -> None:
        """Generated PDF file exists and is non-empty."""
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="AAPL", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """PDF is created inside the specified custom directory."""
        custom_dir = tmp_path / "reports" / "backtest"
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="QQQ", output_dir=custom_dir)
        pdf_path = report.generate()
        assert pdf_path.parent == custom_dir
        assert pdf_path.exists()

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """generate() creates output_dir if it does not exist."""
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        assert not nested_dir.exists()
        result = _make_backtest_result()
        report = BacktestReport(result, ticker="NVDA", output_dir=nested_dir)
        report.generate()
        assert nested_dir.exists()

    def test_single_fold_report(self, tmp_path: Path) -> None:
        """Report generation works with a single fold."""
        fold = _make_fold_result(path_id=0, sharpe=1.2)
        result = BacktestResult(
            fold_results=[fold],
            n_paths=1,
            mean_sharpe=1.2,
            std_sharpe=0.0,
            pct_positive_sharpe=1.0,
            mean_max_drawdown=-0.05,
            mean_cagr=0.10,
            mean_n_trades=5.0,
            mean_total_costs=0.005,
        )
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_many_folds_report(self, tmp_path: Path) -> None:
        """Report generation handles many folds (> 10 paths, rotated labels)."""
        result = _make_backtest_result(n_folds=15)
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()


# ---------------------------------------------------------------------------
# TestPlotMethods
# ---------------------------------------------------------------------------


class TestPlotMethods:
    """Tests for individual plot rendering methods."""

    def test_equity_curves_no_data(self, tmp_path: Path) -> None:
        """Handles folds with empty or single-point equity curves gracefully."""
        fold = FoldResult(
            path_id=0,
            test_groups=(0,),
            sharpe=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            n_train=100,
            n_test=10,
            n_returns=0,
            n_trades=0,
            total_costs=0.0,
            equity_curve=[1.0],  # single point, no curve to plot
        )
        result = BacktestResult(
            fold_results=[fold],
            n_paths=1,
            mean_sharpe=0.0,
            std_sharpe=0.0,
            pct_positive_sharpe=0.0,
            mean_max_drawdown=0.0,
            mean_cagr=0.0,
            mean_n_trades=0.0,
            mean_total_costs=0.0,
        )
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        # Should not raise even with minimal equity curve data
        pdf_path = report.generate()
        assert pdf_path.exists()

    def test_sharpe_distribution(self, tmp_path: Path) -> None:
        """Sharpe distribution plot works with mixed positive/negative values."""
        folds = [
            _make_fold_result(path_id=i, sharpe=val)
            for i, val in enumerate([-2.0, -0.5, 0.0, 0.8, 1.5, 2.3])
        ]
        result = BacktestResult(
            fold_results=folds,
            n_paths=6,
            mean_sharpe=sum(f.sharpe for f in folds) / 6,
            std_sharpe=1.0,
            pct_positive_sharpe=3 / 6,
            mean_max_drawdown=-0.05,
            mean_cagr=0.08,
            mean_n_trades=5.0,
            mean_total_costs=0.005,
        )
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_drawdown_bars(self, tmp_path: Path) -> None:
        """Drawdown bar chart renders correctly with severe and mild drawdowns."""
        folds = [
            _make_fold_result(path_id=i, max_drawdown=dd)
            for i, dd in enumerate([-0.02, -0.08, -0.15, -0.25, -0.05])
        ]
        result = BacktestResult(
            fold_results=folds,
            n_paths=5,
            mean_sharpe=1.0,
            std_sharpe=0.3,
            pct_positive_sharpe=1.0,
            mean_max_drawdown=sum(f.max_drawdown for f in folds) / 5,
            mean_cagr=0.10,
            mean_n_trades=4.0,
            mean_total_costs=0.003,
        )
        report = BacktestReport(result, ticker="NVDA", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_metrics_table(self, tmp_path: Path) -> None:
        """Metrics table renders without error for standard input."""
        result = _make_backtest_result(n_folds=3)
        report = BacktestReport(result, ticker="AAPL", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_few_sharpes_no_violin(self, tmp_path: Path) -> None:
        """With < 4 folds, violin plot is skipped but strip plot works."""
        folds = [_make_fold_result(path_id=i, sharpe=0.5 * i) for i in range(3)]
        result = BacktestResult(
            fold_results=folds,
            n_paths=3,
            mean_sharpe=0.5,
            std_sharpe=0.25,
            pct_positive_sharpe=2 / 3,
            mean_max_drawdown=-0.05,
            mean_cagr=0.08,
            mean_n_trades=5.0,
            mean_total_costs=0.005,
        )
        report = BacktestReport(result, ticker="SPY", output_dir=tmp_path)
        pdf_path = report.generate()
        assert pdf_path.exists()
