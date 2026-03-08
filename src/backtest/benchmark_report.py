"""Walk-forward benchmark report generator with PDF output.

Produces a multi-page PDF comparing portfolio performance against
a buy-and-hold benchmark.

Pages:

1. **Equity curve**: portfolio vs benchmark (log scale).
2. **Drawdown chart**: filled area showing drawdown over time.
3. **Metrics table**: all metrics from ``compute_benchmark_metrics``.
4. **Rolling Sharpe**: 252-day rolling Sharpe for portfolio vs benchmark.
5. **Weight heatmap**: top 15 assets by average weight over time.
6. **Turnover chart**: bar chart of turnover per rebalance event.

Usage::

    from src.backtest.benchmark_report import BenchmarkReport

    report = BenchmarkReport(result, benchmark_name="S&P 500 (SPY)")
    pdf_path = report.generate()
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from src.backtest.walk_forward import RebalanceRecord, WalkForwardResult

matplotlib.use("Agg")  # Non-interactive backend for PDF generation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("data/outputs")
_FIG_WIDTH = 11.0
_FIG_HEIGHT = 8.5
_ROLLING_WINDOW = 252
_TOP_N_ASSETS = 15


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class BenchmarkReport:
    """Generates a multi-page PDF benchmark report.

    Args:
        result: Walk-forward backtest result.
        benchmark_name: Display name for the benchmark (e.g. "S&P 500").
        output_dir: Directory to save the PDF.
    """

    def __init__(
        self,
        result: WalkForwardResult,
        benchmark_name: str = "S&P 500 (SPY)",
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        if result.equity_curve.height < 2:
            raise ValueError("WalkForwardResult has insufficient data for report")

        self.result = result
        self.benchmark_name = benchmark_name
        self.output_dir = Path(output_dir)

    def generate(self) -> Path:
        """Generate the full PDF report.

        Returns:
            Path to the saved PDF file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.output_dir / "benchmark_report.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            pages = [
                self._page_equity_curve,
                self._page_drawdown,
                self._page_metrics_table,
                self._page_rolling_sharpe,
                self._page_weight_heatmap,
                self._page_turnover,
            ]
            for builder in pages:
                fig = builder()
                pdf.savefig(fig)
                plt.close(fig)

        logger.info("Benchmark report saved to {}", pdf_path)
        return pdf_path

    # ------------------------------------------------------------------
    # Page 1: Equity Curve
    # ------------------------------------------------------------------

    def _page_equity_curve(self) -> matplotlib.figure.Figure:
        """Portfolio vs benchmark equity curve (log scale)."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

        dates = self.result.equity_curve["date"].to_list()
        portfolio = self.result.equity_curve["portfolio_value"].to_list()
        benchmark = self.result.equity_curve["benchmark_value"].to_list()

        ax.plot(dates, portfolio, label="Portfolio", color="#2196F3", linewidth=1.5)
        ax.plot(dates, benchmark, label=self.benchmark_name, color="#FF9800", linewidth=1.5, linestyle="--")

        ax.set_yscale("log")
        ax.set_title("Equity Curve — Portfolio vs Benchmark", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (log scale)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Page 2: Drawdown
    # ------------------------------------------------------------------

    def _page_drawdown(self) -> matplotlib.figure.Figure:
        """Drawdown chart (filled area)."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

        portfolio = self.result.equity_curve["portfolio_value"].to_list()
        dates = self.result.equity_curve["date"].to_list()

        # Compute drawdown series
        dd = _compute_drawdown(portfolio)

        ax.fill_between(dates, dd, 0, color="#F44336", alpha=0.4, label="Drawdown")
        ax.plot(dates, dd, color="#D32F2F", linewidth=0.8)

        max_dd = min(dd) if dd else 0.0
        ax.set_title(
            f"Portfolio Drawdown  |  Max DD: {max_dd:.2%}",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Page 3: Metrics Table
    # ------------------------------------------------------------------

    def _page_metrics_table(self) -> matplotlib.figure.Figure:
        """Full metrics summary table."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
        ax.axis("off")

        metrics = self.result.metrics
        if not metrics:
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        # Format metrics for display
        rows = _format_metrics_rows(metrics)

        table = ax.table(
            cellText=[[r[1], r[2]] for r in rows],
            rowLabels=[r[0] for r in rows],
            colLabels=["Value", "Category"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)

        # Style header
        for j in range(2):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(rows) + 1):
            color = "#F2F7FB" if i % 2 == 0 else "white"
            table[i, -1].set_facecolor(color)  # row label
            for j in range(2):
                table[i, j].set_facecolor(color)

        ax.set_title(
            "Performance Metrics — Portfolio vs Benchmark",
            fontsize=14, fontweight="bold", pad=20,
        )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Page 4: Rolling Sharpe
    # ------------------------------------------------------------------

    def _page_rolling_sharpe(self) -> matplotlib.figure.Figure:
        """Rolling 252-day Sharpe ratio for portfolio and benchmark."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

        port_ret = self.result.daily_returns["portfolio_return"].to_list()
        bench_ret = self.result.daily_returns["benchmark_return"].to_list()
        dates = self.result.daily_returns["date"].to_list()
        rf = self.result.config.rf
        trading_days = self.result.config.trading_days_per_year

        if len(port_ret) < _ROLLING_WINDOW:
            ax.text(0.5, 0.5, f"Insufficient data for {_ROLLING_WINDOW}-day rolling Sharpe",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Rolling Sharpe Ratio", fontsize=14, fontweight="bold")
            fig.tight_layout()
            return fig

        port_sharpe = _rolling_sharpe(port_ret, _ROLLING_WINDOW, rf, trading_days)
        bench_sharpe = _rolling_sharpe(bench_ret, _ROLLING_WINDOW, rf, trading_days)
        rolling_dates = dates[_ROLLING_WINDOW - 1:]

        ax.plot(rolling_dates, port_sharpe, label="Portfolio", color="#2196F3", linewidth=1.2)
        ax.plot(rolling_dates, bench_sharpe, label=self.benchmark_name, color="#FF9800",
                linewidth=1.2, linestyle="--")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(y=1.0, color="green", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_title(f"Rolling {_ROLLING_WINDOW}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Page 5: Weight Heatmap
    # ------------------------------------------------------------------

    def _page_weight_heatmap(self) -> matplotlib.figure.Figure:
        """Heatmap of top N asset weights over rebalances."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

        history = self.result.rebalance_history
        if len(history) < 2:
            ax.text(0.5, 0.5, "Insufficient rebalance history for heatmap",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Portfolio Weight Evolution", fontsize=14, fontweight="bold")
            fig.tight_layout()
            return fig

        # Collect all tickers and compute average weight
        all_tickers: set[str] = set()
        for rec in history:
            all_tickers.update(rec.weights.keys())

        avg_weights: dict[str, float] = {}
        for t in all_tickers:
            avg_weights[t] = sum(r.weights.get(t, 0.0) for r in history) / len(history)

        # Top N by average weight
        top_tickers = sorted(avg_weights, key=lambda t: avg_weights[t], reverse=True)[:_TOP_N_ASSETS]

        # Build matrix: rows=tickers, cols=rebalance dates
        rebal_dates = [r.date for r in history]
        matrix = np.zeros((len(top_tickers), len(history)))
        for j, rec in enumerate(history):
            for i, t in enumerate(top_tickers):
                matrix[i, j] = rec.weights.get(t, 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0)
        ax.set_yticks(range(len(top_tickers)))
        ax.set_yticklabels(top_tickers, fontsize=8)

        # Show subset of x-tick labels to avoid clutter
        n_labels = min(10, len(rebal_dates))
        step = max(1, len(rebal_dates) // n_labels)
        ax.set_xticks(range(0, len(rebal_dates), step))
        ax.set_xticklabels(
            [str(rebal_dates[i]) for i in range(0, len(rebal_dates), step)],
            rotation=45, ha="right", fontsize=7,
        )

        fig.colorbar(im, ax=ax, label="Weight", shrink=0.8)
        ax.set_title(
            f"Top {len(top_tickers)} Assets — Weight Evolution",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel("Rebalance Date")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Page 6: Turnover
    # ------------------------------------------------------------------

    def _page_turnover(self) -> matplotlib.figure.Figure:
        """Bar chart of turnover per rebalance event."""
        fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

        history = self.result.rebalance_history
        if not history:
            ax.text(0.5, 0.5, "No rebalance history available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("Turnover per Rebalance", fontsize=14, fontweight="bold")
            fig.tight_layout()
            return fig

        rebal_dates = [r.date for r in history]
        turnovers = [r.turnover for r in history]

        # Color bars: retrained = darker
        colors = ["#1565C0" if r.retrained else "#90CAF9" for r in history]

        ax.bar(range(len(turnovers)), turnovers, color=colors, width=0.8)

        # X-axis labels (subset)
        n_labels = min(15, len(rebal_dates))
        step = max(1, len(rebal_dates) // n_labels)
        ax.set_xticks(range(0, len(rebal_dates), step))
        ax.set_xticklabels(
            [str(rebal_dates[i]) for i in range(0, len(rebal_dates), step)],
            rotation=45, ha="right", fontsize=7,
        )

        avg_turnover = sum(turnovers) / len(turnovers) if turnovers else 0.0
        ax.axhline(y=avg_turnover, color="red", linewidth=1, linestyle="--",
                   label=f"Average: {avg_turnover:.3f}")

        ax.set_title("Turnover per Rebalance Event", fontsize=14, fontweight="bold")
        ax.set_xlabel("Rebalance Event")
        ax.set_ylabel("Turnover (sum |Δw|)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_drawdown(values: list[float]) -> list[float]:
    """Compute drawdown series from portfolio values."""
    if not values:
        return []
    peak = values[0]
    dd: list[float] = []
    for v in values:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak != 0 else 0.0)
    return dd


def _rolling_sharpe(
    returns: list[float],
    window: int,
    rf: float,
    trading_days: int,
) -> list[float]:
    """Compute rolling annualised Sharpe ratio.

    Args:
        returns: Daily simple returns.
        window: Rolling window size.
        rf: Annualised risk-free rate.
        trading_days: Trading days per year.

    Returns:
        List of rolling Sharpe values (length = len(returns) - window + 1).
    """
    rf_daily = rf / trading_days
    result: list[float] = []
    for i in range(window - 1, len(returns)):
        chunk = returns[i - window + 1: i + 1]
        excess = [r - rf_daily for r in chunk]
        mean_e = sum(excess) / len(excess)
        var_e = sum((e - mean_e) ** 2 for e in excess) / (len(excess) - 1)
        std_e = math.sqrt(var_e) if var_e > 0 else 0.0
        sharpe = (mean_e / std_e) * math.sqrt(trading_days) if std_e > 0 else 0.0
        result.append(sharpe)
    return result


def _format_metrics_rows(metrics: dict[str, float]) -> list[tuple[str, str, str]]:
    """Format metrics dict into display rows: (label, value, category).

    Returns:
        List of (label, formatted_value, category) tuples.
    """
    formatters: list[tuple[str, str, str, str]] = [
        # (key, label, format, category)
        ("cagr", "CAGR", "{:.2%}", "Return"),
        ("total_return", "Total Return", "{:.2%}", "Return"),
        ("benchmark_total_return", "Benchmark Return", "{:.2%}", "Return"),
        ("annualized_volatility", "Annualized Volatility", "{:.2%}", "Risk"),
        ("max_drawdown", "Max Drawdown", "{:.2%}", "Risk"),
        ("max_drawdown_duration_days", "Max DD Duration (days)", "{:.0f}", "Risk"),
        ("calmar_ratio", "Calmar Ratio", "{:.3f}", "Risk"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.3f}", "Risk-Adjusted"),
        ("sortino_ratio", "Sortino Ratio", "{:.3f}", "Risk-Adjusted"),
        ("information_ratio", "Information Ratio", "{:.3f}", "vs Benchmark"),
        ("alpha", "Alpha (annualized)", "{:.4f}", "vs Benchmark"),
        ("beta", "Beta", "{:.3f}", "vs Benchmark"),
        ("tracking_error", "Tracking Error", "{:.2%}", "vs Benchmark"),
        ("hit_rate_monthly", "Monthly Hit Rate", "{:.1%}", "vs Benchmark"),
        ("avg_annual_turnover", "Avg Annual Turnover", "{:.2f}", "Turnover"),
        ("avg_positions", "Avg Positions", "{:.1f}", "Turnover"),
    ]

    rows: list[tuple[str, str, str]] = []
    for key, label, fmt, category in formatters:
        val = metrics.get(key, 0.0)
        rows.append((label, fmt.format(val), category))
    return rows
