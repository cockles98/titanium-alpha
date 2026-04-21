"""Backtest report generator with PDF output.

Produces a multi-page PDF report from :class:`BacktestResult` containing:

- **Page 1**: summary metrics table and equity curves overlay.
- **Page 2**: Sharpe ratio violin/strip plot and max-drawdown bar chart.

Usage::

    from src.backtest.cpcv import CPCVBacktester, TransactionCosts
    from src.backtest.report import BacktestReport

    result = backtester.run(df, model_factory)
    report = BacktestReport(result, ticker="SPY")
    pdf_path = report.generate()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from src.backtest.cpcv import BacktestResult

matplotlib.use("Agg")  # Non-interactive backend for PDF generation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("data/outputs")
_FIG_WIDTH = 11.0
_FIG_HEIGHT = 8.5


# ---------------------------------------------------------------------------
# BacktestReport
# ---------------------------------------------------------------------------


class BacktestReport:
    """Generates a PDF backtest report from CPCV results.

    Args:
        result: Aggregated backtest result from ``CPCVBacktester.run()``.
        ticker: Ticker symbol for the report title.
        output_dir: Directory to save the PDF.  Defaults to
            ``data/outputs/``.
    """

    def __init__(
        self,
        result: BacktestResult,
        ticker: str,
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        if not result.fold_results:
            raise ValueError("BacktestResult has no fold_results to report")

        self.result = result
        self.ticker = ticker
        self.output_dir = Path(output_dir)

    def generate(self) -> Path:
        """Generate the full PDF report.

        Returns:
            Path to the saved PDF file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.output_dir / f"backtest_report_{self.ticker}.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            # Page 1: metrics table + equity curves
            fig1 = self._page_metrics_and_equity()
            pdf.savefig(fig1)
            plt.close(fig1)

            # Page 2: Sharpe distribution + drawdown bars
            fig2 = self._page_sharpe_and_drawdown()
            pdf.savefig(fig2)
            plt.close(fig2)

        logger.info("Backtest report saved to {}", pdf_path)
        return pdf_path

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _page_metrics_and_equity(self) -> matplotlib.figure.Figure:
        """Build page 1: summary metrics table (top) + equity curves (bottom)."""
        fig, axes = plt.subplots(
            2, 1, figsize=(_FIG_WIDTH, _FIG_HEIGHT),
            gridspec_kw={"height_ratios": [1, 2]},
        )
        fig.suptitle(
            f"CPCV Backtest Report — {self.ticker}",
            fontsize=14, fontweight="bold", y=0.98,
        )

        self._render_metrics_table(axes[0])
        self._plot_equity_curves(axes[1])

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def _page_sharpe_and_drawdown(self) -> matplotlib.figure.Figure:
        """Build page 2: Sharpe distribution (left) + drawdown bars (right)."""
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_WIDTH, _FIG_HEIGHT))
        fig.suptitle(
            f"CPCV Distribution Analysis — {self.ticker}",
            fontsize=14, fontweight="bold", y=0.98,
        )

        self._plot_sharpe_distribution(axes[0])
        self._plot_drawdown(axes[1])

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _render_metrics_table(self, ax: matplotlib.axes.Axes) -> None:
        """Render summary metrics as a table."""
        ax.axis("off")
        r = self.result

        headers = [
            "Path", "Test Groups", "Sharpe", "Max DD",
            "CAGR", "Trades", "Costs",
        ]
        rows: list[list[str]] = []
        for f in r.fold_results:
            rows.append([
                str(f.path_id + 1),
                str(f.test_groups),
                f"{f.sharpe:.3f}",
                f"{f.max_drawdown:.3f}",
                f"{f.cagr:.3f}",
                str(f.n_trades),
                f"{f.total_costs:.4f}",
            ])

        # Add summary row
        rows.append([
            "AVG", "—",
            f"{r.mean_sharpe:.3f} ± {r.std_sharpe:.3f}",
            f"{r.mean_max_drawdown:.3f}",
            f"{r.mean_cagr:.3f}",
            f"{r.mean_n_trades:.1f}",
            f"{r.mean_total_costs:.4f}",
        ])

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)

        # Style header row
        for j in range(len(headers)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Style summary row
        last_row = len(rows)
        for j in range(len(headers)):
            table[last_row, j].set_facecolor("#D6E4F0")
            table[last_row, j].set_text_props(fontweight="bold")

        ax.set_title(
            f"Metrics Summary  |  {r.n_paths} paths  |  "
            f"{r.pct_positive_sharpe:.0%} positive Sharpe",
            fontsize=10, pad=10,
        )

    def _plot_equity_curves(self, ax: matplotlib.axes.Axes) -> None:
        """Overlay equity curves for all folds."""
        has_curves = any(
            len(f.equity_curve) > 1 for f in self.result.fold_results
        )
        if not has_curves:
            ax.text(
                0.5, 0.5, "No equity curve data available",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("Equity Curves")
            return

        cmap = plt.get_cmap("tab20")
        for i, f in enumerate(self.result.fold_results):
            if len(f.equity_curve) > 1:
                color = cmap(i % 20)
                ax.plot(
                    f.equity_curve,
                    label=f"Path {f.path_id + 1}",
                    alpha=0.7,
                    linewidth=0.8,
                    color=color,
                )

        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Return Period")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Equity Curves by Path")
        ax.legend(fontsize=6, ncol=5, loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_sharpe_distribution(self, ax: matplotlib.axes.Axes) -> None:
        """Violin/strip plot of Sharpe ratios across folds."""
        sharpes = [f.sharpe for f in self.result.fold_results]

        if len(sharpes) >= 4:
            sns.violinplot(y=sharpes, ax=ax, inner=None, color="#4472C4", alpha=0.3)
        sns.stripplot(y=sharpes, ax=ax, color="#4472C4", size=6, jitter=0.1)

        ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, label="Zero")
        ax.axhline(
            y=self.result.mean_sharpe, color="green",
            linestyle="-", linewidth=1.0,
            label=f"Mean = {self.result.mean_sharpe:.3f}",
        )
        ax.set_ylabel("Sharpe Ratio (Annualised)")
        ax.set_title("Sharpe Ratio Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_drawdown(self, ax: matplotlib.axes.Axes) -> None:
        """Bar chart of max drawdown per fold."""
        path_ids = [f"P{f.path_id + 1}" for f in self.result.fold_results]
        drawdowns = [f.max_drawdown for f in self.result.fold_results]

        colors = ["#C44E52" if d < -0.10 else "#4C72B0" for d in drawdowns]
        ax.bar(path_ids, drawdowns, color=colors, alpha=0.8)

        ax.axhline(
            y=self.result.mean_max_drawdown, color="orange",
            linestyle="--", linewidth=1.0,
            label=f"Mean = {self.result.mean_max_drawdown:.3f}",
        )
        ax.set_xlabel("Path")
        ax.set_ylabel("Max Drawdown")
        ax.set_title("Maximum Drawdown by Path")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        if len(path_ids) > 10:
            ax.tick_params(axis="x", rotation=45)
