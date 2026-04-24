"""Streamlit dashboard for Titanium Alpha hedge fund.

Four tabs:
    0. **Benchmark** -- walk-forward equity curve, drawdown, metrics, rolling
       Sharpe, weight heatmap
    1. **Performance** -- portfolio weights donut, decision table, metric cards
    2. **War Room** -- agent debate replay with per-agent chat bubbles
    3. **Microstructure** -- PatchTST quantile fan chart per ticker

All data loaded from flat files (JSON/Parquet) in ``data/outputs/``.
No direct PostgreSQL access from the dashboard layer.

Usage::

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data/outputs")

# Resolve benchmark ticker from config (used in chart labels)
try:
    from src.config import load_benchmark as _load_benchmark
    _BENCHMARK_TICKER = _load_benchmark()
except Exception:
    _BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------------

_EXPECTED_SCHEMA_VERSION = "1.2"
_MIN_DISPLAY_WEIGHT = 1e-8

_DARK_BG = "#0E1117"
_CARD_BG = "#1E2130"
_ACCENT_BLUE = "#4A90D9"
_ACCENT_GREEN = "#43A047"
_ACCENT_RED = "#E53935"
_ACCENT_GOLD = "#FFB300"
_TEXT = "#FAFAFA"

AGENT_STYLES: dict[str, dict[str, str]] = {
    "technical": {
        "color": _ACCENT_BLUE,
        "icon": "📊",
        "label": "Technical Analyst",
    },
    "fundamental": {
        "color": _ACCENT_GREEN,
        "icon": "📰",
        "label": "Fundamental Analyst",
    },
    "bear": {
        "color": _ACCENT_RED,
        "icon": "🐻",
        "label": "Bear Agent",
    },
    "pm": {
        "color": _ACCENT_GOLD,
        "icon": "💼",
        "label": "Portfolio Manager",
    },
}

ACTION_COLORS: dict[str, str] = {
    "BUY": _ACCENT_GREEN,
    "HOLD": _ACCENT_GOLD,
    "SELL": _ACCENT_RED,
}

# Dynamic color palette for many tickers (Plotly qualitative)
_PLOTLY_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#4A90D9", "#43A047", "#E53935", "#FFB300", "#7B1FA2",
    "#0097A7", "#F4511E", "#388E3C", "#1976D2", "#C2185B",
    "#00838F", "#558B2F", "#AD1457", "#6A1B9A", "#EF6C00",
    "#00695C", "#283593", "#BF360C", "#1B5E20", "#4527A0",
]


def _ticker_color(index: int) -> str:
    """Return a color for a ticker by index (cycles through palette)."""
    return _PLOTLY_COLORS[index % len(_PLOTLY_COLORS)]

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_decisions() -> dict[str, Any] | None:
    """Load decisions.json. Returns None if missing."""
    path = DATA_DIR / "decisions.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_debate_history() -> dict[str, Any] | None:
    """Load debate_history.json. Returns None if missing."""
    path = DATA_DIR / "debate_history.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_forecast() -> dict[str, list[dict[str, Any]]] | None:
    """Load forecast.parquet as dict grouped by ticker.

    Returns None if the file is missing or Polars is unavailable.
    """
    path = DATA_DIR / "forecast.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        df = pl.read_parquet(path)
        id_col = "ticker" if "ticker" in df.columns else "unique_id"
        result: dict[str, list[dict[str, Any]]] = {}
        for ticker in df[id_col].unique().to_list():
            rows = df.filter(pl.col(id_col) == ticker)
            result[ticker] = rows.to_dicts()
        return result
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_predictions() -> dict[str, dict[str, Any]] | None:
    """Load predictions.parquet as dict keyed by ticker.

    Returns None if the file is missing.
    """
    path = DATA_DIR / "predictions.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        df = pl.read_parquet(path)
        result: dict[str, dict[str, Any]] = {}
        for row in df.to_dicts():
            result[row["ticker"]] = row
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark data loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_benchmark_equity() -> Any | None:
    """Load benchmark_equity.parquet. Returns Polars DataFrame or None."""
    path = DATA_DIR / "benchmark_equity.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        return pl.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_benchmark_metrics() -> dict[str, float] | None:
    """Load benchmark_metrics.json. Returns dict or None."""
    path = DATA_DIR / "benchmark_metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_benchmark_weights() -> Any | None:
    """Load benchmark_weights.parquet. Returns Polars DataFrame or None."""
    path = DATA_DIR / "benchmark_weights.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        return pl.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_ticker_returns() -> Any | None:
    """Load ticker_returns.parquet (wide daily returns date × ticker).

    Returns ``None`` if the file is missing — it is only produced by the
    walk-forward backtester when the result carries a populated
    ``ticker_returns`` DataFrame, so legacy benchmark runs may not have it.
    """
    path = DATA_DIR / "ticker_returns.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        return pl.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_cpcv_paths() -> Any | None:
    """Load cpcv_paths.parquet (long format: config, path_id, date, equity, sharpe).

    Produced by ``src.backtest.cpcv_oos.save_cpcv_paths`` after a
    ``validate(..., collect_equity=True)`` run on the champion config.
    Returns ``None`` when the file is absent so the Phase 10 spaghetti chart
    can degrade gracefully.
    """
    path = DATA_DIR / "cpcv_paths.parquet"
    if not path.exists():
        return None
    try:
        import polars as pl

        return pl.read_parquet(path)
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_cpcv_total_trials() -> int | None:
    """Sum ``total_trials`` across ``validation_tier*_results.json`` files.

    The 3-tier grid search persists one JSON per tier; the Deflated Sharpe
    Ratio correction in Phase 11 needs the aggregated trial count to
    compute the expected-max Sharpe reference line. Returns ``None`` when
    no tier files are present so the chart can skip the DSR line
    gracefully.
    """
    total = 0
    found = False
    for path in sorted(DATA_DIR.glob("validation_tier*_results.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            trials = data.get("total_trials")
            if isinstance(trials, (int, float)) and trials > 0:
                total += int(trials)
                found = True
        except Exception:
            continue
    return total if found else None


# ---------------------------------------------------------------------------
# Tab 0: Benchmark
# ---------------------------------------------------------------------------


def _format_metric_value(key: str, value: float) -> str:
    """Format a metric value for display."""
    pct_keys = {
        "cagr", "total_return", "benchmark_total_return",
        "annualized_volatility", "max_drawdown", "tracking_error",
        "hit_rate_monthly",
    }
    if key in pct_keys:
        return f"{value:.2%}"
    if key == "max_drawdown_duration_days":
        return f"{value:.0f}"
    if key in ("alpha",):
        return f"{value:.4f}"
    return f"{value:.3f}"


def _metric_color(key: str, value: float) -> str:
    """Return green if metric is favorable, red if unfavorable."""
    # Higher is better
    higher_better = {
        "cagr", "total_return", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "information_ratio", "alpha", "hit_rate_monthly",
    }
    # Lower is better (less negative)
    lower_better = {"max_drawdown", "max_drawdown_duration_days"}

    if key in higher_better:
        return _ACCENT_GREEN if value > 0 else _ACCENT_RED
    if key in lower_better:
        return _ACCENT_RED if value < -0.10 else _ACCENT_GREEN
    return _TEXT


_METRIC_LABELS: dict[str, str] = {
    "cagr": "CAGR",
    "total_return": "Total Return",
    "benchmark_total_return": "Benchmark Return",
    "annualized_volatility": "Ann. Volatility",
    "max_drawdown": "Max Drawdown",
    "max_drawdown_duration_days": "Max DD Duration (days)",
    "calmar_ratio": "Calmar Ratio",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "information_ratio": "Information Ratio",
    "alpha": "Alpha (ann.)",
    "beta": "Beta",
    "tracking_error": "Tracking Error",
    "hit_rate_monthly": "Monthly Hit Rate",
    "avg_annual_turnover": "Avg Annual Turnover",
    "avg_positions": "Avg Positions",
}


def _chart_benchmark_equity(equity_df: Any, log_scale: bool = False) -> go.Figure:
    """Plotly equity curve: portfolio vs benchmark."""
    dates = equity_df["date"].to_list()
    portfolio = equity_df["portfolio_value"].to_list()
    benchmark = equity_df["benchmark_value"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio, name="Portfolio",
        line=dict(color=_ACCENT_BLUE, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=benchmark, name=f"Benchmark ({_BENCHMARK_TICKER})",
        line=dict(color=_ACCENT_GOLD, width=2, dash="dash"),
    ))

    yaxis_type = "log" if log_scale else "linear"
    fig.update_layout(
        title=dict(text="Equity Curve — Portfolio vs Benchmark", font=dict(size=16, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        yaxis_type=yaxis_type,
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT)),
        height=450,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _compute_drawdown(values: list[float]) -> list[float]:
    """Compute drawdown series from equity values."""
    peak = values[0]
    dd = []
    for v in values:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak != 0 else 0.0)
    return dd


def _chart_benchmark_drawdown(equity_df: Any) -> go.Figure:
    """Plotly drawdown chart (filled area) — portfolio vs benchmark."""
    dates = equity_df["date"].to_list()
    port_dd = _compute_drawdown(equity_df["portfolio_value"].to_list())
    bench_dd = _compute_drawdown(equity_df["benchmark_value"].to_list())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=port_dd, fill="tozeroy", name="Portfolio",
        line=dict(color=_ACCENT_RED, width=1),
        fillcolor="rgba(229, 57, 53, 0.3)",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=bench_dd, name=f"Benchmark ({_BENCHMARK_TICKER})",
        line=dict(color=_ACCENT_GOLD, width=1, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text="Drawdown — Portfolio vs Benchmark", font=dict(size=16, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT)),
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


_MONTH_LABELS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _compute_monthly_returns(equity_df: Any) -> Any:
    """Compute within-month compounded returns for portfolio and benchmark.

    Daily returns are derived from ``portfolio_value`` / ``benchmark_value``
    and compounded inside each (year, month) bucket as ``prod(1+r) - 1``.
    This gives a well-defined return for partial months (e.g. the first
    month of the backtest) without reaching past the first observation.

    Args:
        equity_df: Polars DataFrame with columns ``date``, ``portfolio_value``
            and ``benchmark_value`` (as produced by the walk-forward pipeline).

    Returns:
        Polars DataFrame with columns ``year`` (Int32), ``month`` (Int8),
        ``port_ret`` (Float64), ``spy_ret`` (Float64), sorted by (year, month).
        Only months with at least one daily return are included.
    """
    import polars as pl

    df = (
        equity_df.sort("date")
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0)
            .alias("port_dret"),
            (pl.col("benchmark_value") / pl.col("benchmark_value").shift(1) - 1.0)
            .alias("spy_dret"),
        ])
        .drop_nulls(["port_dret", "spy_dret"])
    )
    return (
        df.group_by(["year", "month"])
        .agg([
            ((pl.col("port_dret") + 1.0).product() - 1.0).alias("port_ret"),
            ((pl.col("spy_dret") + 1.0).product() - 1.0).alias("spy_ret"),
        ])
        .sort(["year", "month"])
    )


def _compute_annual_returns(equity_df: Any) -> Any:
    """Compute within-year compounded returns for portfolio and benchmark.

    Args:
        equity_df: Polars DataFrame with columns ``date``, ``portfolio_value``
            and ``benchmark_value``.

    Returns:
        Polars DataFrame with columns ``year`` (Int32), ``port_ret``, ``spy_ret``.
    """
    import polars as pl

    df = (
        equity_df.sort("date")
        .with_columns([
            pl.col("date").dt.year().alias("year"),
            (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0)
            .alias("port_dret"),
            (pl.col("benchmark_value") / pl.col("benchmark_value").shift(1) - 1.0)
            .alias("spy_dret"),
        ])
        .drop_nulls(["port_dret", "spy_dret"])
    )
    return (
        df.group_by("year")
        .agg([
            ((pl.col("port_dret") + 1.0).product() - 1.0).alias("port_ret"),
            ((pl.col("spy_dret") + 1.0).product() - 1.0).alias("spy_ret"),
        ])
        .sort("year")
    )


def _chart_calendar_heatmap(equity_df: Any) -> go.Figure | None:
    """Monthly-returns calendar heatmap (12 months + Annual column + Mean row).

    Rows are years (most recent on top). Columns are Jan..Dec followed by a
    bold ``Annual`` column showing the full-year compounded return. A bottom
    ``Mean`` row shows the average return per calendar month across covered
    years (seasonality view). Months outside the backtest coverage render as
    empty cells (null) — never as 0% — so partial first/last years stay
    visually distinct from zero-return months.

    Args:
        equity_df: Polars DataFrame with columns ``date``, ``portfolio_value``,
            ``benchmark_value``.

    Returns:
        Plotly Figure, or ``None`` if fewer than 2 months of data are available.
    """
    import polars as pl

    monthly = _compute_monthly_returns(equity_df)
    if monthly.height < 2:
        return None
    annual = _compute_annual_returns(equity_df)

    years_desc = sorted(monthly["year"].unique().to_list(), reverse=True)
    x_labels = list(_MONTH_LABELS) + ["Annual"]

    # Build {(year, month): ret} lookup for portfolio returns only.
    by_ym = {
        (row["year"], row["month"]): row["port_ret"]
        for row in monthly.select(["year", "month", "port_ret"]).to_dicts()
    }
    by_year = {
        row["year"]: row["port_ret"]
        for row in annual.select(["year", "port_ret"]).to_dicts()
    }

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for y in years_desc:
        row_z: list[float | None] = []
        row_text: list[str] = []
        for m in range(1, 13):
            ret = by_ym.get((y, m))
            row_z.append(ret)
            row_text.append(f"{ret * 100:+.1f}%" if ret is not None else "")
        ann = by_year.get(y)
        row_z.append(ann)
        row_text.append(f"<b>{ann * 100:+.1f}%</b>" if ann is not None else "")
        z.append(row_z)
        text.append(row_text)

    # Bonus: mean row (seasonality) — simple mean over years for each month + annual.
    mean_row_z: list[float | None] = []
    mean_row_text: list[str] = []
    for m in range(1, 13):
        vals = [by_ym[(y, m)] for y in years_desc if (y, m) in by_ym]
        if vals:
            mean = sum(vals) / len(vals)
            mean_row_z.append(mean)
            mean_row_text.append(f"<i>{mean * 100:+.1f}%</i>")
        else:
            mean_row_z.append(None)
            mean_row_text.append("")
    year_vals = [by_year[y] for y in years_desc if y in by_year]
    if year_vals:
        mean_ann = sum(year_vals) / len(year_vals)
        mean_row_z.append(mean_ann)
        mean_row_text.append(f"<b><i>{mean_ann * 100:+.1f}%</i></b>")
    else:
        mean_row_z.append(None)
        mean_row_text.append("")
    z.append(mean_row_z)
    text.append(mean_row_text)

    y_labels = [str(y) for y in years_desc] + ["Mean"]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale="RdYlGn",
        zmid=0.0,
        zmin=-0.15,
        zmax=0.15,
        hovertemplate=(
            "Year: %{y}<br>Month: %{x}<br>Return: %{z:+.2%}<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Return", font=dict(color=_TEXT)),
            tickformat=".0%",
            tickfont=dict(color=_TEXT),
        ),
        xgap=2,
        ygap=2,
    ))
    # Visual separator between Dec and the Annual column.
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=11.5, x1=11.5, y0=0, y1=1,
        line=dict(color="#666666", width=2),
    )
    fig.update_layout(
        title=dict(
            text="Monthly Returns Heatmap — Portfolio",
            font=dict(size=16, color=_TEXT),
        ),
        xaxis=dict(
            title="Month",
            side="top",
            tickfont=dict(color=_TEXT),
            showgrid=False,
        ),
        yaxis=dict(
            title="Year",
            autorange="reversed",
            tickfont=dict(color=_TEXT),
            showgrid=False,
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        height=max(320, 34 * len(y_labels) + 120),
        margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


def _detect_drawdown_periods(
    equity: list[float],
    dates: list[Any],
    min_depth: float = 0.01,
) -> list[dict[str, Any]]:
    """Identify peak-to-recovery drawdown events in an equity series.

    Walks the series tracking the running high-water mark. A drawdown event
    opens the first day equity dips below the prior peak, tracks the trough,
    and closes when equity reclaims that peak (``ongoing=False``) or when the
    series ends still underwater (``ongoing=True``).

    Durations are measured in **trading days** (index differences in the
    ``equity`` array), not calendar days, so they stay consistent with the
    rest of the dashboard, which operates on a business-day grid.

    Args:
        equity: Equity values ordered chronologically. Must align with ``dates``.
        dates: Aligned date-like values (used only for event labelling, never
            for duration arithmetic).
        min_depth: Minimum absolute fractional depth to retain (``0.01`` = 1%).
            Events shallower than this are filtered out to avoid micro-DD noise.

    Returns:
        List of event dicts sorted by depth ascending (deepest first). Keys:
        ``start`` (peak date), ``trough`` (min date), ``end`` (recovery date or
        ``None``), ``depth`` (negative float, e.g. ``-0.21``),
        ``duration_days`` (peak → recovery in trading days),
        ``recovery_days`` (trough → recovery in trading days, or ``None`` if
        ongoing), ``ongoing``.
    """
    n = len(equity)
    if n != len(dates) or n < 2:
        return []

    events: list[dict[str, Any]] = []
    peak_val = equity[0]
    peak_idx = 0
    in_dd = False
    trough_val = peak_val
    trough_idx = 0

    for i in range(1, n):
        v = equity[i]
        if not in_dd:
            if v >= peak_val:
                peak_val = v
                peak_idx = i
            else:
                in_dd = True
                trough_val = v
                trough_idx = i
        else:
            if v < trough_val:
                trough_val = v
                trough_idx = i
            if v >= peak_val:
                depth = (trough_val - peak_val) / peak_val if peak_val else 0.0
                events.append({
                    "start": dates[peak_idx],
                    "trough": dates[trough_idx],
                    "end": dates[i],
                    "depth": depth,
                    "duration_days": i - peak_idx,
                    "recovery_days": i - trough_idx,
                    "ongoing": False,
                })
                in_dd = False
                peak_val = v
                peak_idx = i

    if in_dd:
        depth = (trough_val - peak_val) / peak_val if peak_val else 0.0
        last_idx = n - 1
        events.append({
            "start": dates[peak_idx],
            "trough": dates[trough_idx],
            "end": None,
            "depth": depth,
            "duration_days": last_idx - peak_idx,
            "recovery_days": None,
            "ongoing": True,
        })

    events = [e for e in events if abs(e["depth"]) >= min_depth]
    events.sort(key=lambda e: e["depth"])  # most negative first
    return events


def _format_dd_label(event: dict[str, Any]) -> str:
    """Short y-axis label for a drawdown event: ``YYYY-MM → YYYY-MM (Nd)*``."""
    start = event["start"]
    start_s = start.strftime("%Y-%m") if hasattr(start, "strftime") else str(start)
    if event["ongoing"]:
        return f"{start_s} → ongoing ({event['duration_days']}d)*"
    end = event["end"]
    end_s = end.strftime("%Y-%m") if hasattr(end, "strftime") else str(end)
    return f"{start_s} → {end_s} ({event['duration_days']}d)"


def _chart_top_drawdowns(
    equity_df: Any,
    n: int = 10,
    min_depth: float = 0.01,
) -> go.Figure | None:
    """Horizontal bar chart of the top-N deepest drawdowns.

    Bars are sorted deepest-first (top of the chart). Depth is shown as a
    negative percentage; darker red = deeper. Ongoing drawdowns (series ends
    still underwater) are marked with a trailing asterisk in the label.

    Args:
        equity_df: Polars DataFrame with ``date`` and ``portfolio_value``.
        n: Maximum number of drawdown events to display (default ``10``).
        min_depth: Minimum absolute fractional depth to retain (default ``0.01``).

    Returns:
        Plotly Figure, or ``None`` if the series has no drawdown at least
        ``min_depth`` deep.
    """
    equity = equity_df["portfolio_value"].to_list()
    dates = equity_df["date"].to_list()
    events = _detect_drawdown_periods(equity, dates, min_depth=min_depth)
    if not events:
        return None

    events = events[:n]
    labels = [_format_dd_label(e) for e in events]
    depths = [e["depth"] for e in events]
    depth_pct_abs = [abs(d) for d in depths]

    customdata = [
        [
            e["trough"].strftime("%Y-%m-%d") if hasattr(e["trough"], "strftime") else str(e["trough"]),
            e["recovery_days"] if e["recovery_days"] is not None else float("nan"),
            "ongoing" if e["ongoing"] else "recovered",
        ]
        for e in events
    ]

    fig = go.Figure(data=go.Bar(
        x=depths,
        y=labels,
        orientation="h",
        marker=dict(
            color=depth_pct_abs,
            colorscale="Reds",
            cmin=0.0,
            cmax=max(depth_pct_abs + [0.05]),
            showscale=False,
            line=dict(color="#222", width=0.5),
        ),
        text=[f"{d * 100:+.2f}%" for d in depths],
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Depth: %{x:+.2%}<br>"
            "Trough: %{customdata[0]}<br>"
            "Recovery: %{customdata[1]} trading days<br>"
            "Status: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=f"Top {len(events)} Drawdowns — Portfolio",
            font=dict(size=16, color=_TEXT),
        ),
        xaxis=dict(
            title="Depth",
            tickformat=".0%",
            zeroline=True,
            zerolinecolor="#444",
            gridcolor="#222",
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",  # deepest at top
            tickfont=dict(color=_TEXT),
            showgrid=False,
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        height=max(320, 36 * len(events) + 120),
        margin=dict(t=50, b=40, l=200, r=80),
        showlegend=False,
    )
    return fig


def _render_top_drawdowns_table(equity_df: Any, n: int = 10, min_depth: float = 0.01) -> None:
    """Render a companion table with the same drawdown events as the bar chart."""
    equity = equity_df["portfolio_value"].to_list()
    dates = equity_df["date"].to_list()
    events = _detect_drawdown_periods(equity, dates, min_depth=min_depth)[:n]
    if not events:
        return

    def _fmt_date(d: Any) -> str:
        if d is None:
            return "—"
        return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

    rows = []
    for rank, e in enumerate(events, start=1):
        rows.append({
            "#": rank,
            "Start (peak)": _fmt_date(e["start"]),
            "Trough": _fmt_date(e["trough"]),
            "Recovery": _fmt_date(e["end"]),
            "Depth": f"{e['depth'] * 100:+.2f}%",
            "Duration (td)": e["duration_days"],
            "Trough→Rec. (td)": "—" if e["recovery_days"] is None else e["recovery_days"],
            "Status": "Ongoing" if e["ongoing"] else "Recovered",
        })
    st.dataframe(rows, width="stretch", hide_index=True)


def _portfolio_daily_returns(equity_df: Any) -> list[float]:
    """Extract daily simple returns from ``portfolio_value`` dropping the first NaN."""
    values = equity_df["portfolio_value"].to_list()
    return [values[i] / values[i - 1] - 1.0 for i in range(1, len(values))]


def _return_distribution_stats(rets: list[float]) -> dict[str, float]:
    """Compute mean, std, skew, excess kurtosis, VaR 5%, CVaR 5% of a return series.

    Args:
        rets: Daily simple returns (length ≥ 2 required).

    Returns:
        Dict with keys ``mu``, ``sigma``, ``skew``, ``kurt`` (excess kurtosis
        with Fisher's definition: 0 for a normal distribution), ``var5``
        (5th percentile, negative for a losing tail), ``cvar5`` (mean of
        returns below VaR 5%).
    """
    import numpy as np
    import scipy.stats as stats

    arr = np.asarray(rets, dtype=float)
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=True, bias=False))
    var5 = float(np.quantile(arr, 0.05))
    tail = arr[arr <= var5]
    cvar5 = float(tail.mean()) if tail.size else var5
    return {
        "mu": mu,
        "sigma": sigma,
        "skew": skew,
        "kurt": kurt,
        "var5": var5,
        "cvar5": cvar5,
    }


def _chart_return_distribution(equity_df: Any) -> go.Figure | None:
    """Two-panel chart: return histogram (with normal fit) + QQ-plot vs normal.

    Left panel overlays a fitted normal PDF on the empirical histogram and
    shades the left tail beyond VaR 5% in red (tail-risk proof). Right panel
    plots empirical quantiles against theoretical normal quantiles with a
    reference line ``y = σ·x + μ`` — points bending away from the line at
    either extreme indicate fat tails / skew.

    Args:
        equity_df: Polars DataFrame with ``portfolio_value`` column.

    Returns:
        Plotly Figure with 2 subplots, or ``None`` if fewer than 30 daily
        returns are available (below which skew/kurtosis are unreliable).
    """
    import numpy as np
    import scipy.stats as stats
    from plotly.subplots import make_subplots

    rets = _portfolio_daily_returns(equity_df)
    if len(rets) < 30:
        return None

    arr = np.asarray(rets, dtype=float)
    s = _return_distribution_stats(rets)
    mu, sigma, var5, cvar5 = s["mu"], s["sigma"], s["var5"], s["cvar5"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Daily Return Distribution", "Q-Q Plot vs Normal"),
        horizontal_spacing=0.12,
    )

    # --- Column 1: histogram + normal PDF overlay + VaR5 shading ---
    fig.add_trace(
        go.Histogram(
            x=arr,
            histnorm="probability density",
            name="Empirical",
            marker=dict(color=_ACCENT_BLUE, line=dict(color="#111", width=0.5)),
            opacity=0.75,
            nbinsx=60,
            hovertemplate="Return: %{x:+.2%}<br>Density: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    xs = np.linspace(arr.min(), arr.max(), 200)
    pdf = stats.norm.pdf(xs, mu, sigma)
    fig.add_trace(
        go.Scatter(
            x=xs, y=pdf, mode="lines",
            name="Normal fit",
            line=dict(color=_ACCENT_GOLD, width=2, dash="dash"),
            hovertemplate="x=%{x:+.2%}<br>φ(x)=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Shade the left tail < VaR 5%.
    fig.add_vrect(
        x0=float(arr.min()), x1=var5,
        fillcolor=_ACCENT_RED, opacity=0.18,
        line_width=0,
        row=1, col=1,
    )
    # VaR and CVaR reference lines.
    fig.add_vline(
        x=var5, line=dict(color=_ACCENT_RED, width=1.5, dash="dot"),
        annotation_text=f"Daily VaR 5% = {var5:+.2%}",
        annotation_position="top left",
        annotation_font=dict(color=_ACCENT_RED, size=10),
        row=1, col=1,
    )
    fig.add_vline(
        x=cvar5, line=dict(color=_ACCENT_RED, width=1.5),
        annotation_text=f"Daily CVaR 5% = {cvar5:+.2%}",
        annotation_position="bottom left",
        annotation_font=dict(color=_ACCENT_RED, size=10),
        row=1, col=1,
    )

    # --- Column 2: QQ-plot with reference line y = σ·x + μ ---
    n = len(arr)
    # Use (i-0.5)/n plotting positions (Blom-like, standard in QQ plots).
    ppos = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(ppos)
    empirical = np.sort(arr)
    fig.add_trace(
        go.Scatter(
            x=theoretical, y=empirical,
            mode="markers",
            name="Empirical",
            marker=dict(color=_ACCENT_BLUE, size=4, opacity=0.65),
            hovertemplate=(
                "Theoretical z=%{x:.2f}<br>Empirical=%{y:+.2%}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )
    lo, hi = float(theoretical.min()), float(theoretical.max())
    ref_xs = np.array([lo, hi])
    ref_ys = ref_xs * sigma + mu
    fig.add_trace(
        go.Scatter(
            x=ref_xs, y=ref_ys,
            mode="lines",
            name="Normal reference",
            line=dict(color=_ACCENT_GOLD, width=2, dash="dash"),
            hovertemplate="ref: y=σx+μ<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # --- Layout ---
    stats_title = (
        f"Return Distribution — Skew: {s['skew']:+.2f} | "
        f"Excess Kurt: {s['kurt']:+.2f} | "
        f"Daily VaR 5%: {var5:+.2%} | Daily CVaR 5%: {cvar5:+.2%}"
    )
    fig.update_layout(
        title=dict(text=stats_title, font=dict(size=15, color=_TEXT)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT), orientation="h",
                    yanchor="bottom", y=-0.18, xanchor="center", x=0.25),
        height=450,
        margin=dict(t=70, b=60, l=60, r=20),
        bargap=0.02,
    )
    fig.update_xaxes(
        title_text="Daily Return", tickformat=".1%",
        gridcolor="#222", zerolinecolor="#444",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Density",
        gridcolor="#222", zerolinecolor="#444",
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text="Theoretical Quantiles (z)",
        gridcolor="#222", zerolinecolor="#444",
        row=1, col=2,
    )
    fig.update_yaxes(
        title_text="Empirical Return", tickformat=".1%",
        gridcolor="#222", zerolinecolor="#444",
        row=1, col=2,
    )
    return fig


def _rolling_regression(
    port_ret: Any,
    spy_ret: Any,
    window: int,
) -> tuple[Any, Any, Any]:
    """Rolling OLS of portfolio returns on benchmark returns.

    For every trailing window of size ``window`` ending at index ``t``:

    * ``beta = Cov(port, spy) / Var(spy)``
    * ``alpha_daily = mean(port) - beta * mean(spy)``
    * ``alpha_annual = alpha_daily * 252``
    * ``correlation = Corr(port, spy)``

    Args:
        port_ret: Portfolio daily returns (1-D array-like).
        spy_ret: Benchmark daily returns, same length.
        window: Rolling window size in trading days (≥ 2).

    Returns:
        Three numpy arrays aligned with the input: ``beta``, ``alpha_annual``,
        ``correlation``. The first ``window-1`` positions are ``NaN`` (warmup).
        If ``Var(spy)`` inside a window is zero, ``beta`` is ``NaN`` for that t.
    """
    import numpy as np

    p = np.asarray(port_ret, dtype=float)
    s = np.asarray(spy_ret, dtype=float)
    n = p.size
    beta = np.full(n, np.nan)
    alpha_ann = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    if window < 2 or n < window:
        return beta, alpha_ann, corr

    for t in range(window - 1, n):
        wp = p[t - window + 1: t + 1]
        ws = s[t - window + 1: t + 1]
        mp = wp.mean()
        ms = ws.mean()
        dp = wp - mp
        ds = ws - ms
        var_s = float((ds * ds).sum() / (window - 1))
        cov_ps = float((dp * ds).sum() / (window - 1))
        if var_s <= 0:
            continue
        b = cov_ps / var_s
        beta[t] = b
        alpha_ann[t] = (mp - b * ms) * 252.0
        std_p = float(np.sqrt((dp * dp).sum() / (window - 1)))
        std_s = float(np.sqrt(var_s))
        if std_p > 0 and std_s > 0:
            corr[t] = cov_ps / (std_p * std_s)
    return beta, alpha_ann, corr


def _chart_rolling_market_relationship(
    equity_df: Any, window: int = 126
) -> go.Figure | None:
    """Stacked 3-panel chart: rolling beta, alpha (annualized), correlation vs SPY.

    Args:
        equity_df: Polars DataFrame with ``date``, ``portfolio_value``,
            ``benchmark_value``.
        window: Rolling window in trading days (default ``126`` ≈ 6 months).

    Returns:
        Plotly Figure, or ``None`` if insufficient data.
    """
    import numpy as np
    from plotly.subplots import make_subplots

    port_vals = equity_df["portfolio_value"].to_list()
    spy_vals = equity_df["benchmark_value"].to_list()
    dates = equity_df["date"].to_list()
    if len(port_vals) < window + 2:
        return None

    port_ret = np.array(
        [port_vals[i] / port_vals[i - 1] - 1.0 for i in range(1, len(port_vals))]
    )
    spy_ret = np.array(
        [spy_vals[i] / spy_vals[i - 1] - 1.0 for i in range(1, len(spy_vals))]
    )
    ret_dates = dates[1:]
    beta, alpha_ann, corr = _rolling_regression(port_ret, spy_ret, window)

    def _mean_finite(arr: Any) -> float:
        vals = [v for v in arr if v == v]  # filter NaN
        return float(sum(vals) / len(vals)) if vals else float("nan")

    mean_beta = _mean_finite(beta)
    mean_alpha = _mean_finite(alpha_ann)
    mean_corr = _mean_finite(corr)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"Rolling Beta (mean: {mean_beta:+.2f})",
            f"Rolling Alpha — Annualized (mean: {mean_alpha:+.2%})",
            f"Rolling Correlation (mean: {mean_corr:+.2f})",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=ret_dates, y=beta, name="Beta",
            line=dict(color=_ACCENT_BLUE, width=1.5),
            hovertemplate="%{x}<br>β=%{y:.3f}<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_hline(y=1.0, line=dict(color=_ACCENT_GOLD, dash="dash", width=1), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=ret_dates, y=alpha_ann, name="Alpha (ann.)",
            line=dict(color=_ACCENT_GREEN, width=1.5),
            hovertemplate="%{x}<br>α=%{y:+.2%}<extra></extra>",
        ), row=2, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color=_ACCENT_GOLD, dash="dash", width=1), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=ret_dates, y=corr, name="Correlation",
            line=dict(color=_ACCENT_BLUE, width=1.5, dash="dot"),
            hovertemplate="%{x}<br>ρ=%{y:.3f}<extra></extra>",
        ), row=3, col=1,
    )
    fig.add_hline(y=0.0, line=dict(color=_ACCENT_GOLD, dash="dash", width=1), row=3, col=1)

    fig.update_layout(
        title=dict(
            text=f"Market Relationship — Rolling {window}-Day OLS vs {_BENCHMARK_TICKER}",
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=False,
        height=600,
        margin=dict(t=70, b=40, l=60, r=20),
    )
    fig.update_xaxes(gridcolor="#222", zerolinecolor="#444")
    fig.update_yaxes(gridcolor="#222", zerolinecolor="#444")
    fig.update_yaxes(title_text="β", row=1, col=1)
    fig.update_yaxes(title_text="α (ann.)", tickformat=".1%", row=2, col=1)
    fig.update_yaxes(title_text="ρ", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def _compute_capture_ratios(equity_df: Any) -> dict[str, float] | None:
    """Compute up-capture, down-capture, ratio and positive-month % vs benchmark.

    Monthly compounded returns are classified by the sign of the benchmark
    return in the same month. ``up_capture`` is the ratio of the mean
    portfolio return to the mean benchmark return across ``spy > 0`` months,
    expressed as a percentage (``85.0`` means the portfolio captures 85% of
    the benchmark's average upside). ``down_capture`` is the analogous ratio
    on ``spy < 0`` months (``< 100`` means the portfolio loses less on the
    downside). ``up_down_ratio`` is ``up_capture / down_capture`` (also in
    %): higher means the portfolio behaves asymmetrically in its favor.

    Args:
        equity_df: Polars DataFrame with columns ``date``, ``portfolio_value``,
            ``benchmark_value``.

    Returns:
        Dict with ``up_capture``, ``down_capture``, ``up_down_ratio``,
        ``positive_month_pct`` (portfolio), ``benchmark_positive_month_pct``
        and monthly counts, or ``None`` if fewer than 2 months are available.
        Individual values may be ``NaN`` when the corresponding subset is
        empty or degenerate (e.g. no down months in a bull-only window).
    """
    import numpy as np

    monthly = _compute_monthly_returns(equity_df)
    if monthly.height < 2:
        return None

    port = monthly["port_ret"].to_numpy()
    spy = monthly["spy_ret"].to_numpy()

    up_mask = spy > 0
    dn_mask = spy < 0
    n_up = int(up_mask.sum())
    n_dn = int(dn_mask.sum())

    def _safe_mean(arr: Any) -> float:
        return float(arr.mean()) if arr.size > 0 else float("nan")

    port_up = _safe_mean(port[up_mask])
    port_dn = _safe_mean(port[dn_mask])
    spy_up = _safe_mean(spy[up_mask])
    spy_dn = _safe_mean(spy[dn_mask])

    up_capture = (port_up / spy_up * 100.0) if np.isfinite(spy_up) and spy_up != 0.0 else float("nan")
    down_capture = (port_dn / spy_dn * 100.0) if np.isfinite(spy_dn) and spy_dn != 0.0 else float("nan")

    if np.isfinite(up_capture) and np.isfinite(down_capture) and down_capture != 0.0:
        up_down_ratio = up_capture / down_capture * 100.0
    else:
        up_down_ratio = float("nan")

    total = len(port)
    positive_month_pct = float((port > 0).sum()) / total * 100.0
    benchmark_positive_month_pct = float((spy > 0).sum()) / total * 100.0

    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "up_down_ratio": float(up_down_ratio),
        "positive_month_pct": positive_month_pct,
        "benchmark_positive_month_pct": benchmark_positive_month_pct,
        "n_up_months": n_up,
        "n_down_months": n_dn,
        "n_total_months": total,
    }


def _chart_capture_ratios(equity_df: Any) -> go.Figure | None:
    """Bar chart with up-capture, down-capture, up/down ratio and positive month %.

    Each bar is expressed as a percentage. A horizontal dashed line at 100%
    marks the benchmark baseline: bars above the line capture more upside,
    bars below capture less downside. Colors are semantic (green = favorable,
    red = adverse) relative to each metric's preferred direction.

    Args:
        equity_df: Polars DataFrame with ``date``, ``portfolio_value``,
            ``benchmark_value``.

    Returns:
        Plotly Figure, or ``None`` if fewer than 2 months of data are
        available.
    """
    ratios = _compute_capture_ratios(equity_df)
    if ratios is None:
        return None

    labels = ["Up-Capture", "Down-Capture", "Up/Down Ratio", "Positive Month %"]
    values = [
        ratios["up_capture"],
        ratios["down_capture"],
        ratios["up_down_ratio"],
        ratios["positive_month_pct"],
    ]

    def _favorable(metric: str, v: float) -> bool:
        if v != v:  # NaN
            return False
        if metric == "Up-Capture":
            return v > 100.0
        if metric == "Down-Capture":
            return v < 100.0
        if metric == "Up/Down Ratio":
            return v > 100.0
        if metric == "Positive Month %":
            return v > 50.0
        return False

    colors = [
        "#888888" if v != v else (_ACCENT_GREEN if _favorable(m, v) else _ACCENT_RED)
        for m, v in zip(labels, values)
    ]
    text = [("n/a" if v != v else f"{v:.1f}%") for v in values]

    hover = (
        f"Up months: {ratios['n_up_months']} | "
        f"Down months: {ratios['n_down_months']} | "
        f"Total: {ratios['n_total_months']}"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(color="#111", width=1)),
        text=text,
        textposition="outside",
        textfont=dict(color=_TEXT, size=13),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>" + hover + "<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=100.0,
        line=dict(color=_ACCENT_GOLD, dash="dash", width=1),
        annotation_text=f"{_BENCHMARK_TICKER} baseline (100%)",
        annotation_position="top right",
        annotation_font=dict(color=_ACCENT_GOLD, size=11),
    )

    finite_vals = [v for v in values if v == v]
    y_top = max(finite_vals + [110.0]) * 1.20 if finite_vals else 120.0
    y_bottom = min(finite_vals + [0.0]) - 15.0 if finite_vals else -15.0

    fig.update_layout(
        title=dict(
            text=(
                f"Up / Down Capture vs {_BENCHMARK_TICKER} "
                f"(n={ratios['n_total_months']} months: "
                f"{ratios['n_up_months']} up / {ratios['n_down_months']} down)"
            ),
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        yaxis=dict(
            title="% of benchmark",
            gridcolor="#222",
            zerolinecolor="#444",
            range=[y_bottom, y_top],
        ),
        xaxis=dict(gridcolor="#222", zerolinecolor="#444"),
        height=420,
        margin=dict(t=60, b=50, l=60, r=20),
        bargap=0.35,
    )
    return fig


def _chart_capm_scatter(equity_df: Any) -> go.Figure | None:
    """CAPM scatter of daily portfolio vs benchmark returns with OLS line.

    Each point is a trading day: x = benchmark daily return, y = portfolio
    daily return. A single full-sample OLS line visualizes
    ``R_p = α + β · R_m``. Slope is the static beta; intercept is the daily
    alpha, annualized as ``α_daily × 252`` for the legend and annotation.
    R² is computed from the correlation between the two return series.

    Args:
        equity_df: Polars DataFrame with ``date``, ``portfolio_value``,
            ``benchmark_value``.

    Returns:
        Plotly Figure, or ``None`` if fewer than 3 daily returns are
        available or the benchmark has zero variance.
    """
    import numpy as np

    port_vals = equity_df["portfolio_value"].to_list()
    spy_vals = equity_df["benchmark_value"].to_list()
    if len(port_vals) < 4:
        return None

    port_ret = np.array(
        [port_vals[i] / port_vals[i - 1] - 1.0 for i in range(1, len(port_vals))]
    )
    spy_ret = np.array(
        [spy_vals[i] / spy_vals[i - 1] - 1.0 for i in range(1, len(spy_vals))]
    )

    if spy_ret.var() <= 0.0:
        return None

    slope, intercept = np.polyfit(spy_ret, port_ret, 1)
    beta = float(slope)
    alpha_daily = float(intercept)
    alpha_annual = alpha_daily * 252.0
    r_squared = float(np.corrcoef(spy_ret, port_ret)[0, 1] ** 2)

    xs = np.linspace(float(spy_ret.min()), float(spy_ret.max()), 100)
    ys = slope * xs + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_ret,
        y=port_ret,
        mode="markers",
        marker=dict(size=4, opacity=0.45, color=_ACCENT_BLUE),
        name="Daily returns",
        hovertemplate=(
            f"{_BENCHMARK_TICKER}: %{{x:+.2%}}<br>"
            "Portfolio: %{y:+.2%}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=_ACCENT_GOLD, width=2),
        name=f"OLS fit (β={beta:.2f}, α_ann={alpha_annual:+.2%})",
        hoverinfo="skip",
    ))
    fig.add_hline(y=0.0, line=dict(color="#444", width=1))
    fig.add_vline(x=0.0, line=dict(color="#444", width=1))

    fig.add_annotation(
        text=(
            f"β = {beta:.3f}<br>"
            f"α (ann.) = {alpha_annual:+.2%}<br>"
            f"R² = {r_squared:.3f}<br>"
            f"n = {len(port_ret):,}"
        ),
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        bgcolor="rgba(14,17,23,0.85)",
        bordercolor=_ACCENT_GOLD,
        borderwidth=1,
        font=dict(color=_TEXT, size=12),
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=f"CAPM Scatter — Portfolio vs {_BENCHMARK_TICKER} Daily Returns",
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            title=f"{_BENCHMARK_TICKER} daily return",
            tickformat=".1%",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        yaxis=dict(
            title="Portfolio daily return",
            tickformat=".1%",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
        ),
        showlegend=True,
        height=500,
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def _chart_rolling_sharpe(equity_df: Any, window: int = 252) -> go.Figure | None:
    """Rolling Sharpe ratio chart. Returns None if insufficient data."""
    import math

    portfolio = equity_df["portfolio_value"].to_list()
    benchmark = equity_df["benchmark_value"].to_list()
    dates = equity_df["date"].to_list()

    if len(portfolio) < window + 1:
        return None

    # Risk-free rate (geometric daily) consistent with rest of pipeline
    rf_annual = 0.05
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252) - 1.0

    # Compute daily returns from equity values (skip first day)
    port_rets = [(portfolio[i] / portfolio[i - 1] - 1.0) for i in range(1, len(portfolio))]
    bench_rets = [(benchmark[i] / benchmark[i - 1] - 1.0) for i in range(1, len(benchmark))]
    dates = dates[1:]

    # Rolling Sharpe (excess returns over rf)
    def _rolling(rets: list[float], w: int) -> list[float | None]:
        result: list[float | None] = [None] * (w - 1)
        for i in range(w - 1, len(rets)):
            chunk = rets[i - w + 1: i + 1]
            excess = [r - rf_daily for r in chunk]
            mean_ex = sum(excess) / len(excess)
            var_ex = sum((e - mean_ex) ** 2 for e in excess) / (len(excess) - 1)
            std_ex = math.sqrt(var_ex) if var_ex > 0 else 0.0
            sharpe = (mean_ex / std_ex) * math.sqrt(252) if std_ex > 0 else 0.0
            result.append(sharpe)
        return result

    port_sharpe = _rolling(port_rets, window)
    bench_sharpe = _rolling(bench_rets, window)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=port_sharpe, name="Portfolio",
        line=dict(color=_ACCENT_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=bench_sharpe, name="Benchmark",
        line=dict(color=_ACCENT_GOLD, width=1.5, dash="dash"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=dict(text=f"Rolling {window}-Day Sharpe Ratio", font=dict(size=16, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT)),
        height=400,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _compute_turnover_per_rebalance(weights_df: Any) -> Any:
    """Collapse the long-format weights DataFrame to one row per rebalance date.

    ``turnover`` and ``costs`` are produced by the walk-forward backtester as
    portfolio-level scalars per rebalance event and replicated across every
    ticker row sharing that date. This helper picks the first row per date
    and keeps only dates where a real rebalance happened
    (``turnover > 0``).

    Args:
        weights_df: Long-format Polars DataFrame with at least ``date``,
            ``turnover``, ``costs`` columns.

    Returns:
        Polars DataFrame with columns ``date``, ``turnover``, ``costs``,
        sorted by date. ``turnover`` is in fractional units (sum of
        ``|Δw|`` across tickers).
    """
    import polars as pl

    return (
        weights_df
        .group_by("date")
        .agg([
            pl.col("turnover").first().alias("turnover"),
            pl.col("costs").first().alias("costs"),
        ])
        .filter(pl.col("turnover") > 0.0)
        .sort("date")
    )


def _reconstruct_gross_equity(equity_df: Any, weights_df: Any) -> Any:
    """Reconstruct a counterfactual gross-equity series (zero transaction costs).

    At each rebalance day ``r`` the walk-forward backtester deducts a dollar
    cost ``c_r`` from the start-of-day portfolio value ``sod_r`` *before* the
    day's drift is applied. End-of-day net equity is therefore

    .. code-block:: text

        net[r] = (sod[r] - c_r) * (1 + drift[r])

    where ``sod[r] ≈ net[r-1]``. The counterfactual gross portfolio pays no
    costs, so its drift is the same but its base is ``sod[r]``:

    .. code-block:: text

        gross_factor[t] = Π_{r ≤ t}  sod[r] / (sod[r] - c_r)
        gross[t]        = net[t] * gross_factor[t]

    This identity is exact under the backtester's accounting; it does not
    require re-running the simulation.

    Args:
        equity_df: Polars DataFrame with ``date`` and ``portfolio_value``.
        weights_df: Long-format weights DataFrame with ``date`` and ``costs``.

    Returns:
        ``equity_df`` sorted by date with an additional
        ``portfolio_value_gross`` column. If ``costs`` is missing or all
        zeros, ``portfolio_value_gross`` equals ``portfolio_value``.
    """
    import polars as pl

    eq = equity_df.sort("date")
    if "costs" not in weights_df.columns:
        return eq.with_columns(
            pl.col("portfolio_value").alias("portfolio_value_gross")
        )

    costs_per_date = (
        weights_df
        .group_by("date")
        .agg(pl.col("costs").first().alias("_cost"))
    )
    joined = (
        eq.with_columns(pl.col("portfolio_value").shift(1).alias("_sod"))
        .join(costs_per_date, on="date", how="left")
        .with_columns(pl.col("_cost").fill_null(0.0))
    )

    factor = 1.0
    factors: list[float] = []
    for row in joined.iter_rows(named=True):
        sod = row["_sod"]
        cost = row["_cost"] or 0.0
        if sod is not None and sod > 0.0 and cost > 0.0 and sod > cost:
            factor *= sod / (sod - cost)
        factors.append(factor)

    return (
        joined.with_columns(pl.Series("_factor", factors))
        .with_columns(
            (pl.col("portfolio_value") * pl.col("_factor"))
            .alias("portfolio_value_gross")
        )
        .drop(["_sod", "_cost", "_factor"])
    )


def _compute_effective_n(weights_row: Any) -> float:
    """Herfindahl-based effective number of positions (cash counts as 1 slot).

    ``Effective N = 1 / Σ w_i²`` where the sum is taken over the supplied
    ticker weights *plus* an implicit cash position ``cash = max(0, 1 - Σw)``.
    Counting cash as a slot keeps the metric bounded and intuitive: a
    portfolio that is 50% cash + 1 stock returns 2, and a fully-invested
    uniform-N portfolio returns N.

    Args:
        weights_row: 1-D array-like of non-negative ticker weights (long-only).
            Zeros are allowed and contribute nothing to HHI.

    Returns:
        Effective number of positions (``float``); ``NaN`` if the full
        vector (including cash) sums to zero.
    """
    import numpy as np

    w = np.asarray(list(weights_row), dtype=float)
    w = w[w > 0.0]
    cash = max(0.0, 1.0 - float(w.sum()))
    hhi = cash * cash + float(np.sum(w * w))
    if hhi <= 0.0:
        return float("nan")
    return 1.0 / hhi


def _compute_gini(weights_row: Any) -> float:
    """Gini coefficient of the portfolio weight vector.

    Zero weights are dropped; implicit cash (``1 - Σw``) is appended as a
    position *only when strictly positive*, so a fully-invested uniform-N
    portfolio correctly returns ``Gini = 0`` instead of being penalized for
    having zero cash.

    ``0`` = all positions equal (including cash, when present).
    ``(n-1)/n`` = single position absorbs everything.

    Args:
        weights_row: 1-D array-like of non-negative ticker weights.

    Returns:
        Gini coefficient in ``[0, 1)`` or ``NaN`` if the vector is empty or
        sums to zero.
    """
    import numpy as np

    w = np.asarray(list(weights_row), dtype=float)
    w = w[w > 0.0]
    cash = max(0.0, 1.0 - float(w.sum()))
    if cash > 0.0:
        w = np.concatenate([w, np.array([cash])])
    if w.size == 0 or w.sum() == 0.0:
        return float("nan")
    w = np.sort(w)
    n = w.size
    cum = np.cumsum(w)
    return float(
        (2.0 * float(np.sum(np.arange(1, n + 1) * w)) - (n + 1) * float(cum[-1]))
        / (n * float(cum[-1]))
    )


def _chart_concentration_evolution(weights_df: Any) -> go.Figure | None:
    """Dual-axis time series: Effective N (left) and Gini coefficient (right).

    One point per rebalance date sourced from the long-format weights
    parquet. Cash is treated per the semantics of ``_compute_effective_n``
    (counted as a slot) and ``_compute_gini`` (appended only when positive).
    A dotted reference line at ``Effective N = 1 / 0.06 ≈ 16.67`` marks
    the equivalent number of positions when every slot sits at the HRP
    ``max_weight = 0.06`` cap — the book crosses above this line whenever
    weights are distributed more evenly than the cap would force.

    Args:
        weights_df: Long-format Polars DataFrame with at least ``date`` and
            ``weight`` columns.

    Returns:
        Plotly Figure, or ``None`` if the weight history is missing or empty.
    """
    import polars as pl
    from plotly.subplots import make_subplots

    if weights_df is None or "weight" not in weights_df.columns:
        return None

    per_date = (
        weights_df
        .group_by("date")
        .agg(pl.col("weight").alias("weights_list"))
        .sort("date")
    )
    if per_date.height == 0:
        return None

    dates = per_date["date"].to_list()
    eff_ns: list[float] = []
    ginis: list[float] = []
    for row in per_date.iter_rows(named=True):
        w = row["weights_list"]
        eff_ns.append(_compute_effective_n(w))
        ginis.append(_compute_gini(w))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=dates, y=eff_ns,
            name="Effective N",
            mode="lines+markers",
            line=dict(color=_ACCENT_BLUE, width=1.8),
            marker=dict(size=4),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Effective N: %{y:.2f}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=ginis,
            name="Gini",
            mode="lines+markers",
            line=dict(color=_ACCENT_GOLD, width=1.5, dash="dash"),
            marker=dict(size=4),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Gini: %{y:.3f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    max_weight_ref = 1.0 / 0.06
    fig.add_shape(
        type="line",
        xref="paper", x0=0.0, x1=1.0,
        yref="y",
        y0=max_weight_ref, y1=max_weight_ref,
        line=dict(color="#666", dash="dot", width=1),
    )
    fig.add_annotation(
        text=(
            f"Equivalent N at HRP max_weight=0.06 ({max_weight_ref:.1f})"
        ),
        xref="paper", yref="y",
        x=0.98, y=max_weight_ref,
        showarrow=False,
        font=dict(color="#888", size=10),
        xanchor="right", yanchor="bottom",
    )

    fig.update_layout(
        title=dict(
            text="Portfolio Concentration Over Time",
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        height=450,
        margin=dict(t=60, b=40, l=60, r=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date", gridcolor="#222", zerolinecolor="#444")
    fig.update_yaxes(
        title_text="Effective N (1 / Σw²)",
        secondary_y=False, gridcolor="#222", zerolinecolor="#444",
    )
    fig.update_yaxes(
        title_text="Gini coefficient",
        secondary_y=True, range=[0.0, 1.0], showgrid=False,
    )
    return fig


def _compute_contribution_per_ticker(
    weights_df: Any, returns_df: Any
) -> dict[str, float]:
    """Additive per-ticker contribution to total portfolio return (pp).

    For every rebalance interval ``[r, r_next]`` the contribution of
    ticker ``i`` is ``w_r[i] * (Π(1 + r_{i,d}) - 1)`` — i.e. the weight set
    at rebalance ``r`` times the compound simple return of ticker ``i``
    until the next rebalance (or the end of the backtest). Summing over
    intervals gives the additive attribution to total portfolio return.

    The sum across tickers approximates total realized portfolio return
    within a few percentage points because cash carry, transaction costs
    and across-interval compounding are not attributed to any ticker
    (this is standard for Brinson-style attribution).

    Args:
        weights_df: Long-format Polars DataFrame with at least ``date``,
            ``ticker``, ``weight`` (rows for non-zero rebalance allocations).
        returns_df: Wide Polars DataFrame with a ``date`` column and one
            column per ticker holding daily simple returns.

    Returns:
        Mapping ``{ticker: contribution_pp}`` in percentage points. Tickers
        with exactly zero net contribution are omitted. Empty dict when
        either input is missing the required columns.
    """
    import numpy as np
    import polars as pl

    if weights_df is None or returns_df is None:
        return {}
    if "weight" not in weights_df.columns or "ticker" not in weights_df.columns:
        return {}
    if "date" not in returns_df.columns:
        return {}

    rebalance_dates = (
        weights_df.sort("date")["date"].unique(maintain_order=True).to_list()
    )
    if not rebalance_dates:
        return {}

    returns_sorted = returns_df.sort("date")
    return_dates = returns_sorted["date"].to_list()
    if not return_dates:
        return {}
    return_tickers = set(returns_sorted.columns) - {"date"}

    contributions: dict[str, float] = {}
    for i, r_date in enumerate(rebalance_dates):
        r_next = (
            rebalance_dates[i + 1]
            if i + 1 < len(rebalance_dates)
            else return_dates[-1]
        )
        interval = returns_sorted.filter(
            (pl.col("date") > r_date) & (pl.col("date") <= r_next)
        )
        if interval.height == 0:
            continue

        weights_row = weights_df.filter(pl.col("date") == r_date)
        for w_row in weights_row.iter_rows(named=True):
            ticker = w_row["ticker"]
            if ticker not in return_tickers:
                continue
            weight = float(w_row["weight"])
            if weight == 0.0:
                continue
            rets = interval[ticker].to_numpy()
            rets = np.where(np.isfinite(rets), rets, 0.0)
            if rets.size == 0:
                continue
            compound = float(np.prod(1.0 + rets) - 1.0)
            contributions[ticker] = (
                contributions.get(ticker, 0.0) + weight * compound
            )

    return {t: v * 100.0 for t, v in contributions.items() if v != 0.0}


def _chart_contribution_waterfall(
    contributions: dict[str, float],
    top_n: int = 15,
) -> go.Figure | None:
    """Waterfall of per-ticker contribution (pp): Start → top_n → Others → Total.

    Selection of the top-N is by absolute contribution so that large
    detractors also surface instead of being hidden in the bucket.
    Displayed order within the top-N is signed-descending so the story
    reads left-to-right from biggest positive to biggest negative.

    Args:
        contributions: Mapping ``ticker → contribution_pp`` (produced by
            :func:`_compute_contribution_per_ticker`).
        top_n: Number of tickers to surface individually. Defaults to 15.

    Returns:
        Plotly Figure, or ``None`` if ``contributions`` is empty.
    """
    if not contributions:
        return None

    by_abs = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
    top = by_abs[:top_n]
    rest = by_abs[top_n:]
    top.sort(key=lambda kv: -kv[1])

    top_labels = [t for t, _ in top]
    top_values = [float(v) for _, v in top]
    others_value = float(sum(v for _, v in rest)) if rest else 0.0
    total_value = float(sum(v for _, v in by_abs))

    x_labels: list[str] = ["Start"] + top_labels
    measures: list[str] = ["absolute"] + ["relative"] * len(top_labels)
    y_vals: list[float] = [0.0] + top_values
    text_vals: list[str] = [""] + [f"{v:+.2f}pp" for v in top_values]

    if rest:
        x_labels.append(f"Others ({len(rest)})")
        measures.append("relative")
        y_vals.append(others_value)
        text_vals.append(f"{others_value:+.2f}pp")

    x_labels.append("Total")
    measures.append("total")
    y_vals.append(total_value)
    text_vals.append(f"{total_value:+.2f}pp")

    fig = go.Figure(go.Waterfall(
        x=x_labels,
        measure=measures,
        y=y_vals,
        text=text_vals,
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        connector=dict(line=dict(color="#555", dash="dot", width=1)),
        increasing=dict(marker=dict(color=_ACCENT_GREEN)),
        decreasing=dict(marker=dict(color=_ACCENT_RED)),
        totals=dict(marker=dict(color=_ACCENT_BLUE)),
        hovertemplate="<b>%{x}</b><br>%{y:+.3f}pp<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=(
                "Return Attribution — Additive Contribution to Total Return "
                f"(sum: {total_value:+.2f}pp)"
            ),
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            title="",
            tickangle=-40,
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        yaxis=dict(
            title="Contribution (pp)",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        showlegend=False,
        height=520,
        margin=dict(t=60, b=100, l=60, r=20),
    )
    return fig


def _chart_cpcv_spaghetti(
    paths_df: Any,
    config_filter: str | None = None,
) -> go.Figure | None:
    """CPCV-OOS equity spaghetti — 15 individual paths + IQR band + median.

    Each path is rendered as a thin semi-transparent line; the cross-path
    25th/50th/75th percentiles at every date are overlaid as a shaded band
    with a bold median line. Tightly clustered paths = low path-dependency
    = robust configuration. The chart expects paths already normalised to
    the same baseline (see :meth:`ValidationResult.to_paths_frame`).

    Args:
        paths_df: Long-format Polars DataFrame with columns ``config``,
            ``path_id``, ``date``, ``equity``, ``sharpe``.
        config_filter: Optional config name. When the parquet holds
            multiple configs, plot only rows where ``config == filter``.
            Defaults to the first distinct config if omitted.

    Returns:
        Plotly Figure, or ``None`` if the input is empty or missing
        required columns.
    """
    import polars as pl

    if paths_df is None:
        return None
    required = {"path_id", "date", "equity"}
    if not required.issubset(set(paths_df.columns)):
        return None
    if paths_df.height == 0:
        return None

    if "config" in paths_df.columns:
        if config_filter is None:
            config_filter = (
                paths_df["config"].unique(maintain_order=True).to_list()[0]
            )
        df = paths_df.filter(pl.col("config") == config_filter)
    else:
        df = paths_df
        config_filter = config_filter or "cpcv"

    if df.height == 0:
        return None

    path_ids = sorted(df["path_id"].unique().to_list())
    n_paths = len(path_ids)

    sharpes: list[float] = []
    if "sharpe" in df.columns:
        sharpe_per_path = (
            df.group_by("path_id")
            .agg(pl.col("sharpe").first())
            .sort("path_id")
        )
        sharpes = [
            float(v)
            for v in sharpe_per_path["sharpe"].to_list()
            if v is not None and v == v
        ]

    fig = go.Figure()

    for pid in path_ids:
        path = df.filter(pl.col("path_id") == pid).sort("date")
        raw_dates = path["date"].to_list()
        raw_equity = path["equity"].to_list()
        # CPCV paths covering non-adjacent test blocks (e.g. blocks (1,3))
        # have a train-period gap between the blocks. Break the Plotly line
        # at any > 7-day date jump so Plotly does not interpolate a diagonal
        # "slingshot" across months where this path was off the test set.
        x_vals: list[Any] = []
        y_vals: list[Any] = []
        for i, (d, eq) in enumerate(zip(raw_dates, raw_equity)):
            if i > 0 and (d - raw_dates[i - 1]).days > 7:
                x_vals.append(None)
                y_vals.append(None)
            x_vals.append(d)
            y_vals.append(eq)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=_ACCENT_BLUE, width=1),
            opacity=0.28,
            name=f"Path {pid}",
            connectgaps=False,
            hovertemplate=(
                f"Path {pid}<br>%{{x|%Y-%m-%d}}<br>"
                "Equity: %{y:.3f}<extra></extra>"
            ),
            showlegend=False,
        ))

    agg = (
        df.group_by("date")
        .agg([
            pl.col("equity").quantile(0.25).alias("q25"),
            pl.col("equity").quantile(0.50).alias("med"),
            pl.col("equity").quantile(0.75).alias("q75"),
        ])
        .sort("date")
    )
    agg_dates = agg["date"].to_list()

    fig.add_trace(go.Scatter(
        x=agg_dates,
        y=agg["q75"].to_list(),
        mode="lines",
        line=dict(color="rgba(74,144,217,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=agg_dates,
        y=agg["q25"].to_list(),
        mode="lines",
        line=dict(color="rgba(74,144,217,0)"),
        fill="tonexty",
        fillcolor="rgba(74,144,217,0.20)",
        name="IQR 25–75%",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=agg_dates,
        y=agg["med"].to_list(),
        mode="lines",
        line=dict(color=_ACCENT_GOLD, width=2.5),
        name="Median",
        hovertemplate=(
            "Median<br>%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>"
        ),
    ))

    if sharpes:
        sharpe_range = (
            f"Sharpe range across paths: {min(sharpes):.2f} — "
            f"{max(sharpes):.2f} (median {sorted(sharpes)[len(sharpes)//2]:.2f})"
        )
    else:
        sharpe_range = "Sharpe per path not available in parquet"

    fig.update_layout(
        title=dict(
            text=(
                f"CPCV-OOS Path Distribution — {n_paths} combinatorial paths "
                f"({config_filter})"
            ),
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            title="Date",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        yaxis=dict(
            title="Equity (base 1.0)",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        height=500,
        margin=dict(t=70, b=50, l=60, r=20),
        hovermode="closest",
    )
    fig.add_annotation(
        text=sharpe_range,
        xref="paper", yref="paper",
        x=0.99, y=0.02,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(color="#AAAAAA", size=11),
        bgcolor="rgba(14,17,23,0.7)",
    )
    return fig


# ---------------------------------------------------------------------------
# Phase 11 — Sharpe violin across CPCV paths
# ---------------------------------------------------------------------------


def _compute_sharpe_per_path(paths_df: Any) -> list[float]:
    """Extract one Sharpe ratio per CPCV path from a long-format paths frame.

    The ``sharpe`` column in ``cpcv_paths.parquet`` is duplicated across
    every row of the same ``path_id``; this helper collapses the frame
    back to one value per path and drops non-finite entries so downstream
    statistics (mean, std, PSR) do not blow up.

    Args:
        paths_df: Polars DataFrame with at least ``path_id`` and
            ``sharpe`` columns.

    Returns:
        Sharpe ratios sorted by ``path_id``. Empty list when the frame is
        missing the required columns, is empty, or every value is NaN.
    """
    import math

    import polars as pl

    if paths_df is None:
        return []
    if paths_df.height == 0:
        return []
    if not {"path_id", "sharpe"}.issubset(set(paths_df.columns)):
        return []

    agg = (
        paths_df.group_by("path_id")
        .agg(pl.col("sharpe").first().alias("sharpe"))
        .sort("path_id")
    )
    out: list[float] = []
    for value in agg["sharpe"].to_list():
        if value is None:
            continue
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            continue
        out.append(fv)
    return out


def _expected_max_sharpe(sigma: float, n_trials: int) -> float | None:
    """Expected maximum Sharpe across ``n_trials`` independent draws.

    Euler–Mascheroni approximation for the max of standard normal variates,
    scaled by ``sigma`` (the cross-sample Sharpe dispersion used as a
    proxy for σ(SR)). Mirrors the core of Bailey & Lopez de Prado's
    Deflated Sharpe Ratio formula. Returns ``None`` when the inputs are
    degenerate so the caller can skip the reference line gracefully.

    Args:
        sigma: Standard deviation of Sharpe across candidate strategies
            (typically the std across CPCV paths).
        n_trials: Number of strategy configurations tested.

    Returns:
        Expected-maximum Sharpe in the same units as ``sigma``, or
        ``None`` when ``n_trials < 2`` or ``sigma <= 0``.
    """
    import math

    if n_trials is None or n_trials < 2:
        return None
    if sigma is None or sigma <= 0:
        return None

    euler_mascheroni = 0.5772156649
    try:
        z1 = _inv_normal_cdf_approx(1.0 - 1.0 / n_trials)
        z2 = _inv_normal_cdf_approx(1.0 - 1.0 / (n_trials * math.e))
    except ValueError:
        return None
    e_max_z = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2
    return float(sigma) * float(e_max_z)


def _inv_normal_cdf_approx(p: float) -> float:
    """Abramowitz & Stegun 26.2.23 approximation to the inverse normal CDF.

    Same algorithm used in :func:`src.backtest.cpcv_oos._inv_normal_cdf`;
    duplicated here so the dashboard does not import heavy modules at
    startup. Accurate to ~4.5e-4 on ``(0, 1)``.
    """
    import math

    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p < 0.5:
        return -_inv_normal_cdf_approx(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (
        1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    )


def _probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    sharpe_benchmark: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> float | None:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Probability that the *true* annualised Sharpe exceeds
    ``sharpe_benchmark`` given ``n_obs`` daily observations. Returns a
    value in ``[0, 1]``; above 0.95 is the standard confidence level.

    Args:
        observed_sharpe: Annualised Sharpe realised on the test window.
        n_obs: Number of daily return observations behind the Sharpe.
        sharpe_benchmark: Annualised Sharpe to test against (default 0).
        skewness: Sample skewness of the underlying returns.
        kurtosis: Full sample kurtosis (normal = 3.0) of the returns.
        periods_per_year: Annualisation factor.

    Returns:
        PSR probability in ``[0, 1]``, or ``None`` if the inputs are
        insufficient to compute it (n_obs < 2 or zero SR variance).
    """
    import math

    if n_obs is None or n_obs < 2:
        return None

    sr_daily = observed_sharpe / math.sqrt(periods_per_year)
    bench_daily = sharpe_benchmark / math.sqrt(periods_per_year)

    var_sr_daily = (
        1.0
        + 0.5 * sr_daily * sr_daily
        - skewness * sr_daily
        + (kurtosis - 3.0) / 4.0 * sr_daily * sr_daily
    ) / n_obs
    if var_sr_daily <= 0:
        return None

    z = (sr_daily - bench_daily) / math.sqrt(var_sr_daily)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _chart_sharpe_violin(
    paths_df: Any,
    oos_sharpe: float | None = None,
    dsr_threshold: float | None = None,
    psr: float | None = None,
    n_trials: int | None = None,
    config_filter: str | None = None,
) -> go.Figure | None:
    """Violin plot of per-path Sharpe ratios with statistical references.

    Companion to the CPCV spaghetti (Phase 10): where the spaghetti shows
    equity dispersion, the violin shows **where the Sharpe actually lands
    across the 15 combinatorial paths**. Horizontal reference lines
    contextualise the distribution:

    - **DSR expected max (dashed grey):** the Sharpe a best-of-``n_trials``
      noise strategy would produce under the Bailey & Lopez de Prado
      approximation. The path distribution should sit clearly above it.
    - **Walk-forward OOS (solid gold):** the single-path Sharpe realised
      on the production backtest. Inside the IQR means the OOS run is
      consistent with the CPCV cloud.

    A side stats box surfaces path count, mean, std, % positive and the
    Probabilistic Sharpe Ratio (PSR) for the walk-forward OOS Sharpe.

    Args:
        paths_df: Long-format Polars DataFrame with ``path_id`` and
            ``sharpe`` columns (same schema as
            :func:`_chart_cpcv_spaghetti`).
        oos_sharpe: Walk-forward OOS Sharpe (annualised). ``None`` skips
            the gold reference line.
        dsr_threshold: Pre-computed DSR expected-max Sharpe. When
            ``None`` and ``n_trials`` is provided, computed internally
            from the path-Sharpe std.
        psr: Pre-computed Probabilistic Sharpe Ratio (0..1). Shown in
            the side stats when provided.
        n_trials: Number of configurations the grid searched. Used to
            derive ``dsr_threshold`` when it is not supplied explicitly.
        config_filter: If the parquet holds multiple configs, only rows
            matching this label are plotted. Defaults to the first
            distinct config.

    Returns:
        Plotly Figure, or ``None`` when the paths frame is missing
        required columns or has fewer than 2 valid Sharpes.
    """
    import statistics

    import polars as pl

    if paths_df is None:
        return None
    if not {"path_id", "sharpe"}.issubset(set(paths_df.columns)):
        return None
    if paths_df.height == 0:
        return None

    if "config" in paths_df.columns:
        if config_filter is None:
            config_filter = (
                paths_df["config"].unique(maintain_order=True).to_list()[0]
            )
        df = paths_df.filter(pl.col("config") == config_filter)
    else:
        df = paths_df
        config_filter = config_filter or "cpcv"

    sharpes = _compute_sharpe_per_path(df)
    if len(sharpes) < 2:
        return None

    n_paths = len(sharpes)
    mean_sharpe = statistics.fmean(sharpes)
    stdev = statistics.stdev(sharpes)
    n_positive = sum(1 for s in sharpes if s > 0)
    pct_positive = n_positive / n_paths

    if dsr_threshold is None and n_trials is not None:
        dsr_threshold = _expected_max_sharpe(stdev, int(n_trials))

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=sharpes,
            name="CPCV paths",
            box_visible=True,
            meanline_visible=True,
            points="all",
            pointpos=0.0,
            jitter=0.25,
            line=dict(color=_ACCENT_BLUE, width=1.5),
            fillcolor="rgba(74,144,217,0.25)",
            marker=dict(
                color=_ACCENT_BLUE,
                size=7,
                opacity=0.85,
                line=dict(color=_DARK_BG, width=1),
            ),
            hovertemplate="Path Sharpe: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    if dsr_threshold is not None:
        fig.add_hline(
            y=float(dsr_threshold),
            line=dict(color="#AAAAAA", dash="dash", width=1.5),
            annotation_text=f"DSR expected max = {float(dsr_threshold):.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#AAAAAA", size=11),
        )

    if oos_sharpe is not None:
        fig.add_hline(
            y=float(oos_sharpe),
            line=dict(color=_ACCENT_GOLD, dash="solid", width=2),
            annotation_text=(
                f"Walk-forward OOS = {float(oos_sharpe):.2f}"
            ),
            annotation_position="bottom left",
            annotation_font=dict(color=_ACCENT_GOLD, size=11),
        )

    stats_lines = [
        f"<b>Paths:</b> {n_paths}",
        f"<b>Mean:</b> {mean_sharpe:.3f}",
        f"<b>Std:</b> {stdev:.3f}",
        f"<b>% positive:</b> {pct_positive:.0%}",
    ]
    if psr is not None:
        stats_lines.append(f"<b>PSR:</b> {float(psr):.2f}")
    stats_text = "<br>".join(stats_lines)
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(color=_TEXT, size=11),
        bgcolor="rgba(14,17,23,0.8)",
        bordercolor="#444",
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        title=dict(
            text=(
                "Sharpe Distribution Across CPCV Paths — "
                f"{n_paths} paths ({config_filter})"
            ),
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            title="Annualised Sharpe",
            gridcolor="#222",
            zerolinecolor="#444",
        ),
        height=460,
        margin=dict(t=70, b=40, l=60, r=30),
        showlegend=False,
    )
    return fig


def _chart_turnover_and_costs(
    equity_df: Any, weights_df: Any
) -> go.Figure | None:
    """Two-panel chart: turnover per rebalance + gross vs net equity.

    Row 1 is a bar chart of per-rebalance turnover (as a percentage). Row 2
    overlays the (reconstructed) gross equity index and the realized net
    equity index, base-100 at t=0, with a red-shaded area marking the
    cumulative cost drag. The title reports the total dollar cost paid and
    the subtitle reports annualized drag in basis points per year (from the
    CAGR gap).

    Args:
        equity_df: Polars DataFrame with ``date``, ``portfolio_value``.
        weights_df: Long-format weights DataFrame with ``date``,
            ``turnover``, ``costs``.

    Returns:
        Plotly Figure, or ``None`` if either input is missing required
        columns or there are no rebalance events.
    """
    import polars as pl
    from plotly.subplots import make_subplots

    if equity_df is None or weights_df is None:
        return None
    required = {"date", "turnover", "costs"}
    if not required.issubset(set(weights_df.columns)):
        return None
    if "portfolio_value" not in equity_df.columns:
        return None

    turnover_df = _compute_turnover_per_rebalance(weights_df)
    if turnover_df.height == 0:
        return None

    gross_df = _reconstruct_gross_equity(equity_df, weights_df)

    net_vals = gross_df["portfolio_value"].to_list()
    gross_vals = gross_df["portfolio_value_gross"].to_list()
    dates = gross_df["date"].to_list()
    if not net_vals:
        return None

    base_net = float(net_vals[0])
    base_gross = float(gross_vals[0])
    if base_net <= 0 or base_gross <= 0:
        return None

    net_idx = [v / base_net * 100.0 for v in net_vals]
    gross_idx = [v / base_gross * 100.0 for v in gross_vals]

    total_cost_dollars = float(
        weights_df.group_by("date")
        .agg(pl.col("costs").first())
        .select(pl.col("costs").sum())
        .item()
    )
    n_days = len(dates)
    n_years = n_days / 252.0 if n_days > 0 else 1.0

    cagr_net = (net_idx[-1] / 100.0) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    cagr_gross = (gross_idx[-1] / 100.0) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    drag_bps = (cagr_gross - cagr_net) * 10000.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=(
            f"Turnover per Rebalance ({turnover_df.height} events)",
            f"Gross vs Net Equity (drag ≈ {drag_bps:+.1f} bps/year)",
        ),
        row_heights=[0.35, 0.65],
    )

    fig.add_trace(
        go.Bar(
            x=turnover_df["date"].to_list(),
            y=[v * 100.0 for v in turnover_df["turnover"].to_list()],
            marker=dict(color=_ACCENT_BLUE),
            name="Turnover",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Turnover: %{y:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=gross_idx, mode="lines",
            line=dict(color=_ACCENT_GOLD, width=1.5, dash="dash"),
            name="Gross (no costs)",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Gross: %{y:.2f}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=net_idx, mode="lines",
            line=dict(color=_ACCENT_BLUE, width=1.8),
            name="Net (after costs)",
            fill="tonexty",
            fillcolor="rgba(229,57,53,0.15)",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Net: %{y:.2f}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Trading Costs — Total ${total_cost_dollars:,.0f} paid "
                f"over {n_years:.1f}y"
            ),
            font=dict(size=15, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            x=0.01, y=0.55,
            xanchor="left", yanchor="top",
        ),
        height=620,
        margin=dict(t=70, b=40, l=60, r=20),
    )
    fig.update_xaxes(gridcolor="#222", zerolinecolor="#444")
    fig.update_yaxes(gridcolor="#222", zerolinecolor="#444")
    fig.update_yaxes(title_text="Turnover (%)", row=1, col=1)
    fig.update_yaxes(title_text="Index (base 100)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def _chart_weight_heatmap(weights_df: Any) -> go.Figure | None:
    """Plotly heatmap of top 15 asset weights over time."""
    try:
        import polars as pl
    except ImportError:
        return None

    if weights_df is None or weights_df.height < 2:
        return None

    # Compute average weight per ticker
    avg = weights_df.group_by("ticker").agg(pl.col("weight").mean().alias("avg_w"))
    top_tickers = (
        avg.sort("avg_w", descending=True)
        .head(15)["ticker"]
        .to_list()
    )

    # Filter to top tickers
    filtered = weights_df.filter(pl.col("ticker").is_in(top_tickers))

    # Pivot: rows=tickers, cols=dates
    pivot = filtered.pivot(on="date", index="ticker", values="weight").fill_null(0.0)

    # Order tickers by average weight
    ticker_order = top_tickers
    date_cols = [c for c in pivot.columns if c != "ticker"]

    z_data = []
    y_labels = []
    for t in ticker_order:
        row = pivot.filter(pl.col("ticker") == t)
        if row.height > 0:
            z_data.append([row[c][0] for c in date_cols])
            y_labels.append(t)

    if not z_data:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=date_cols,
        y=y_labels,
        colorscale="YlOrRd",
        zmin=0,
        colorbar=dict(title="Weight", tickformat=".0%"),
    ))

    fig.update_layout(
        title=dict(text=f"Top {len(y_labels)} Assets — Weight Evolution", font=dict(size=16, color=_TEXT)),
        xaxis_title="Rebalance Date",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        height=450,
        margin=dict(t=50, b=80, l=80, r=20),
    )
    return fig


def _load_validation_results() -> dict[str, Any] | None:
    """Load CPCV-OOS validation_results.json if available."""
    path = DATA_DIR / "validation_results.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Validated strategy configuration (Session 39 — CPCV-OOS 3-tier, 547 configs)
_VALIDATED_CONFIG: dict[str, Any] = {
    "Model Factory": "NaiveModelFactory(lookback=5)",
    "rebalance_every": 15,
    "retrain_every": 126,
    "lookback_days": 756,
    "costs": "slippage=5bps & commission=10bps",
    "min_rebalance_delta": 0.02,
    "target_vol": "10% annualized (63-day lookback, 0.5-1.0 leverage)",
    "HRP linkage": "ward",
    "HRP shrinkage": "Ledoit-Wolf",
    "HRP correlation": "pearson",
    "max_weight": "min(0.06, 2/n)",
}

# Key findings from 3-tier CPCV-OOS grid search (547 configs)
_STRESS_FINDINGS: list[str] = [
    "Triweekly rebalance (rb=15) validated via 3-tier CPCV-OOS (547 configs).",
    "Vol targeting at 10%: single biggest driver (+0.035 Sharpe, MaxDD halved).",
    "Ward linkage + Ledoit-Wolf shrinkage: confirmed dominant HRP structure.",
    "max_weight relaxed from min(0.25, 2/n) to min(0.06, 2/n): +0.026 Sharpe.",
    "top_n and killswitch both harmful — disabled.",
    "Transaction costs: slippage 5bps + commission 10bps fully reflected.",
    "Record: Sharpe=0.712, CAGR=13.35%, MaxDD=-18.43%, Beta=0.532.",
]


def tab_benchmark(
    equity_df: Any | None,
    metrics: dict[str, float] | None,
    weights_df: Any | None,
    ticker_returns_df: Any | None = None,
) -> None:
    """Render the Benchmark tab."""
    st.header("Benchmark — Walk-Forward Backtest")

    if equity_df is None:
        st.warning(
            "No benchmark data found. Run `make benchmark-fast` to generate "
            "benchmark results."
        )
        return

    # --- Strategy Configuration expander
    with st.expander("Strategy Configuration (CPCV-OOS validated)", expanded=False):
        st.caption("Parameters validated via 3-tier CPCV-OOS grid search (547 configs, 15 paths each).")
        cfg_col1, cfg_col2 = st.columns(2)
        cfg_items = list(_VALIDATED_CONFIG.items())
        mid_cfg = len(cfg_items) // 2
        with cfg_col1:
            for k, v in cfg_items[:mid_cfg]:
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:3px 6px; border-bottom:1px solid #333;'>"
                    f"<span style='color:#aaa;'>{k}</span>"
                    f"<span style='color:{_ACCENT_BLUE}; font-weight:bold;'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with cfg_col2:
            for k, v in cfg_items[mid_cfg:]:
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:3px 6px; border-bottom:1px solid #333;'>"
                    f"<span style='color:#aaa;'>{k}</span>"
                    f"<span style='color:{_ACCENT_BLUE}; font-weight:bold;'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("**Key Findings (Stress Test & Optimization):**")
        for finding in _STRESS_FINDINGS:
            st.markdown(f"- {finding}")

    # --- Validation Results expander (if available)
    validation = _load_validation_results()
    if validation and "configs" in validation:
        with st.expander("CPCV-OOS Validation Results", expanded=False):
            st.caption(f"Generated: {validation.get('generated_at', 'N/A')}")
            configs = validation["configs"]
            rows = []
            for name, data in configs.items():
                rows.append({
                    "Config": name,
                    "Mean Sharpe": f"{data.get('mean_sharpe', 0):.3f}",
                    "Std": f"{data.get('std_sharpe', 0):.3f}",
                    "Pct+": f"{data.get('pct_positive', 0):.0%}",
                    "DSR p-value": f"{data.get('p_value', 0):.3f}",
                    "Accepted": "YES" if data.get("accepted") else "no",
                })
            st.dataframe(rows, width="stretch", hide_index=True)

    # --- CPCV-OOS path distribution spaghetti (Phase 10)
    cpcv_paths = load_cpcv_paths()
    if cpcv_paths is not None:
        st.subheader("CPCV-OOS Path Distribution")
        fig_spaghetti = _chart_cpcv_spaghetti(cpcv_paths)
        if fig_spaghetti is not None:
            st.plotly_chart(fig_spaghetti, width="stretch")
            st.caption(
                "Each thin line is one of the 15 C(6,2) combinatorial "
                "CPCV-OOS test-period equity trajectories for the champion "
                "config, each normalised to 1.0 at its own first test date. "
                "The blue band is the 25–75% quantile of equity across "
                "paths present at each date; the gold line is the median. "
                "Breaks in a line mark the train-period gap of a path whose "
                "two test blocks are non-adjacent (e.g. blocks 1 and 3). "
                "Paths that share a start date collapse to identical "
                "equity because the naive model factory has no trained "
                "state to vary across folds — the visible dispersion "
                "therefore reflects different calendar windows (and "
                "different base dates for the 1.0 normalisation), not "
                "different trained models. The genuine path-dependency "
                "signal lives in the Sharpe distribution below, where "
                "each path is computed on a different subset of dates."
            )

        # --- Phase 11 — Sharpe distribution across CPCV paths
        oos_sharpe = None
        if metrics and "sharpe_ratio" in metrics:
            try:
                oos_sharpe = float(metrics["sharpe_ratio"])
            except (TypeError, ValueError):
                oos_sharpe = None

        n_trials = load_cpcv_total_trials()

        psr = None
        if oos_sharpe is not None and equity_df is not None and equity_df.height >= 2:
            n_obs_psr = int(equity_df.height - 1)
            psr = _probabilistic_sharpe_ratio(oos_sharpe, n_obs_psr)

        fig_violin = _chart_sharpe_violin(
            cpcv_paths,
            oos_sharpe=oos_sharpe,
            n_trials=n_trials,
            psr=psr,
        )
        if fig_violin is not None:
            st.plotly_chart(fig_violin, width="stretch")
            dsr_note = (
                f" — DSR deflation uses n_trials={n_trials} from the "
                "3-tier grid search"
                if n_trials
                else ""
            )
            st.caption(
                "Violin of the 15 per-path annualised Sharpes. Each "
                "Sharpe is computed on a different combination of test "
                "blocks (1–3 years of daily returns), so the spread "
                "captures regime sensitivity of the champion across "
                "non-overlapping calendar windows. The gold line is the "
                "realised walk-forward OOS Sharpe — landing inside the "
                "IQR means the full-sample run is consistent with the "
                "CPCV cloud (not a lucky fold). The dashed line is the "
                "Deflated-Sharpe expected maximum from best-of-n noise "
                f"strategies{dsr_note}; it is the benchmark for the "
                "grid-search winner as a whole, not a per-path threshold "
                "— so don't expect every path to exceed it. The formal "
                "statistical evidence that the edge survives multiple "
                "testing is the PSR in the stats box (≥ 0.95 is the "
                "standard confidence level)."
            )
        st.divider()

    # Equity curve with log toggle
    col_toggle, _ = st.columns([1, 4])
    with col_toggle:
        log_scale = st.checkbox("Log scale", value=False, key="bench_log")

    fig_equity = _chart_benchmark_equity(equity_df, log_scale=log_scale)
    st.plotly_chart(fig_equity, width="stretch")

    # Drawdown
    fig_dd = _chart_benchmark_drawdown(equity_df)
    st.plotly_chart(fig_dd, width="stretch")

    st.divider()

    # Top-N drawdowns ranked (Phase 2)
    st.subheader("Top Drawdowns — Ranked")
    fig_top_dd = _chart_top_drawdowns(equity_df, n=10, min_depth=0.01)
    if fig_top_dd is not None:
        st.plotly_chart(fig_top_dd, width="stretch")
        st.caption(
            "Peak-to-recovery drawdown events, sorted by depth (deepest at top). "
            "Only drawdowns ≥1% are shown. Ongoing drawdowns (series ends "
            "underwater) are flagged with a trailing asterisk. Durations are "
            "in trading days."
        )
        with st.expander("Drawdown detail table", expanded=False):
            _render_top_drawdowns_table(equity_df, n=10, min_depth=0.01)
    else:
        st.info("No drawdowns of at least 1% detected in this equity series.")

    st.divider()

    # Monthly returns calendar heatmap (Phase 1)
    st.subheader("Monthly Returns Heatmap")
    fig_cal = _chart_calendar_heatmap(equity_df)
    if fig_cal is not None:
        st.plotly_chart(fig_cal, width="stretch")
        st.caption(
            "Each cell shows portfolio return compounded within that month. "
            "The Annual column shows full-year compounded return; the Mean row "
            "shows the average across covered years (seasonality). Empty cells "
            "= months outside the backtest window."
        )
    else:
        st.info(
            "Not enough data to build the monthly heatmap — need at least "
            "2 months of equity history."
        )

    st.divider()

    # Metrics table
    if metrics:
        st.subheader("Performance Metrics")
        # Split metrics into two columns
        keys = list(_METRIC_LABELS.keys())
        mid = len(keys) // 2
        col_left, col_right = st.columns(2)

        with col_left:
            for key in keys[:mid]:
                if key in metrics:
                    val = metrics[key]
                    color = _metric_color(key, val)
                    label = _METRIC_LABELS[key]
                    formatted = _format_metric_value(key, val)
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:4px 8px; border-bottom:1px solid #333;'>"
                        f"<span style='color:#aaa;'>{label}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>{formatted}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        with col_right:
            for key in keys[mid:]:
                if key in metrics:
                    val = metrics[key]
                    color = _metric_color(key, val)
                    label = _METRIC_LABELS[key]
                    formatted = _format_metric_value(key, val)
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:4px 8px; border-bottom:1px solid #333;'>"
                        f"<span style='color:#aaa;'>{label}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>{formatted}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No metrics available.")

    st.divider()

    # Up / Down capture vs benchmark (Phase 5)
    st.subheader(f"Up / Down Capture vs {_BENCHMARK_TICKER}")
    fig_capture = _chart_capture_ratios(equity_df)
    if fig_capture is not None:
        st.plotly_chart(fig_capture, width="stretch")
        st.caption(
            f"Monthly returns classified by the sign of {_BENCHMARK_TICKER}'s "
            "return. **Up-Capture > 100%** means the portfolio captures more "
            f"upside than {_BENCHMARK_TICKER} in up months; "
            "**Down-Capture < 100%** means it loses less in down months. "
            "The Up/Down ratio summarizes the asymmetry (higher is better). "
            "Positive Month % shows how often the portfolio returns are "
            "positive (random chance ≈ 50%)."
        )
    else:
        st.info(
            "Insufficient data for capture ratios — need at least 2 months "
            "of equity history."
        )

    st.divider()

    # Return distribution + QQ (Phase 3)
    st.subheader("Return Distribution")
    fig_dist = _chart_return_distribution(equity_df)
    if fig_dist is not None:
        st.plotly_chart(fig_dist, width="stretch")
        st.caption(
            "Left: histogram of daily returns with a fitted normal PDF "
            "(dashed gold). The red-shaded region marks the left tail below "
            "VaR 5%; CVaR 5% is the mean return inside that tail. "
            "Right: Q-Q plot vs the fitted normal — points bending away "
            "from the line at the extremes are visual evidence of fat tails."
        )
    else:
        st.info(
            "Insufficient data for the return distribution — need at least "
            "30 daily returns."
        )

    st.divider()

    # Rolling Sharpe with slider
    st.subheader("Rolling Sharpe Ratio")
    window = st.slider(
        "Window (days)", min_value=60, max_value=504,
        value=252, step=21, key="bench_sharpe_window",
    )
    fig_sharpe = _chart_rolling_sharpe(equity_df, window=window)
    if fig_sharpe:
        st.plotly_chart(fig_sharpe, width="stretch")
    else:
        st.info(f"Insufficient data for {window}-day rolling Sharpe.")

    st.divider()

    # Market relationship — rolling beta/alpha/correlation (Phase 4)
    st.subheader(f"Market Relationship vs {_BENCHMARK_TICKER}")
    mr_window = st.slider(
        "Regression window (days)",
        min_value=60, max_value=252, value=126, step=21,
        key="bench_market_rel_window",
    )
    fig_mr = _chart_rolling_market_relationship(equity_df, window=mr_window)
    if fig_mr:
        st.plotly_chart(fig_mr, width="stretch")
        st.caption(
            "Rolling OLS of daily portfolio returns on benchmark returns. "
            "Top: beta (reference = 1 = market exposure). "
            "Middle: alpha, annualized (reference = 0 = no excess return). "
            "Bottom: linear correlation (reference = 0 = independent). "
            "A portfolio generating real alpha should show β near or below 1 "
            "with α persistently above 0."
        )
    else:
        st.info(f"Insufficient data for {mr_window}-day rolling OLS.")

    st.divider()

    # CAPM scatter with OLS regression (Phase 6)
    st.subheader(f"CAPM Scatter vs {_BENCHMARK_TICKER}")
    fig_capm = _chart_capm_scatter(equity_df)
    if fig_capm is not None:
        st.plotly_chart(fig_capm, width="stretch")
        st.caption(
            "Each point is a single trading day. The gold line is the "
            "full-sample OLS fit of `R_portfolio = α + β · R_benchmark`. "
            "The slope (β) is the static market exposure; the intercept, "
            "annualized by 252, is the alpha. R² quantifies how much of "
            "the portfolio variance is explained by benchmark moves — "
            "lower R² leaves more room for idiosyncratic alpha."
        )
    else:
        st.info(
            "Insufficient data for the CAPM scatter — need at least 3 "
            "daily returns with non-zero benchmark variance."
        )

    st.divider()

    # Weight heatmap
    st.subheader("Portfolio Weight Evolution")
    fig_heatmap = _chart_weight_heatmap(weights_df)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, width="stretch")
    else:
        st.info("No weight history available.")

    st.divider()

    # Portfolio concentration — effective N and Gini (Phase 8)
    st.subheader("Portfolio Concentration Over Time")
    fig_conc = _chart_concentration_evolution(weights_df)
    if fig_conc is not None:
        st.plotly_chart(fig_conc, width="stretch")
        st.caption(
            "**Effective N** (blue, left axis) = 1 / Σwᵢ² is the number "
            "of equally-weighted positions the book is equivalent to; "
            "implicit cash counts as a single slot, so a 50% cash + 1 "
            "stock portfolio scores 2. **Gini** (gold, right axis) is the "
            "inequality of the held positions (cash appended when positive) "
            "— 0 = perfectly equal, approaching 1 = fully concentrated. "
            "The dotted reference line marks the equivalent N if every "
            "slot sits at the HRP max_weight cap (0.06) — values above the "
            "line indicate a more balanced book than the cap alone would "
            "force."
        )
    else:
        st.info("Weight history unavailable for concentration metrics.")

    st.divider()

    # Trading costs — turnover + gross vs net equity (Phase 7)
    st.subheader("Trading Costs")
    fig_costs = _chart_turnover_and_costs(equity_df, weights_df)
    if fig_costs is not None:
        st.plotly_chart(fig_costs, width="stretch")
        st.caption(
            "Top: fraction of capital turned over at each rebalance "
            "(sum of |Δw| across tickers). Bottom: the gold dashed line "
            "is the counterfactual gross equity reconstructed by adding "
            "back the transaction costs deducted at every rebalance "
            "(15 bps default); the blue solid line is the realized net "
            "equity. The red shaded area is the cumulative cost drag."
        )
    else:
        st.info(
            "Cost data unavailable — weights parquet lacks turnover / costs "
            "columns, or no rebalance events occurred."
        )

    st.divider()

    # Return attribution waterfall (Phase 9)
    st.subheader("Return Attribution")
    if ticker_returns_df is not None and weights_df is not None:
        contributions = _compute_contribution_per_ticker(
            weights_df, ticker_returns_df
        )
        fig_waterfall = _chart_contribution_waterfall(contributions, top_n=15)
        if fig_waterfall is not None:
            st.plotly_chart(fig_waterfall, width="stretch")
            st.caption(
                "Each bar is the additive contribution of a single ticker "
                "to total portfolio return, expressed in percentage points. "
                "For every rebalance interval, the ticker's weight at the "
                "rebalance is multiplied by its compound simple return over "
                "the interval; the values are summed across all intervals. "
                "The top-15 tickers by |contribution| are shown individually; "
                "the remainder are aggregated into ``Others``. The final blue "
                "bar sums all contributions — the small gap versus the "
                "realized CAGR × years is due to cash carry, transaction "
                "costs and across-interval compounding."
            )
        else:
            st.info(
                "No ticker contributions to attribute — weights and returns "
                "did not align on any shared ticker."
            )
    else:
        st.info(
            "Return attribution requires ``ticker_returns.parquet`` — "
            "generated automatically by the next ``make benchmark-fast`` "
            "run after the Phase 9 backtester change."
        )

    # PDF export
    st.divider()
    pdf_path = DATA_DIR / "benchmark_report.pdf"
    if pdf_path.exists():
        with pdf_path.open("rb") as f:
            st.download_button(
                label="Download Benchmark Report (PDF)",
                data=f.read(),
                file_name="titanium_benchmark_report.pdf",
                mime="application/pdf",
                width="stretch",
            )
    else:
        st.caption(
            "PDF report not generated yet. Run `make benchmark` to produce "
            "`data/outputs/benchmark_report.pdf`."
        )


# ---------------------------------------------------------------------------
# Tab 1: Performance
# ---------------------------------------------------------------------------


def _latest_benchmark_weights(weights_df: Any) -> dict[str, float]:
    """Extract the most recent rebalance weights from benchmark_weights.parquet.

    Args:
        weights_df: Polars DataFrame with columns date, ticker, weight.

    Returns:
        ``{ticker: weight}`` from the latest rebalance date.
    """
    try:
        import polars as pl
    except ImportError:
        return {}

    latest_date = weights_df["date"].max()
    latest = weights_df.filter(pl.col("date") == latest_date)
    return {
        row["ticker"]: row["weight"]
        for row in latest.select(["ticker", "weight"]).to_dicts()
    }


def _render_benchmark_weight_table(weights_df: Any) -> None:
    """Render a table of the latest benchmark weights sorted by weight."""
    try:
        import polars as pl
    except ImportError:
        st.warning("Polars not available.")
        return

    latest_date = weights_df["date"].max()
    latest = (
        weights_df.filter(pl.col("date") == latest_date)
        .select(["ticker", "weight"])
        .sort("weight", descending=True)
    )
    rows = [
        {"Ticker": r["ticker"], "Weight": f"{r['weight']:.4f}"}
        for r in latest.to_dicts()
    ]
    st.caption(f"Rebalance date: {latest_date}")
    st.dataframe(rows, width="stretch", hide_index=True, height=400)


def _chart_weight_donut(
    weights: dict[str, float],
    title: str = "HRP Portfolio Weights",
    top_n: int = 15,
) -> go.Figure:
    """Plotly donut chart of portfolio weights (top N + Others + Cash).

    Args:
        weights: ``{ticker: weight}`` mapping.
        title: Chart title.
        top_n: Number of top tickers to show individually.
    """
    if not weights:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color=_TEXT)),
            paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
            font=dict(color=_TEXT),
        )
        return fig

    # Sort by weight descending (exclude zero-weight tickers)
    sorted_items = sorted(
        ((t, w) for t, w in weights.items() if w > _MIN_DISPLAY_WEIGHT),
        key=lambda x: x[1],
        reverse=True,
    )

    if len(sorted_items) > top_n:
        top = sorted_items[:top_n]
        others_val = sum(v for _, v in sorted_items[top_n:])
        tickers = [t for t, _ in top] + ["Others"]
        values = [v for _, v in top] + [others_val]
    else:
        tickers = [t for t, _ in sorted_items]
        values = [v for _, v in sorted_items]

    # Show implicit cash when invested fraction < 100%
    total_invested = sum(values)
    if total_invested < 1.0 - 1e-6:
        tickers.append("Cash")
        values.append(1.0 - total_invested)

    colors = [_ticker_color(i) for i in range(len(tickers))]
    if "Others" in tickers:
        colors[tickers.index("Others")] = "#555555"
    if "Cash" in tickers:
        colors[tickers.index("Cash")] = "#888888"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=tickers,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=10, color=_TEXT),
                hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
                insidetextorientation="horizontal",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_TEXT)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=False,
        uniformtext=dict(minsize=8, mode="hide"),
        height=550,
        margin=dict(t=50, b=80, l=80, r=80),
    )
    return fig


def _detect_decision_quality(decisions: dict[str, Any]) -> list[str]:
    """Detect quality issues in decisions data.

    Returns:
        List of warning messages (empty if no issues found).
    """
    warnings: list[str] = []
    decs = decisions.get("decisions", [])
    meta = decisions.get("metadata", {})

    if not decs:
        return ["No decisions found."]

    # Staleness check
    ts_str = decisions.get("timestamp", "")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str)
            age_hours = (
                datetime.now(timezone.utc) - ts
            ).total_seconds() / 3600
            if age_hours > 24:
                warnings.append(
                    f"Decisions are **{age_hours:.0f}h old** "
                    f"(generated {ts_str[:16]}). Re-run `make decide`."
                )
        except (ValueError, TypeError):
            pass

    # All same action
    actions = {d["action"] for d in decs}
    if len(actions) == 1:
        warnings.append(
            f"All {len(decs)} tickers classified as **{actions.pop()}** "
            f"-- no differentiation."
        )

    # All same confidence (fallback mode)
    confs = {d["confidence"] for d in decs}
    if len(confs) == 1:
        warnings.append(
            f"All confidences identical ({confs.pop():.2f}) "
            f"-- debate likely failed with no PatchTST fallback."
        )

    # Confidence source
    source = meta.get("confidence_source", "")
    if source in ("none", ""):
        warnings.append(
            "Confidence source: **none** -- "
            "no debate or PatchTST signal used."
        )

    # Schema version mismatch
    schema = meta.get("schema_version", "")
    if schema and schema != _EXPECTED_SCHEMA_VERSION:
        warnings.append(
            f"Old schema version ({schema}). "
            f"Re-run `make decide` to update."
        )

    # HRP config validation against CPCV-OOS validated params
    hrp_cfg = meta.get("hrp_config", {})
    if hrp_cfg:
        linkage = hrp_cfg.get("linkage_method", "")
        shrinkage = hrp_cfg.get("shrinkage", None)
        if linkage and linkage != "ward":
            warnings.append(
                f"HRP linkage is **{linkage}** "
                f"(validated: ward). Weights may be suboptimal."
            )
        if shrinkage is not True:
            warnings.append(
                "HRP shrinkage **disabled** "
                "(validated: Ledoit-Wolf). Covariance may be noisy."
            )

    # Max-weight violations
    hrp_weights = decisions.get("hrp_final_weights", {})
    max_w = hrp_cfg.get("max_weight", 0.25)
    violations = [t for t, w in hrp_weights.items() if w > max_w + 1e-6]
    if violations:
        warnings.append(
            f"**{len(violations)} tickers** exceed max_weight "
            f"({max_w:.4f}). Re-run `make decide` to fix."
        )

    return warnings


def _render_metric_cards(decisions: dict[str, Any]) -> None:
    """Render KPI metric cards."""
    decs = decisions.get("decisions", [])
    meta = decisions.get("metadata", {})
    n_total = len(decs)
    n_buy = sum(1 for d in decs if d["action"] == "BUY")
    n_hold = sum(1 for d in decs if d["action"] == "HOLD")
    n_sell = sum(1 for d in decs if d["action"] == "SELL")
    avg_conf = (
        sum(d["confidence"] for d in decs) / len(decs) if decs else 0.0
    )
    source = meta.get("confidence_source", "N/A")
    invested = meta.get("invested_fraction", 1.0)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("BUY", f"{n_buy}/{n_total}")
    col2.metric("HOLD", f"{n_hold}/{n_total}")
    col3.metric("SELL", f"{n_sell}/{n_total}")
    col4.metric("Avg Confidence", f"{avg_conf:.2f}")
    col5.metric("Invested", f"{invested:.1%}")
    col6.metric("Signal Source", source)


def _render_decision_table(decisions: dict[str, Any]) -> None:
    """Render per-ticker decision table with action filter."""
    decs = decisions.get("decisions", [])
    if not decs:
        st.info("No decisions available.")
        return

    # Action filter
    actions_present = sorted({d["action"] for d in decs})
    selected_actions = st.multiselect(
        "Filter by action",
        actions_present,
        default=actions_present,
        key="perf_action_filter",
    )

    filtered = [d for d in decs if d["action"] in selected_actions]
    sorted_decs = sorted(filtered, key=lambda d: d["weight"], reverse=True)

    # Summary table
    rows = []
    for d in sorted_decs:
        rows.append(
            {
                "Ticker": d["ticker"],
                "Action": d["action"],
                "Weight": f"{d['weight']:.4f}",
                "Confidence": f"{d['confidence']:.2f}",
            }
        )

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
    )

    # Full reasoning in expander
    has_reasoning = any(d.get("reasoning") for d in sorted_decs)
    if has_reasoning:
        with st.expander("Reasoning details", expanded=False):
            for d in sorted_decs:
                reasoning = d.get("reasoning", "")
                if reasoning:
                    color = ACTION_COLORS.get(d["action"], "#888")
                    st.markdown(
                        f"**<span style='color:{color};'>"
                        f"{d['ticker']}</span>** ({d['action']}): "
                        f"{reasoning}",
                        unsafe_allow_html=True,
                    )


def _chart_weight_comparison(
    raw: dict[str, float],
    final: dict[str, float],
    top_n: int = 15,
) -> go.Figure:
    """Bar chart comparing raw vs final HRP weights (top N tickers).

    Args:
        raw: Raw HRP weights.
        final: Final weights (after 3-tier adjustment).
        top_n: Number of top tickers to display.
    """
    # Sort by final weight descending, show top_n
    sorted_tickers = sorted(final.keys(), key=lambda t: final[t], reverse=True)
    tickers = sorted_tickers[:top_n]

    # Detect if tilt was actually applied
    no_tilt = all(
        abs(raw.get(t, 0) - final.get(t, 0)) < 1e-6
        for t in tickers
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Raw HRP",
            x=tickers,
            y=[raw.get(t, 0) for t in tickers],
            marker_color=_ACCENT_BLUE,
            opacity=0.6,
        )
    )
    if not no_tilt:
        fig.add_trace(
            go.Bar(
                name="After Tilt",
                x=tickers,
                y=[final.get(t, 0) for t in tickers],
                marker_color=_ACCENT_GREEN,
            )
        )

    subtitle = f" (Top {top_n})" if len(sorted_tickers) > top_n else ""
    title = (
        f"HRP Weights{subtitle} (no adjustments)"
        if no_tilt
        else f"Raw vs Final Weights{subtitle}"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=_TEXT)),
        barmode="group",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(gridcolor="#333", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#333", title="Weight", tickformat=".1%"),
        height=400,
        margin=dict(t=50, b=80, l=50, r=20),
    )
    return fig


def _render_concentration_metrics(weights: dict[str, float]) -> None:
    """Render portfolio concentration risk metrics.

    Args:
        weights: ``{ticker: weight}`` mapping.
    """
    active = {t: w for t, w in weights.items() if w > _MIN_DISPLAY_WEIGHT}
    if not active:
        return

    n_active = len(active)
    values = sorted(active.values(), reverse=True)
    hhi = sum(w ** 2 for w in values)
    eff_n = 1.0 / hhi if hhi > 0 else 0.0
    top5 = sum(values[:5])
    max_ticker = max(active, key=lambda t: active[t])
    max_weight = active[max_ticker]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Positions", n_active)
    col2.metric("Effective N", f"{eff_n:.1f}")
    col3.metric("Top-5 Weight", f"{top5:.1%}")
    col4.metric(f"Max ({max_ticker})", f"{max_weight:.2%}")

    # Concentration warnings
    equal_weight = 1.0 / n_active if n_active > 0 else 0.0
    concentrated = [
        t for t, w in active.items() if w > 2.0 * equal_weight
    ]
    if concentrated:
        st.warning(
            f"**{len(concentrated)} tickers** exceed 2x equal-weight "
            f"({2.0 * equal_weight:.2%}): "
            f"{', '.join(sorted(concentrated)[:5])}"
            f"{'...' if len(concentrated) > 5 else ''}"
        )
    if top5 > 0.50 and n_active >= 10:
        st.warning(
            f"Top-5 concentration is **{top5:.1%}** — "
            f"more than half the portfolio in 5 assets."
        )


def _chart_confidence_histogram(
    decisions: dict[str, Any],
) -> go.Figure | None:
    """Plotly histogram of confidence scores grouped by action.

    Args:
        decisions: Full decisions.json dict.

    Returns:
        Plotly figure or None if no decisions.
    """
    decs = decisions.get("decisions", [])
    if not decs:
        return None

    fig = go.Figure()
    for action, color in (
        ("BUY", _ACCENT_GREEN),
        ("HOLD", _ACCENT_GOLD),
        ("SELL", _ACCENT_RED),
    ):
        confs = [d["confidence"] for d in decs if d["action"] == action]
        if confs:
            fig.add_trace(go.Histogram(
                x=confs, name=action, marker_color=color,
                opacity=0.7, nbinsx=20,
            ))

    # Auto-scale x-axis to data range with small padding
    all_confs = [d["confidence"] for d in decs]
    c_min, c_max = min(all_confs), max(all_confs)
    pad = max((c_max - c_min) * 0.15, 0.01)

    fig.update_layout(
        title=dict(
            text="Confidence Distribution", font=dict(size=16, color=_TEXT),
        ),
        barmode="stack",
        xaxis=dict(
            title="Confidence",
            range=[c_min - pad, c_max + pad],
            gridcolor="#333",
        ),
        yaxis=dict(title="Count", gridcolor="#333"),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT)),
        height=300,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def _chart_action_distribution(
    decisions: dict[str, Any],
) -> go.Figure | None:
    """Plotly donut chart of BUY/HOLD/SELL action counts.

    Args:
        decisions: Full decisions.json dict.

    Returns:
        Plotly figure or None if no decisions.
    """
    decs = decisions.get("decisions", [])
    if not decs:
        return None

    counts: dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for d in decs:
        action = d.get("action", "")
        if action in counts:
            counts[action] += 1

    labels = [a for a, c in counts.items() if c > 0]
    values = [counts[a] for a in labels]
    colors = [ACTION_COLORS.get(a, "#888") for a in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+value",
        textfont=dict(size=12, color=_TEXT),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])
    fig.update_layout(
        title=dict(
            text="Action Distribution", font=dict(size=16, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=False,
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def _render_weight_delta(
    current_weights: dict[str, float],
    bench_weights_df: Any,
) -> None:
    """Show weight changes between current decisions and last benchmark rebalance.

    Args:
        current_weights: Current portfolio weights from decisions.json.
        bench_weights_df: Polars DataFrame from benchmark_weights.parquet.
    """
    import polars as pl

    latest_date = bench_weights_df["date"].max()
    prev = {
        r["ticker"]: r["weight"]
        for r in bench_weights_df.filter(
            pl.col("date") == latest_date
        ).select(["ticker", "weight"]).to_dicts()
    }

    all_tickers = set(current_weights) | set(prev)
    deltas = []
    for t in all_tickers:
        curr = current_weights.get(t, 0.0)
        old = prev.get(t, 0.0)
        delta = curr - old
        if abs(delta) > 1e-6:
            deltas.append({
                "ticker": t, "current": curr,
                "previous": old, "delta": delta,
            })

    if not deltas:
        st.info("No weight changes from last benchmark rebalance.")
        return

    deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)

    entered = [d for d in deltas if d["previous"] < _MIN_DISPLAY_WEIGHT and d["current"] > _MIN_DISPLAY_WEIGHT]
    exited = [d for d in deltas if d["current"] < _MIN_DISPLAY_WEIGHT and d["previous"] > _MIN_DISPLAY_WEIGHT]
    turnover = sum(abs(d["delta"]) for d in deltas) / 2

    col1, col2, col3 = st.columns(3)
    col1.metric("New Positions", len(entered))
    col2.metric("Exited", len(exited))
    col3.metric("Est. Turnover", f"{turnover:.1%}")

    st.caption(f"vs benchmark rebalance {latest_date}")
    rows = [
        {
            "Ticker": d["ticker"],
            "Previous": f"{d['previous']:.4f}",
            "Current": f"{d['current']:.4f}",
            "Delta": f"{d['delta']:+.4f}",
        }
        for d in deltas[:15]
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def tab_performance(
    decisions: dict[str, Any] | None,
    bench_weights: Any | None = None,
    bench_metrics: dict[str, float] | None = None,
) -> None:
    """Render the Performance tab.

    Falls back to benchmark weight data when decisions.json is missing
    or stale (fewer tickers than the benchmark universe).
    """
    st.header("Portfolio Performance")

    # --- Benchmark metrics (always shown when available) ---
    if bench_metrics:
        st.caption("Walk-Forward Backtest Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe", f"{bench_metrics.get('sharpe_ratio', 0):.3f}")
        col2.metric("CAGR", f"{bench_metrics.get('cagr', 0):.2%}")
        col3.metric("Max DD", f"{bench_metrics.get('max_drawdown', 0):.2%}")
        col4.metric("Positions", f"{bench_metrics.get('avg_positions', 0):.0f}")

    # --- Determine data source ---
    has_decisions = decisions is not None and decisions.get("decisions")
    has_bench = bench_weights is not None

    if has_decisions:
        assert decisions is not None  # narrowed by has_decisions
        # Quality warnings
        quality_warnings = _detect_decision_quality(decisions)
        if quality_warnings:
            with st.expander(
                f"Data Quality Issues ({len(quality_warnings)})",
                expanded=True,
            ):
                for w in quality_warnings:
                    st.warning(w)

        ts = decisions.get("timestamp", "N/A")
        meta = decisions.get("metadata", {})
        invested_frac = meta.get("invested_fraction", 1.0)
        caption = f"Last agent decision: {ts}"
        if invested_frac < 1.0 - 1e-6:
            caption += f"  |  Invested: {invested_frac:.1%}  |  Cash: {1.0 - invested_frac:.1%}"
        st.caption(caption)
        _render_metric_cards(decisions)

        # Use per-decision weights (actual allocation after 3-tier adjustment)
        decs = decisions.get("decisions", [])
        weights_final = {d["ticker"]: d["weight"] for d in decs}
        weights_raw = decisions.get("hrp_raw_weights", {})
    elif has_bench:
        st.info(
            "Agent decisions not available. Showing latest benchmark "
            "weights from walk-forward backtest."
        )
        weights_final = _latest_benchmark_weights(bench_weights)
        weights_raw = weights_final  # no raw/tilt distinction in benchmark
    else:
        st.warning(
            "No decision or benchmark data available. "
            "Run `make benchmark-fast` or `make decide`."
        )
        return

    # --- Concentration metrics ---
    st.divider()
    st.subheader("Concentration Risk")
    _render_concentration_metrics(weights_final)

    # --- Charts side by side ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            _chart_weight_donut(weights_final), width="stretch"
        )
    with col_right:
        st.plotly_chart(
            _chart_weight_comparison(weights_raw, weights_final),
            width="stretch",
        )

    # --- Confidence distribution + action breakdown ---
    if has_decisions:
        assert decisions is not None  # narrowed by has_decisions
        fig_conf = _chart_confidence_histogram(decisions)
        fig_action = _chart_action_distribution(decisions)
        if fig_conf or fig_action:
            col_conf, col_action = st.columns([3, 1])
            if fig_conf:
                with col_conf:
                    st.plotly_chart(fig_conf, width="stretch")
            if fig_action:
                with col_action:
                    st.plotly_chart(fig_action, width="stretch")

    st.divider()

    # --- Weight delta vs benchmark ---
    if has_decisions and has_bench:
        st.subheader("Weight Changes vs Last Rebalance")
        _render_weight_delta(weights_final, bench_weights)
        st.divider()

    # --- Decision table ---
    if has_decisions:
        assert decisions is not None  # narrowed by has_decisions
        st.subheader("Decisions")
        _render_decision_table(decisions)
        cluster = decisions.get("cluster_order", [])
        if cluster:
            st.caption(f"HRP Cluster Order: {' → '.join(cluster)}")

    # --- Benchmark weights (always shown when available) ---
    if has_bench:
        if has_decisions:
            st.divider()
        st.subheader("Latest Benchmark Weights (Walk-Forward)")
        _render_benchmark_weight_table(bench_weights)


# ---------------------------------------------------------------------------
# Tab 2: War Room
# ---------------------------------------------------------------------------


def _render_agent_report(report: dict[str, Any]) -> None:
    """Render a single agent report as a styled chat bubble."""
    agent = report.get("agent", "unknown")
    style = AGENT_STYLES.get(agent, {"color": "#888", "icon": "?", "label": agent})

    signal = report.get("signal", "N/A")
    confidence = report.get("confidence", 0.0)
    reasoning = report.get("reasoning", "No reasoning provided.")
    factors = report.get("key_factors", [])
    sources = report.get("sources_cited", [])

    st.markdown(
        f"""<div style="
            border-left: 4px solid {style['color']};
            padding: 12px 16px;
            margin: 8px 0;
            background: {_CARD_BG};
            border-radius: 0 8px 8px 0;
        ">
        <strong style="color: {style['color']};">
            {style['icon']} {style['label']}
        </strong>
        <span style="float: right; color: #888;">
            {signal.upper()} | conf: {confidence:.2f}
        </span>
        <hr style="border-color: #333; margin: 8px 0;">
        <p style="color: {_TEXT}; margin: 4px 0;">{reasoning}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if factors:
        with st.expander("Key Factors"):
            for f in factors:
                st.markdown(f"- {f}")

    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- {s}")


def _render_final_decision_card(decision: dict[str, Any]) -> None:
    """Render highlighted final decision card."""
    action = decision.get("action", "N/A")
    color = ACTION_COLORS.get(action, "#888")

    st.markdown(
        f"""<div style="
            border: 2px solid {color};
            padding: 16px;
            margin: 12px 0;
            background: {_CARD_BG};
            border-radius: 8px;
            text-align: center;
        ">
        <h3 style="color: {color}; margin: 0;">
            {AGENT_STYLES['pm']['icon']} FINAL DECISION: {action}
        </h3>
        <p style="color: {_TEXT}; margin: 8px 0;">
            Weight: <strong>{decision.get('weight', decision.get('suggested_weight', 0)):.4f}</strong> |
            Confidence: <strong>{decision.get('confidence', 0):.2f}</strong>
        </p>
        <p style="color: #AAA; font-size: 0.9em;">
            {decision.get('reasoning', '')[:200]}
        </p>
        </div>""",
        unsafe_allow_html=True,
    )

    dissent = decision.get("dissenting_view", "")
    if dissent:
        st.markdown(
            f"""<div style="
                border-left: 3px solid {_ACCENT_RED};
                padding: 8px 12px;
                margin: 4px 0;
                background: #1a1a2e;
                border-radius: 0 4px 4px 0;
                font-size: 0.85em;
                color: #CCC;
            ">
            <strong style="color: {_ACCENT_RED};">🐻 Dissent:</strong> {dissent}
            </div>""",
            unsafe_allow_html=True,
        )


def _replay_debate(reports: list[dict[str, Any]], delay: float = 1.5) -> None:
    """Animate debate replay — shows reports one by one with typing effect.

    Args:
        reports: Agent reports to replay.
        delay: Seconds between each report appearing.
    """
    for i, report in enumerate(reports):
        agent = report.get("agent", "unknown")
        style = AGENT_STYLES.get(
            agent, {"icon": "?", "label": agent, "color": "#888"}
        )

        # Typing indicator
        placeholder = st.empty()
        placeholder.markdown(
            f"""<div style="
                border-left: 4px solid {style['color']};
                padding: 12px 16px; margin: 8px 0;
                background: {_CARD_BG}; border-radius: 0 8px 8px 0;
                opacity: 0.6;
            ">
            <strong style="color: {style['color']};">
                {style['icon']} {style['label']}
            </strong>
            <span style="color: #888; margin-left: 8px;">typing...</span>
            </div>""",
            unsafe_allow_html=True,
        )
        time.sleep(delay)

        # Replace with full report
        placeholder.empty()
        _render_agent_report(report)


def _run_live_debate_thread(
    tickers: list[str],
    result_queue: queue.Queue,  # type: ignore[type-arg]
) -> None:
    """Worker thread that runs agent debate and posts updates to queue.

    Args:
        tickers: Tickers to analyse.
        result_queue: Queue for posting (event_type, data) tuples.
    """
    try:
        import sys
        from pathlib import Path

        # Ensure project root is on sys.path so "src." imports resolve
        _project_root = str(Path(__file__).resolve().parents[2])
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from src.agents.graph import run_agent_debate

        def on_node(ticker: str, node: str, output: dict) -> None:
            result_queue.put(("node", ticker, node, output))

        decisions, states = run_agent_debate(
            tickers=tickers, on_node_complete=on_node
        )
        result_queue.put(("done", decisions, states))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


# Node-name to agent-key mapping for display
_NODE_TO_AGENT: dict[str, str] = {
    "load_context": "context",
    "rag_retrieval": "rag",
    "technical": "technical",
    "fundamental": "fundamental",
    "bear": "bear",
    "portfolio_manager": "pm",
}


def tab_war_room(
    decisions: dict[str, Any],
    debate: dict[str, Any] | None,
) -> None:
    """Render the War Room tab with replay and live run modes."""
    st.header("War Room — Agent Debate")

    tickers = decisions.get("tickers", [])
    if not tickers:
        st.warning("No tickers in decisions.")
        return

    selected = st.selectbox("Select Ticker", tickers, key="war_room_ticker")

    # --- Mode selector ---
    col_replay, col_live = st.columns(2)

    # Replay button (only if debate history exists)
    has_debate = debate is not None and selected in debate
    with col_replay:
        replay_clicked = st.button(
            "Replay Debate",
            disabled=not has_debate,
            key="replay_btn",
            width="stretch",
        )

    with col_live:
        is_running = st.session_state.get("live_running", False)
        live_clicked = st.button(
            "Run Live Debate" if not is_running else "Running...",
            disabled=is_running,
            key="live_btn",
            width="stretch",
        )

    st.divider()

    # --- Replay mode ---
    if replay_clicked and has_debate:
        assert debate is not None  # narrowed by has_debate
        ticker_state = debate[selected]
        reports = ticker_state.get("reports", [])
        if reports:
            _replay_debate(reports, delay=1.5)
        # Show final decision after replay
        decs = decisions.get("decisions", [])
        for d in decs:
            if d["ticker"] == selected:
                time.sleep(1.0)
                _render_final_decision_card(d)
                break
        return

    # --- Live run mode ---
    if live_clicked and not is_running:
        st.session_state["live_running"] = True
        st.session_state["live_events"] = []
        st.session_state["live_queue"] = queue.Queue()

        t = threading.Thread(
            target=_run_live_debate_thread,
            args=([selected], st.session_state["live_queue"]),
            daemon=True,
        )
        t.start()
        st.rerun()

    if st.session_state.get("live_running", False):
        _render_live_debate()
        return

    # --- Static mode (default) ---
    if has_debate:
        assert debate is not None  # narrowed by has_debate
        ticker_state = debate[selected]
        reports = ticker_state.get("reports", [])

        # Debate log timeline
        debate_log = ticker_state.get("debate_log", [])
        if debate_log:
            with st.expander("Debate Timeline", expanded=False):
                for entry in debate_log:
                    st.text(entry)

        # Agent reports
        if reports:
            for report in reports:
                _render_agent_report(report)
        else:
            st.info("No individual agent reports available for this ticker.")
    else:
        st.info(
            "No debate history available. Click **Run Live Debate** to "
            "start a new debate, or run `make decide` first."
        )

    # Always show final decision from decisions.json
    st.divider()
    decs = decisions.get("decisions", [])
    for d in decs:
        if d["ticker"] == selected:
            _render_final_decision_card(d)
            break


def _render_live_debate() -> None:
    """Poll the live debate queue and render updates."""
    q: queue.Queue = st.session_state.get("live_queue", queue.Queue())  # type: ignore[type-arg]
    events: list = st.session_state.get("live_events", [])

    # Drain queue
    while True:
        try:
            event = q.get_nowait()
            events.append(event)
        except queue.Empty:
            break

    st.session_state["live_events"] = events

    # Check if done
    is_done = any(e[0] in ("done", "error") for e in events)

    # Render events
    for event in events:
        if event[0] == "node":
            _, ticker, node_name, output = event

            # Show node completion
            reports = output.get("reports", [])
            if reports:
                for r in reports:
                    _render_agent_report(r)
            elif output.get("final_decision"):
                decision = output["final_decision"]
                _render_final_decision_card(decision)
            else:
                # Context/RAG nodes
                log = output.get("debate_log", [])
                if log:
                    st.markdown(
                        f"""<div style="
                            padding: 8px 12px; margin: 4px 0;
                            background: {_CARD_BG};
                            border-radius: 4px;
                            color: #888; font-size: 0.85em;
                        ">⚙️ {log[-1]}</div>""",
                        unsafe_allow_html=True,
                    )

        elif event[0] == "error":
            st.error(f"Debate failed: {event[1]}")

        elif event[0] == "done":
            st.success("Debate complete!")

    if is_done:
        st.session_state["live_running"] = False
        # Clear caches so new data appears
        load_decisions.clear()
        load_debate_history.clear()
    else:
        # Progress bar based on completed nodes (6 per ticker)
        completed_nodes = {
            (e[1], e[2]) for e in events if e[0] == "node"
        }
        total_expected = 6  # load_context, rag, technical, fundamental, bear, pm
        progress = min(len(completed_nodes) / total_expected, 0.99)
        st.progress(
            progress,
            text=f"Agents are deliberating... ({len(completed_nodes)}/{total_expected} nodes)",
        )
        time.sleep(1.0)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 3: Microstructure
# ---------------------------------------------------------------------------


def _chart_quantile_fan(
    forecast_rows: list[dict[str, Any]],
    ticker: str,
    last_close: float | None = None,
) -> go.Figure:
    """Plotly fan chart with quantile confidence bands.

    Args:
        forecast_rows: List of forecast row dicts from forecast.parquet.
        ticker: Ticker symbol for chart title.
        last_close: Last known close price.  When provided, a horizontal
            reference line is drawn so the user can visually compare the
            forecast against the current price.
    """
    # Detect quantile columns
    if not forecast_rows:
        fig = go.Figure()
        fig.update_layout(
            title=f"No forecast data for {ticker}",
            paper_bgcolor=_DARK_BG,
            plot_bgcolor=_DARK_BG,
            font=dict(color=_TEXT),
        )
        return fig

    sample = forecast_rows[0]
    _raw_q_cols = [
        c for c in sample.keys() if c.startswith("PatchTST-") and c not in ("date", "ds")
    ]

    def _col_to_quantile(col: str) -> float:
        """Return the quantile level for sorting (old and new NeuralForecast naming)."""
        if "-lo-" in col:
            return (100.0 - float(col.split("-lo-")[1])) / 200.0
        if "-hi-" in col:
            return (100.0 + float(col.split("-hi-")[1])) / 200.0
        if "median" in col:
            return 0.5
        # Old format: PatchTST-q0.1, PatchTST-q0.25, etc.
        try:
            return float(col.split("-q")[1])
        except (IndexError, ValueError):
            return 0.5

    q_cols = sorted(_raw_q_cols, key=_col_to_quantile)

    # Extract dates and quantile values
    dates = [r.get("ds", r.get("date", i)) for i, r in enumerate(forecast_rows)]
    # Convert dates to string for Plotly
    dates_str = [str(d) for d in dates]

    fig = go.Figure()

    # Last close reference line (drawn first so bands layer on top)
    if last_close is not None:
        fig.add_hline(
            y=last_close,
            line_dash="dot",
            line_color=_ACCENT_GOLD,
            opacity=0.8,
            annotation_text=f"Close: {last_close:,.2f}",
            annotation_position="top left",
            annotation_font_color=_ACCENT_GOLD,
            annotation_font_size=11,
        )

    # Fan bands (outer to inner for proper layering)
    band_pairs = []
    if len(q_cols) >= 5:
        # q0.1 - q0.9 (90% band)
        band_pairs.append((q_cols[0], q_cols[-1], "90% CI", "rgba(74,144,217,0.15)"))
        # q0.25 - q0.75 (50% band)
        band_pairs.append((q_cols[1], q_cols[-2], "50% CI", "rgba(74,144,217,0.30)"))

    for q_low, q_high, name, fill_color in band_pairs:
        low_vals = [r.get(q_low, 0) for r in forecast_rows]
        high_vals = [r.get(q_high, 0) for r in forecast_rows]

        fig.add_trace(
            go.Scatter(
                x=dates_str,
                y=high_vals,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dates_str,
                y=low_vals,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                name=name,
                hoverinfo="skip",
            )
        )

    # Median line
    if q_cols:
        median_col = q_cols[len(q_cols) // 2]
        median_vals = [r.get(median_col, 0) for r in forecast_rows]
        fig.add_trace(
            go.Scatter(
                x=dates_str,
                y=median_vals,
                mode="lines",
                name="Median",
                line=dict(color=_ACCENT_BLUE, width=2),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"PatchTST Forecast — {ticker}",
            font=dict(size=16, color=_TEXT),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(gridcolor="#333", title="Date"),
        yaxis=dict(gridcolor="#333", title="Forecast Value"),
        height=450,
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(font=dict(color=_TEXT)),
    )
    return fig


def _render_prediction_card(pred: dict[str, Any], ticker: str) -> None:
    """Render prediction summary for a ticker."""
    prob_up = pred.get("prob_up", 0.0)
    expected_ret = pred.get("expected_return", 0.0)

    direction = "UP" if prob_up > 0.5 else "DOWN"

    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"{ticker} Direction",
        direction,
        delta=f"{prob_up:.1%} probability",
    )
    col2.metric("P(Up)", f"{prob_up:.2%}")
    col3.metric("Expected Return", f"{expected_ret:.4f}")


def _chart_ticker_weight_history(
    weights_df: Any,
    ticker: str,
) -> go.Figure | None:
    """Line chart showing a single ticker's weight evolution over time."""
    try:
        import polars as pl
    except ImportError:
        return None

    filtered = weights_df.filter(pl.col("ticker") == ticker).sort("date")
    if filtered.height < 2:
        return None

    dates = filtered["date"].to_list()
    weights = filtered["weight"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=weights, mode="lines",
        fill="tozeroy", fillcolor="rgba(74,144,217,0.2)",
        line=dict(color=_ACCENT_BLUE, width=2),
        name="Weight",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Weight Evolution", font=dict(size=16, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Portfolio Weight",
        yaxis_tickformat=".1%",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _chart_ticker_cumulative_return(
    equity_df: Any,
    ticker: str,
) -> go.Figure | None:
    """Line chart of a ticker's cumulative return vs portfolio vs benchmark."""
    try:
        # Availability probe: the function gracefully returns None when polars is unavailable.
        import polars as pl  # noqa: F401
    except ImportError:
        return None

    if equity_df is None or equity_df.height < 2:
        return None

    dates = equity_df["date"].to_list()
    port_vals = equity_df["portfolio_value"].to_list()
    bench_vals = equity_df["benchmark_value"].to_list()

    # Normalise to 1.0
    port_norm = [v / port_vals[0] for v in port_vals]
    bench_norm = [v / bench_vals[0] for v in bench_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=port_norm, name="Portfolio",
        line=dict(color=_ACCENT_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=bench_norm, name=f"Benchmark ({_BENCHMARK_TICKER})",
        line=dict(color=_ACCENT_GOLD, width=1.5, dash="dash"),
    ))

    fig.update_layout(
        title=dict(
            text="Cumulative Returns — Portfolio vs Benchmark",
            font=dict(size=16, color=_TEXT),
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return (1.0 = start)",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT)),
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _render_ticker_stats(weights_df: Any, ticker: str) -> None:
    """Render per-ticker statistics from benchmark weight history."""
    import polars as pl

    filtered = weights_df.filter(pl.col("ticker") == ticker).sort("date")
    if filtered.height == 0:
        return

    avg_weight = filtered["weight"].mean()
    max_weight = filtered["weight"].max()
    min_weight = filtered["weight"].min()
    n_rebalances = filtered.height
    n_retrained = filtered.filter(pl.col("retrained")).height

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Weight", f"{avg_weight:.2%}")
    col2.metric("Max Weight", f"{max_weight:.2%}")
    col3.metric("Min Weight", f"{min_weight:.2%}")
    col4.metric("Rebalances", n_rebalances)
    col5.metric("Retrains", n_retrained)


def tab_microstructure(
    forecast: dict[str, list[dict[str, Any]]] | None,
    predictions: dict[str, dict[str, Any]] | None,
    bench_weights: Any | None = None,
    bench_equity: Any | None = None,
) -> None:
    """Render the Microstructure tab.

    Shows per-ticker analysis: PatchTST forecasts (when available)
    and benchmark weight/return data for all tickers in the universe.
    """
    st.header("Microstructure — Per-Ticker Analysis")

    # Build ticker list from all available sources
    available: set[str] = set()
    if forecast:
        available.update(forecast.keys())
    if predictions:
        available.update(predictions.keys())
    if bench_weights is not None:
        try:
            available.update(bench_weights["ticker"].unique().to_list())
        except Exception:
            pass

    if not available:
        st.warning(
            "No ticker data available. "
            "Run `make benchmark-fast` or `make predict` first."
        )
        return

    tickers = sorted(available)
    selected = st.selectbox(
        "Select Ticker", tickers, key="micro_ticker"
    )

    # --- PatchTST prediction summary ---
    if predictions and selected in predictions:
        _render_prediction_card(predictions[selected], selected)
        st.divider()

    # --- PatchTST fan chart ---
    if forecast and selected in forecast:
        st.subheader("PatchTST Quantile Forecast")
        # Pass last_close for reference line (if available in predictions)
        close_ref = None
        if predictions and selected in predictions:
            close_ref = predictions[selected].get("last_close")
        fig = _chart_quantile_fan(forecast[selected], selected, last_close=close_ref)
        st.plotly_chart(fig, width="stretch")
        st.divider()

    # --- Benchmark per-ticker stats ---
    if bench_weights is not None:
        import polars as pl

        has_ticker = bench_weights.filter(
            pl.col("ticker") == selected
        ).height > 0

        if has_ticker:
            st.subheader(f"{selected} — Benchmark Stats")
            _render_ticker_stats(bench_weights, selected)

            fig_wh = _chart_ticker_weight_history(bench_weights, selected)
            if fig_wh:
                st.plotly_chart(fig_wh, width="stretch")
        else:
            st.info(f"No benchmark weight data for {selected}.")

    # --- Portfolio vs benchmark cumulative return ---
    if bench_equity is not None:
        fig_cr = _chart_ticker_cumulative_return(
            bench_equity, selected
        )
        if fig_cr:
            st.plotly_chart(fig_cr, width="stretch")

    # --- No data at all for this ticker ---
    if (
        (forecast is None or selected not in forecast)
        and (predictions is None or selected not in predictions)
        and bench_weights is None
    ):
        st.info(f"No data available for {selected}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render sidebar with About, methodology link, and GitHub."""
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "**Titanium Alpha** is a research project combining "
            "deep learning forecasts (PatchTST), an agentic "
            "debate layer (LangGraph), hierarchical risk parity "
            "allocation (HRP), and CPCV-OOS validation."
        )
        st.divider()
        st.markdown("### Links")
        st.markdown("- [GitHub repository](https://github.com/)")
        st.markdown("- [Methodology (docs/)](https://github.com/)")
        st.markdown("- [Benchmark analysis](https://github.com/)")
        st.divider()
        st.caption(
            "Walk-forward record: Sharpe=0.712, CAGR=13.35%, "
            "MaxDD=-18.43%, Beta=0.532 (rf=5%)."
        )


def _format_last_update(decisions: dict[str, Any] | None) -> str:
    """Return a short 'last update' string from decisions timestamp."""
    if not decisions or not decisions.get("timestamp"):
        return "no decisions.json yet"
    ts = decisions["timestamp"]
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, AttributeError):
        return str(ts)[:19]


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="Titanium Alpha",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Load all data
    decisions = load_decisions()
    debate = load_debate_history()
    forecast = load_forecast()
    predictions = load_predictions()
    bench_equity = load_benchmark_equity()
    bench_metrics = load_benchmark_metrics()
    bench_weights = load_benchmark_weights()
    bench_ticker_returns = load_ticker_returns()

    _render_sidebar()

    # Title + last update
    last_update = _format_last_update(decisions)
    st.markdown(
        f"""<h1 style="text-align: center; color: #4A90D9;">
        🏦 Titanium Alpha
        </h1>
        <p style="text-align: center; color: #888; margin-top: -10px;">
        Agentic Multi-Strategy Hedge Fund Dashboard
        </p>
        <p style="text-align: center; color: #666; font-size: 0.85em; margin-top: -8px;">
        Last update: <strong>{last_update}</strong>
        </p>""",
        unsafe_allow_html=True,
    )

    # Tabs
    tab0, tab1, tab2, tab3 = st.tabs(
        ["📊 Benchmark", "📈 Performance", "⚔️ War Room", "🔬 Microstructure"]
    )

    with tab0:
        tab_benchmark(
            bench_equity, bench_metrics, bench_weights, bench_ticker_returns
        )

    with tab1:
        tab_performance(decisions, bench_weights, bench_metrics)

    with tab2:
        if decisions:
            tab_war_room(decisions, debate)
        else:
            st.warning(
                "No decision data available. Run `make decide` from the "
                "project root to generate `data/outputs/decisions.json`, "
                "then refresh this page."
            )

    with tab3:
        tab_microstructure(forecast, predictions, bench_weights, bench_equity)


if __name__ == "__main__":
    main()
