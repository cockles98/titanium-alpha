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
    # Neutral
    neutral = {
        "beta", "annualized_volatility", "tracking_error",
        "benchmark_total_return", "avg_annual_turnover", "avg_positions",
    }

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


def _chart_benchmark_drawdown(equity_df: Any) -> go.Figure:
    """Plotly drawdown chart (filled area)."""
    dates = equity_df["date"].to_list()
    values = equity_df["portfolio_value"].to_list()

    # Compute drawdown
    peak = values[0]
    dd = []
    for v in values:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak if peak != 0 else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=dd, fill="tozeroy", name="Drawdown",
        line=dict(color=_ACCENT_RED, width=1),
        fillcolor="rgba(229, 57, 53, 0.3)",
    ))

    fig.update_layout(
        title=dict(text="Portfolio Drawdown", font=dict(size=16, color=_TEXT)),
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def _chart_rolling_sharpe(equity_df: Any, window: int = 252) -> go.Figure | None:
    """Rolling Sharpe ratio chart. Returns None if insufficient data."""
    import math

    try:
        import polars as pl
        dr_path = DATA_DIR / "benchmark_equity.parquet"
        df = pl.read_parquet(dr_path)
    except Exception:
        return None

    portfolio = df["portfolio_value"].to_list()
    benchmark = df["benchmark_value"].to_list()
    dates = df["date"].to_list()

    if len(portfolio) < window + 1:
        return None

    # Compute daily returns from equity values
    port_rets = [0.0] + [(portfolio[i] / portfolio[i - 1] - 1.0) for i in range(1, len(portfolio))]
    bench_rets = [0.0] + [(benchmark[i] / benchmark[i - 1] - 1.0) for i in range(1, len(benchmark))]

    # Rolling Sharpe
    def _rolling(rets: list[float], w: int) -> list[float | None]:
        result: list[float | None] = [None] * (w - 1)
        for i in range(w - 1, len(rets)):
            chunk = rets[i - w + 1: i + 1]
            mean_r = sum(chunk) / len(chunk)
            var_r = sum((r - mean_r) ** 2 for r in chunk) / (len(chunk) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 0.0
            sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
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


# Validated strategy configuration (Session 35 — CPCV-OOS approved)
_VALIDATED_CONFIG: dict[str, Any] = {
    "Model Factory": "NaiveModelFactory(lookback=1)",
    "rebalance_every": 1,
    "retrain_every": 126,
    "lookback_days": 63,
    "costs": "slippage=5bps + commission=10bps",
    "min_rebalance_delta": 0.01,
    "target_vol": 0.15,
    "vol_lookback": 21,
    "confidence_tilt_cap": 1.0,
    "max_weight": 0.15,
}

# Key findings from stress testing
_STRESS_FINDINGS: list[str] = [
    "Alpha is ultra-short-term: 1-day momentum + daily rebalance maximises Sharpe.",
    "Strategy supports up to ~30 bps total cost with OOS Sharpe > 1.5.",
    "At 50 bps cost Sharpe drops to ~0.89; at 75 bps strategy breaks.",
    "Vol targeting (15%, 21d lookback) crushes tail kurtosis from ~26 to ~9.4.",
    "Max weight cap at 15% per asset — above this, no performance gain.",
    "Bug fix: port_ret now uses start-of-day value (costs reflected in Sharpe).",
]


def tab_benchmark(
    equity_df: Any | None,
    metrics: dict[str, float] | None,
    weights_df: Any | None,
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
        st.caption("Parameters validated via Deflated Sharpe Ratio on 15 CPCV-OOS paths.")
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
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # Equity curve with log toggle
    col_toggle, _ = st.columns([1, 4])
    with col_toggle:
        log_scale = st.checkbox("Log scale", value=False, key="bench_log")

    fig_equity = _chart_benchmark_equity(equity_df, log_scale=log_scale)
    st.plotly_chart(fig_equity, use_container_width=True)

    # Drawdown
    fig_dd = _chart_benchmark_drawdown(equity_df)
    st.plotly_chart(fig_dd, use_container_width=True)

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

    # Rolling Sharpe with slider
    st.subheader("Rolling Sharpe Ratio")
    window = st.slider(
        "Window (days)", min_value=60, max_value=504,
        value=252, step=21, key="bench_sharpe_window",
    )
    fig_sharpe = _chart_rolling_sharpe(equity_df, window=window)
    if fig_sharpe:
        st.plotly_chart(fig_sharpe, use_container_width=True)
    else:
        st.info(f"Insufficient data for {window}-day rolling Sharpe.")

    st.divider()

    # Weight heatmap
    st.subheader("Portfolio Weight Evolution")
    fig_heatmap = _chart_weight_heatmap(weights_df)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No weight history available.")


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
    import polars as pl

    latest_date = weights_df["date"].max()
    latest = weights_df.filter(pl.col("date") == latest_date)
    return {
        row["ticker"]: row["weight"]
        for row in latest.select(["ticker", "weight"]).to_dicts()
    }


def _render_benchmark_weight_table(weights_df: Any) -> None:
    """Render a table of the latest benchmark weights sorted by weight."""
    import polars as pl

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
    st.dataframe(rows, use_container_width=True, hide_index=True, height=400)


def _chart_weight_donut(
    weights: dict[str, float],
    title: str = "HRP Portfolio Weights",
    top_n: int = 15,
) -> go.Figure:
    """Plotly donut chart of portfolio weights (top N + Others).

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

    # Sort by weight descending
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_items) > top_n:
        top = sorted_items[:top_n]
        others_val = sum(v for _, v in sorted_items[top_n:])
        tickers = [t for t, _ in top] + ["Others"]
        values = [v for _, v in top] + [others_val]
    else:
        tickers = [t for t, _ in sorted_items]
        values = [v for _, v in sorted_items]

    colors = [_ticker_color(i) for i in range(len(tickers))]
    if len(sorted_items) > top_n:
        colors[-1] = "#555555"  # gray for "Others"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=tickers,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textfont=dict(size=11, color=_TEXT),
                hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_TEXT)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=True,
        legend=dict(font=dict(color=_TEXT, size=10)),
        height=450,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def _render_metric_cards(decisions: dict[str, Any]) -> None:
    """Render KPI metric cards."""
    decs = decisions.get("decisions", [])
    n_buy = sum(1 for d in decs if d["action"] == "BUY")
    n_hold = sum(1 for d in decs if d["action"] == "HOLD")
    n_sell = sum(1 for d in decs if d["action"] == "SELL")
    avg_conf = (
        sum(d["confidence"] for d in decs) / len(decs) if decs else 0.0
    )
    n_obs = decisions.get("metadata", {}).get("n_observations", "N/A")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("BUY", n_buy)
    col2.metric("HOLD", n_hold)
    col3.metric("SELL", n_sell)
    col4.metric("Avg Confidence", f"{avg_conf:.2f}")
    col5.metric("Observations", n_obs)


def _render_decision_table(decisions: dict[str, Any]) -> None:
    """Render per-ticker decision table."""
    decs = decisions.get("decisions", [])
    if not decs:
        st.info("No decisions available.")
        return

    # Build table data
    rows = []
    for d in decs:
        rows.append(
            {
                "Ticker": d["ticker"],
                "Action": d["action"],
                "Weight": f"{d['weight']:.4f}",
                "Confidence": f"{d['confidence']:.2f}",
                "Reasoning": d.get("reasoning", "")[:120] + "...",
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def _chart_weight_comparison(
    raw: dict[str, float],
    final: dict[str, float],
    top_n: int = 20,
) -> go.Figure:
    """Bar chart comparing raw vs tilted HRP weights (top N tickers).

    Args:
        raw: Raw HRP weights.
        final: Tilted/final weights.
        top_n: Number of top tickers to display.
    """
    # Sort by final weight descending, show top_n
    sorted_tickers = sorted(final.keys(), key=lambda t: final[t], reverse=True)
    tickers = sorted_tickers[:top_n]

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
    fig.add_trace(
        go.Bar(
            name="After Tilt",
            x=tickers,
            y=[final.get(t, 0) for t in tickers],
            marker_color=_ACCENT_GREEN,
        )
    )

    subtitle = f" (Top {top_n})" if len(sorted_tickers) > top_n else ""
    fig.update_layout(
        title=dict(
            text=f"Raw vs Confidence-Tilted Weights{subtitle}",
            font=dict(size=16, color=_TEXT),
        ),
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

    # --- Determine data source ---
    has_decisions = decisions is not None and decisions.get("decisions")
    has_bench = bench_weights is not None

    if has_decisions:
        ts = decisions.get("timestamp", "N/A")
        st.caption(f"Last agent decision: {ts}")
        _render_metric_cards(decisions)

        weights_final = decisions.get("hrp_final_weights", {})
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

    # --- Metric summary from benchmark ---
    if bench_metrics and not has_decisions:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe", f"{bench_metrics.get('sharpe_ratio', 0):.3f}")
        col2.metric("CAGR", f"{bench_metrics.get('cagr', 0):.2%}")
        col3.metric("Max DD", f"{bench_metrics.get('max_drawdown', 0):.2%}")
        col4.metric("Positions", f"{bench_metrics.get('avg_positions', 0):.0f}")

    # --- Charts side by side ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            _chart_weight_donut(weights_final), use_container_width=True
        )
    with col_right:
        st.plotly_chart(
            _chart_weight_comparison(weights_raw, weights_final),
            use_container_width=True,
        )

    # --- Decision table ---
    if has_decisions:
        st.subheader("Decisions")
        _render_decision_table(decisions)
        cluster = decisions.get("cluster_order", [])
        if cluster:
            st.caption(f"HRP Cluster Order: {' → '.join(cluster)}")
    elif has_bench:
        st.subheader("Latest Benchmark Weights")
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
            Weight: <strong>{decision.get('weight', 0):.4f}</strong> |
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
            use_container_width=True,
        )

    with col_live:
        is_running = st.session_state.get("live_running", False)
        live_clicked = st.button(
            "Run Live Debate" if not is_running else "Running...",
            disabled=is_running,
            key="live_btn",
            use_container_width=True,
        )

    st.divider()

    # --- Replay mode ---
    if replay_clicked and has_debate:
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
            agent_key = _NODE_TO_AGENT.get(node_name, node_name)
            style = AGENT_STYLES.get(
                agent_key,
                {"icon": "⚙️", "label": node_name, "color": "#888"},
            )

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
        # Still running — show spinner and auto-refresh
        st.markdown(
            f"""<div style="
                text-align: center; padding: 20px;
                color: {_ACCENT_GOLD};
            ">⏳ Agents are deliberating...</div>""",
            unsafe_allow_html=True,
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

    color = _ACCENT_GREEN if prob_up > 0.5 else _ACCENT_RED
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
    weights_df: Any,
    ticker: str,
) -> go.Figure | None:
    """Line chart of a ticker's cumulative return vs portfolio vs benchmark."""
    try:
        import polars as pl
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
            text=f"Cumulative Returns — Portfolio vs Benchmark",
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
        st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig_wh, use_container_width=True)
        else:
            st.info(f"No benchmark weight data for {selected}.")

    # --- Portfolio vs benchmark cumulative return ---
    if bench_equity is not None:
        fig_cr = _chart_ticker_cumulative_return(
            bench_equity, bench_weights, selected
        )
        if fig_cr:
            st.plotly_chart(fig_cr, use_container_width=True)

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


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="Titanium Alpha",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Title
    st.markdown(
        """<h1 style="text-align: center; color: #4A90D9;">
        🏦 Titanium Alpha
        </h1>
        <p style="text-align: center; color: #888; margin-top: -10px;">
        Agentic Multi-Strategy Hedge Fund Dashboard
        </p>""",
        unsafe_allow_html=True,
    )

    # Load all data
    decisions = load_decisions()
    debate = load_debate_history()
    forecast = load_forecast()
    predictions = load_predictions()
    bench_equity = load_benchmark_equity()
    bench_metrics = load_benchmark_metrics()
    bench_weights = load_benchmark_weights()

    # Tabs
    tab0, tab1, tab2, tab3 = st.tabs(
        ["📊 Benchmark", "📈 Performance", "⚔️ War Room", "🔬 Microstructure"]
    )

    with tab0:
        tab_benchmark(bench_equity, bench_metrics, bench_weights)

    with tab1:
        tab_performance(decisions, bench_weights, bench_metrics)

    with tab2:
        if decisions:
            tab_war_room(decisions, debate)
        else:
            st.warning("No decision data available. Run `make decide` first.")

    with tab3:
        tab_microstructure(forecast, predictions, bench_weights, bench_equity)


if __name__ == "__main__":
    main()
