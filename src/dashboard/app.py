"""Streamlit dashboard for Titanium Alpha hedge fund.

Three tabs:
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
# Tab 1: Performance
# ---------------------------------------------------------------------------


def _chart_weight_donut(decisions: dict[str, Any]) -> go.Figure:
    """Plotly donut chart of HRP final weights."""
    weights = decisions.get("hrp_final_weights", {})
    tickers = list(weights.keys())
    values = list(weights.values())

    colors = [_ACCENT_BLUE, _ACCENT_GREEN, _ACCENT_RED, _ACCENT_GOLD]
    # Extend colors if more tickers
    while len(colors) < len(tickers):
        colors.append("#888888")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=tickers,
                values=values,
                hole=0.55,
                marker=dict(colors=colors[: len(tickers)]),
                textinfo="label+percent",
                textfont=dict(size=14, color=_TEXT),
                hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="HRP Portfolio Weights", font=dict(size=18, color=_TEXT)),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        showlegend=True,
        legend=dict(font=dict(color=_TEXT)),
        height=400,
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


def _chart_weight_comparison(decisions: dict[str, Any]) -> go.Figure:
    """Bar chart comparing raw vs tilted HRP weights."""
    raw = decisions.get("hrp_raw_weights", {})
    final = decisions.get("hrp_final_weights", {})
    tickers = list(final.keys())

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
    fig.update_layout(
        title=dict(
            text="Raw vs Confidence-Tilted Weights",
            font=dict(size=16, color=_TEXT),
        ),
        barmode="group",
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font=dict(color=_TEXT),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", title="Weight"),
        height=350,
        margin=dict(t=50, b=30, l=50, r=20),
    )
    return fig


def tab_performance(decisions: dict[str, Any]) -> None:
    """Render the Performance tab."""
    st.header("Portfolio Performance")

    # Timestamp
    ts = decisions.get("timestamp", "N/A")
    st.caption(f"Last decision: {ts}")

    # Metric cards
    _render_metric_cards(decisions)

    # Charts side by side
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            _chart_weight_donut(decisions), use_container_width=True
        )
    with col_right:
        st.plotly_chart(
            _chart_weight_comparison(decisions), use_container_width=True
        )

    # Decision table
    st.subheader("Decisions")
    _render_decision_table(decisions)

    # Cluster order
    cluster = decisions.get("cluster_order", [])
    if cluster:
        st.caption(f"HRP Cluster Order: {' → '.join(cluster)}")


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
) -> go.Figure:
    """Plotly fan chart with quantile confidence bands."""
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
    q_cols = sorted(
        [c for c in sample.keys() if c.startswith("PatchTST-") and c not in ("date", "ds")]
    )

    # Extract dates and quantile values
    dates = [r.get("ds", r.get("date", i)) for i, r in enumerate(forecast_rows)]
    # Convert dates to string for Plotly
    dates_str = [str(d) for d in dates]

    fig = go.Figure()

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


def tab_microstructure(
    forecast: dict[str, list[dict[str, Any]]] | None,
    predictions: dict[str, dict[str, Any]] | None,
) -> None:
    """Render the Microstructure tab."""
    st.header("Microstructure — PatchTST Forecasts")

    if forecast is None and predictions is None:
        st.warning(
            "No forecast or prediction data available. "
            "Run `make predict` first."
        )
        return

    # Ticker selector
    available = set()
    if forecast:
        available.update(forecast.keys())
    if predictions:
        available.update(predictions.keys())

    tickers = sorted(available)
    if not tickers:
        st.warning("No tickers found in forecast data.")
        return

    selected = st.selectbox(
        "Select Ticker", tickers, key="micro_ticker"
    )

    # Prediction summary
    if predictions and selected in predictions:
        _render_prediction_card(predictions[selected], selected)
        st.divider()

    # Fan chart
    if forecast and selected in forecast:
        fig = _chart_quantile_fan(forecast[selected], selected)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No quantile forecast available for {selected}.")


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

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["📈 Performance", "⚔️ War Room", "🔬 Microstructure"]
    )

    with tab1:
        if decisions:
            tab_performance(decisions)
        else:
            st.warning(
                "No decision data found. Run `make decide` to generate "
                "portfolio decisions."
            )

    with tab2:
        if decisions:
            tab_war_room(decisions, debate)
        else:
            st.warning("No decision data available. Run `make decide` first.")

    with tab3:
        tab_microstructure(forecast, predictions)


if __name__ == "__main__":
    main()
