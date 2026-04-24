"""Tests for dashboard Phase 2 — top drawdowns ranked."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_top_drawdowns,
    _detect_drawdown_periods,
    _format_dd_label,
)


def _dates(n: int, start: date = date(2024, 1, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# _detect_drawdown_periods
# ---------------------------------------------------------------------------


def test_detector_returns_empty_for_monotonic_series():
    equity = [100.0, 101.0, 102.0, 103.0, 104.0]
    out = _detect_drawdown_periods(equity, _dates(5))
    assert out == []


def test_detector_returns_empty_for_empty_or_singleton():
    assert _detect_drawdown_periods([], []) == []
    assert _detect_drawdown_periods([100.0], [date(2024, 1, 1)]) == []


def test_detector_finds_simple_v_shape():
    # Peak=100 at idx 0, trough=80 at idx 3, recovery at idx 6.
    equity = [100.0, 95.0, 85.0, 80.0, 90.0, 95.0, 100.0, 105.0]
    dates = _dates(8)
    events = _detect_drawdown_periods(equity, dates, min_depth=0.01)
    assert len(events) == 1
    e = events[0]
    assert e["depth"] == pytest.approx(-0.20, abs=1e-9)
    assert e["start"] == dates[0]
    assert e["trough"] == dates[3]
    assert e["end"] == dates[6]
    assert e["ongoing"] is False
    assert e["duration_days"] == 6  # dates[6] - dates[0]
    assert e["recovery_days"] == 3  # dates[6] - dates[3]


def test_detector_handles_ongoing_drawdown():
    # Series ends underwater → must mark ongoing=True, end=None, recovery_days=None.
    equity = [100.0, 105.0, 110.0, 100.0, 95.0, 90.0]
    dates = _dates(6)
    events = _detect_drawdown_periods(equity, dates, min_depth=0.01)
    assert len(events) == 1
    e = events[0]
    assert e["ongoing"] is True
    assert e["end"] is None
    assert e["recovery_days"] is None
    assert e["depth"] == pytest.approx((90.0 - 110.0) / 110.0, rel=1e-9)
    assert e["start"] == dates[2]  # peak at idx 2
    assert e["trough"] == dates[5]


def test_detector_finds_multiple_independent_events():
    # Two recovered drawdowns, each ~10%.
    equity = [
        100.0, 95.0, 90.0, 100.0,  # DD1 peak→trough→recov
        110.0, 105.0, 99.0, 110.0,  # DD2 peak→trough→recov
    ]
    dates = _dates(8)
    events = _detect_drawdown_periods(equity, dates, min_depth=0.01)
    assert len(events) == 2
    # Sorted deepest-first.
    assert events[0]["depth"] <= events[1]["depth"]


def test_detector_filters_below_min_depth():
    # 0.5% dip should be filtered out with default min_depth=0.01.
    equity = [100.0, 99.5, 100.5]
    events = _detect_drawdown_periods(equity, _dates(3), min_depth=0.01)
    assert events == []
    # With relaxed threshold, it appears.
    events_loose = _detect_drawdown_periods(equity, _dates(3), min_depth=0.001)
    assert len(events_loose) == 1


def test_detector_tracks_deepest_trough_within_event():
    # Initial dip to 90, then deeper dip to 75, then recovery.
    equity = [100.0, 90.0, 92.0, 80.0, 75.0, 85.0, 100.0]
    dates = _dates(7)
    events = _detect_drawdown_periods(equity, dates, min_depth=0.01)
    assert len(events) == 1
    e = events[0]
    assert e["trough"] == dates[4]  # deeper trough at idx 4, not idx 1
    assert e["depth"] == pytest.approx(-0.25, abs=1e-9)


def test_detector_sorts_deepest_first():
    equity = [
        100.0, 98.0, 100.0,   # small DD, -2%
        120.0, 90.0, 120.0,   # big DD, -25%
        130.0, 125.0, 130.0,  # medium DD, ~-3.85%
    ]
    events = _detect_drawdown_periods(equity, _dates(9), min_depth=0.01)
    depths = [e["depth"] for e in events]
    assert depths == sorted(depths)  # ascending = deepest first


# ---------------------------------------------------------------------------
# _format_dd_label
# ---------------------------------------------------------------------------


def test_format_label_recovered():
    event = {
        "start": date(2022, 1, 15),
        "trough": date(2022, 6, 1),
        "end": date(2023, 3, 10),
        "depth": -0.1,
        "duration_days": 419,
        "recovery_days": 282,
        "ongoing": False,
    }
    label = _format_dd_label(event)
    assert "2022-01" in label
    assert "2023-03" in label
    assert "419d" in label
    assert "*" not in label


def test_format_label_ongoing_has_asterisk():
    event = {
        "start": date(2026, 2, 10),
        "trough": date(2026, 4, 5),
        "end": None,
        "depth": -0.07,
        "duration_days": 49,
        "recovery_days": None,
        "ongoing": True,
    }
    label = _format_dd_label(event)
    assert label.endswith("*")
    assert "ongoing" in label
    assert "49d" in label


# ---------------------------------------------------------------------------
# _chart_top_drawdowns
# ---------------------------------------------------------------------------


def _synthetic_equity_with_known_dds() -> pl.DataFrame:
    """Build an equity frame with exactly 3 drawdowns of known depth."""
    # DD #1: -15%  |  DD #2: -5%  |  DD #3: -10%
    equity = [
        100.0,
        # DD1: 100 → 85 → 100  (depth -15%)
        90.0, 85.0, 92.0, 100.0,
        # DD2: 110 → 104.5 → 110  (depth -5%)
        110.0, 104.5, 110.0,
        # DD3: 120 → 108 → 120  (depth -10%)
        120.0, 108.0, 120.0,
    ]
    return pl.DataFrame({
        "date": _dates(len(equity)),
        "portfolio_value": equity,
        "benchmark_value": equity,  # irrelevant for this chart
    })


def test_chart_returns_none_when_no_drawdown():
    df = pl.DataFrame({
        "date": _dates(5),
        "portfolio_value": [100.0, 101.0, 102.0, 103.0, 104.0],
        "benchmark_value": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    assert _chart_top_drawdowns(df) is None


def test_chart_returns_figure_with_single_bar_trace():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


def test_chart_respects_top_n():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df, n=2)
    assert len(fig.data[0].x) == 2


def test_chart_sorts_deepest_at_top():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df, n=10)
    depths = list(fig.data[0].x)
    # Depths are negative floats sorted ascending (most negative first).
    assert depths == sorted(depths)
    # y-axis reversed so most-negative (deepest) lands at the top visually.
    assert fig.layout.yaxis.autorange == "reversed"


def test_chart_tooltip_mentions_depth_and_status():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df)
    hover = fig.data[0].hovertemplate
    assert "Depth" in hover
    assert "Status" in hover
    assert "Recovery" in hover


def test_chart_uses_dark_theme():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_text_labels_show_depth_percent():
    df = _synthetic_equity_with_known_dds()
    fig = _chart_top_drawdowns(df)
    texts = list(fig.data[0].text)
    assert all("%" in t for t in texts)
    assert all(t.startswith("-") for t in texts)  # all are negative depths


def test_chart_filters_below_min_depth():
    # Build equity with one 0.5% and one 5% drawdown.
    equity = [
        100.0, 99.5, 100.0,           # 0.5% DD (below default threshold)
        110.0, 104.5, 110.0,          # 5% DD
    ]
    df = pl.DataFrame({
        "date": _dates(len(equity)),
        "portfolio_value": equity,
        "benchmark_value": equity,
    })
    fig = _chart_top_drawdowns(df, min_depth=0.01)
    assert len(fig.data[0].x) == 1  # only the 5% DD remains
