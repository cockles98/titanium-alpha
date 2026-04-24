"""Tests for dashboard Phase 5 — up/down capture ratios."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_capture_ratios,
    _compute_capture_ratios,
)


def _equity_from_monthly_returns(
    port_monthly: list[float],
    spy_monthly: list[float],
    start: date = date(2020, 1, 1),
    start_value: float = 1_000_000.0,
) -> pl.DataFrame:
    """Build a daily-frequency DataFrame with one row per calendar month.

    The row at the first day of calendar month ``k+1`` carries the compounded
    effect of ``port_monthly[k]`` / ``spy_monthly[k]`` relative to month ``k``'s
    anchor row. Because ``_compute_monthly_returns`` computes a single daily
    return in month ``k+1`` from the previous row and assigns it to the month
    ``k+1`` bucket, the helper's monthly output equals ``(port_monthly,
    spy_monthly)`` exactly.
    """
    assert len(port_monthly) == len(spy_monthly)
    rows: list[tuple[date, float, float]] = [(start, start_value, start_value)]
    y, m = start.year, start.month
    p, s = start_value, start_value
    for pr, sr in zip(port_monthly, spy_monthly):
        m += 1
        if m > 12:
            m = 1
            y += 1
        p = p * (1.0 + pr)
        s = s * (1.0 + sr)
        rows.append((date(y, m, 1), p, s))
    return pl.DataFrame({
        "date": [r[0] for r in rows],
        "portfolio_value": [r[1] for r in rows],
        "benchmark_value": [r[2] for r in rows],
    })


# ---------------------------------------------------------------------------
# _compute_capture_ratios
# ---------------------------------------------------------------------------


def test_capture_returns_none_for_insufficient_history():
    # 2 rows → 1 monthly return → below the 2-month minimum.
    df = _equity_from_monthly_returns([0.01], [0.02])
    assert _compute_capture_ratios(df) is None


def test_capture_matches_known_inputs():
    port = [0.03, -0.01, 0.005, -0.015]
    spy = [0.02, -0.02, 0.01, -0.01]
    df = _equity_from_monthly_returns(port, spy)

    res = _compute_capture_ratios(df)
    assert res is not None

    # Up months: indices 0 (+0.02) and 2 (+0.01).
    # port_up_mean = (0.03 + 0.005) / 2 = 0.0175
    # spy_up_mean  = (0.02 + 0.01)  / 2 = 0.015
    expected_up = 0.0175 / 0.015 * 100.0
    assert res["up_capture"] == pytest.approx(expected_up, rel=1e-6)

    # Down months: indices 1 (-0.02) and 3 (-0.01).
    # port_dn_mean = (-0.01 + -0.015) / 2 = -0.0125
    # spy_dn_mean  = (-0.02 + -0.01)  / 2 = -0.015
    expected_dn = (-0.0125) / (-0.015) * 100.0
    assert res["down_capture"] == pytest.approx(expected_dn, rel=1e-6)

    expected_ratio = expected_up / expected_dn * 100.0
    assert res["up_down_ratio"] == pytest.approx(expected_ratio, rel=1e-6)

    assert res["n_total_months"] == 4
    assert res["n_up_months"] == 2
    assert res["n_down_months"] == 2
    # Portfolio positive months: 0.03, 0.005 → 2 of 4 = 50%.
    assert res["positive_month_pct"] == pytest.approx(50.0)
    assert res["benchmark_positive_month_pct"] == pytest.approx(50.0)


def test_capture_perfect_tracker_is_one_hundred_pct():
    port = [0.02, -0.01, 0.03, -0.02, 0.01]
    df = _equity_from_monthly_returns(port, port)
    res = _compute_capture_ratios(df)
    assert res is not None
    assert res["up_capture"] == pytest.approx(100.0, rel=1e-6)
    assert res["down_capture"] == pytest.approx(100.0, rel=1e-6)
    assert res["up_down_ratio"] == pytest.approx(100.0, rel=1e-6)


def test_capture_defensive_portfolio_beats_baseline():
    # Bigger upside and smaller downside than benchmark.
    port = [0.03, -0.005, 0.02, -0.004]
    spy = [0.02, -0.03, 0.015, -0.02]
    df = _equity_from_monthly_returns(port, spy)
    res = _compute_capture_ratios(df)
    assert res is not None
    assert res["up_capture"] > 100.0
    assert res["down_capture"] < 100.0
    assert res["up_down_ratio"] > 100.0


def test_capture_handles_no_down_months():
    port = [0.02, 0.03, 0.01]
    spy = [0.01, 0.015, 0.008]
    df = _equity_from_monthly_returns(port, spy)
    res = _compute_capture_ratios(df)
    assert res is not None
    assert math.isfinite(res["up_capture"])
    assert math.isnan(res["down_capture"])
    assert math.isnan(res["up_down_ratio"])
    assert res["n_down_months"] == 0
    assert res["positive_month_pct"] == pytest.approx(100.0)


def test_capture_handles_no_up_months():
    port = [-0.01, -0.02, -0.005]
    spy = [-0.015, -0.03, -0.01]
    df = _equity_from_monthly_returns(port, spy)
    res = _compute_capture_ratios(df)
    assert res is not None
    assert math.isnan(res["up_capture"])
    assert math.isfinite(res["down_capture"])
    assert math.isnan(res["up_down_ratio"])
    assert res["n_up_months"] == 0
    assert res["positive_month_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _chart_capture_ratios
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_insufficient_data():
    df = _equity_from_monthly_returns([0.01], [0.02])
    assert _chart_capture_ratios(df) is None


def test_chart_has_four_bars_with_expected_labels():
    rng = np.random.default_rng(0)
    port = rng.normal(0.0, 0.03, size=36).tolist()
    spy = rng.normal(0.0, 0.03, size=36).tolist()
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert list(bar.x) == [
        "Up-Capture",
        "Down-Capture",
        "Up/Down Ratio",
        "Positive Month %",
    ]
    assert len(bar.y) == 4


def test_chart_has_baseline_hline_at_100():
    rng = np.random.default_rng(1)
    port = rng.normal(0.0, 0.03, size=24).tolist()
    spy = rng.normal(0.0, 0.03, size=24).tolist()
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    shapes = list(fig.layout.shapes or [])
    ys = {float(sh.y0) for sh in shapes}
    assert 100.0 in ys


def test_chart_uses_dark_theme():
    rng = np.random.default_rng(2)
    port = rng.normal(0.0, 0.03, size=24).tolist()
    spy = rng.normal(0.0, 0.03, size=24).tolist()
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_text_labels_use_percent():
    port = [0.02, -0.01]
    spy = [0.015, -0.02]
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    texts = [str(t) for t in fig.data[0].text]
    assert all(("%" in t) or (t == "n/a") for t in texts)


def test_chart_title_reports_month_counts():
    rng = np.random.default_rng(3)
    port = rng.normal(0.0, 0.03, size=24).tolist()
    spy = rng.normal(0.0, 0.03, size=24).tolist()
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    title_text = (fig.layout.title.text or "").lower()
    assert "month" in title_text
    assert "up" in title_text
    assert "down" in title_text


def test_chart_colors_reflect_favorability():
    # Portfolio captures much more upside and much less downside than SPY.
    port = [0.04, -0.003, 0.02, -0.004]
    spy = [0.02, -0.02, 0.01, -0.015]
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    colors = list(fig.data[0].marker.color)
    green = "#43A047"
    # up-capture > 100 → favorable; down-capture < 100 → favorable;
    # ratio > 100 → favorable.
    assert colors[0] == green
    assert colors[1] == green
    assert colors[2] == green


def test_chart_nan_bar_uses_neutral_color():
    # No down months → down_capture is NaN → neutral grey bar.
    port = [0.02, 0.03, 0.01]
    spy = [0.01, 0.015, 0.008]
    df = _equity_from_monthly_returns(port, spy)
    fig = _chart_capture_ratios(df)
    colors = list(fig.data[0].marker.color)
    texts = list(fig.data[0].text)
    # Index 1 = Down-Capture. With all-up history it is n/a.
    assert texts[1] == "n/a"
    assert colors[1] == "#888888"
