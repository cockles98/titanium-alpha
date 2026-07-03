"""Tests for dashboard Phase 18 — vol targeting trajectory."""

from __future__ import annotations

import math
from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import _chart_vol_targeting


def _equity(
    dates: list[date],
    leverages: list[float],
    realized_vols: list[float | None],
) -> pl.DataFrame:
    """Build a synthetic equity_curve-like DataFrame."""
    n = len(dates)
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": [1.0e6] * n,
        "benchmark_value": [1.0e6] * n,
        "leverage": leverages,
        "realized_vol_63d": realized_vols,
    })


def _date_range(n: int, start: date = date(2020, 1, 2)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Missing-data behaviour
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_missing_columns():
    """Old parquets without Phase 18 columns must degrade gracefully."""
    df = pl.DataFrame({
        "date": _date_range(5),
        "portfolio_value": [1.0e6] * 5,
        "benchmark_value": [1.0e6] * 5,
    })
    assert _chart_vol_targeting(df) is None


def test_chart_returns_none_for_none_input():
    assert _chart_vol_targeting(None) is None


def test_chart_returns_none_for_empty_dataframe():
    df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "leverage": pl.Float64,
            "realized_vol_63d": pl.Float64,
        }
    )
    assert _chart_vol_targeting(df) is None


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_chart_returns_figure_with_two_subplots():
    df = _equity(
        _date_range(80),
        leverages=[0.85] * 80,
        realized_vols=[None] * 19 + [0.12] * 61,
    )
    fig = _chart_vol_targeting(df)
    assert isinstance(fig, go.Figure)
    # Each row holds one Scatter trace = 2 traces total.
    assert len(fig.data) == 2
    # Axes: x1, x2, y1, y2 (Plotly subplot convention).
    assert any(k.startswith("xaxis") for k in fig.layout)
    assert any(k.startswith("yaxis") for k in fig.layout)


def test_chart_has_three_reference_hlines():
    """min_leverage + max_leverage + target_vol must each get an hline."""
    df = _equity(
        _date_range(80),
        leverages=[0.7] * 80,
        realized_vols=[None] * 19 + [0.10] * 61,
    )
    fig = _chart_vol_targeting(
        df, target_vol=0.10, min_leverage=0.5, max_leverage=1.0,
    )
    assert fig is not None
    # add_hline emits a layout shape per call.
    shapes = list(fig.layout.shapes or [])
    assert len(shapes) >= 3


def test_chart_uses_dark_theme():
    df = _equity(_date_range(10), [1.0] * 10, [None] * 10)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_top_subplot_is_step_function():
    """Leverage line must use horizontal-vertical step shape."""
    df = _equity(_date_range(20), [0.85] * 20, [None] * 20)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    leverage_trace = fig.data[0]
    assert leverage_trace.line.shape == "hv"


def test_chart_realized_vol_does_not_connect_gaps():
    """During warmup, realized_vol is None; the line must break, not interpolate."""
    df = _equity(
        _date_range(20),
        leverages=[1.0] * 20,
        realized_vols=[None] * 5 + [0.10] * 15,
    )
    fig = _chart_vol_targeting(df)
    assert fig is not None
    rv_trace = fig.data[1]
    assert rv_trace.connectgaps is False


def test_chart_y_axis_is_percentage_format_for_vol_subplot():
    df = _equity(_date_range(20), [1.0] * 20, [0.10] * 20)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    # yaxis2 corresponds to row 2 (vol panel).
    assert fig.layout.yaxis2.tickformat == ".0%"


def test_chart_leverage_subplot_y_range_includes_clamps():
    df = _equity(_date_range(20), [0.7] * 20, [None] * 20)
    fig = _chart_vol_targeting(
        df, min_leverage=0.5, max_leverage=1.0,
    )
    assert fig is not None
    rng = list(fig.layout.yaxis.range)
    # Must extend at least to the clamps (with a small visual margin).
    assert rng[0] <= 0.5
    assert rng[1] >= 1.0


# ---------------------------------------------------------------------------
# Saturation reporting
# ---------------------------------------------------------------------------


def test_chart_saturation_pct_reflects_max_clamp():
    """If 50% of days hit the max clamp, the title must report it."""
    leverages = [1.0] * 50 + [0.7] * 50
    df = _equity(_date_range(100), leverages, [None] * 100)
    fig = _chart_vol_targeting(
        df, target_vol=0.10, min_leverage=0.5, max_leverage=1.0,
    )
    assert fig is not None
    titles = [a.text for a in fig.layout.annotations if a.text]
    joined = " | ".join(titles)
    assert "50% of days at max" in joined


def test_chart_saturation_pct_reflects_min_clamp():
    leverages = [0.5] * 30 + [0.8] * 70
    df = _equity(_date_range(100), leverages, [None] * 100)
    fig = _chart_vol_targeting(
        df, target_vol=0.10, min_leverage=0.5, max_leverage=1.0,
    )
    assert fig is not None
    titles = [a.text for a in fig.layout.annotations if a.text]
    joined = " | ".join(titles)
    assert "30% at min" in joined


def test_chart_target_vol_in_subtitle():
    df = _equity(_date_range(20), [1.0] * 20, [0.10] * 20)
    fig = _chart_vol_targeting(df, target_vol=0.10)
    assert fig is not None
    titles = [a.text for a in fig.layout.annotations if a.text]
    joined = " | ".join(titles)
    assert "Target (10%)" in joined or "target = 10%" in joined


def test_chart_handles_all_null_realized_vol():
    """Pure warmup window: realized_vol all null, chart must still render."""
    df = _equity(_date_range(30), [1.0] * 30, [None] * 30)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    rv_trace = fig.data[1]
    # All y values null, but trace still constructed.
    assert len(rv_trace.x) == 30


def test_chart_realized_vol_values_passed_through_unmodified():
    """Plotly sees the same numeric vol values we put in."""
    rv = [None] * 10 + [0.10, 0.11, 0.12, 0.09, 0.13]
    df = _equity(_date_range(15), [1.0] * 15, rv)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    ys = list(fig.data[1].y)
    # Tail values match exactly.
    for v_in, v_out in zip(rv[-5:], ys[-5:]):
        assert v_out == pytest.approx(v_in)
    # Warmup values come back as None / NaN equivalents.
    for v_out in ys[:10]:
        assert v_out is None or (isinstance(v_out, float) and math.isnan(v_out))


def test_chart_main_title_mentions_vol_targeting():
    df = _equity(_date_range(20), [1.0] * 20, [0.10] * 20)
    fig = _chart_vol_targeting(df)
    assert fig is not None
    title = fig.layout.title.text or ""
    assert "Vol Targeting" in title or "Risk Management" in title
