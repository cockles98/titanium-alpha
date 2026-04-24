"""Tests for dashboard Phase 1 — monthly returns calendar heatmap."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_calendar_heatmap,
    _compute_annual_returns,
    _compute_monthly_returns,
)


def _synthetic_equity(
    start: date,
    n_days: int,
    daily_port: float = 0.001,
    daily_spy: float = 0.0005,
    start_value: float = 1_000_000.0,
) -> pl.DataFrame:
    """Build a daily equity frame with constant per-day geometric growth.

    This makes returns deterministic so tests can assert numerical values.
    """
    dates = [start + timedelta(days=i) for i in range(n_days)]
    port = [start_value * (1.0 + daily_port) ** i for i in range(n_days)]
    spy = [start_value * (1.0 + daily_spy) ** i for i in range(n_days)]
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": port,
        "benchmark_value": spy,
    })


# ---------------------------------------------------------------------------
# _compute_monthly_returns
# ---------------------------------------------------------------------------


def test_monthly_returns_basic_shape():
    df = _synthetic_equity(date(2024, 1, 1), n_days=90)
    out = _compute_monthly_returns(df)
    assert set(out.columns) == {"year", "month", "port_ret", "spy_ret"}
    # Jan, Feb, Mar — 3 months covered.
    assert out.height == 3
    assert out["month"].to_list() == [1, 2, 3]
    assert out["year"].to_list() == [2024, 2024, 2024]


def test_monthly_returns_compounds_daily_correctly():
    # 20 days of constant +0.5% daily return within a single month.
    df = _synthetic_equity(date(2024, 6, 1), n_days=20, daily_port=0.005, daily_spy=0.0)
    out = _compute_monthly_returns(df)
    june = out.filter((pl.col("year") == 2024) & (pl.col("month") == 6))
    assert june.height == 1
    # 19 daily returns of +0.5% compounded (first day has no prior → no return).
    expected = (1.005 ** 19) - 1.0
    assert june["port_ret"][0] == pytest.approx(expected, rel=1e-6)
    assert june["spy_ret"][0] == pytest.approx(0.0, abs=1e-9)


def test_monthly_returns_excludes_uncovered_months():
    """Apr 15 start → Jan, Feb, Mar of the start year must not appear."""
    df = _synthetic_equity(date(2016, 4, 15), n_days=270)  # to ~Jan 2017
    out = _compute_monthly_returns(df)
    months_2016 = sorted(out.filter(pl.col("year") == 2016)["month"].to_list())
    # Q1 2016 uncovered; Apr–Dec covered (Apr present — it has ≥ 1 daily return).
    assert 1 not in months_2016
    assert 2 not in months_2016
    assert 3 not in months_2016
    assert 4 in months_2016
    assert 12 in months_2016


def test_monthly_returns_sorted():
    df = _synthetic_equity(date(2023, 1, 1), n_days=400)
    out = _compute_monthly_returns(df)
    years_months = list(zip(out["year"].to_list(), out["month"].to_list()))
    assert years_months == sorted(years_months)


# ---------------------------------------------------------------------------
# _compute_annual_returns
# ---------------------------------------------------------------------------


def test_annual_returns_compounds_full_year():
    # ~252 trading days of +0.1% daily → expected annual ≈ (1.001**251 - 1).
    df = _synthetic_equity(date(2024, 1, 1), n_days=252, daily_port=0.001)
    out = _compute_annual_returns(df)
    y2024 = out.filter(pl.col("year") == 2024)
    assert y2024.height == 1
    assert y2024["port_ret"][0] == pytest.approx((1.001 ** 251) - 1.0, rel=1e-6)


def test_annual_returns_partial_year_is_within_year_return():
    # 10 days (early Jan), +1% daily → partial 2024 return should be ~1.01**9 - 1.
    df = _synthetic_equity(date(2024, 1, 1), n_days=10, daily_port=0.01)
    out = _compute_annual_returns(df)
    y2024 = out.filter(pl.col("year") == 2024)
    assert y2024.height == 1
    assert y2024["port_ret"][0] == pytest.approx((1.01 ** 9) - 1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# _chart_calendar_heatmap
# ---------------------------------------------------------------------------


def test_heatmap_returns_none_when_insufficient_data():
    # Only 5 days — single month, below the 2-month minimum.
    df = _synthetic_equity(date(2024, 3, 1), n_days=5)
    assert _chart_calendar_heatmap(df) is None


def test_heatmap_returns_figure_with_heatmap_trace():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    trace = fig.data[0]
    assert trace.type == "heatmap"


def test_heatmap_columns_have_annual_last():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    x_labels = list(fig.data[0].x)
    assert x_labels[-1] == "Annual"
    assert x_labels[:12] == [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]


def test_heatmap_rows_have_mean_last_and_recent_year_first():
    df = _synthetic_equity(date(2022, 1, 1), n_days=800)  # ~2 years 2022-2024
    fig = _chart_calendar_heatmap(df)
    y_labels = list(fig.data[0].y)
    assert y_labels[-1] == "Mean"
    years_only = [lbl for lbl in y_labels if lbl != "Mean"]
    # Most recent year in first position (renders at top due to reversed axis).
    assert int(years_only[0]) == max(int(y) for y in years_only)


def test_heatmap_partial_months_are_none_not_zero():
    """Jan-Mar 2016 have no data and must render as None, not 0% (green bias)."""
    df = _synthetic_equity(date(2016, 4, 15), n_days=300)  # spans into 2017
    fig = _chart_calendar_heatmap(df)
    z = fig.data[0].z
    y_labels = list(fig.data[0].y)

    # Locate the 2016 row.
    row_2016_idx = y_labels.index("2016")
    row_2016 = z[row_2016_idx]
    # Columns 0..2 = Jan, Feb, Mar (not covered).
    assert row_2016[0] is None
    assert row_2016[1] is None
    assert row_2016[2] is None
    # Column 3 = April (covered from Apr 15).
    assert row_2016[3] is not None


def test_heatmap_has_zmid_zero_for_divergent_scale():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    trace = fig.data[0]
    assert trace.zmid == 0.0
    assert trace.zmin == -0.15
    assert trace.zmax == 0.15
    assert trace.colorscale is not None  # RdYlGn named scale resolves to tuple


def test_heatmap_cell_text_format():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400, daily_port=0.01)
    fig = _chart_calendar_heatmap(df)
    text = fig.data[0].text
    # At least one non-empty cell with a '%' marker and explicit sign.
    flat = [t for row in text for t in row]
    non_empty = [t for t in flat if t]
    assert non_empty, "heatmap should have at least one populated cell"
    assert any("%" in t for t in non_empty)
    assert any(t.startswith("+") or "+" in t or "-" in t for t in non_empty)


def test_heatmap_has_divider_shape_between_dec_and_annual():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    shapes = list(fig.layout.shapes or [])
    assert len(shapes) >= 1
    assert any(abs(s.x0 - 11.5) < 1e-6 for s in shapes)


def test_heatmap_y_axis_reversed_so_recent_year_on_top():
    df = _synthetic_equity(date(2020, 1, 1), n_days=2000)
    fig = _chart_calendar_heatmap(df)
    assert fig.layout.yaxis.autorange == "reversed"


def test_heatmap_uses_dark_theme():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_heatmap_hover_mentions_year_month_return():
    df = _synthetic_equity(date(2024, 1, 1), n_days=400)
    fig = _chart_calendar_heatmap(df)
    hover = fig.data[0].hovertemplate
    assert "Year" in hover
    assert "Month" in hover
    assert "Return" in hover


def test_heatmap_on_real_shape_without_errors():
    """Smoke test on a multi-year synthetic run that mimics the benchmark."""
    # 3650 calendar days ≈ 10 years starting 2016-04-19 → spans 2016..2026.
    df = _synthetic_equity(date(2016, 4, 19), n_days=3650)
    fig = _chart_calendar_heatmap(df)
    assert fig is not None
    y_labels = list(fig.data[0].y)
    years = {int(lbl) for lbl in y_labels if lbl != "Mean"}
    # Must span at least 2016..2025 (partial 2026 may also be present).
    assert {2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025} <= years
    assert y_labels[-1] == "Mean"
