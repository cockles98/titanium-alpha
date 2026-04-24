"""Tests for dashboard Phase 7 — turnover and transaction cost drag."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.app import (
    _chart_turnover_and_costs,
    _compute_turnover_per_rebalance,
    _reconstruct_gross_equity,
)


def _make_weights_long(
    events: list[tuple[date, float, float]],
    tickers: tuple[str, ...] = ("AAA", "BBB"),
) -> pl.DataFrame:
    """Synthesize a long-format weights parquet.

    Each ``events`` entry is ``(date, turnover, dollar_cost)`` replicated
    across the given tickers, matching the layout produced by the walk-
    forward backtester.
    """
    rows: list[dict] = []
    for d, turnover, cost in events:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "weight": 1.0 / len(tickers),
                "turnover": turnover,
                "costs": cost,
                "retrained": False,
            })
    return pl.DataFrame(rows)


def _make_equity(values: list[float], start: date = date(2024, 1, 1)) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": values,
        "benchmark_value": [v for v in values],
    })


# ---------------------------------------------------------------------------
# _compute_turnover_per_rebalance
# ---------------------------------------------------------------------------


def test_turnover_collapses_to_one_row_per_date():
    events = [
        (date(2024, 1, 2), 0.30, 100.0),
        (date(2024, 1, 9), 0.15, 50.0),
    ]
    w = _make_weights_long(events)
    out = _compute_turnover_per_rebalance(w)
    assert out.height == 2
    assert out["turnover"].to_list() == [0.30, 0.15]
    assert out["costs"].to_list() == [100.0, 50.0]


def test_turnover_filters_zero_events():
    events = [
        (date(2024, 1, 2), 0.30, 100.0),
        (date(2024, 1, 3), 0.0, 0.0),  # non-rebalance day
        (date(2024, 1, 9), 0.20, 70.0),
    ]
    w = _make_weights_long(events)
    out = _compute_turnover_per_rebalance(w)
    assert out.height == 2
    assert date(2024, 1, 3) not in out["date"].to_list()


def test_turnover_sorted_by_date():
    events = [
        (date(2024, 3, 1), 0.10, 20.0),
        (date(2024, 1, 1), 0.50, 100.0),
        (date(2024, 2, 1), 0.30, 60.0),
    ]
    w = _make_weights_long(events)
    out = _compute_turnover_per_rebalance(w)
    assert out["date"].to_list() == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]


# ---------------------------------------------------------------------------
# _reconstruct_gross_equity
# ---------------------------------------------------------------------------


def test_gross_equals_net_when_no_costs():
    # 5 days of net equity, no rebalance events → gross == net everywhere.
    eq = _make_equity([100.0, 101.0, 102.0, 103.0, 104.0])
    w = _make_weights_long([])
    out = _reconstruct_gross_equity(eq, w)
    assert "portfolio_value_gross" in out.columns
    np.testing.assert_allclose(
        out["portfolio_value_gross"].to_numpy(),
        out["portfolio_value"].to_numpy(),
    )


def test_gross_geq_net_always():
    eq = _make_equity([100.0, 99.5, 100.2, 101.0, 102.0, 102.5])
    events = [
        (date(2024, 1, 3), 0.25, 0.25),   # cost on day 3
        (date(2024, 1, 5), 0.40, 0.40),   # cost on day 5
    ]
    w = _make_weights_long(events)
    out = _reconstruct_gross_equity(eq, w)
    diff = out["portfolio_value_gross"].to_numpy() - out["portfolio_value"].to_numpy()
    assert (diff >= -1e-9).all()
    # Gap must be strictly positive after a cost event has occurred.
    assert diff[-1] > 0.0


def test_gross_reconstruction_matches_analytical_identity():
    # Simple scenario: start 100, 1 rebalance on day 2 with cost=1.0, then hold.
    # net series:
    #   day 0: 100
    #   day 1: 100 (baseline, no drift)
    #   day 2: (100 - 1.0) * 1.05 = 103.95  (5% drift after cost)
    #   day 3: 103.95 (hold)
    # gross series:
    #   day 0: 100
    #   day 1: 100
    #   day 2: 100 * 1.05 = 105.0
    #   day 3: 105.0
    eq = _make_equity([100.0, 100.0, 103.95, 103.95])
    events = [(date(2024, 1, 3), 0.01, 1.0)]  # cost on day index 2
    w = _make_weights_long(events)
    out = _reconstruct_gross_equity(eq, w)
    gross = out["portfolio_value_gross"].to_list()
    # day 0, 1: unchanged; factor kicks in on day 2.
    np.testing.assert_allclose(gross[0], 100.0)
    np.testing.assert_allclose(gross[1], 100.0)
    np.testing.assert_allclose(gross[2], 105.0, rtol=1e-9)
    np.testing.assert_allclose(gross[3], 105.0, rtol=1e-9)


def test_gross_handles_missing_costs_column():
    eq = _make_equity([100.0, 101.0, 102.0])
    w_no_costs = pl.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 2)],
        "ticker": ["AAA", "AAA"],
        "weight": [0.5, 0.5],
    })
    out = _reconstruct_gross_equity(eq, w_no_costs)
    np.testing.assert_allclose(
        out["portfolio_value_gross"].to_numpy(),
        out["portfolio_value"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# _chart_turnover_and_costs
# ---------------------------------------------------------------------------


def test_chart_returns_none_when_weights_lack_required_columns():
    eq = _make_equity([100.0, 101.0])
    w = pl.DataFrame({"date": [date(2024, 1, 1)], "ticker": ["AAA"], "weight": [1.0]})
    assert _chart_turnover_and_costs(eq, w) is None


def test_chart_returns_none_when_no_rebalance_events():
    eq = _make_equity([100.0, 101.0, 102.0])
    w = _make_weights_long([
        (date(2024, 1, 1), 0.0, 0.0),
        (date(2024, 1, 2), 0.0, 0.0),
    ])
    assert _chart_turnover_and_costs(eq, w) is None


def test_chart_builds_two_panels_with_three_traces():
    eq = _make_equity([100.0 + 0.1 * i for i in range(20)])
    events = [
        (date(2024, 1, 3), 0.20, 0.20),
        (date(2024, 1, 10), 0.15, 0.15),
        (date(2024, 1, 17), 0.10, 0.10),
    ]
    w = _make_weights_long(events)
    fig = _chart_turnover_and_costs(eq, w)
    assert isinstance(fig, go.Figure)
    # Row 1: 1 bar trace for turnover; Row 2: 2 line traces (gross, net).
    assert len(fig.data) == 3
    kinds = sorted({t.type for t in fig.data})
    assert "bar" in kinds
    assert "scatter" in kinds


def test_chart_uses_dark_theme():
    eq = _make_equity([100.0 + 0.1 * i for i in range(10)])
    w = _make_weights_long([(date(2024, 1, 3), 0.2, 0.2)])
    fig = _chart_turnover_and_costs(eq, w)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_title_shows_total_dollar_cost():
    eq = _make_equity([100.0 + 0.1 * i for i in range(10)])
    events = [
        (date(2024, 1, 3), 0.2, 12.34),
        (date(2024, 1, 7), 0.1, 87.66),
    ]
    w = _make_weights_long(events)
    fig = _chart_turnover_and_costs(eq, w)
    title_text = fig.layout.title.text or ""
    # Total should be $100.
    assert "$100" in title_text


def test_chart_subtitle_reports_bps_per_year():
    eq = _make_equity([100.0 + 0.1 * i for i in range(40)])
    events = [(date(2024, 1, 3), 0.2, 0.5), (date(2024, 1, 17), 0.1, 0.3)]
    w = _make_weights_long(events)
    fig = _chart_turnover_and_costs(eq, w)
    subtitles = [a.text for a in (fig.layout.annotations or [])]
    joined = " | ".join(subtitles)
    assert "bps/year" in joined
    assert "drag" in joined.lower()


def test_chart_bar_values_are_in_percent():
    eq = _make_equity([100.0 + 0.05 * i for i in range(30)])
    events = [(date(2024, 1, 3), 0.25, 0.25)]  # turnover = 25%
    w = _make_weights_long(events)
    fig = _chart_turnover_and_costs(eq, w)
    bar_traces = [t for t in fig.data if t.type == "bar"]
    assert bar_traces
    ys = list(bar_traces[0].y)
    assert ys == [25.0]
